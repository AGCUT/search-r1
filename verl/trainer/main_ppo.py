# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em, qrecc_em, qrecc_em_v2
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np

def _select_rm_score_fn(data_source, reward_fn_type='em'):
    """
    Select reward scoring function based on data source and reward function type.

    Args:
        data_source: Dataset name (e.g., 'nq', 'qrecc_plan_b')
        reward_fn_type: Reward function type
            - 'em': Exact Match (strict matching, 0 or 1)
            - 'subem': Substring Exact Match (more lenient, 0 or 1)
            - 'f1': Token-level F1 score (partial credit, 0 to 1)
            - 'em_f1': Combined EM + F1 (full score for EM, partial for F1)

    Returns:
        Scoring function
    """
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em
    elif data_source in ['qrecc_plan_b', 'qrecc_plan_a', 'qrecc', 'mini_qrecc']:
        if reward_fn_type == 'subem':
            print(f"[RewardFn] Using SubEM (substring match) for {data_source}")
            return qrecc_em.compute_score_subem
        elif reward_fn_type == 'f1':
            print(f"[RewardFn] Using F1 (token-level F1 score) for {data_source}")
            return qrecc_em.compute_score_f1
        elif reward_fn_type == 'em_f1':
            print(f"[RewardFn] Using EM+F1 (combined scoring) for {data_source}")
            return qrecc_em.compute_score_em_f1
        elif reward_fn_type == 'hybrid':
            print(f"[RewardFn] Using Hybrid (format-aware F1) for {data_source}")
            return qrecc_em_v2.compute_score_hybrid
        else:
            print(f"[RewardFn] Using EM (exact match) for {data_source}")
            return qrecc_em.compute_score_em
    else:
        # Fallback: use qrecc_em for unknown data sources
        print(f"[Warning] Unknown data_source: {data_source}, using qrecc_em.compute_score_em")
        return qrecc_em.compute_score_em


class RewardManager():
    """The reward manager.

    Supports multiple reward function types:
    - 'em': Exact Match (strict, default) - 0 or 1
    - 'subem': Substring Exact Match (more lenient) - 0 or 1
    - 'f1': Token-level F1 score (partial credit) - 0 to 1
    - 'em_f1': Combined EM + F1 (full for EM, partial for F1) - 0 to 1
    - 'hybrid': Format-aware F1 scoring with tiered rewards
    """

    def __init__(self, tokenizer, num_examine, format_score=0., reward_fn_type='em') -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        self.reward_fn_type = reward_fn_type
        print(f"[RewardManager] Initialized with reward_fn_type={reward_fn_type}")

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        all_scores = []

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score with reward function type
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source, self.reward_fn_type)

            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=self.format_score)

            reward_tensor[i, valid_response_length - 1] = score
            all_scores.append(score)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        # Print reward statistics for each batch
        if len(all_scores) > 0:
            print(f"[Reward] mean: {np.mean(all_scores):.4f}, max: {np.max(all_scores):.4f}, min: {np.min(all_scores):.4f}, std: {np.std(all_scores):.4f}, count: {len(all_scores)}")

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    # Get reward function type from config (default: 'em')
    reward_fn_type = getattr(config.algorithm, 'reward_fn', 'em')
    print(f"[Config] Using reward function type: {reward_fn_type}")

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, reward_fn_type=reward_fn_type)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, reward_fn_type=reward_fn_type)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
