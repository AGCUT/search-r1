#!/bin/bash
#
# Evaluation script for QReCC using veRL framework with F1 Score
# This script supports full retrieval and multiple evaluation metrics
#
# Usage:
#   bash configs/qrecc/evaluate_qrecc_with_f1.sh
#
# Before running:
# 1. Set CHECKPOINT_PATH to your trained model checkpoint
# 2. Ensure retrieval server is running: http://127.0.0.1:8000/retrieve

# ============================================================================
# Configuration
# ============================================================================
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ============================================================================
# IMPORTANT: Set your checkpoint path here
# ============================================================================
export CHECKPOINT_PATH="/usr/yuque/guo/searchr1/verl_checkpoints/nq_hotpotqa_train-search-r1-ppo-qwen2.5-3b-it-bm25-em/actor/global_step_200"

# Or use base model for baseline evaluation
# export CHECKPOINT_PATH="Qwen/Qwen2.5-3B"

# ============================================================================
# Evaluation Metric Configuration
# ============================================================================
# Options:
#   'em'      - Exact Match (strict, 0 or 1)
#   'subem'   - Substring Exact Match (more lenient, 0 or 1)
#   'f1'      - Token-level F1 score (0.0 to 1.0, best for long answers)
#   'em_f1'   - Combined EM + F1 scoring
#   'hybrid'  - Format-aware F1 (experimental)
#
# Recommended for QReCC: 'f1' (because answers are often long)
export REWARD_FN='f1'

# ============================================================================
# Data Configuration
# ============================================================================
export DATA_DIR='data/qrecc_raw'

# For QReCC raw data, we need to process it first
# If you already have processed qrecc_plan_b data, use:
# export DATA_DIR='data/qrecc_plan_b'

# ============================================================================
# vLLM Configuration
# ============================================================================
export VLLM_ATTENTION_BACKEND=XFORMERS

# ============================================================================
# Output Configuration
# ============================================================================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export OUTPUT_DIR="results/qrecc_eval_${REWARD_FN}_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

# ============================================================================
# Launch Evaluation (val_only mode)
# ============================================================================
echo "============================================================================"
echo "Starting QReCC Evaluation with ${REWARD_FN} metric"
echo "============================================================================"
echo "Checkpoint:     $CHECKPOINT_PATH"
echo "Data Directory: $DATA_DIR"
echo "Reward Fn:      $REWARD_FN"
echo "GPUs:           $CUDA_VISIBLE_DEVICES"
echo "Output:         $OUTPUT_DIR"
echo "============================================================================"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=2000 \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    +algorithm.reward_fn=$REWARD_FN \
    actor_rollout_ref.model.path=$CHECKPOINT_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.015 \
    critic.model.path=$CHECKPOINT_PATH \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size=8 \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.grad_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    trainer.critic_warmup=0 \
    trainer.logger=[] \
    +trainer.val_only=true \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    max_turns=2 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee $OUTPUT_DIR/eval.log

echo "============================================================================"
echo "Evaluation Complete!"
echo "============================================================================"
echo "Output saved to: $OUTPUT_DIR"
echo "Log file: $OUTPUT_DIR/eval.log"
echo ""
echo "To view results:"
echo "  tail -100 $OUTPUT_DIR/eval.log"
echo "============================================================================"
