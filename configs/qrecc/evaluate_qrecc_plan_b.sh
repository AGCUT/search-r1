#!/bin/bash
#
# Evaluation script for QReCC Plan B using verl framework
#
# This script runs evaluation in val_only mode, which uses the full
# PPO infrastructure including retrieval server integration.
#
# Usage:
#   bash configs/qrecc/evaluate_qrecc_plan_b.sh
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
export CHECKPOINT_PATH="verl_checkpoints/qrecc-plan-b-ppo-qwen3-4b-em/checkpoint_1000"

# Or use base model for baseline evaluation
# export CHECKPOINT_PATH="Qwen/Qwen3-4B"

# ============================================================================
# Data Configuration
# ============================================================================
export DATA_DIR='data/qrecc_plan_b'

# ============================================================================
# vLLM Configuration
# ============================================================================
export VLLM_ATTENTION_BACKEND=XFORMERS

# ============================================================================
# Launch Evaluation (val_only mode)
# ============================================================================
echo "============================================================================"
echo "Starting QReCC Plan B Evaluation"
echo "============================================================================"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Data Directory: $DATA_DIR"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "============================================================================"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
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
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    max_turns=2 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3

echo "============================================================================"
echo "Evaluation Complete!"
echo "============================================================================"