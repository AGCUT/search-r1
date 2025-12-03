#!/bin/bash
#
# Training script for QReCC Plan B (Using Ground-truth Rewritten Queries)
#
# Plan B Approach:
# - Uses rewritten standalone queries from QReCC dataset
# - Simplified baseline without conversation history
# - Model learns reasoning and search with clear, standalone questions
#
# Usage:
#   # Fresh start
#   bash configs/qrecc/train_qrecc_ppo_plan_b.sh
#
#   # Resume from checkpoint (set RESUME_STEP)
#   RESUME_STEP=100 bash configs/qrecc/train_qrecc_ppo_plan_b.sh
#
# Requirements:
# 1. Processed dataset: data/qrecc_plan_b/train.parquet and test.parquet
# 2. Retrieval server running: http://127.0.0.1:8000/retrieve
# 3. Local model downloaded (or HuggingFace access)

# ============================================================================
# GPU Configuration
# ============================================================================
# Using 2x A800 GPUs for training (GPU 4 for E5 retrieval server)
export CUDA_VISIBLE_DEVICES=6,7

# ============================================================================
# Data Configuration
# ============================================================================
export DATA_DIR='data/qrecc_plan_b'

# ============================================================================
# Reward Function Configuration
# ============================================================================
# Options: em (exact match), subem (substring match), f1, hybrid
REWARD_FN=${REWARD_FN:-"hybrid"}
echo "Using reward function: $REWARD_FN"

# ============================================================================
# Model / Experiment Configuration
# ============================================================================
# Primary model: Qwen2.5-3B-Instruct (ChatML 格式)
export BASE_MODEL='/usr/yuque/guo/models/qwen2.5-3b-instruct'
# Create unique experiment name per run to avoid collisions
RUN_TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
DEFAULT_EXPERIMENT_NAME="qrecc-plan-b-ppo-qwen2.5-3b-instruct-${REWARD_FN}-${RUN_TIMESTAMP}"
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-$DEFAULT_EXPERIMENT_NAME}
# Show final experiment name
echo "Experiment Name: $EXPERIMENT_NAME"

# ============================================================================
# Resume Configuration (set RESUME_STEP to resume from checkpoint)
# ============================================================================
# Example: RESUME_STEP=100 bash configs/qrecc/train_qrecc_ppo_plan_b.sh
RESUME_STEP=${RESUME_STEP:-""}
CHECKPOINT_DIR="verl_checkpoints/$EXPERIMENT_NAME"

if [ -n "$RESUME_STEP" ]; then
    RESUME_PATH="$CHECKPOINT_DIR/global_step_${RESUME_STEP}"
    if [ -d "$RESUME_PATH" ]; then
        echo "============================================================================"
        echo "RESUMING from checkpoint: $RESUME_PATH"
        echo "============================================================================"
        RESUME_ARGS="actor_rollout_ref.model.path=$RESUME_PATH/actor critic.model.path=$RESUME_PATH/critic +trainer.resume_step=$RESUME_STEP"
    else
        echo "ERROR: Checkpoint not found: $RESUME_PATH"
        echo "Available checkpoints:"
        ls -la $CHECKPOINT_DIR/ 2>/dev/null || echo "No checkpoints found"
        exit 1
    fi
else
    RESUME_ARGS=""
fi

# Alternative models:
# export BASE_MODEL='/usr/yuque/guo/models/qwen2.5-3b'  # Base 版本（不推荐）
# export EXPERIMENT_NAME='qrecc-plan-b-ppo-qwen2.5-3b-em'

# export BASE_MODEL='meta-llama/Llama-3.2-3B'
# export EXPERIMENT_NAME='qrecc-plan-b-ppo-llama3.2-3b-em'

# export BASE_MODEL='Qwen/Qwen2.5-7B'
# export EXPERIMENT_NAME='qrecc-plan-b-ppo-qwen2.5-7b-em'

# ============================================================================
# Logging Configuration
# ============================================================================
WAND_PROJECT='Search-R1-QReCC'

# ============================================================================
# vLLM Configuration (for inference during rollout)
# ============================================================================
# Use XFORMERS backend for better compatibility
export VLLM_ATTENTION_BACKEND=XFORMERS

# ============================================================================
# Launch PPO Training
# ============================================================================
echo "============================================================================"
echo "Starting QReCC Plan B Training"
echo "============================================================================"
echo "Model: $BASE_MODEL"
echo "Experiment: $EXPERIMENT_NAME"
echo "Data Directory: $DATA_DIR"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "============================================================================"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=128 \
    data.val_batch_size=64 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=800 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.dtype=bfloat16 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.top_k=50 \
    actor_rollout_ref.actor.state_masking=true \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.015 \
    critic.model.path=$BASE_MODEL \
    critic.model.dtype=bfloat16 \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size=4 \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.grad_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    +algorithm.reward_fn=$REWARD_FN \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=500 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=2 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    $RESUME_ARGS \
    2>&1 | tee -a logs/$EXPERIMENT_NAME.log

echo "============================================================================"
echo "Training Complete!"
echo "============================================================================"
echo "Checkpoints saved to: verl_checkpoints/$EXPERIMENT_NAME"
echo "Logs saved to: logs/$EXPERIMENT_NAME.log"
echo "============================================================================"
