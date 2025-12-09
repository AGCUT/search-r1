#!/bin/bash
#
# Compare base model vs trained model on QReCC with F1 Score
# This script uses the full veRL framework with retrieval support
#
# Usage:
#   bash configs/qrecc/compare_base_vs_trained_f1.sh
#
# Before running:
# 1. Start retrieval server: bash retrieval_launch.sh
# 2. Verify server is running: curl http://127.0.0.1:8000/retrieve

# ============================================================================
# Configuration
# ============================================================================
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Base model
export BASE_MODEL="Qwen/Qwen2.5-3B"

# Trained checkpoint
export TRAINED_CHECKPOINT="/usr/yuque/guo/searchr1/verl_checkpoints/nq_hotpotqa_train-search-r1-ppo-qwen2.5-3b-it-bm25-em/actor/global_step_200"

# Data directory
export DATA_DIR='data/qrecc_raw'

# Evaluation metric
# Options: 'em', 'f1', 'subem', 'em_f1', 'hybrid'
# Recommended for QReCC: 'f1'
export REWARD_FN='f1'

# vLLM config
export VLLM_ATTENTION_BACKEND=XFORMERS

# Output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export OUTPUT_BASE_DIR="results/qrecc_comparison_${REWARD_FN}_${TIMESTAMP}"
mkdir -p $OUTPUT_BASE_DIR

# ============================================================================
# Helper function to run evaluation
# ============================================================================
run_evaluation() {
    local MODEL_PATH=$1
    local MODEL_NAME=$2
    local OUTPUT_DIR=$3

    echo ""
    echo "========================================================================"
    echo "Evaluating: $MODEL_NAME"
    echo "========================================================================"
    echo "Model Path: $MODEL_PATH"
    echo "Output Dir: $OUTPUT_DIR"
    echo "Metric:     $REWARD_FN"
    echo "========================================================================"
    echo ""

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
        actor_rollout_ref.model.path=$MODEL_PATH \
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
        critic.model.path=$MODEL_PATH \
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

    echo ""
    echo "✓ $MODEL_NAME evaluation complete"
    echo "  Results saved to: $OUTPUT_DIR"
    echo ""
}

# ============================================================================
# Main Execution
# ============================================================================
echo "============================================================================"
echo "QReCC Model Comparison - Base vs Trained"
echo "============================================================================"
echo "Base Model:         $BASE_MODEL"
echo "Trained Checkpoint: $TRAINED_CHECKPOINT"
echo "Data Directory:     $DATA_DIR"
echo "Evaluation Metric:  $REWARD_FN"
echo "Output Directory:   $OUTPUT_BASE_DIR"
echo "GPUs:               $CUDA_VISIBLE_DEVICES"
echo "============================================================================"
echo ""
echo "This will run TWO evaluations:"
echo "  1. Base Model (Qwen2.5-3B)"
echo "  2. Trained Model (your checkpoint)"
echo ""
echo "⚠️  Make sure retrieval server is running at http://127.0.0.1:8000/retrieve"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Evaluate base model
run_evaluation "$BASE_MODEL" "Base Model" "$OUTPUT_BASE_DIR/base_model"

# Free GPU memory
echo "Waiting 30 seconds to free GPU memory..."
sleep 30

# Evaluate trained model
run_evaluation "$TRAINED_CHECKPOINT" "Trained Model" "$OUTPUT_BASE_DIR/trained_model"

# ============================================================================
# Extract and Compare Results
# ============================================================================
echo ""
echo "============================================================================"
echo "COMPARISON REPORT"
echo "============================================================================"
echo ""

# Extract scores from logs
BASE_SCORE=$(grep -oP "(?<=val/test_score/)[^:]+:\s*\K\d+\.\d+" "$OUTPUT_BASE_DIR/base_model/eval.log" | tail -1)
TRAINED_SCORE=$(grep -oP "(?<=val/test_score/)[^:]+:\s*\K\d+\.\d+" "$OUTPUT_BASE_DIR/trained_model/eval.log" | tail -1)

# Extract search statistics
BASE_SEARCH=$(grep -oP "(?<=env/number_of_valid_search:\s)\d+\.\d+" "$OUTPUT_BASE_DIR/base_model/eval.log" | tail -1)
TRAINED_SEARCH=$(grep -oP "(?<=env/number_of_valid_search:\s)\d+\.\d+" "$OUTPUT_BASE_DIR/trained_model/eval.log" | tail -1)

# Extract generation time (if available)
BASE_TIME=$(grep -oP "(?<=timing/gen:\s)\d+\.\d+" "$OUTPUT_BASE_DIR/base_model/eval.log" | tail -1)
TRAINED_TIME=$(grep -oP "(?<=timing/gen:\s)\d+\.\d+" "$OUTPUT_BASE_DIR/trained_model/eval.log" | tail -1)

# Fallback to average_score if test_score not found
if [ -z "$BASE_SCORE" ]; then
    BASE_SCORE=$(grep -oP "(?<=average_score:\s)\d+\.\d+" "$OUTPUT_BASE_DIR/base_model/eval.log" | tail -1)
fi

if [ -z "$TRAINED_SCORE" ]; then
    TRAINED_SCORE=$(grep -oP "(?<=average_score:\s)\d+\.\d+" "$OUTPUT_BASE_DIR/trained_model/eval.log" | tail -1)
fi

echo "Metric: ${REWARD_FN^^} Score"
echo "--------------------------------------------------------------------"
printf "%-25s %-20s %-20s %-15s\n" "Metric" "Base Model" "Trained Model" "Change"
echo "--------------------------------------------------------------------"

if [ ! -z "$BASE_SCORE" ] && [ ! -z "$TRAINED_SCORE" ]; then
    IMPROVEMENT=$(python3 -c "print(f'{float('$TRAINED_SCORE') - float('$BASE_SCORE'):.4f}')")
    IMPROVEMENT_PCT=$(python3 -c "print(f'{(float('$TRAINED_SCORE') - float('$BASE_SCORE')) / max(float('$BASE_SCORE'), 0.0001) * 100:.2f}')")
    printf "%-25s %-20s %-20s %-15s\n" "${REWARD_FN^^} Score" "$BASE_SCORE" "$TRAINED_SCORE" "+$IMPROVEMENT (+${IMPROVEMENT_PCT}%)"

    SCORE_IMPROVEMENT=$IMPROVEMENT
else
    echo "⚠️  Could not extract ${REWARD_FN} scores from logs."
    SCORE_IMPROVEMENT="0"
fi

# Display search statistics
if [ ! -z "$BASE_SEARCH" ] && [ ! -z "$TRAINED_SEARCH" ]; then
    SEARCH_DIFF=$(python3 -c "print(f'{float('$TRAINED_SEARCH') - float('$BASE_SEARCH'):.2f}')")
    SEARCH_CHANGE=$(python3 -c "
if float('$BASE_SEARCH') > 0:
    print(f'{(float('$TRAINED_SEARCH') - float('$BASE_SEARCH')) / float('$BASE_SEARCH') * 100:.1f}')
else:
    print('N/A')
")
    printf "%-25s %-20s %-20s %-15s\n" "Avg Searches/Question" "$BASE_SEARCH" "$TRAINED_SEARCH" "$SEARCH_DIFF (${SEARCH_CHANGE}%)"
fi

# Display generation time
if [ ! -z "$BASE_TIME" ] && [ ! -z "$TRAINED_TIME" ]; then
    TIME_DIFF=$(python3 -c "print(f'{float('$TRAINED_TIME') - float('$BASE_TIME'):.2f}')")
    printf "%-25s %-20s %-20s %-15s\n" "Generation Time (s)" "$BASE_TIME" "$TRAINED_TIME" "$TIME_DIFF"
fi

echo "--------------------------------------------------------------------"
echo ""

# Summary
if [ ! -z "$SCORE_IMPROVEMENT" ]; then
    if (( $(echo "$SCORE_IMPROVEMENT > 0" | bc -l) )); then
        echo "✓ The trained model shows IMPROVEMENT over the base model"
        echo ""
        echo "Key Improvements:"
        echo "  • ${REWARD_FN^^} Score: +$IMPROVEMENT (+${IMPROVEMENT_PCT}%)"
        if [ ! -z "$BASE_SEARCH" ] && [ ! -z "$TRAINED_SEARCH" ]; then
            echo "  • Search Usage: The trained model makes $TRAINED_SEARCH searches/question (vs $BASE_SEARCH for base)"
            if (( $(echo "$TRAINED_SEARCH > $BASE_SEARCH" | bc -l) )); then
                echo "    → Model learned to use search more actively"
            elif (( $(echo "$TRAINED_SEARCH < $BASE_SEARCH" | bc -l) )); then
                echo "    → Model learned to be more selective with search"
            fi
        fi
    elif (( $(echo "$SCORE_IMPROVEMENT < 0" | bc -l) )); then
        echo "✗ The trained model performs WORSE than the base model"
    else
        echo "= The models perform EQUALLY"
    fi
fi

echo ""
echo "============================================================================"
echo "Full Results"
echo "============================================================================"
echo "Base Model Log:    $OUTPUT_BASE_DIR/base_model/eval.log"
echo "Trained Model Log: $OUTPUT_BASE_DIR/trained_model/eval.log"
echo ""
echo "To view detailed logs:"
echo "  tail -100 $OUTPUT_BASE_DIR/base_model/eval.log"
echo "  tail -100 $OUTPUT_BASE_DIR/trained_model/eval.log"
echo ""
echo "To extract all metrics:"
echo "  grep 'env/' $OUTPUT_BASE_DIR/base_model/eval.log"
echo "  grep 'env/' $OUTPUT_BASE_DIR/trained_model/eval.log"
echo "============================================================================"
