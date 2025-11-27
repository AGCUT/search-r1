#!/bin/bash
###############################################################################
# 步骤3: 模型训练脚本
# 功能: 使用LoRA微调Qwen2.5-VL-32B
# GPU使用: GPU 0,1,2,3,4 (5张GPU，避开正在使用的5,6,7)
# 预计耗时: 6-8小时
# 运行位置: 项目根目录 (pdf/)
###############################################################################

set -e  # 遇到错误立即退出

echo "=========================================="
echo "开始模型训练"
echo "开始时间: $(date)"
echo "=========================================="

# 配置GPU - 使用5张GPU（避开GPU 5,6,7）
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export MAX_PIXELS=1229312  # 训练时使用较低分辨率以节省显存

# 配置路径（请根据实际情况修改）
export PROJECT_ROOT=/data/coding
export BASE_MODEL_PATH=$PROJECT_ROOT/llm_model/Qwen/Qwen2___5-VL-32B-Instruct
export TRAIN_DATASET_PATH=$(pwd)/ccks2025_pdf_multimodal/round_b/train_b_dataset_for_image_0801.jsonl
export OUTPUT_DIR=$PROJECT_ROOT/lora_qwen25_vl_32b_b

echo "配置信息:"
echo "  GPU: $CUDA_VISIBLE_DEVICES (5张GPU)"
echo "  MAX_PIXELS: $MAX_PIXELS"
echo "  基座模型: $BASE_MODEL_PATH"
echo "  训练数据: $TRAIN_DATASET_PATH"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# 检查文件是否存在
if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "错误: 基座模型路径不存在: $BASE_MODEL_PATH"
    exit 1
fi

if [ ! -f "$TRAIN_DATASET_PATH" ]; then
    echo "错误: 训练数据集不存在: $TRAIN_DATASET_PATH"
    echo "请先运行 Jupyter notebook 构造训练集"
    echo "  cd ccks2025_pdf_multimodal/round_b"
    echo "  jupyter notebook finetune训练集构造_v2.ipynb"
    exit 1
fi

# 检查GPU状态
echo "当前GPU状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | head -n 5
echo ""

# 检查训练数据
echo "训练数据统计:"
wc -l "$TRAIN_DATASET_PATH"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 开始训练
echo "=========================================="
echo "正在启动训练..."
echo "=========================================="
echo "训练参数:"
echo "  - LoRA rank: 8"
echo "  - LoRA alpha: 32"
echo "  - Learning rate: 1e-4"
echo "  - Epochs: 5"
echo "  - Batch size per device: 1"
echo "  - Gradient accumulation: 16"
echo "  - Effective batch size: 5 × 1 × 16 = 80"
echo ""

# 切换到 round_b 目录运行训练
cd ccks2025_pdf_multimodal/round_b

# 运行训练脚本
swift sft \
--model "$BASE_MODEL_PATH" \
--dataset "$TRAIN_DATASET_PATH" \
--train_type lora \
--device_map auto \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--split_dataset_ratio 0.1 \
--output_dir "$OUTPUT_DIR" \
--num_train_epochs 5 \
--lorap_lr_ratio 10 \
--save_steps 10 \
--eval_steps 10 \
--save_total_limit 4 \
--logging_steps 5 \
--seed 42 \
--learning_rate 1e-4 \
--init_weights true \
--lora_rank 8 \
--lora_alpha 32 \
--adam_beta1 0.9 \
--adam_beta2 0.95 \
--adam_epsilon 1e-08 \
--weight_decay 0.1 \
--gradient_accumulation_steps 16 \
--max_grad_norm 1 \
--lr_scheduler_type cosine \
--warmup_ratio 0.05 \
--warmup_steps 0 \
--gradient_checkpointing false \
2>&1 | tee train.log

# 返回项目根目录
cd ../..

echo ""
echo "=========================================="
echo "训练完成！"
echo "结束时间: $(date)"
echo "=========================================="

# 显示保存的checkpoints
echo ""
echo "保存的checkpoints:"
ls -lh "$OUTPUT_DIR"/checkpoint-* 2>/dev/null || echo "未找到checkpoint"

echo ""
echo "下一步: 合并LoRA权重并运行推理"
echo "  bash scripts/03_inference.sh"