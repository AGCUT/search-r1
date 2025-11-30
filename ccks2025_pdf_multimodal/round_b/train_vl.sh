#!/bin/bash
# 模型微调脚本
# 使用 ms-swift 对 Qwen2.5-VL-32B 进行 LoRA 微调
# 适配 4 × A800 80GB

# 配置路径（请根据实际情况修改）
MODEL_PATH="/usr/yuque/guo/pdf_processer/llm_model/Qwen/Qwen2.5-VL-32B-Instruct"  # 32B模型路径
DATASET_PATH="/usr/yuque/guo/pdf_processer/ccks2025_pdf_multimodal/round_b/train_dataset_for_image.jsonl"  # 训练数据
OUTPUT_DIR="/usr/yuque/guo/pdf_processer/lora_output_32b"  # LoRA权重输出目录

# GPU配置
GPUS="4,5,6,7"

echo "=============================================="
echo "开始模型微调 (Qwen2.5-VL-32B)"
echo "=============================================="
echo "模型路径: ${MODEL_PATH}"
echo "数据集: ${DATASET_PATH}"
echo "输出目录: ${OUTPUT_DIR}"
echo "使用GPU: ${GPUS}"
echo "=============================================="

MAX_PIXELS=1229312 CUDA_VISIBLE_DEVICES=${GPUS} swift sft \
    --model ${MODEL_PATH} \
    --dataset ${DATASET_PATH} \
    --train_type lora \
    --device_map auto \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --split_dataset_ratio 0.1 \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 5 \
    --lorap_lr_ratio 10 \
    --save_steps 50 \
    --eval_steps 50 \
    --save_total_limit 3 \
    --logging_steps 10 \
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
    --gradient_checkpointing true \
    --deepspeed zero2

echo "=============================================="
echo "训练完成！"
echo "LoRA权重保存在: ${OUTPUT_DIR}"
echo "=============================================="
