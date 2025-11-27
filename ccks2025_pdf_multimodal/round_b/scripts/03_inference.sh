#!/bin/bash
###############################################################################
# 步骤4: 模型推理脚本
# 功能: 使用微调后的模型生成测试集答案
# GPU使用: GPU 0,1,2,3 (4张GPU用于张量并行)
# 预计耗时: 2-3小时
###############################################################################

set -e  # 遇到错误立即退出

echo "=========================================="
echo "开始模型推理"
echo "开始时间: $(date)"
echo "=========================================="

# 配置GPU - 使用4张GPU进行推理
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MAX_PIXELS=1568000  # 推理时使用较高分辨率（约2000 tokens）

# 配置路径（请根据实际情况修改）
export PROJECT_ROOT=/data/coding
export MODEL_PATH=$PROJECT_ROOT/lora_qwen25_vl_32b_b

echo "配置信息:"
echo "  GPU: $CUDA_VISIBLE_DEVICES (4张GPU)"
echo "  MAX_PIXELS: $MAX_PIXELS"
echo "  模型路径: $MODEL_PATH"
echo ""

# 查找最新的checkpoint
LATEST_CHECKPOINT=$(ls -td "$MODEL_PATH"/v*/checkpoint-*-merged 2>/dev/null | head -n 1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "警告: 未找到已合并的checkpoint"
    echo "正在查找未合并的checkpoint..."

    LATEST_CHECKPOINT=$(ls -td "$MODEL_PATH"/checkpoint-* 2>/dev/null | head -n 1)

    if [ -z "$LATEST_CHECKPOINT" ]; then
        echo "错误: 未找到任何checkpoint"
        echo "请先完成训练步骤"
        exit 1
    fi

    echo "找到未合并的checkpoint: $LATEST_CHECKPOINT"
    echo "正在合并LoRA权重..."

    swift export \
        --ckpt_dir "$LATEST_CHECKPOINT" \
        --merge_lora true

    # 重新查找合并后的checkpoint
    LATEST_CHECKPOINT=$(ls -td "$MODEL_PATH"/v*/checkpoint-*-merged 2>/dev/null | head -n 1)

    if [ -z "$LATEST_CHECKPOINT" ]; then
        echo "错误: 合并失败"
        exit 1
    fi
fi

echo "使用checkpoint: $LATEST_CHECKPOINT"
echo ""

# 检查必要的文件
REQUIRED_FILES=(
    "all_train_b_question_vectors.npy"
    "all_test_b_question_vectors.npy"
    "test_b_pdf_img_vectors.npy"
    "test_b_pdf_img_page_num_mapping.csv"
)

echo "检查必要文件..."
ALL_EXIST=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ 缺失: $file"
        ALL_EXIST=false
    fi
done

if [ "$ALL_EXIST" = false ]; then
    echo ""
    echo "错误: 某些必要文件缺失，请先运行预处理步骤"
    exit 1
fi

echo ""

# 检查GPU状态
echo "当前GPU状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | head -n 4
echo ""

# 检查推理脚本
if [ ! -f "test_b_style_refer_215.py" ]; then
    echo "错误: 找不到推理脚本 test_b_style_refer_215.py"
    exit 1
fi

# 自动修改推理脚本中的模型路径
echo "正在更新推理脚本中的模型路径..."
sed -i.bak "s|model_path = \".*\"|model_path = \"$LATEST_CHECKPOINT\"|g" test_b_style_refer_215.py
echo "模型路径已更新为: $LATEST_CHECKPOINT"
echo ""

# 开始推理
echo "=========================================="
echo "正在运行推理..."
echo "=========================================="

python test_b_style_refer_215.py 2>&1 | tee inference.log

# 检查输出文件
echo ""
echo "=========================================="
echo "检查输出文件..."
echo "=========================================="

OUTPUT_FILE="test_b_style_infer_if_need_ck215.jsonl"

if [ -f "$OUTPUT_FILE" ]; then
    line_count=$(wc -l < "$OUTPUT_FILE")
    file_size=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')

    echo "✓ 生成结果文件: $OUTPUT_FILE"
    echo "  - 样本数量: $line_count"
    echo "  - 文件大小: $file_size"
    echo ""

    # 显示前3个结果示例
    echo "结果示例（前3个）:"
    head -n 3 "$OUTPUT_FILE" | python -m json.tool 2>/dev/null || head -n 3 "$OUTPUT_FILE"

    echo ""
    echo "=========================================="
    echo "推理完成！"
    echo "结束时间: $(date)"
    echo "=========================================="
    echo ""
    echo "结果文件: $OUTPUT_FILE"
    echo "可以提交到竞赛平台了！"

else
    echo "✗ 未找到输出文件: $OUTPUT_FILE"
    echo "推理可能失败，请查看日志: inference.log"
    exit 1
fi