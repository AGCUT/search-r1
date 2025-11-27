#!/bin/bash
###############################################################################
# 步骤1: 数据预处理脚本
# 功能: PDF转图像 + 生成向量嵌入
# GPU使用: GPU 1 (最空闲的GPU)
# 预计耗时: 6-8小时
# 运行位置: 项目根目录 (pdf/)
###############################################################################

set -e  # 遇到错误立即退出

echo "=========================================="
echo "开始数据预处理"
echo "开始时间: $(date)"
echo "=========================================="

# 配置GPU
export CUDA_VISIBLE_DEVICES=1
export MAX_PIXELS=1229312  # 约1280 tokens per image

# 配置路径（请根据实际情况修改）
export PROJECT_ROOT=/data/coding
export PATENT_DATA_DIR=$PROJECT_ROOT/patent_b
export GME_MODEL_PATH=$PROJECT_ROOT/llm_model/iic/gme-Qwen2-VL-7B-Instruct

echo "配置信息:"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "  MAX_PIXELS: $MAX_PIXELS"
echo "  训练数据路径: $PATENT_DATA_DIR/train/"
echo "  测试数据路径: $PATENT_DATA_DIR/test/"
echo "  GME模型路径: $GME_MODEL_PATH"
echo ""

# 检查路径是否存在
if [ ! -d "$PATENT_DATA_DIR/train/documents" ]; then
    echo "错误: 训练数据路径不存在: $PATENT_DATA_DIR/train/documents"
    exit 1
fi

if [ ! -d "$PATENT_DATA_DIR/test/documents" ]; then
    echo "错误: 测试数据路径不存在: $PATENT_DATA_DIR/test/documents"
    exit 1
fi

if [ ! -d "$GME_MODEL_PATH" ]; then
    echo "错误: GME模型路径不存在: $GME_MODEL_PATH"
    exit 1
fi

# 检查Python脚本是否存在
if [ ! -f "ccks2025_pdf_multimodal/round_b/b_train_test_preprocess.py" ]; then
    echo "错误: 找不到预处理脚本 ccks2025_pdf_multimodal/round_b/b_train_test_preprocess.py"
    echo "请确保在项目根目录 (pdf/) 下运行此脚本"
    exit 1
fi

# 检查GPU状态
echo "当前GPU状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | grep "^1,"
echo ""

# 切换到 round_b 目录运行预处理
cd ccks2025_pdf_multimodal/round_b

# 运行预处理脚本
echo "=========================================="
echo "正在运行预处理脚本..."
echo "=========================================="

python b_train_test_preprocess.py 2>&1 | tee preprocess.log

# 检查输出文件
echo ""
echo "=========================================="
echo "检查生成的文件..."
echo "=========================================="

REQUIRED_FILES=(
    "train_b_pdf_img_vectors.npy"
    "train_b_pdf_img_page_num_mapping.csv"
    "all_train_b_question_vectors.npy"
    "test_b_pdf_img_vectors.npy"
    "test_b_pdf_img_page_num_mapping.csv"
    "all_test_b_question_vectors.npy"
)

ALL_EXIST=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        size=$(ls -lh "$file" | awk '{print $5}')
        echo "✓ $file ($size)"
    else
        echo "✗ 缺失: $file"
        ALL_EXIST=false
    fi
done

# 返回项目根目录
cd ../..

echo ""
if [ "$ALL_EXIST" = true ]; then
    echo "=========================================="
    echo "预处理完成！"
    echo "结束时间: $(date)"
    echo "=========================================="
    echo ""
    echo "下一步: 运行 Jupyter notebook 构造训练集"
    echo "  cd ccks2025_pdf_multimodal/round_b"
    echo "  jupyter notebook finetune训练集构造_v2.ipynb"
    exit 0
else
    echo "=========================================="
    echo "预处理失败！某些输出文件缺失"
    echo "请查看日志文件: ccks2025_pdf_multimodal/round_b/preprocess.log"
    echo "=========================================="
    exit 1
fi