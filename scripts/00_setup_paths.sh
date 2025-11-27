#!/bin/bash
###############################################################################
# 步骤0: 路径配置脚本
# 功能: 自动修改所有Python脚本中的路径为实际路径
# 使用方法: bash scripts/00_setup_paths.sh /data/coding
###############################################################################

set -e

echo "=========================================="
echo "路径配置工具"
echo "=========================================="

# 获取用户指定的项目根目录
if [ -z "$1" ]; then
    echo "用法: bash scripts/00_setup_paths.sh <项目根目录>"
    echo "示例: bash scripts/00_setup_paths.sh /data/coding"
    echo ""
    echo "当前默认路径: /data/coding"
    read -p "是否使用默认路径? [y/N]: " use_default

    if [[ "$use_default" =~ ^[Yy]$ ]]; then
        PROJECT_ROOT="/data/coding"
    else
        read -p "请输入项目根目录: " PROJECT_ROOT
    fi
else
    PROJECT_ROOT="$1"
fi

# 移除末尾的斜杠
PROJECT_ROOT="${PROJECT_ROOT%/}"

echo ""
echo "项目根目录: $PROJECT_ROOT"
echo ""

# 检查目录是否存在
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "警告: 目录不存在: $PROJECT_ROOT"
    read -p "是否创建该目录? [y/N]: " create_dir

    if [[ "$create_dir" =~ ^[Yy]$ ]]; then
        mkdir -p "$PROJECT_ROOT"
        echo "已创建目录: $PROJECT_ROOT"
    else
        echo "已取消"
        exit 1
    fi
fi

# 定义需要修改的文件（在 round_b 目录中）
FILES_TO_UPDATE=(
    "ccks2025_pdf_multimodal/round_b/b_train_test_preprocess.py"
    "ccks2025_pdf_multimodal/round_b/test_b_style_refer_215.py"
    "ccks2025_pdf_multimodal/round_b/test_b_style_refer_90.py"
    "ccks2025_pdf_multimodal/round_b/gme_inference.py"
)

echo "将要修改以下文件中的路径:"
for file in "${FILES_TO_UPDATE[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (不存在)"
    fi
done

echo ""
read -p "是否继续? [y/N]: " confirm

if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "正在修改文件..."

# 备份原文件
BACKUP_DIR="path_backups_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

for file in "${FILES_TO_UPDATE[@]}"; do
    if [ -f "$file" ]; then
        # 备份
        mkdir -p "$BACKUP_DIR/$(dirname $file)"
        cp "$file" "$BACKUP_DIR/$file"

        # 修改路径
        sed -i.tmp \
            -e "s|/data/coding/patent_qa|$PROJECT_ROOT/patent_b|g" \
            -e "s|/data/coding/patent_b|$PROJECT_ROOT/patent_b|g" \
            -e "s|/data/coding/llm_model|$PROJECT_ROOT/llm_model|g" \
            -e "s|/data/coding/lora_qwen25_vl_32b_b|$PROJECT_ROOT/lora_qwen25_vl_32b_b|g" \
            -e "s|/data/coding/lora_qwen25_vl_32b_for_b|$PROJECT_ROOT/lora_qwen25_vl_32b_for_b|g" \
            "$file"

        # 删除临时文件
        rm -f "$file.tmp"

        echo "  ✓ 已修改: $file"
    fi
done

echo ""
echo "=========================================="
echo "路径配置完成！"
echo "=========================================="
echo ""
echo "原始文件已备份到: $BACKUP_DIR/"
echo ""
echo "新的路径配置:"
echo "  - 项目根目录: $PROJECT_ROOT"
echo "  - 数据目录: $PROJECT_ROOT/patent_b"
echo "  - 模型目录: $PROJECT_ROOT/llm_model"
echo "  - 输出目录: $PROJECT_ROOT/lora_qwen25_vl_32b_b"
echo ""
echo "请确保这些目录存在并包含正确的文件！"
echo ""
echo "下一步: 运行预处理脚本"
echo "  bash scripts/01_preprocess.sh"