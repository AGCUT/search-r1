#!/bin/bash
#
# 将 QReCC test.json 和 train.json 转换为 parquet 格式
#
# 使用方法:
#   bash convert_qrecc_test.sh
#

echo "======================================================================"
echo "QReCC 数据转换 - JSON 转 Parquet"
echo "======================================================================"

# 输入输出路径
INPUT_TEST_FILE="data/qrecc_raw/qrecc_test.json"
OUTPUT_TEST_FILE="data/qrecc_raw/test.parquet"

INPUT_TRAIN_FILE="data/qrecc_raw/qrecc_train.json"
OUTPUT_TRAIN_FILE="data/qrecc_raw/train.parquet"

# 模板类型 (base 用于基础模型, instruct 用于指令微调模型)
TEMPLATE_TYPE="base"

echo "测试集:"
echo "  输入文件: $INPUT_TEST_FILE"
echo "  输出文件: $OUTPUT_TEST_FILE"
echo ""
echo "训练集:"
echo "  输入文件: $INPUT_TRAIN_FILE"
echo "  输出文件: $OUTPUT_TRAIN_FILE"
echo ""
echo "模板类型: $TEMPLATE_TYPE"
echo "======================================================================"

# 检查输入文件是否存在
if [ ! -f "$INPUT_TEST_FILE" ]; then
    echo "错误: 找不到测试集文件 $INPUT_TEST_FILE"
    exit 1
fi

if [ ! -f "$INPUT_TRAIN_FILE" ]; then
    echo "错误: 找不到训练集文件 $INPUT_TRAIN_FILE"
    exit 1
fi

echo ""
echo "======================================================================"
echo "转换测试集..."
echo "======================================================================"

# 运行测试集转换脚本
python scripts/data_process/convert_qrecc_test_to_parquet.py \
    --input_file $INPUT_TEST_FILE \
    --output_file $OUTPUT_TEST_FILE \
    --template_type $TEMPLATE_TYPE

TEST_STATUS=$?

if [ $TEST_STATUS -ne 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✗ 测试集转换失败"
    echo "======================================================================"
    exit 1
fi

echo ""
echo "======================================================================"
echo "转换训练集..."
echo "======================================================================"

# 运行训练集转换脚本
python scripts/data_process/convert_qrecc_test_to_parquet.py \
    --input_file $INPUT_TRAIN_FILE \
    --output_file $OUTPUT_TRAIN_FILE \
    --template_type $TEMPLATE_TYPE

TRAIN_STATUS=$?

# 检查是否成功
if [ $TEST_STATUS -eq 0 ] && [ $TRAIN_STATUS -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✓ 转换成功！"
    echo "======================================================================"
    echo "生成的文件:"
    echo "  测试集: $OUTPUT_TEST_FILE"
    echo "  训练集: $OUTPUT_TRAIN_FILE"
    echo ""
    echo "验证文件:"
    echo "  python -c \"import pandas as pd; df = pd.read_parquet('$OUTPUT_TEST_FILE'); print(f'测试集样本数: {len(df)}'); df2 = pd.read_parquet('$OUTPUT_TRAIN_FILE'); print(f'训练集样本数: {len(df2)}')\""
    echo ""
    echo "现在可以运行评估脚本了:"
    echo "  bash configs/qrecc/compare_base_vs_trained_f1.sh"
    echo "======================================================================"
else
    echo ""
    echo "======================================================================"
    echo "✗ 转换失败"
    echo "======================================================================"
    exit 1
fi