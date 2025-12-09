#!/bin/bash
#
# 将 QReCC test.json 转换为 test.parquet 格式
#
# 使用方法:
#   bash convert_qrecc_test.sh
#

echo "======================================================================"
echo "QReCC 测试数据转换 - JSON 转 Parquet"
echo "======================================================================"

# 输入输出路径
INPUT_FILE="data/qrecc_raw/qrecc_test.json"
OUTPUT_FILE="data/qrecc_raw/test.parquet"

# 模板类型 (base 用于基础模型, instruct 用于指令微调模型)
TEMPLATE_TYPE="base"

echo "输入文件: $INPUT_FILE"
echo "输出文件: $OUTPUT_FILE"
echo "模板类型: $TEMPLATE_TYPE"
echo "======================================================================"

# 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
    echo "错误: 找不到输入文件 $INPUT_FILE"
    exit 1
fi

# 运行转换脚本
python scripts/data_process/convert_qrecc_test_to_parquet.py \
    --input_file $INPUT_FILE \
    --output_file $OUTPUT_FILE \
    --template_type $TEMPLATE_TYPE

# 检查是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✓ 转换成功！"
    echo "======================================================================"
    echo "生成的文件: $OUTPUT_FILE"
    echo ""
    echo "验证文件:"
    echo "  python -c \"import pandas as pd; df = pd.read_parquet('$OUTPUT_FILE'); print(f'总样本数: {len(df)}'); print(df.iloc[0])\""
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