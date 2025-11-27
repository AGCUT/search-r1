#!/bin/bash
###############################################################################
# CCKS 2025 Conda环境自动配置脚本
# 使用方法: bash setup_conda_env.sh
###############################################################################

set -e  # 遇到错误立即退出

echo "=========================================="
echo "CCKS 2025 Conda环境配置脚本"
echo "开始时间: $(date)"
echo "=========================================="
echo ""

# 环境名称
ENV_NAME="ccks2025_pdf_qa"

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查conda是否安装
echo "1. 检查Conda安装..."
if ! command -v conda &> /dev/null; then
    echo -e "${RED}错误: Conda未安装${NC}"
    echo "请先安装Anaconda或Miniconda:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

CONDA_VERSION=$(conda --version)
echo -e "${GREEN}✓ Conda已安装: $CONDA_VERSION${NC}"
echo ""

# 检查CUDA版本
echo "2. 检查CUDA版本..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo -e "${GREEN}✓ 检测到CUDA版本: $CUDA_VERSION${NC}"

    # 根据CUDA版本设置PyTorch下载链接
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    if [ "$CUDA_MAJOR" -ge 12 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        echo "  使用PyTorch CUDA 12.1版本"
    else
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        echo "  使用PyTorch CUDA 11.8版本"
    fi
else
    echo -e "${YELLOW}⚠ 未检测到nvidia-smi，将使用CPU版本PyTorch${NC}"
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
fi
echo ""

# 检查环境是否已存在
echo "3. 检查Conda环境..."
if conda env list | grep -q "^$ENV_NAME "; then
    echo -e "${YELLOW}⚠ 环境 '$ENV_NAME' 已存在${NC}"
    read -p "是否删除并重新创建? [y/N]: " RECREATE
    if [[ "$RECREATE" =~ ^[Yy]$ ]]; then
        echo "删除旧环境..."
        conda env remove -n $ENV_NAME -y
    else
        echo "使用现有环境..."
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate $ENV_NAME
        echo -e "${GREEN}✓ 环境已激活${NC}"
        echo ""
        echo "如果需要重新安装依赖，请运行:"
        echo "  conda activate $ENV_NAME"
        echo "  pip install -r requirements.txt"
        exit 0
    fi
fi
echo ""

# 创建conda环境
echo "4. 创建Conda环境: $ENV_NAME"
echo "  Python版本: 3.10"
conda create -n $ENV_NAME python=3.10 -y

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 环境创建成功${NC}"
else
    echo -e "${RED}✗ 环境创建失败${NC}"
    exit 1
fi
echo ""

# 激活环境
echo "5. 激活环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

if [ "$CONDA_DEFAULT_ENV" = "$ENV_NAME" ]; then
    echo -e "${GREEN}✓ 环境已激活: $ENV_NAME${NC}"
else
    echo -e "${RED}✗ 环境激活失败${NC}"
    exit 1
fi
echo ""

# 升级pip
echo "6. 升级pip..."
pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✓ pip升级完成${NC}"
echo ""

# 安装PyTorch
echo "7. 安装PyTorch (这可能需要几分钟)..."
if [ "$TORCH_INDEX" = "https://download.pytorch.org/whl/cpu" ]; then
    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
else
    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url $TORCH_INDEX
fi

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ PyTorch安装成功${NC}"
else
    echo -e "${RED}✗ PyTorch安装失败${NC}"
    exit 1
fi

# 验证PyTorch
echo "验证PyTorch安装..."
python -c "
import torch
print(f'  PyTorch版本: {torch.__version__}')
print(f'  CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA版本: {torch.version.cuda}')
    print(f'  GPU数量: {torch.cuda.device_count()}')
    print(f'  当前GPU: {torch.cuda.get_device_name(0)}')
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ PyTorch验证通过${NC}"
else
    echo -e "${YELLOW}⚠ PyTorch验证失败，但继续安装其他依赖${NC}"
fi
echo ""

# 检查requirements.txt是否存在
echo "8. 安装其他依赖..."
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}✗ 找不到requirements.txt文件${NC}"
    echo "请确保在项目根目录运行此脚本"
    exit 1
fi

# 安装其他依赖
echo "正在安装依赖包 (这可能需要10-20分钟)..."
pip install -r requirements.txt --no-cache-dir

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 依赖安装成功${NC}"
else
    echo -e "${RED}✗ 部分依赖安装失败${NC}"
    echo "请查看上面的错误信息"
fi
echo ""

# 验证核心库
echo "9. 验证核心库安装..."
python -c "
import sys
errors = []

packages = {
    'torch': 'PyTorch',
    'transformers': 'Transformers',
    'vllm': 'vLLM',
    'swift': 'MS-SWIFT',
    'fitz': 'PyMuPDF',
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'qwen_vl_utils': 'Qwen-VL-Utils',
    'PIL': 'Pillow',
    'tqdm': 'tqdm',
}

for module, name in packages.items():
    try:
        __import__(module)
        print(f'  ✓ {name}')
    except ImportError:
        print(f'  ✗ {name} (未安装)')
        errors.append(name)

if errors:
    print(f'\n缺少以下依赖: {', '.join(errors)}')
    sys.exit(1)
else:
    print('\n✓ 所有核心库已安装')
"

VERIFY_STATUS=$?
echo ""

# 显示环境信息
echo "=========================================="
echo "环境配置完成！"
echo "=========================================="
echo ""
echo "环境名称: $ENV_NAME"
echo "Python版本: $(python --version)"
echo "安装路径: $(which python)"
echo ""

if [ $VERIFY_STATUS -eq 0 ]; then
    echo -e "${GREEN}✓ 所有依赖验证通过${NC}"
else
    echo -e "${YELLOW}⚠ 部分依赖验证失败，请手动检查${NC}"
fi

echo ""
echo "使用方法:"
echo "  1. 激活环境:"
echo "     conda activate $ENV_NAME"
echo ""
echo "  2. 运行环境检查:"
echo "     cd ccks2025_pdf_multimodal/round_b"
echo "     bash scripts/check_environment.sh"
echo ""
echo "  3. 开始复现:"
echo "     bash scripts/00_setup_paths.sh /data/coding"
echo "     bash scripts/01_preprocess.sh"
echo ""
echo "停用环境:"
echo "  conda deactivate"
echo ""
echo "删除环境:"
echo "  conda env remove -n $ENV_NAME"
echo ""

# 显示下一步
echo "=========================================="
echo "下一步操作"
echo "=========================================="
echo ""
echo "1. 保持此终端打开（环境已激活）"
echo "2. 运行环境检查:"
echo "   cd ccks2025_pdf_multimodal/round_b && bash scripts/check_environment.sh"
echo ""
echo "3. 如果检查通过，开始复现:"
echo "   查看 REPRODUCTION_GUIDE.md 了解详细步骤"
echo ""

# 询问是否立即运行检查
read -p "是否现在运行环境检查? [y/N]: " RUN_CHECK
if [[ "$RUN_CHECK" =~ ^[Yy]$ ]]; then
    if [ -f "ccks2025_pdf_multimodal/round_b/scripts/check_environment.sh" ]; then
        echo ""
        echo "运行环境检查..."
        cd ccks2025_pdf_multimodal/round_b
        bash scripts/check_environment.sh
    else
        echo -e "${YELLOW}未找到检查脚本，请手动运行${NC}"
    fi
fi

echo ""
echo "结束时间: $(date)"