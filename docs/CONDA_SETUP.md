# Conda 环境配置指南

本指南提供使用Conda配置CCKS 2025项目运行环境的详细步骤。

## 📋 前置要求

- Anaconda 或 Miniconda 已安装
- CUDA 12.1+ 已安装（用于GPU支持）
- 至少50GB磁盘空间（用于conda环境和依赖包）

## 🚀 快速配置（推荐）

### 方法1: 使用environment.yml（一键创建）

```bash
# 创建conda环境（自动安装所有依赖）
conda env create -f environment.yml

# 激活环境
conda activate ccks2025_pdf_qa

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 方法2: 使用requirements.txt（手动创建）

```bash
# 1. 创建新的conda环境
conda create -n ccks2025_pdf_qa python=3.10 -y

# 2. 激活环境
conda activate ccks2025_pdf_qa

# 3. 安装PyTorch (根据你的CUDA版本选择)
# 对于CUDA 12.1:
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 对于CUDA 11.8:
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 4. 安装其他依赖
pip install -r requirements.txt

# 5. 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 🔍 详细步骤说明

### 步骤1: 检查CUDA版本

```bash
# 检查系统CUDA版本
nvidia-smi

# 查看CUDA版本信息
nvcc --version
```

根据输出的CUDA版本（如12.4），选择合适的PyTorch版本。

### 步骤2: 创建Conda环境

```bash
# 创建Python 3.10环境
conda create -n ccks2025_pdf_qa python=3.10 -y

# 激活环境
conda activate ccks2025_pdf_qa

# 验证Python版本
python --version  # 应该显示 Python 3.10.x
```

### 步骤3: 安装PyTorch

⚠️ **重要**: 必须先安装PyTorch，再安装其他依赖

```bash
# 方案A: CUDA 12.1+ (推荐)
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# 方案B: CUDA 11.8
# pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

# 验证PyTorch安装
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

预期输出:
```
2.2.0+cu121  # 或类似版本
True         # 表示CUDA可用
```

### 步骤4: 安装核心依赖

```bash
# 安装SWIFT训练框架
pip install ms-swift>=2.0.0

# 安装vLLM推理引擎
pip install vllm>=0.4.0

# 安装Transformers和加速库
pip install transformers>=4.40.0 accelerate>=0.27.0 peft>=0.10.0

# 验证核心库
python -c "import swift; import vllm; import transformers; print('核心库安装成功')"
```

### 步骤5: 安装其他依赖

```bash
# 一次性安装所有剩余依赖
pip install -r requirements.txt

# 或者手动安装主要依赖
pip install qwen-vl-utils PyMuPDF numpy pandas pillow modelscope jupyter tqdm
```

### 步骤6: 验证完整安装

```bash
# 运行环境检查脚本
cd ccks2025_pdf_multimodal/round_b
bash scripts/check_environment.sh
```

## 📦 依赖包说明

### 必需依赖 (Required)

| 包名 | 版本 | 用途 |
|------|------|------|
| torch | >=2.1.0 | 深度学习框架 |
| transformers | >=4.40.0 | 模型加载和训练 |
| vllm | >=0.4.0 | 高效推理引擎 |
| ms-swift | >=2.0.0 | 模型训练框架 |
| PyMuPDF | >=1.23.0 | PDF处理 |
| qwen-vl-utils | latest | Qwen视觉语言工具 |
| numpy | >=1.24.0 | 数值计算 |
| pandas | >=2.0.0 | 数据处理 |

### 推荐依赖 (Recommended)

| 包名 | 版本 | 用途 |
|------|------|------|
| accelerate | >=0.27.0 | 分布式训练加速 |
| deepspeed | >=0.12.0 | 高效训练优化 |
| tensorboard | >=2.14.0 | 训练可视化 |
| wandb | >=0.16.0 | 实验跟踪 |
| jupyter | >=1.0.0 | 交互式开发 |

### 可选依赖 (Optional)

```bash
# 模型量化（如果需要）
pip install bitsandbytes>=0.41.0  # 8-bit/4-bit量化
pip install auto-gptq>=0.5.0      # GPTQ量化

# 数据增强（如果需要）
pip install albumentations>=1.3.0  # 图像增强
```

## 🔧 常见问题解决

### 问题1: PyTorch CUDA版本不匹配

**症状**: `torch.cuda.is_available()` 返回 `False`

**解决方案**:
```bash
# 1. 卸载现有PyTorch
pip uninstall torch torchvision torchaudio -y

# 2. 检查CUDA版本
nvidia-smi | grep "CUDA Version"

# 3. 安装匹配的PyTorch
# 对于CUDA 12.1+:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 对于CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 问题2: vLLM安装失败

**症状**: `pip install vllm` 报错

**解决方案**:
```bash
# vLLM需要特定版本的依赖，可能需要降级某些包
pip install vllm==0.4.3 --no-build-isolation

# 或者从源码安装
pip install git+https://github.com/vllm-project/vllm.git
```

### 问题3: ms-swift安装失败

**症状**: `pip install ms-swift` 报错

**解决方案**:
```bash
# 确保已安装PyTorch
pip install torch

# 使用国内镜像安装
pip install ms-swift -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或者从源码安装
pip install git+https://github.com/modelscope/swift.git
```

### 问题4: PyMuPDF (fitz) 导入错误

**症状**: `ModuleNotFoundError: No module named 'fitz'`

**解决方案**:
```bash
# 确保安装的是PyMuPDF，而不是fitz
pip uninstall fitz PyMuPDF -y
pip install PyMuPDF>=1.23.0

# 验证
python -c "import fitz; print(fitz.__doc__)"
```

### 问题5: 内存不足 (OOM)

**症状**: 安装过程中内存耗尽

**解决方案**:
```bash
# 分批安装依赖
pip install torch torchvision torchaudio
pip install transformers accelerate
pip install vllm
pip install ms-swift
# ... 逐个安装其他包

# 或者使用 --no-cache-dir
pip install -r requirements.txt --no-cache-dir
```

## 🌐 使用国内镜像加速

### 配置pip镜像

```bash
# 临时使用清华镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 永久配置
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 配置conda镜像

```bash
# 编辑 ~/.condarc
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
conda config --set show_channel_urls yes
```

### 配置HuggingFace镜像

```bash
# 在 ~/.bashrc 或 ~/.zshrc 中添加
export HF_ENDPOINT=https://hf-mirror.com

# 或在脚本中临时设置
HF_ENDPOINT=https://hf-mirror.com python your_script.py
```

## 📊 环境验证清单

运行以下命令验证环境配置是否正确：

```bash
# 1. 检查Python版本
python --version  # 应该是 3.10.x

# 2. 检查PyTorch和CUDA
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'CUDA版本: {torch.version.cuda}')
print(f'GPU数量: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'当前GPU: {torch.cuda.get_device_name(0)}')
"

# 3. 检查核心库
python -c "
try:
    import torch; print('✓ torch')
    import transformers; print('✓ transformers')
    import vllm; print('✓ vllm')
    import swift; print('✓ ms-swift')
    import fitz; print('✓ PyMuPDF')
    import numpy; print('✓ numpy')
    import pandas; print('✓ pandas')
    import qwen_vl_utils; print('✓ qwen-vl-utils')
    print('\n所有核心库已安装！')
except ImportError as e:
    print(f'✗ 缺少依赖: {e}')
"

# 4. 运行完整检查
cd ccks2025_pdf_multimodal/round_b
bash scripts/check_environment.sh
```

## 🎯 环境管理

### 激活/停用环境

```bash
# 激活环境
conda activate ccks2025_pdf_qa

# 停用环境
conda deactivate
```

### 查看环境信息

```bash
# 查看所有conda环境
conda env list

# 查看当前环境安装的包
conda list

# 查看pip安装的包
pip list

# 导出环境配置
conda env export > environment_backup.yml
pip freeze > requirements_backup.txt
```

### 删除环境

```bash
# 停用当前环境
conda deactivate

# 删除环境
conda env remove -n ccks2025_pdf_qa

# 清理缓存
conda clean --all -y
```

## 📋 完整安装脚本

将以下内容保存为 `setup_conda_env.sh`：

```bash
#!/bin/bash
set -e

echo "=========================================="
echo "CCKS 2025 Conda环境配置脚本"
echo "=========================================="

# 环境名称
ENV_NAME="ccks2025_pdf_qa"

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "错误: Conda未安装"
    exit 1
fi

# 创建conda环境
echo "创建conda环境: $ENV_NAME"
conda create -n $ENV_NAME python=3.10 -y

# 激活环境
echo "激活环境: $ENV_NAME"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# 安装PyTorch (CUDA 12.1)
echo "安装PyTorch..."
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# 验证PyTorch
python -c "import torch; assert torch.cuda.is_available(), 'CUDA不可用'; print('✓ PyTorch安装成功')"

# 安装其他依赖
echo "安装其他依赖..."
pip install -r requirements.txt

# 验证安装
echo "验证安装..."
python -c "
import torch, transformers, vllm, swift, fitz, numpy, pandas
print('✓ 所有核心库安装成功')
"

echo ""
echo "=========================================="
echo "环境配置完成！"
echo "=========================================="
echo ""
echo "使用方法:"
echo "  conda activate $ENV_NAME"
echo ""
echo "下一步:"
echo "  bash scripts/check_environment.sh"
```

运行脚本：
```bash
chmod +x setup_conda_env.sh
bash setup_conda_env.sh
```

## 🔗 相关资源

- **PyTorch安装**: https://pytorch.org/get-started/locally/
- **Conda文档**: https://docs.conda.io/
- **requirements.txt**: 本项目根目录
- **environment.yml**: 本项目根目录
- **检查脚本**: `scripts/check_environment.sh`

---

**下一步**: 环境配置完成后，请查看 `REPRODUCTION_GUIDE.md` 开始复现项目。