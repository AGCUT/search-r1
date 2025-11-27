# 文件导航索引

## 🎯 快速查找

**我是新手，想快速开始** → [docs/QUICKSTART.md](docs/QUICKSTART.md)

**我想了解详细步骤** → [docs/REPRODUCTION_GUIDE.md](docs/REPRODUCTION_GUIDE.md)

**我想配置Conda环境** → [docs/CONDA_SETUP.md](docs/CONDA_SETUP.md) 或 `bash scripts/setup_conda_env.sh`

**我想查看脚本说明** → [scripts/README.md](scripts/README.md)

**我想了解项目架构** → [ccks2025_pdf_multimodal/CLAUDE.md](ccks2025_pdf_multimodal/CLAUDE.md)

**我想深入了解技术** → [ccks2025_pdf_multimodal/技术分析报告.md](ccks2025_pdf_multimodal/技术分析报告.md)

## 📂 完整文件结构

### 📁 根目录文件

| 文件 | 说明 | 用途 |
|------|------|------|
| [README.md](README.md) | 项目主页 | 项目概览和快速开始 |
| [FILE_INDEX.md](FILE_INDEX.md) | 本文件 | 文件导航索引 |
| [requirements.txt](requirements.txt) | Python依赖 | `pip install -r requirements.txt` |
| [environment.yml](environment.yml) | Conda配置 | `conda env create -f environment.yml` |

### 📁 scripts/ - 运行脚本

| 脚本 | 功能 | GPU | 耗时 |
|------|------|-----|------|
| [setup_conda_env.sh](scripts/setup_conda_env.sh) | Conda环境配置 | - | 10-20分钟 |
| [check_environment.sh](scripts/check_environment.sh) | 环境检查 | - | <1分钟 |
| [00_setup_paths.sh](scripts/00_setup_paths.sh) | 路径配置 | - | <1分钟 |
| [01_preprocess.sh](scripts/01_preprocess.sh) | 数据预处理 | GPU 1 | 6-8小时 |
| [02_train.sh](scripts/02_train.sh) | 模型训练 | GPU 0-4 | 6-8小时 |
| [03_inference.sh](scripts/03_inference.sh) | 模型推理 | GPU 0-3 | 2-3小时 |
| [README.md](scripts/README.md) | 脚本说明 | - | - |

**使用方法**: 在项目根目录运行，例如 `bash scripts/01_preprocess.sh`

### 📁 docs/ - 文档

| 文档 | 类型 | 详细程度 | 适合人群 |
|------|------|----------|----------|
| [QUICKSTART.md](docs/QUICKSTART.md) | 快速开始 | ⭐ 简洁 | 新手 |
| [REPRODUCTION_GUIDE.md](docs/REPRODUCTION_GUIDE.md) | 完整指南 | ⭐⭐⭐⭐⭐ 详细 | 所有人 |
| [CONDA_SETUP.md](docs/CONDA_SETUP.md) | 环境配置 | ⭐⭐⭐⭐ 详细 | 需要配置环境的人 |
| [README_REPRODUCTION.md](docs/README_REPRODUCTION.md) | 总览文档 | ⭐⭐⭐ 中等 | 想了解概览的人 |
| [gpu.md](docs/gpu.md) | GPU状态 | ⭐ 简洁 | 参考用 |

### 📁 ccks2025_pdf_multimodal/ - 项目代码

#### 核心文档
| 文档 | 说明 |
|------|------|
| [CLAUDE.md](ccks2025_pdf_multimodal/CLAUDE.md) | 项目架构和技术细节（非常详细） |
| [技术分析报告.md](ccks2025_pdf_multimodal/技术分析报告.md) | 深度技术分析和改进建议 |
| [README.md](ccks2025_pdf_multimodal/README.md) | 原始项目说明 |

#### round_b/ - 复赛代码（主要使用）
| 文件 | 说明 |
|------|------|
| `b_train_test_preprocess.py` | 预处理核心代码：PDF转图像+向量生成 |
| `test_b_style_refer_215.py` | 推理核心代码：模型推理+答案生成 |
| `test_b_style_refer_90.py` | 备选推理代码：使用checkpoint-90 |
| `gme_inference.py` | GME嵌入模型封装 |
| `finetune训练集构造_v2.ipynb` | 训练集构造notebook |
| `train_vl_32b.sh` | 原始训练脚本（参考用） |

#### round_a/ - 初赛代码（探索性，参考用）

#### choice_pipeline/ - 备选方案（参考用）

## 🚦 使用流程

### 第一次使用

1. **阅读文档** (5分钟)
   - 阅读 [README.md](README.md) 了解项目概览
   - 阅读 [docs/QUICKSTART.md](docs/QUICKSTART.md) 了解快速开始

2. **配置环境** (10-20分钟)
   ```bash
   bash scripts/setup_conda_env.sh
   conda activate ccks2025_pdf_qa
   ```

3. **检查环境** (<1分钟)
   ```bash
   bash scripts/check_environment.sh
   ```

4. **配置路径** (<1分钟)
   ```bash
   bash scripts/00_setup_paths.sh /data/coding
   ```

5. **开始复现** (约17小时)
   ```bash
   tmux new -s ccks2025
   bash scripts/01_preprocess.sh  # 6-8h
   # 手动运行notebook构造训练集
   bash scripts/02_train.sh       # 6-8h
   bash scripts/03_inference.sh   # 2-3h
   ```

### 遇到问题时

1. **环境问题** → 查看 [docs/CONDA_SETUP.md](docs/CONDA_SETUP.md)
2. **脚本问题** → 查看 [scripts/README.md](scripts/README.md)
3. **路径问题** → 重新运行 `bash scripts/00_setup_paths.sh`
4. **GPU问题** → 查看 [docs/gpu.md](docs/gpu.md) 和调整脚本中的 `CUDA_VISIBLE_DEVICES`
5. **训练问题** → 查看 [docs/REPRODUCTION_GUIDE.md](docs/REPRODUCTION_GUIDE.md) 的"常见问题排查"章节

### 深入学习

1. **了解架构** → [ccks2025_pdf_multimodal/CLAUDE.md](ccks2025_pdf_multimodal/CLAUDE.md)
2. **技术分析** → [ccks2025_pdf_multimodal/技术分析报告.md](ccks2025_pdf_multimodal/技术分析报告.md)
3. **原理理解** → 阅读代码和jupyter notebook

## 📊 文件关系图

```
用户使用流程：
  README.md (项目主页)
     ↓
  QUICKSTART.md (快速开始)
     ↓
  setup_conda_env.sh (环境配置)
     ↓
  check_environment.sh (环境检查)
     ↓
  00_setup_paths.sh (路径配置)
     ↓
  01_preprocess.sh → round_b/b_train_test_preprocess.py (预处理)
     ↓
  finetune训练集构造_v2.ipynb (构造训练集)
     ↓
  02_train.sh → round_b/train_vl_32b.sh (训练)
     ↓
  03_inference.sh → round_b/test_b_style_refer_215.py (推理)
     ↓
  test_b_style_infer_if_need_ck215.jsonl (结果)

技术理解流程：
  CLAUDE.md (架构概览)
     ↓
  技术分析报告.md (深度分析)
     ↓
  代码阅读 (round_b/*.py)
     ↓
  Notebook学习 (round_b/*.ipynb)
```

## 🔍 按需查找

### 我想...

#### 快速开始
- **快速上手** → [QUICKSTART.md](docs/QUICKSTART.md)
- **一键配置** → `bash scripts/setup_conda_env.sh`
- **环境检查** → `bash scripts/check_environment.sh`

#### 配置环境
- **Conda完整指南** → [CONDA_SETUP.md](docs/CONDA_SETUP.md)
- **依赖列表** → [requirements.txt](requirements.txt)
- **环境配置** → [environment.yml](environment.yml)
- **故障排查** → [CONDA_SETUP.md#常见问题解决](docs/CONDA_SETUP.md#常见问题解决)

#### 运行项目
- **完整步骤** → [REPRODUCTION_GUIDE.md](docs/REPRODUCTION_GUIDE.md)
- **脚本说明** → [scripts/README.md](scripts/README.md)
- **GPU配置** → [README.md#服务器配置](README.md#服务器配置)

#### 理解技术
- **架构概览** → [CLAUDE.md](ccks2025_pdf_multimodal/CLAUDE.md)
- **技术分析** → [技术分析报告.md](ccks2025_pdf_multimodal/技术分析报告.md)
- **代码理解** → `ccks2025_pdf_multimodal/round_b/*.py`

#### 解决问题
- **OOM错误** → [scripts/README.md#Q1-OOM错误](scripts/README.md#Q1-OOM错误)
- **路径错误** → [scripts/README.md#Q2-路径错误](scripts/README.md#Q2-路径错误)
- **训练中断** → [README_REPRODUCTION.md#Q4-训练中断如何恢复](docs/README_REPRODUCTION.md#Q4-训练中断如何恢复)
- **环境问题** → [CONDA_SETUP.md#常见问题解决](docs/CONDA_SETUP.md#常见问题解决)

## 📱 快捷命令

### 环境相关
```bash
# 配置环境
bash scripts/setup_conda_env.sh
conda activate ccks2025_pdf_qa

# 检查环境
bash scripts/check_environment.sh

# 查看GPU
nvidia-smi
```

### 运行相关
```bash
# 配置路径
bash scripts/00_setup_paths.sh /data/coding

# 启动tmux
tmux new -s ccks2025

# 运行流程
bash scripts/01_preprocess.sh
bash scripts/02_train.sh
bash scripts/03_inference.sh
```

### 监控相关
```bash
# 查看日志
tail -f ccks2025_pdf_multimodal/round_b/preprocess.log
tail -f ccks2025_pdf_multimodal/round_b/train.log
tail -f ccks2025_pdf_multimodal/round_b/inference.log

# 查看进度
ls -lh ccks2025_pdf_multimodal/round_b/*_vectors.npy
ls -lh /data/coding/lora_qwen25_vl_32b_b/checkpoint-*

# GPU监控
watch -n 5 nvidia-smi
```

## ⚡ 关键文件速查

| 我需要... | 查看文件 |
|-----------|----------|
| 快速开始5分钟 | [QUICKSTART.md](docs/QUICKSTART.md) |
| 完整复现步骤 | [REPRODUCTION_GUIDE.md](docs/REPRODUCTION_GUIDE.md) |
| 环境配置帮助 | [CONDA_SETUP.md](docs/CONDA_SETUP.md) |
| 脚本使用说明 | [scripts/README.md](scripts/README.md) |
| 项目架构理解 | [CLAUDE.md](ccks2025_pdf_multimodal/CLAUDE.md) |
| 深度技术分析 | [技术分析报告.md](ccks2025_pdf_multimodal/技术分析报告.md) |
| 依赖包列表 | [requirements.txt](requirements.txt) |
| 环境配置文件 | [environment.yml](environment.yml) |

---

**找不到需要的信息？**
1. 查看 [README.md](README.md)
2. 运行 `bash scripts/check_environment.sh`
3. 查看对应文档的目录
