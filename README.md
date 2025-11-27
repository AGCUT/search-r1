# CCKS 2025 专利问答多模态项目

本项目提供在8卡A800服务器上完整复现CCKS 2025专利问答多模态竞赛的自动化脚本和详细文档。

## 📁 项目结构

```
pdf/
├── scripts/                          # 🔧 运行脚本目录
│   ├── setup_conda_env.sh           # Conda环境自动配置
│   ├── check_environment.sh         # 环境检查工具
│   ├── 00_setup_paths.sh            # 路径配置
│   ├── 01_preprocess.sh             # 数据预处理 (GPU 1, 6-8h)
│   ├── 02_train.sh                  # 模型训练 (GPU 0-4, 6-8h)
│   ├── 03_inference.sh              # 模型推理 (GPU 0-3, 2-3h)
│   └── README.md                    # 脚本使用说明
│
├── docs/                             # 📚 文档目录
│   ├── QUICKSTART.md                # 5分钟快速开始
│   ├── REPRODUCTION_GUIDE.md        # 完整复现指南
│   ├── CONDA_SETUP.md               # Conda环境配置详细说明
│   ├── README_REPRODUCTION.md       # 总览文档
│   └── gpu.md                       # GPU状态记录
│
├── ccks2025_pdf_multimodal/         # 🎯 项目代码目录
│   ├── round_a/                     # 初赛代码（探索性）
│   ├── round_b/                     # 复赛代码（生产级）
│   │   ├── b_train_test_preprocess.py      # 预处理核心代码
│   │   ├── test_b_style_refer_215.py       # 推理核心代码
│   │   ├── gme_inference.py                # GME嵌入模型封装
│   │   ├── finetune训练集构造_v2.ipynb     # 训练集构造
│   │   └── train_vl_32b.sh                 # 原始训练脚本
│   ├── CLAUDE.md                    # 项目架构文档
│   └── 技术分析报告.md               # 深度技术分析
│
├── data/                             # 📦 数据目录（需要下载）
│   ├── original_problems.zip
│   ├── preliminary_dataset.zip
│   └── semi_final_dataset.zip
│
├── requirements.txt                  # Python依赖列表
├── environment.yml                   # Conda环境配置
└── README.md                         # 本文件
```

## 🚀 快速开始

### 1. 配置环境（一键完成）

```bash
# 在项目根目录运行
bash scripts/setup_conda_env.sh

# 激活环境
conda activate ccks2025_pdf_qa
```

### 2. 检查环境

```bash
bash scripts/check_environment.sh
```

### 3. 配置路径

```bash
bash scripts/00_setup_paths.sh /data/coding
```

### 4. 开始复现

```bash
# 启动tmux会话
tmux new -s ccks2025

# 预处理 (6-8小时)
bash scripts/01_preprocess.sh

# 构造训练集 (手动运行notebook)
cd ccks2025_pdf_multimodal/round_b
jupyter notebook finetune训练集构造_v2.ipynb
cd ../..

# 训练 (6-8小时)
bash scripts/02_train.sh

# 推理 (2-3小时)
bash scripts/03_inference.sh
```

## 📚 文档导航

### 新手入门
1. **[QUICKSTART.md](docs/QUICKSTART.md)** - 5分钟快速开始指南
2. **[REPRODUCTION_GUIDE.md](docs/REPRODUCTION_GUIDE.md)** - 完整详细的复现步骤
3. **[scripts/README.md](scripts/README.md)** - 脚本使用说明

### 环境配置
- **[CONDA_SETUP.md](docs/CONDA_SETUP.md)** - Conda环境配置详细指南
- **[requirements.txt](requirements.txt)** - Python依赖包列表
- **[environment.yml](environment.yml)** - Conda环境配置文件

### 技术文档
- **[CLAUDE.md](ccks2025_pdf_multimodal/CLAUDE.md)** - 项目架构和技术细节
- **[技术分析报告.md](ccks2025_pdf_multimodal/技术分析报告.md)** - 深度技术分析

## 💻 服务器配置

当前GPU配置 (2025-11-25):
- **GPU 0**: 可用约14GB
- **GPU 1**: ✅ 可用约74GB (最空闲)
- **GPU 2**: 可用约41GB
- **GPU 3**: 可用约57GB
- **GPU 4**: 可用约16GB
- **GPU 5-7**: ❌ 正在使用，请避开

**脚本GPU使用策略**:
```
预处理: GPU 1 (单卡)
训练:   GPU 0,1,2,3,4 (5卡)
推理:   GPU 0,1,2,3 (4卡)
```

## 🎯 复现流程概览

```
步骤0: 环境准备
  ↓ (1小时)
步骤1: 数据预处理 [GPU 1]
  ↓ (6-8小时)
步骤2: 构造训练集 [手动]
  ↓ (30分钟)
步骤3: 模型训练 [GPU 0-4]
  ↓ (6-8小时)
步骤4: 模型推理 [GPU 0-3]
  ↓ (2-3小时)
完成！
```

**总耗时**: 约17小时（可在tmux后台运行）

## 📦 所需资源

### 模型 (~78GB)
- GME-Qwen2-VL-7B-Instruct: ~14GB
- Qwen2.5-VL-32B-Instruct: ~64GB

### 数据集
- 训练集: ~1000个PDF
- 测试集: ~800个PDF

### 硬件要求
- 至少5张A800 GPU (80GB)
- 磁盘空间: >200GB
- 内存: >128GB

## 🔍 核心特性

### 自动化脚本
- ✅ 一键环境配置
- ✅ 自动路径配置
- ✅ 完整的错误检查
- ✅ 详细的进度日志
- ✅ 自动checkpoint管理

### GPU优化
- ✅ 根据当前GPU占用智能分配
- ✅ 支持5卡/8卡训练配置
- ✅ 自动OOM检测和建议

### 文档完善
- ✅ 5分钟快速开始指南
- ✅ 详细的复现步骤文档
- ✅ 完整的故障排查指南
- ✅ 深度技术分析报告

## 🐛 常见问题

### Q: 环境配置失败？
A: 查看 [CONDA_SETUP.md](docs/CONDA_SETUP.md) 的故障排查章节

### Q: GPU显存不足？
A: 查看 [scripts/README.md](scripts/README.md) 的OOM解决方案

### Q: 路径配置错误？
A: 运行 `bash scripts/00_setup_paths.sh /your/actual/path`

### Q: 训练中断怎么办？
A: SWIFT会自动从最新checkpoint恢复，直接重新运行 `bash scripts/02_train.sh`

## 📞 获取帮助

### 文档
- **快速问题**: 查看 [QUICKSTART.md](docs/QUICKSTART.md)
- **详细步骤**: 查看 [REPRODUCTION_GUIDE.md](docs/REPRODUCTION_GUIDE.md)
- **脚本说明**: 查看 [scripts/README.md](scripts/README.md)

### 运行检查
```bash
# 环境检查
bash scripts/check_environment.sh

# 查看GPU状态
nvidia-smi

# 查看日志
tail -f ccks2025_pdf_multimodal/round_b/train.log
```

### 原始项目
- **竞赛页面**: https://tianchi.aliyun.com/competition/entrance/532357/information
- **技术博客**: https://zhuanlan.zhihu.com/p/1937478905532506972
- **数据分享**: https://pan.baidu.com/s/10fYbAaSNf-Nv7x1jsMiqzw (提取码: r96k)

## ✅ 检查清单

开始复现前请确认：
- [ ] 已查看GPU状态，确认可用GPU
- [ ] 已配置Conda环境: `bash scripts/setup_conda_env.sh`
- [ ] 已运行环境检查: `bash scripts/check_environment.sh`
- [ ] 已下载模型 (GME + Qwen2.5-VL-32B, ~78GB)
- [ ] 已下载数据集 (训练集 + 测试集)
- [ ] 已配置路径: `bash scripts/00_setup_paths.sh /data/coding`
- [ ] 已启动tmux: `tmux new -s ccks2025`
- [ ] 磁盘空间充足 (>200GB)

## 🎉 预期结果

复现成功后你将得到：
- 训练集和测试集的向量文件 (~500MB)
- 微调后的模型checkpoints (~5GB)
- 测试集预测结果: `ccks2025_pdf_multimodal/round_b/test_b_style_infer_if_need_ck215.jsonl`

---

## 🌟 项目亮点

本项目不仅提供了完整的代码，还包含：
- 🚀 **自动化程度高**: 从环境配置到推理，全程自动化
- 📖 **文档详尽**: 5份文档覆盖从入门到深度分析
- 🔧 **工程化完善**: 错误检查、日志记录、进度监控
- 💡 **GPU优化**: 根据实际情况智能分配GPU资源
- 🎯 **开箱即用**: 一键脚本，降低使用门槛

**立即开始**: `bash scripts/setup_conda_env.sh`
