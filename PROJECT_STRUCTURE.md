# 项目结构说明

## 📊 完整目录树

```
pdf/                                    # 项目根目录
│
├── 📄 README.md                        # 项目主页和快速开始
├── 📄 FILE_INDEX.md                    # 文件导航索引
├── 📄 PROJECT_STRUCTURE.md             # 本文件：项目结构说明
├── 📄 requirements.txt                 # Python依赖列表
├── 📄 environment.yml                  # Conda环境配置
│
├── 📁 scripts/                         # 🔧 运行脚本目录
│   ├── setup_conda_env.sh             # Conda环境自动配置脚本
│   ├── check_environment.sh           # 环境检查工具
│   ├── 00_setup_paths.sh              # 路径配置脚本
│   ├── 01_preprocess.sh               # 数据预处理脚本 (GPU 1)
│   ├── 02_train.sh                    # 模型训练脚本 (GPU 0-4)
│   ├── 03_inference.sh                # 模型推理脚本 (GPU 0-3)
│   └── README.md                      # 脚本详细使用说明
│
├── 📁 docs/                            # 📚 文档目录
│   ├── QUICKSTART.md                  # 5分钟快速开始指南
│   ├── REPRODUCTION_GUIDE.md          # 完整详细的复现指南
│   ├── CONDA_SETUP.md                 # Conda环境配置详细说明
│   ├── README_REPRODUCTION.md         # 复现总览文档
│   └── gpu.md                         # GPU状态记录
│
├── 📁 ccks2025_pdf_multimodal/        # 🎯 项目代码目录
│   │
│   ├── 📄 CLAUDE.md                   # 项目架构和技术细节文档
│   ├── 📄 技术分析报告.md              # 深度技术分析报告
│   ├── 📄 README.md                   # 原始项目说明
│   │
│   ├── 📁 round_b/                    # 复赛代码（主要使用）
│   │   ├── b_train_test_preprocess.py        # 预处理核心代码
│   │   ├── test_b_style_refer_215.py         # 推理核心代码
│   │   ├── test_b_style_refer_90.py          # 备选推理代码
│   │   ├── gme_inference.py                  # GME嵌入模型封装
│   │   ├── finetune训练集构造_v2.ipynb       # 训练集构造notebook
│   │   ├── train_vl_32b.sh                   # 原始训练脚本
│   │   └── (运行时生成的文件)
│   │       ├── *_vectors.npy                 # 向量文件
│   │       ├── *_mapping.csv                 # 映射文件
│   │       ├── train_b_dataset_*.jsonl       # 训练数据
│   │       └── test_b_style_*.jsonl          # 推理结果
│   │
│   ├── 📁 round_a/                    # 初赛代码（探索性，参考用）
│   │   ├── finetune训练集构造.ipynb
│   │   ├── gme_inference.py
│   │   ├── run_test_qwen3_32b.py
│   │   ├── test_img_*.py
│   │   ├── train_*.sh
│   │   └── *.ipynb
│   │
│   ├── 📁 choice_pipeline/            # 备选方案（参考用）
│   │   ├── choice_rag3_*.py
│   │   ├── gme_inference.py
│   │   ├── train_*.sh
│   │   └── *.ipynb
│   │
│   └── 📁 pic/                        # 文档图片
│       └── *.png
│
└── 📁 data/                            # 📦 数据目录
    ├── original_problems.zip
    ├── preliminary_dataset.zip
    └── semi_final_dataset.zip
```

## 🎯 目录设计原则

### 1. 根目录 (pdf/)
- **作用**: 项目入口，包含核心配置文件
- **运行位置**: 所有脚本都应在此目录运行
- **包含文件**:
  - 主文档 (README.md)
  - 配置文件 (requirements.txt, environment.yml)
  - 导航文件 (FILE_INDEX.md, PROJECT_STRUCTURE.md)

### 2. scripts/ 目录
- **作用**: 存放所有自动化运行脚本
- **特点**:
  - 独立于round_a和round_b
  - 可以操作不同的测试集
  - 从项目根目录调用
- **命名规范**:
  - 数字前缀表示执行顺序 (00, 01, 02, 03)
  - 描述性名称 (setup_paths, preprocess, train, inference)

### 3. docs/ 目录
- **作用**: 存放所有文档
- **文档分类**:
  - 入门文档: QUICKSTART.md
  - 详细文档: REPRODUCTION_GUIDE.md
  - 专题文档: CONDA_SETUP.md
  - 总览文档: README_REPRODUCTION.md

### 4. ccks2025_pdf_multimodal/ 目录
- **作用**: 原始项目代码，保持结构不变
- **子目录**:
  - round_a: 初赛探索性代码
  - round_b: 复赛生产级代码（主要使用）
  - choice_pipeline: 备选实现方案
- **特点**: 不在此目录内放置运行脚本

### 5. data/ 目录
- **作用**: 存放下载的数据集
- **注意**: 实际数据应存放在服务器上的 `/data/coding/patent_b/`

## 🔄 文件流转关系

### 环境准备阶段
```
requirements.txt
environment.yml
   ↓
scripts/setup_conda_env.sh
   ↓
conda环境: ccks2025_pdf_qa
```

### 路径配置阶段
```
scripts/00_setup_paths.sh
   ↓
修改 ccks2025_pdf_multimodal/round_b/*.py
```

### 数据处理阶段
```
scripts/01_preprocess.sh
   ↓
ccks2025_pdf_multimodal/round_b/b_train_test_preprocess.py
   ↓
生成: round_b/*_vectors.npy, *_mapping.csv
```

### 训练准备阶段
```
ccks2025_pdf_multimodal/round_b/finetune训练集构造_v2.ipynb
   ↓
使用: round_b/*_vectors.npy
   ↓
生成: round_b/train_b_dataset_*.jsonl
```

### 模型训练阶段
```
scripts/02_train.sh
   ↓
使用: round_b/train_b_dataset_*.jsonl
   ↓
生成: /data/coding/lora_qwen25_vl_32b_b/checkpoint-*
```

### 模型推理阶段
```
scripts/03_inference.sh
   ↓
ccks2025_pdf_multimodal/round_b/test_b_style_refer_215.py
   ↓
使用: checkpoint-*, *_vectors.npy
   ↓
生成: round_b/test_b_style_infer_*.jsonl
```

## 📝 文件命名规范

### 脚本文件
- **前缀编号**: `00_`, `01_`, `02_`, `03_` 表示执行顺序
- **描述性名称**: 清晰说明功能
- **扩展名**: `.sh` for bash scripts

### 文档文件
- **全大写**: 重要文档 (README.md, QUICKSTART.md)
- **描述性**: 说明内容 (CONDA_SETUP.md)
- **扩展名**: `.md` for Markdown

### 数据文件
- **前缀**: `train_b_`, `test_b_`, `all_`
- **描述**: 说明内容 (vectors, mapping, dataset)
- **扩展名**: `.npy`, `.csv`, `.jsonl`

## 🎯 使用场景

### 场景1: 首次使用
1. 阅读 `README.md` 了解项目
2. 阅读 `docs/QUICKSTART.md` 快速开始
3. 运行 `scripts/setup_conda_env.sh` 配置环境
4. 运行 `scripts/check_environment.sh` 检查环境
5. 运行 `scripts/00_setup_paths.sh` 配置路径
6. 按顺序运行 01, 02, 03 脚本

### 场景2: 查找文档
1. 打开 `FILE_INDEX.md` 查看文件导航
2. 根据需求查找对应文档
3. 或使用 `FILE_INDEX.md` 中的快速查找

### 场景3: 运行不同测试集
```bash
# 所有脚本从根目录运行
cd /path/to/pdf

# 运行 round_b 测试集（默认）
bash scripts/01_preprocess.sh
bash scripts/02_train.sh
bash scripts/03_inference.sh

# 如需运行 round_a，修改脚本中的路径即可
# round_a 和 round_b 是独立的目录，互不干扰
```

### 场景4: 查看技术细节
1. 查看 `ccks2025_pdf_multimodal/CLAUDE.md` 了解架构
2. 查看 `ccks2025_pdf_multimodal/技术分析报告.md` 深度分析
3. 阅读 `ccks2025_pdf_multimodal/round_b/` 中的代码

## ⚙️ 运行位置说明

### ✅ 正确的运行位置

所有脚本都应该在项目根目录 (`pdf/`) 运行：

```bash
# 正确 ✓
cd /path/to/pdf
bash scripts/01_preprocess.sh
bash scripts/02_train.sh
bash scripts/03_inference.sh
```

### ❌ 错误的运行位置

不要在 round_a 或 round_b 目录中运行脚本：

```bash
# 错误 ✗
cd /path/to/pdf/ccks2025_pdf_multimodal/round_b
bash ../../scripts/01_preprocess.sh  # 路径会错误
```

## 🔍 文件角色说明

### 配置文件
- `requirements.txt`: pip依赖列表
- `environment.yml`: conda环境配置
- `gpu.md`: GPU状态记录（参考）

### 运行脚本
- `setup_conda_env.sh`: 环境配置（一次性）
- `check_environment.sh`: 环境检查（随时可用）
- `00_setup_paths.sh`: 路径配置（一次性或更新时）
- `01_preprocess.sh`: 预处理（每个数据集运行一次）
- `02_train.sh`: 训练（每次训练运行）
- `03_inference.sh`: 推理（每次推理运行）

### 入口文档
- `README.md`: 项目主页
- `FILE_INDEX.md`: 文件导航
- `PROJECT_STRUCTURE.md`: 本文件

### 指导文档
- `docs/QUICKSTART.md`: 5分钟快速开始
- `docs/REPRODUCTION_GUIDE.md`: 完整复现指南
- `docs/CONDA_SETUP.md`: Conda配置指南
- `scripts/README.md`: 脚本使用说明

### 技术文档
- `ccks2025_pdf_multimodal/CLAUDE.md`: 架构文档
- `ccks2025_pdf_multimodal/技术分析报告.md`: 技术分析

### 代码文件
- `round_b/*.py`: 核心代码
- `round_b/*.ipynb`: 交互式notebook
- `round_b/*.sh`: 原始脚本（参考）

## 📊 文件数量统计

- **脚本**: 6个 (scripts/ 目录)
- **文档**: 11个 (根目录 + docs/ + ccks2025_pdf_multimodal/)
- **配置**: 2个 (requirements.txt, environment.yml)
- **核心代码**: 4个主要Python文件 (round_b/)
- **Notebook**: 1个主要 (round_b/)

## 🎉 设计优势

1. **清晰分离**: 脚本、文档、代码各自独立
2. **易于维护**: 统一的脚本和文档位置
3. **便于扩展**: 可轻松添加新脚本或文档
4. **避免混乱**: round_a 和 round_b 保持独立
5. **易于导航**: FILE_INDEX.md 提供快速导航

---

**下一步**: 查看 [README.md](README.md) 开始使用项目
