# 运行脚本说明

本目录包含用于复现CCKS 2025专利问答项目的自动化脚本。

## 📁 脚本列表

| 脚本 | 功能 | GPU使用 | 耗时 | 运行位置 |
|------|------|---------|------|----------|
| `setup_conda_env.sh` | Conda环境配置 | 无 | 10-20分钟 | 项目根目录 |
| `check_environment.sh` | 环境检查 | 无 | <1分钟 | 项目根目录 |
| `00_setup_paths.sh` | 路径配置 | 无 | <1分钟 | 项目根目录 |
| `01_preprocess.sh` | 数据预处理 | GPU 1 | 6-8小时 | 项目根目录 |
| `02_train.sh` | 模型训练 | GPU 0-4 | 6-8小时 | 项目根目录 |
| `03_inference.sh` | 模型推理 | GPU 0-3 | 2-3小时 | 项目根目录 |

## 🚀 使用步骤

### 前提条件
- 已在项目根目录 (`pdf/`) 下
- 已激活conda环境（如果使用）: `conda activate ccks2025_pdf_qa`

### 0. 配置Conda环境（首次运行）

```bash
# 在项目根目录运行
bash scripts/setup_conda_env.sh
conda activate ccks2025_pdf_qa
```

### 1. 环境检查

```bash
bash scripts/check_environment.sh
```

这个脚本会检查：
- Python版本和依赖包
- CUDA和GPU状态
- 模型和数据路径
- 磁盘空间
- 项目结构

### 2. 配置路径

```bash
bash scripts/00_setup_paths.sh /data/coding
```

这个脚本会：
- 自动修改 `round_b` 中Python脚本的路径
- 备份原始文件
- 验证目录结构

### 3. 数据预处理

```bash
# 在tmux会话中运行
tmux new -s preprocess
bash scripts/01_preprocess.sh
# Ctrl+B, D 分离会话
```

**功能**:
- PDF转JPG图像 (600 DPI)
- 生成图像向量 (GME嵌入, 3584维)
- 生成问题向量
- 验证输出文件

**输出文件** (位于 `ccks2025_pdf_multimodal/round_b/`):
- `train_b_pdf_img_vectors.npy` (~270MB)
- `train_b_pdf_img_page_num_mapping.csv`
- `all_train_b_question_vectors.npy`
- `test_b_pdf_img_vectors.npy` (~200MB)
- `test_b_pdf_img_page_num_mapping.csv`
- `all_test_b_question_vectors.npy`

### 4. 构造训练集

```bash
cd ccks2025_pdf_multimodal/round_b
jupyter notebook finetune训练集构造_v2.ipynb
```

按顺序执行notebook中的所有单元格。

**输出**: `train_b_dataset_for_image_0801.jsonl`

### 5. 模型训练

```bash
# 返回项目根目录
cd ../..

# 在tmux会话中运行
tmux new -s train
bash scripts/02_train.sh
# Ctrl+B, D 分离会话
```

**功能**:
- LoRA微调Qwen2.5-VL-32B
- 每10步保存checkpoint
- 保留最近4个checkpoint

**输出**: Checkpoints保存在 `/data/coding/lora_qwen25_vl_32b_b/`

### 6. 模型推理

```bash
# 在tmux会话中运行
tmux new -s inference
bash scripts/03_inference.sh
# Ctrl+B, D 分离会话
```

**功能**:
- 自动查找最新checkpoint
- 自动合并LoRA权重（如需要）
- 生成测试集答案

**输出**: `ccks2025_pdf_multimodal/round_b/test_b_style_infer_if_need_ck215.jsonl`

## 📋 脚本详细说明

### setup_conda_env.sh

**用途**: 一键配置Conda环境

**功能**:
- 自动检测CUDA版本
- 创建名为 `ccks2025_pdf_qa` 的环境
- 安装Python 3.10
- 安装匹配的PyTorch版本
- 安装所有依赖包
- 验证安装

**使用方法**:
```bash
bash scripts/setup_conda_env.sh
```

### check_environment.sh

**用途**: 全面的环境检查工具

**检查项**:
1. Python版本 (需要 >= 3.10)
2. 核心依赖包 (torch, transformers, vllm, swift, 等)
3. CUDA和GPU状态
4. 项目路径配置
5. 磁盘空间 (需要 > 200GB)
6. 运行脚本是否存在
7. tmux是否安装
8. 项目结构完整性

**使用方法**:
```bash
bash scripts/check_environment.sh
```

### 00_setup_paths.sh

**用途**: 自动配置所有Python脚本中的路径

**参数**:
- `$1`: 项目根目录路径 (默认: `/data/coding`)

**修改的文件**:
- `ccks2025_pdf_multimodal/round_b/b_train_test_preprocess.py`
- `ccks2025_pdf_multimodal/round_b/test_b_style_refer_215.py`
- `ccks2025_pdf_multimodal/round_b/test_b_style_refer_90.py`
- `ccks2025_pdf_multimodal/round_b/gme_inference.py`

**使用方法**:
```bash
bash scripts/00_setup_paths.sh /data/coding
```

**备份**: 原始文件会备份到 `path_backups_<timestamp>/` 目录

### 01_preprocess.sh

**用途**: 数据预处理自动化

**GPU**: GPU 1 (单卡，最空闲)

**环境变量**:
- `CUDA_VISIBLE_DEVICES=1`
- `MAX_PIXELS=1229312`

**配置路径** (在脚本中修改):
- `PROJECT_ROOT`: 项目根目录
- `PATENT_DATA_DIR`: 数据目录
- `GME_MODEL_PATH`: GME模型路径

**使用方法**:
```bash
bash scripts/01_preprocess.sh
```

**监控进度**:
```bash
# 查看日志
tail -f ccks2025_pdf_multimodal/round_b/preprocess.log

# 查看生成的文件
ls -lh ccks2025_pdf_multimodal/round_b/*_vectors.npy
```

### 02_train.sh

**用途**: 模型训练自动化

**GPU**: GPU 0,1,2,3,4 (5卡并行)

**环境变量**:
- `CUDA_VISIBLE_DEVICES=0,1,2,3,4`
- `MAX_PIXELS=1229312`

**训练参数**:
- LoRA rank: 8
- LoRA alpha: 32
- Learning rate: 1e-4
- Epochs: 5
- Batch size per device: 1
- Gradient accumulation: 16
- Effective batch size: 80

**使用方法**:
```bash
bash scripts/02_train.sh
```

**监控训练**:
```bash
# 查看日志
tail -f ccks2025_pdf_multimodal/round_b/train.log

# 查看checkpoints
ls -lh /data/coding/lora_qwen25_vl_32b_b/checkpoint-*

# 监控GPU
watch -n 5 nvidia-smi
```

### 03_inference.sh

**用途**: 模型推理自动化

**GPU**: GPU 0,1,2,3 (4卡并行)

**环境变量**:
- `CUDA_VISIBLE_DEVICES=0,1,2,3`
- `MAX_PIXELS=1568000`

**功能**:
- 自动查找最新checkpoint
- 自动合并LoRA权重（如需要）
- 更新推理脚本中的模型路径
- 生成测试集答案

**使用方法**:
```bash
bash scripts/03_inference.sh
```

**监控推理**:
```bash
# 查看日志
tail -f ccks2025_pdf_multimodal/round_b/inference.log

# 查看进度
wc -l ccks2025_pdf_multimodal/round_b/test_b_style_infer_if_need_ck215.jsonl
```

## 🐛 常见问题

### Q1: OOM错误

**症状**: 训练或推理时出现 CUDA Out of Memory

**解决方案**:
```bash
# 方案1: 降低MAX_PIXELS
# 编辑对应脚本，将 MAX_PIXELS 降低
export MAX_PIXELS=819200  # 从1229312降低

# 方案2: 使用更少的GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 从5卡减少到4卡

# 方案3: 增加梯度累积步数
# 编辑 02_train.sh
--gradient_accumulation_steps 32  # 从16增加到32
```

### Q2: 路径错误

**症状**: FileNotFoundError

**解决方案**:
```bash
# 重新运行路径配置脚本
bash scripts/00_setup_paths.sh /your/actual/path

# 手动检查需要修改的文件
grep -r "/data/coding" ccks2025_pdf_multimodal/round_b/*.py
```

### Q3: 脚本权限错误

**症状**: Permission denied

**解决方案**:
```bash
# 添加执行权限
chmod +x scripts/*.sh

# 或单独添加
chmod +x scripts/01_preprocess.sh
```

### Q4: tmux会话断开

**症状**: SSH断开后进程终止

**解决方案**:
```bash
# 查看所有tmux会话
tmux ls

# 重新连接会话
tmux attach -t preprocess

# 创建新会话
tmux new -s ccks2025
```

### Q5: GPU被占用

**症状**: GPU已满无法运行

**解决方案**:
```bash
# 查看GPU状态
nvidia-smi

# 修改脚本使用其他GPU
# 编辑对应脚本，修改 CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=1,2,3  # 使用不同的GPU
```

## 📈 性能优化

### 1. 加速预处理
- 使用SSD存储中间文件
- 增加CPU核心数进行并行PDF转换

### 2. 加速训练
- 使用DeepSpeed进行更高效的分布式训练
- 增大有效batch size

### 3. 加速推理
- 增加批处理大小 (`max_num_seqs=4`)
- 使用INT8量化

## 🔧 自定义配置

### 修改GPU分配

编辑对应脚本中的 `CUDA_VISIBLE_DEVICES`:

```bash
# 01_preprocess.sh
export CUDA_VISIBLE_DEVICES=1  # 改为其他GPU

# 02_train.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4  # 改为其他GPU组合

# 03_inference.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 改为其他GPU组合
```

### 修改路径配置

编辑对应脚本中的路径变量:

```bash
export PROJECT_ROOT=/your/path
export PATENT_DATA_DIR=$PROJECT_ROOT/patent_b
export GME_MODEL_PATH=$PROJECT_ROOT/llm_model/iic/gme-Qwen2-VL-7B-Instruct
```

### 修改训练参数

编辑 `02_train.sh` 中的参数:

```bash
--lora_rank 16              # 从8改为16
--num_train_epochs 10       # 从5改为10
--learning_rate 5e-5        # 从1e-4改为5e-5
```

## 📝 日志文件

所有脚本都会生成日志文件：

- `ccks2025_pdf_multimodal/round_b/preprocess.log` - 预处理日志
- `ccks2025_pdf_multimodal/round_b/train.log` - 训练日志
- `ccks2025_pdf_multimodal/round_b/inference.log` - 推理日志

查看日志:
```bash
# 实时查看
tail -f ccks2025_pdf_multimodal/round_b/train.log

# 查找错误
grep -i error ccks2025_pdf_multimodal/round_b/train.log

# 查找警告
grep -i warning ccks2025_pdf_multimodal/round_b/train.log
```

## 🎯 完整执行流程

```bash
# 1. 配置环境（首次）
bash scripts/setup_conda_env.sh
conda activate ccks2025_pdf_qa

# 2. 环境检查
bash scripts/check_environment.sh

# 3. 配置路径
bash scripts/00_setup_paths.sh /data/coding

# 4. 启动tmux
tmux new -s ccks2025

# 5. 预处理 (6-8h)
bash scripts/01_preprocess.sh
# Ctrl+B, D 分离

# 6. 构造训练集 (30min)
cd ccks2025_pdf_multimodal/round_b
jupyter notebook finetune训练集构造_v2.ipynb
cd ../..

# 7. 训练 (6-8h)
bash scripts/02_train.sh
# Ctrl+B, D 分离

# 8. 推理 (2-3h)
bash scripts/03_inference.sh

# 9. 完成！
echo "结果: ccks2025_pdf_multimodal/round_b/test_b_style_infer_if_need_ck215.jsonl"
```

## 📞 获取帮助

- **详细指南**: `docs/REPRODUCTION_GUIDE.md`
- **快速开始**: `docs/QUICKSTART.md`
- **Conda配置**: `docs/CONDA_SETUP.md`
- **项目文档**: `ccks2025_pdf_multimodal/CLAUDE.md`

---

**重要提示**: 所有脚本都应该在项目根目录 (`pdf/`) 下运行，而不是在 `round_a` 或 `round_b` 目录中运行。
