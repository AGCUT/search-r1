# 文件索引和说明

## 📂 目录结构

```
round_b/
├── scripts/                          # 自动化运行脚本
│   ├── check_environment.sh         # ⭐ 环境检查工具
│   ├── 00_setup_paths.sh            # ⭐ 路径配置脚本
│   ├── 01_preprocess.sh             # ⭐ 预处理脚本 (GPU 1)
│   ├── 02_train.sh                  # ⭐ 训练脚本 (GPU 0-4)
│   ├── 03_inference.sh              # ⭐ 推理脚本 (GPU 0-3)
│   └── README.md                    # 脚本使用说明
│
├── b_train_test_preprocess.py       # 核心预处理代码
├── test_b_style_refer_215.py        # 核心推理代码
├── test_b_style_refer_90.py         # 备选推理代码
├── gme_inference.py                 # GME嵌入模型封装
├── finetune训练集构造_v2.ipynb      # 训练集构造notebook
├── train_vl_32b.sh                  # 原始训练脚本
│
└── FILE_INDEX.md                    # 本文件
```

## 📄 文件说明

### 运行脚本 (scripts/)

#### check_environment.sh ⭐⭐⭐⭐⭐
**用途**: 运行前的环境检查工具
**功能**:
- 检查Python版本和依赖包
- 检查CUDA和GPU状态
- 检查模型和数据路径
- 检查磁盘空间
- 检查tmux安装

**使用方法**:
```bash
bash scripts/check_environment.sh
```

**何时使用**: 开始复现前的第一步

---

#### 00_setup_paths.sh ⭐⭐⭐⭐⭐
**用途**: 自动配置所有脚本中的路径
**功能**:
- 批量修改Python脚本中的路径
- 自动备份原始文件
- 验证目录结构

**使用方法**:
```bash
bash scripts/00_setup_paths.sh /data/coding
```

**何时使用**: 环境检查通过后的第二步

---

#### 01_preprocess.sh ⭐⭐⭐⭐⭐
**用途**: 数据预处理自动化脚本
**GPU**: GPU 1 (单卡)
**耗时**: 6-8小时
**功能**:
- PDF转JPG图像 (600 DPI)
- 生成图像向量 (GME嵌入)
- 生成问题向量
- 验证输出文件

**输出文件**:
- `train_b_pdf_img_vectors.npy` (~270MB)
- `train_b_pdf_img_page_num_mapping.csv`
- `all_train_b_question_vectors.npy`
- `test_b_pdf_img_vectors.npy` (~200MB)
- `test_b_pdf_img_page_num_mapping.csv`
- `all_test_b_question_vectors.npy`

**使用方法**:
```bash
tmux new -s preprocess
bash scripts/01_preprocess.sh
# Ctrl+B, D 分离会话
```

**何时使用**: 路径配置完成后

---

#### 02_train.sh ⭐⭐⭐⭐⭐
**用途**: 模型训练自动化脚本
**GPU**: GPU 0,1,2,3,4 (5卡并行)
**耗时**: 6-8小时
**功能**:
- LoRA微调Qwen2.5-VL-32B
- 自动保存checkpoints
- 监控训练进度

**输出文件**:
- `/data/coding/lora_qwen25_vl_32b_b/checkpoint-*` (~5GB)

**使用方法**:
```bash
tmux new -s train
bash scripts/02_train.sh
# Ctrl+B, D 分离会话
```

**何时使用**: 训练集构造完成后

**重要参数**:
- `MAX_PIXELS=1229312` - 图像分辨率
- `lora_rank=8` - LoRA rank
- `num_train_epochs=5` - 训练轮次
- `gradient_accumulation_steps=16` - 梯度累积

---

#### 03_inference.sh ⭐⭐⭐⭐⭐
**用途**: 模型推理自动化脚本
**GPU**: GPU 0,1,2,3 (4卡并行)
**耗时**: 2-3小时
**功能**:
- 自动查找最新checkpoint
- 自动合并LoRA权重（如需要）
- 生成测试集答案
- 验证输出格式

**输出文件**:
- `test_b_style_infer_if_need_ck215.jsonl` (~1MB)

**使用方法**:
```bash
tmux new -s inference
bash scripts/03_inference.sh
# Ctrl+B, D 分离会话
```

**何时使用**: 训练完成后

---

### 核心Python文件

#### b_train_test_preprocess.py ⭐⭐⭐⭐⭐
**用途**: 数据预处理的核心实现
**被调用**: `01_preprocess.sh`
**主要功能**:
1. PDF转JPG (使用PyMuPDF/fitz)
2. 图像向量化 (使用GME模型)
3. 问题向量化
4. 保存映射关系

**关键代码段**:
```python
# PDF转图像
pdf_document = fitz.open(pdf_path)
page = pdf_document.load_page(i)
pix = page.get_pixmap(dpi=600)
pix.save(output_path)

# 生成向量
gme = GmeQwen2VL(model_name='gme-Qwen2-VL-7B-Instruct')
embeddings = gme.get_image_embeddings(images=[image_path])
```

**配置项**:
- `base_dir` - 数据路径
- `MAX_PIXELS` - 图像分辨率
- `CUDA_VISIBLE_DEVICES` - GPU选择

---

#### test_b_style_refer_215.py ⭐⭐⭐⭐⭐
**用途**: 推理的核心实现
**被调用**: `03_inference.sh`
**主要功能**:
1. 加载微调后的模型
2. 向量检索 (top-2相似页面)
3. 风格一致性控制 (检索相似训练样本)
4. 生成答案

**关键流程**:
```python
# 1. 检索相似页面
similar_pages = get_similar_image_embedding(question_idx, top_k=2)

# 2. 检索相似训练问题（用于风格参考）
similar_questions = get_similar_question_embedding(question_idx, top_k=2)

# 3. 生成答案
answer = vl_model.generate(images + question + style_examples)

# 4. 答案精炼
final_answer = extract_concise_answer(answer, style_examples)
```

**配置项**:
- `model_path` - 模型checkpoint路径
- `MAX_PIXELS` - 推理分辨率 (1568000)
- `CUDA_VISIBLE_DEVICES` - GPU选择 (0,1,2,3)

---

#### gme_inference.py ⭐⭐⭐⭐
**用途**: GME嵌入模型的封装
**被调用**: `b_train_test_preprocess.py`, `test_b_style_refer_*.py`
**主要功能**:
- 封装GME-Qwen2-VL-7B模型
- 提供统一的图像和文本嵌入接口
- 智能图像缩放 (smart_resize)

**主要接口**:
```python
class GmeQwen2VL:
    def get_image_embeddings(self, images: List[str]) -> torch.Tensor:
        # 返回图像嵌入向量 (3584维)
        pass

    def get_text_embeddings(self, texts: List[str]) -> torch.Tensor:
        # 返回文本嵌入向量 (3584维)
        pass
```

**参数说明**:
- `min_image_tokens=256` - 最小图像token数
- `max_image_tokens=1280` - 最大图像token数
- `max_pixels` - 由环境变量控制

---

#### finetune训练集构造_v2.ipynb ⭐⭐⭐⭐
**用途**: 构造训练数据集
**使用方式**: Jupyter Notebook
**主要功能**:
1. 加载预处理的向量
2. 对每个训练问题进行相似页面检索
3. 构造多模态训练样本
4. 保存为JSONL格式

**输出**:
- `train_b_dataset_for_image_0801.jsonl`

**何时使用**: 预处理完成后，训练前

---

### 备选文件

#### test_b_style_refer_90.py
**用途**: 使用checkpoint-90的推理脚本
**说明**: 备选方案，如果checkpoint-215效果不好可以尝试

#### train_vl_32b.sh
**用途**: 原始训练脚本
**说明**: `02_train.sh`是基于此脚本优化的版本

---

## 🔍 文件依赖关系

```
预处理阶段:
gme_inference.py ← b_train_test_preprocess.py ← 01_preprocess.sh

训练集构造:
*_vectors.npy ← finetune训练集构造_v2.ipynb → train_dataset.jsonl

训练阶段:
train_dataset.jsonl ← train_vl_32b.sh ← 02_train.sh → checkpoints

推理阶段:
checkpoints + *_vectors.npy ← test_b_style_refer_215.py ← 03_inference.sh → results.jsonl
```

## 📋 使用顺序

1. ✅ **check_environment.sh** - 检查环境
2. ✅ **00_setup_paths.sh** - 配置路径
3. ⏳ **01_preprocess.sh** - 预处理 (6-8h)
4. 📝 **finetune训练集构造_v2.ipynb** - 构造训练集 (30min)
5. ⏳ **02_train.sh** - 训练 (6-8h)
6. ⏳ **03_inference.sh** - 推理 (2-3h)

## 🎯 关键配置对照表

| 配置项 | 预处理 | 训练 | 推理 |
|--------|--------|------|------|
| **GPU** | GPU 1 | GPU 0-4 | GPU 0-3 |
| **MAX_PIXELS** | 1229312 | 1229312 | 1568000 |
| **模型** | GME-7B | Qwen2.5-VL-32B | 微调后32B |
| **耗时** | 6-8h | 6-8h | 2-3h |

## 💡 使用建议

### 首次使用
1. 按顺序阅读: `QUICKSTART.md` → `REPRODUCTION_GUIDE.md` → 本文件
2. 运行: `check_environment.sh` → `00_setup_paths.sh`
3. 开始复现

### 调试时
1. 查看日志文件: `*.log`
2. 检查输出文件是否生成
3. 查看GPU状态: `nvidia-smi`

### 遇到错误
1. 查看 `REPRODUCTION_GUIDE.md` 的"常见问题排查"
2. 查看 `scripts/README.md` 的常见问题
3. 检查路径配置是否正确

## 📞 快速帮助

```bash
# 查看某个脚本的功能
head -n 20 scripts/01_preprocess.sh

# 查看脚本中的配置
grep "export" scripts/*.sh

# 查看Python文件中的路径
grep "/data/coding" *.py

# 查看所有生成的文件
ls -lh *.npy *.csv *.jsonl
```

---

**提示**: 所有标记⭐⭐⭐⭐⭐的文件都是复现必需的核心文件。