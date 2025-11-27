# CCKS 2025 专利问答多模态项目完整复现指南

本指南提供在8卡A800服务器上完整复现该项目的详细步骤。

## 服务器环境信息

- **GPU配置**: 8 × NVIDIA A800-SXM4-80GB
- **当前GPU占用情况**:
  - GPU 0: 67.9GB/80GB 已使用 (可用约14GB)
  - GPU 1: 7.5GB/80GB 已使用 (可用约74GB) ✅ **最空闲**
  - GPU 2: 40.8GB/80GB 已使用 (可用约41GB)
  - GPU 3: 24.5GB/80GB 已使用 (可用约57GB)
  - GPU 4: 65.5GB/80GB 已使用 (可用约16GB)
  - GPU 5-7: 正在100%使用中 ❌ **避免使用**

## 复现流程概览

```
步骤0: 环境准备 (安装依赖、下载模型、准备数据)
   ↓
步骤1: 数据预处理 (PDF转图像、生成向量) [~6-8小时] [使用GPU 1]
   ↓
步骤2: 构造训练集 (运行Jupyter notebook)
   ↓
步骤3: 模型微调 (LoRA训练) [~6-8小时] [使用GPU 0,1,2,3,4]
   ↓
步骤4: 模型推理 (生成测试结果) [~2-3小时] [使用GPU 0,1,2,3]
```

---

## 步骤0: 环境准备

### 0.1 创建工作目录

```bash
# 定义项目根目录（根据实际情况修改）
export PROJECT_ROOT=/data/coding
export PATENT_DATA_DIR=$PROJECT_ROOT/patent_b

# 创建必要的目录结构
mkdir -p $PROJECT_ROOT
mkdir -p $PATENT_DATA_DIR/train/documents
mkdir -p $PATENT_DATA_DIR/train/pdf_img
mkdir -p $PATENT_DATA_DIR/test/documents
mkdir -p $PATENT_DATA_DIR/test/pdf_img

# 进入项目代码目录
cd /path/to/ccks2025_pdf_multimodal/round_b
```

### 0.2 配置Python环境

#### 方法A: 使用Conda环境（推荐）⭐

```bash
# 一键自动配置（推荐）
bash setup_conda_env.sh

# 这个脚本会自动：
# 1. 创建名为 ccks2025_pdf_qa 的conda环境
# 2. 安装Python 3.10
# 3. 安装PyTorch (根据你的CUDA版本自动选择)
# 4. 安装所有依赖包
# 5. 验证安装
```

**或者手动配置**:

```bash
# 1. 使用environment.yml创建环境
conda env create -f environment.yml
conda activate ccks2025_pdf_qa

# 2. 或者手动创建环境
conda create -n ccks2025_pdf_qa python=3.10 -y
conda activate ccks2025_pdf_qa

# 3. 安装PyTorch (根据CUDA版本选择)
# 对于CUDA 12.1+:
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# 对于CUDA 11.8:
# pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

# 4. 安装其他依赖
pip install -r requirements.txt

# 5. 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

**详细说明**: 查看 `CONDA_SETUP.md` 获取完整的Conda环境配置指南。

#### 方法B: 使用pip直接安装

```bash
# 确保使用Python 3.10+
python --version

# 一键安装所有依赖
pip install -r requirements.txt

# 或手动安装核心依赖
pip install torch>=2.1.0 torchvision torchaudio
pip install ms-swift>=2.0.0
pip install vllm>=0.4.0
pip install transformers>=4.40.0
pip install PyMuPDF>=1.23.0
pip install numpy>=1.24.0 pandas>=2.0.0
pip install pillow>=9.0.0
pip install qwen-vl-utils
pip install tqdm jupyter

# 验证安装
python -c "import swift; import vllm; print('安装成功')"
```

**注意**: 如果使用conda环境，后续所有命令都需要先激活环境：
```bash
conda activate ccks2025_pdf_qa
```

### 0.3 下载模型

#### 方法1: 使用ModelScope（魔搭，推荐）

```bash
# 安装modelscope
pip install modelscope

# 创建模型下载脚本
cat > download_models.py << 'EOF'
from modelscope import snapshot_download

# 下载GME嵌入模型 (约14GB)
print("正在下载 GME-Qwen2-VL-7B-Instruct...")
snapshot_download(
    'iic/gme-Qwen2-VL-7B-Instruct',
    cache_dir='/data/coding/llm_model',
    revision='master'
)

# 下载Qwen2.5-VL-32B基座模型 (约64GB)
print("正在下载 Qwen2.5-VL-32B-Instruct...")
snapshot_download(
    'Qwen/Qwen2.5-VL-32B-Instruct',
    cache_dir='/data/coding/llm_model',
    revision='master'
)

print("模型下载完成！")
EOF

# 运行下载脚本（需要1-3小时，取决于网速）
python download_models.py
```

#### 方法2: 使用HuggingFace

```bash
# 设置HF镜像（如果需要）
export HF_ENDPOINT=https://hf-mirror.com

# 下载模型
huggingface-cli download iic/gme-Qwen2-VL-7B-Instruct \
    --local-dir /data/coding/llm_model/iic/gme-Qwen2-VL-7B-Instruct

huggingface-cli download Qwen/Qwen2.5-VL-32B-Instruct \
    --local-dir /data/coding/llm_model/Qwen/Qwen2___5-VL-32B-Instruct
```

### 0.4 下载并解压数据集

```bash
# 安装ossutil64
cd $PROJECT_ROOT
wget http://gosspublic.alicdn.com/ossutil/1.7.12/ossutil64
chmod +x ossutil64

# 从天池竞赛页面获取数据下载命令
# https://tianchi.aliyun.com/competition/entrance/532357/information
# 复制OSS命令并修改为使用 ./ossutil64

# 示例（替换为实际的OSS路径）:
# ./ossutil64 cp oss://tianchi-competition/ccks2025/train_data.zip ./
# ./ossutil64 cp oss://tianchi-competition/ccks2025/test_data.zip ./

# 解压数据集
unzip train_data.zip -d $PATENT_DATA_DIR/train/documents/
unzip test_data.zip -d $PATENT_DATA_DIR/test/documents/

# 验证数据
ls $PATENT_DATA_DIR/train/documents/*.pdf | wc -l  # 应该看到训练集PDF数量
ls $PATENT_DATA_DIR/test/documents/*.pdf | wc -l   # 应该看到测试集PDF数量
```

### 0.5 启动tmux会话（重要！）

```bash
# 安装tmux（如果未安装）
apt-get update && apt-get install -y tmux

# 创建tmux会话，防止SSH断开影响长时间任务
tmux new -s ccks2025

# tmux常用命令：
# Ctrl+B, D - 分离会话（程序继续运行）
# tmux attach -t ccks2025 - 重新连接会话
# tmux ls - 查看所有会话
```

---

## 步骤1: 数据预处理 [预计6-8小时]

**说明**: 这是最耗时的步骤，需要将所有PDF转为图像，并生成向量嵌入。

### 1.1 修改预处理脚本路径

编辑 `b_train_test_preprocess.py`，确保路径正确：

```python
# 修改以下路径为你的实际路径
base_dir = '/data/coding/patent_b/train/'  # 训练集路径
test_base_dir = '/data/coding/patent_b/test/'  # 测试集路径
gme_model_path = '/data/coding/llm_model/iic/gme-Qwen2-VL-7B-Instruct'
```

### 1.2 运行预处理脚本

```bash
# 在tmux会话中运行
cd /path/to/ccks2025_pdf_multimodal/round_b

# 使用GPU 1（最空闲的GPU）
bash scripts/01_preprocess.sh

# 或者直接运行（如果没有脚本）：
CUDA_VISIBLE_DEVICES=1 MAX_PIXELS=1229312 python b_train_test_preprocess.py
```

**预处理输出文件**:
- `train_b_pdf_img_vectors.npy` - 训练集图像向量 (~270MB)
- `train_b_pdf_img_page_num_mapping.csv` - 训练集页码映射
- `all_train_b_question_vectors.npy` - 训练集问题向量
- `test_b_pdf_img_vectors.npy` - 测试集图像向量 (~200MB)
- `test_b_pdf_img_page_num_mapping.csv` - 测试集页码映射
- `all_test_b_question_vectors.npy` - 测试集问题向量

**进度监控**:
```bash
# 在另一个终端查看进度
watch -n 10 "ls -lh *_vectors.npy 2>/dev/null"
```

---

## 步骤2: 构造训练集

### 2.1 运行Jupyter Notebook

```bash
# 启动Jupyter（在tmux中）
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0

# 如果通过SSH连接，建立端口转发：
# 在本地终端运行: ssh -L 8888:localhost:8888 user@server_ip
```

### 2.2 运行notebook单元格

打开 `finetune训练集构造_v2.ipynb`，按顺序执行所有单元格。

**主要操作**:
1. 加载预处理的向量文件
2. 对每个训练问题进行相似页面检索
3. 构造多模态训练样本（图像+文本）
4. 保存为JSONL格式

**输出文件**:
- `train_b_dataset_for_image_0801.jsonl` - 训练数据集

**验证训练集**:
```bash
wc -l train_b_dataset_for_image_0801.jsonl  # 查看样本数量
head -n 1 train_b_dataset_for_image_0801.jsonl | python -m json.tool  # 查看样本格式
```

---

## 步骤3: 模型微调 [预计6-8小时]

**说明**: 使用LoRA对Qwen2.5-VL-32B进行微调。由于GPU 5-7正在使用，我们使用5张GPU (0,1,2,3,4)。

### 3.1 修改训练脚本

训练脚本已自动调整为使用5张GPU。检查 `scripts/02_train.sh`:

```bash
cat scripts/02_train.sh
```

### 3.2 开始训练

```bash
# 在tmux会话中运行
cd /path/to/ccks2025_pdf_multimodal/round_b

bash scripts/02_train.sh
```

**训练参数说明**:
- `CUDA_VISIBLE_DEVICES=0,1,2,3,4` - 使用5张GPU（避开正在使用的5,6,7）
- `MAX_PIXELS=1229312` - 图像分辨率（约1280 tokens）
- `lora_rank=8, lora_alpha=32` - LoRA参数
- `learning_rate=1e-4` - 学习率
- `num_train_epochs=5` - 训练5个epoch
- `per_device_train_batch_size=1` - 每卡batch size为1
- `gradient_accumulation_steps=16` - 梯度累积16步（有效batch size = 5×1×16 = 80）

**训练监控**:
```bash
# 查看训练日志
tail -f /data/coding/lora_qwen25_vl_32b_b/runs/*/logs/default/*.log

# 查看checkpoint
ls -lh /data/coding/lora_qwen25_vl_32b_b/checkpoint-*

# 监控GPU使用
watch -n 5 nvidia-smi
```

**预期输出**:
- Checkpoint保存在 `/data/coding/lora_qwen25_vl_32b_b/`
- 每10步保存一次checkpoint
- 保留最近4个checkpoint (`save_total_limit=4`)

---

## 步骤4: 模型推理 [预计2-3小时]

### 4.1 合并LoRA权重（如果需要）

```bash
# SWIFT会自动合并，检查是否存在merged目录
ls /data/coding/lora_qwen25_vl_32b_b/v*/checkpoint-*-merged/

# 如果不存在，手动合并：
swift export \
    --ckpt_dir /data/coding/lora_qwen25_vl_32b_b/checkpoint-215 \
    --merge_lora true
```

### 4.2 修改推理脚本路径

编辑 `test_b_style_refer_215.py`，修改以下路径：

```python
# 第8-14行，修改数据路径
train_base_dir = '/data/coding/patent_b/train/'
base_dir = '/data/coding/patent_b/test/'

# 第23行，修改模型路径
model_path = "/data/coding/lora_qwen25_vl_32b_b/v0-20250802-085531/checkpoint-215-merged/"
# 或使用你实际的checkpoint路径
```

### 4.3 运行推理

```bash
# 在tmux会话中运行
cd /path/to/ccks2025_pdf_multimodal/round_b

bash scripts/03_inference.sh

# 或直接运行：
CUDA_VISIBLE_DEVICES=0,1,2,3 MAX_PIXELS=1568000 python test_b_style_refer_215.py
```

**推理输出**:
- `test_b_style_infer_if_need_ck215.jsonl` - 测试集预测结果

**验证结果**:
```bash
wc -l test_b_style_infer_if_need_ck215.jsonl  # 应与测试集问题数量一致
head -n 3 test_b_style_infer_if_need_ck215.jsonl | python -m json.tool
```

---

## 步骤5: 提交结果

```bash
# 检查结果文件
cat test_b_style_infer_if_need_ck215.jsonl | python -m json.tool | less

# 按照竞赛要求格式化结果（如果需要）
# 然后上传到天池平台
```

---

## 常见问题排查

### 问题1: CUDA Out of Memory

**症状**: 训练或推理时出现OOM错误

**解决方案**:
```bash
# 方案1: 降低MAX_PIXELS
export MAX_PIXELS=819200  # 从1229312降低到819200

# 方案2: 降低batch size
# 编辑train_vl_32b.sh，修改：
--per_device_train_batch_size 1  # 改为1（已经是最小）
--gradient_accumulation_steps 32  # 从16增加到32以保持有效batch size

# 方案3: 使用更少的GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/02_train.sh
```

### 问题2: 路径错误

**症状**: FileNotFoundError

**解决方案**:
```bash
# 检查所有路径配置
find . -name "*.py" -exec grep -l "/data/coding" {} \;

# 批量替换路径（谨慎使用）
find . -name "*.py" -exec sed -i 's|/data/coding|/your/actual/path|g' {} \;
```

### 问题3: 模型下载失败

**症状**: Connection timeout

**解决方案**:
```bash
# 使用国内镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载后放置到指定目录
# 从百度网盘下载（参考README.md中的链接）
```

### 问题4: GPU正在被其他进程占用

**症状**: GPU已满无法运行

**解决方案**:
```bash
# 查看GPU进程
nvidia-smi

# 查看特定GPU的进程详情
nvidia-smi -i 5,6,7

# 等待GPU空闲或调整CUDA_VISIBLE_DEVICES使用其他GPU
```

---

## 性能优化建议

### 1. 加速预处理
```bash
# 使用多进程PDF转换（修改b_train_test_preprocess.py）
from multiprocessing import Pool

# 并行转换PDF
with Pool(processes=4) as pool:
    pool.map(convert_pdf_to_images, pdf_file_list)
```

### 2. 加速训练
```bash
# 使用DeepSpeed（编辑训练脚本添加）
--deepspeed ds_config.json
```

### 3. 加速推理
```bash
# 增加批处理大小（编辑test_b_style_refer_215.py）
max_num_seqs=4  # 从1改为4
```

---

## 资源估算

| 阶段 | 时间 | GPU | 显存需求 | 存储需求 |
|------|------|-----|----------|----------|
| 预处理 | 6-8h | 1×A800 | ~30GB | ~500MB(向量) |
| 训练 | 6-8h | 5×A800 | ~60GB/卡 | ~5GB(checkpoints) |
| 推理 | 2-3h | 4×A800 | ~40GB/卡 | ~10MB(结果) |

**总计**: 约14-19小时，需要至少5张A800 GPU

---

## 检查清单

复现前请确认：

- [ ] GPU 0-4可用（至少5张GPU用于训练）
- [ ] 已安装所有Python依赖
- [ ] 已下载GME和Qwen2.5-VL-32B模型（共约78GB）
- [ ] 已下载并解压训练/测试数据集
- [ ] 所有脚本中的路径已修改为实际路径
- [ ] 已启动tmux会话
- [ ] 磁盘空间充足（至少200GB可用）

---

## 快速启动命令

```bash
# 一键启动完整流程（在tmux中）
cd /path/to/ccks2025_pdf_multimodal/round_b

# 步骤1: 预处理
bash scripts/01_preprocess.sh && \

# 步骤2: 构造训练集（需要手动运行notebook）
echo "请运行Jupyter notebook: finetune训练集构造_v2.ipynb" && \

# 步骤3: 训练
bash scripts/02_train.sh && \

# 步骤4: 推理
bash scripts/03_inference.sh

echo "复现完成！结果文件：test_b_style_infer_if_need_ck215.jsonl"
```

---

## 联系与参考

- **竞赛页面**: https://tianchi.aliyun.com/competition/entrance/532357/information
- **技术博客**: https://zhuanlan.zhihu.com/p/1937478905532506972
- **数据分享**: 链接: https://pan.baidu.com/s/10fYbAaSNf-Nv7x1jsMiqzw 提取码: r96k
