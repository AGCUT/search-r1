# 运行脚本说明

本目录包含用于复现CCKS 2025专利问答项目的自动化脚本。

## 脚本列表

| 脚本 | 功能 | GPU使用 | 耗时 |
|------|------|---------|------|
| `00_setup_paths.sh` | 配置路径 | 无 | <1分钟 |
| `01_preprocess.sh` | 数据预处理 | GPU 1 | 6-8小时 |
| `02_train.sh` | 模型训练 | GPU 0,1,2,3,4 | 6-8小时 |
| `03_inference.sh` | 模型推理 | GPU 0,1,2,3 | 2-3小时 |

## 使用步骤

### 0. 配置路径（首次运行必需）

```bash
cd /path/to/ccks2025_pdf_multimodal/round_b

# 自动配置路径
bash scripts/00_setup_paths.sh /data/coding

# 或使用默认路径
bash scripts/00_setup_paths.sh
```

这个脚本会：
- 备份原始文件
- 将所有Python脚本中的路径修改为你的实际路径
- 验证必要的目录结构

### 1. 数据预处理

```bash
# 在tmux会话中运行
tmux new -s preprocess
bash scripts/01_preprocess.sh
```

**说明**:
- 使用GPU 1（最空闲的GPU）
- 将PDF转换为图像（600 DPI）
- 生成图像和问题的向量嵌入（3584维）
- 输出6个文件（训练和测试的向量及映射）

**预期输出**:
```
train_b_pdf_img_vectors.npy (约270MB)
train_b_pdf_img_page_num_mapping.csv
all_train_b_question_vectors.npy
test_b_pdf_img_vectors.npy (约200MB)
test_b_pdf_img_page_num_mapping.csv
all_test_b_question_vectors.npy
```

### 2. 构造训练集

```bash
# 启动Jupyter
jupyter notebook --no-browser --port=8888

# 打开并运行: finetune训练集构造_v2.ipynb
```

**说明**:
- 基于向量检索构造训练样本
- 每个问题检索top-2相似页面
- 生成多模态训练数据（图像+文本）

**预期输出**:
```
train_b_dataset_for_image_0801.jsonl
```

### 3. 模型训练

```bash
# 在tmux会话中运行
tmux new -s train
bash scripts/02_train.sh
```

**说明**:
- 使用5张GPU (0,1,2,3,4)，避开正在使用的GPU 5,6,7
- LoRA微调Qwen2.5-VL-32B
- 每10步保存checkpoint，保留最近4个

**监控训练**:
```bash
# 查看日志
tail -f train.log

# 查看GPU使用
watch -n 5 nvidia-smi

# 查看checkpoints
ls -lh /data/coding/lora_qwen25_vl_32b_b/checkpoint-*
```

### 4. 模型推理

```bash
# 在tmux会话中运行
tmux new -s inference
bash scripts/03_inference.sh
```

**说明**:
- 使用4张GPU (0,1,2,3)进行张量并行
- 自动查找最新的checkpoint
- 如果checkpoint未合并，自动合并LoRA权重
- 生成测试集答案

**预期输出**:
```
test_b_style_infer_if_need_ck215.jsonl
```

## GPU使用策略

根据当前GPU占用情况（GPU 5,6,7正在使用），脚本已优化为：

```
预处理: GPU 1 (最空闲，74GB可用)
训练:   GPU 0,1,2,3,4 (5张卡，而非原始的8张)
推理:   GPU 0,1,2,3 (4张卡)
```

## 常见问题

### 1. OOM错误

**训练时OOM**:
```bash
# 编辑 02_train.sh，降低MAX_PIXELS
export MAX_PIXELS=819200  # 从1229312降低

# 或增加梯度累积步数
--gradient_accumulation_steps 32  # 从16增加
```

**推理时OOM**:
```bash
# 编辑 03_inference.sh，降低MAX_PIXELS
export MAX_PIXELS=1229312  # 从1568000降低

# 或使用更少的GPU
export CUDA_VISIBLE_DEVICES=0,1,2  # 使用3张而非4张
```

### 2. 路径错误

```bash
# 重新运行路径配置
bash scripts/00_setup_paths.sh /your/actual/path

# 手动检查
grep -r "/data/coding" *.py
```

### 3. 训练中断

```bash
# SWIFT支持从checkpoint恢复
# 编辑 02_train.sh，添加:
--resume_from_checkpoint /data/coding/lora_qwen25_vl_32b_b/checkpoint-xxx
```

### 4. GPU占用冲突

```bash
# 实时查看GPU状态
watch -n 2 nvidia-smi

# 调整CUDA_VISIBLE_DEVICES使用其他GPU
export CUDA_VISIBLE_DEVICES=1,2,3  # 避开GPU 0
```

## 脚本特性

所有脚本包含：
- ✅ 自动错误检测（set -e）
- ✅ 文件存在性检查
- ✅ GPU状态监控
- ✅ 详细的进度日志
- ✅ 执行时间记录
- ✅ 输出文件验证

## 一键运行（高级）

如果你确信所有配置正确，可以一键运行所有步骤：

```bash
# 警告: 这将连续运行14-19小时！
cd /path/to/ccks2025_pdf_multimodal/round_b

# 在tmux中运行
tmux new -s ccks2025

# 执行完整流程
bash scripts/00_setup_paths.sh /data/coding && \
bash scripts/01_preprocess.sh && \
echo "请手动运行notebook: finetune训练集构造_v2.ipynb" && \
read -p "完成后按Enter继续..." && \
bash scripts/02_train.sh && \
bash scripts/03_inference.sh

# 分离tmux会话: Ctrl+B, 然后按D
# 重新连接: tmux attach -t ccks2025
```

## 参数调优建议

### 训练参数
```bash
# 更大的LoRA rank（可能提升性能，但更慢）
--lora_rank 16  # 从8改为16

# 更多的训练轮次
--num_train_epochs 10  # 从5改为10

# 不同的学习率
--learning_rate 5e-5  # 从1e-4改为5e-5
```

### 推理参数
```bash
# 批量推理（更快，需要更多显存）
max_num_seqs=4  # 在test_b_style_refer_215.py中修改

# 更高分辨率（可能提升准确率）
export MAX_PIXELS=2352000  # 从1568000增加
```

## 日志文件

所有脚本会生成日志文件：
- `preprocess.log` - 预处理日志
- `train.log` - 训练日志
- `inference.log` - 推理日志

查看日志:
```bash
# 实时查看
tail -f train.log

# 查找错误
grep -i error train.log

# 查找警告
grep -i warning train.log
```

## 性能基准

在8×A800服务器上的预期性能：

| 阶段 | GPU | 时间 | 显存/卡 |
|------|-----|------|---------|
| 预处理 | 1×A800 | 6-8h | ~30GB |
| 训练 | 5×A800 | 6-8h | ~60GB |
| 推理 | 4×A800 | 2-3h | ~40GB |

**注意**: 使用5张GPU训练（而非8张）可能会稍微延长训练时间（约多1-2小时）。