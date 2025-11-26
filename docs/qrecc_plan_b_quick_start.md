# QReCC Plan B - Quick Start Guide

本文档提供QReCC Plan B (使用真实重写查询的基线方案) 的快速入门指南，包括数据集下载、模型加载、训练和评估的完整流程。

## 目录

- [环境准备](#环境准备)
- [数据集下载](#数据集下载)
- [模型下载](#模型下载)
- [数据处理](#数据处理)
- [启动检索服务器](#启动检索服务器)
- [开始训练](#开始训练)
- [模型保存位置](#模型保存位置)
- [模型评估](#模型评估)
- [常见问题](#常见问题)

---

## 环境准备

### 1. 安装依赖

```bash
# 安装项目依赖
pip install -r requirements.txt

# 安装额外的评估依赖
pip install datasets transformers accelerate
```

### 2. 创建必要的目录

```bash
# 创建数据和日志目录
mkdir -p data/qrecc_raw
mkdir -p data/qrecc_plan_b
mkdir -p logs
mkdir -p results
```

---

## 数据集下载

### 方法1: 从HuggingFace自动下载 (推荐)

```bash
# 下载QReCC数据集
python scripts/download_qrecc_dataset.py \
    --output_dir data/qrecc_raw \
    --source huggingface
```

### 方法2: 从GitHub手动下载

```bash
# 备选方案：从官方GitHub下载
python scripts/download_qrecc_dataset.py \
    --output_dir data/qrecc_raw \
    --source github
```

### 方法3: 手动下载

如果自动下载失败，可以手动下载：

1. 访问: https://github.com/apple/ml-qrecc
2. 下载数据集文件
3. 将文件放置到 `data/qrecc_raw/` 目录

**下载后的目录结构：**

```
data/qrecc_raw/
├── train.jsonl    # 训练集
└── test.jsonl     # 测试集
```

---

## 模型下载

### 选项1: 自动下载 (推荐)

训练脚本会自动从HuggingFace下载模型。只需在训练脚本中指定模型名称：

```bash
export BASE_MODEL='Qwen/Qwen3-4B'
```

**首次运行时，模型会被自动下载到：**

- Linux/Mac: `~/.cache/huggingface/hub/`
- Windows: `C:\Users\<你的用户名>\.cache\huggingface\hub\`

### 选项2: 手动下载到本地

如果需要提前下载模型或使用本地模型：

```bash
# 创建模型目录
mkdir -p models/qwen3-4b

# 使用huggingface-cli下载
pip install huggingface-cli

huggingface-cli download Qwen/Qwen3-4B \
    --local-dir models/qwen3-4b \
    --local-dir-use-symlinks False
```

**然后在训练脚本中使用本地路径：**

```bash
export BASE_MODEL='models/qwen3-4b'
```

### 选项3: 使用其他模型

项目支持多种模型，可以在训练脚本中切换：

```bash
# Qwen3-4B (推荐用于Plan B)
export BASE_MODEL='Qwen/Qwen3-4B'

# Qwen3-4B-Instruct (指令微调版本)
export BASE_MODEL='Qwen/Qwen3-4B-Instruct'

# Llama-3.2-3B
export BASE_MODEL='meta-llama/Llama-3.2-3B'

# Qwen2.5-7B (更大的模型)
export BASE_MODEL='Qwen/Qwen2.5-7B'
```

### 查看已下载的模型

```bash
# Linux/Mac
ls ~/.cache/huggingface/hub/

# Windows (PowerShell)
dir $env:USERPROFILE\.cache\huggingface\hub\

# 或者使用Python
python -c "from transformers import snapshot_download; print(snapshot_download('Qwen/Qwen3-4B'))"
```

---

## 数据处理

将下载的原始数据处理成训练所需的格式：

```bash
python scripts/data_process/qrecc_search_plan_b.py \
    --input_dir data/qrecc_raw \
    --output_dir data/qrecc_plan_b \
    --template_type base
```

**参数说明：**

- `--input_dir`: 原始数据目录
- `--output_dir`: 处理后数据保存目录
- `--template_type`: 提示模板类型
  - `base`: 用于基础模型 (推荐)
  - `instruct`: 用于指令微调模型

**处理后的目录结构：**

```
data/qrecc_plan_b/
├── train.parquet    # 训练集 (Parquet格式)
└── test.parquet     # 测试集 (Parquet格式)
```

---

## 启动检索服务器

训练需要一个运行中的检索服务器来提供外部知识。

### 方法1: 启动本地检索服务器 (BM25)

```bash
# 如果使用BM25检索
python search_r1/search/retrieval_server.py \
    --index_path /path/to/your/bm25_index \
    --corpus_path /path/to/your/corpus.jsonl \
    --topk 3 \
    --retriever_name bm25
```

### 方法2: 启动密集检索服务器 (E5)

```bash
# 如果使用密集检索 (如E5)
python search_r1/search/retrieval_server.py \
    --index_path /path/to/your/faiss_index \
    --corpus_path /path/to/your/corpus.jsonl \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model intfloat/e5-base-v2 \
    --faiss_gpu
```

### 验证检索服务器

```bash
# 测试服务器是否正常运行
curl -X POST http://127.0.0.1:8000/retrieve \
    -H "Content-Type: application/json" \
    -d '{"queries": ["What is Python?"], "topk": 3}'
```

**服务器配置说明：**

- 默认端口: `8000`
- 默认地址: `http://127.0.0.1:8000/retrieve`
- 如果需要修改，请同时更新训练脚本中的 `retriever.url`

---

## 开始训练

### 1. 修改训练配置 (可选)

编辑 `configs/qrecc/train_qrecc_ppo_plan_b.sh`:

```bash
# GPU配置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 根据实际GPU数量调整

# 模型选择
export BASE_MODEL='Qwen/Qwen3-4B'  # 或使用本地路径
export EXPERIMENT_NAME='qrecc-plan-b-ppo-qwen3-4b-em'

# 数据配置
export DATA_DIR='data/qrecc_plan_b'
```

### 2. 启动训练

```bash
# 给脚本添加执行权限
chmod +x configs/qrecc/train_qrecc_ppo_plan_b.sh

# 启动训练
bash configs/qrecc/train_qrecc_ppo_plan_b.sh
```

### 3. 监控训练进度

训练日志会保存到：

```
logs/qrecc-plan-b-ppo-qwen3-4b-em.log
```

查看实时日志：

```bash
tail -f logs/qrecc-plan-b-ppo-qwen3-4b-em.log
```

### 4. 使用WandB监控 (可选)

如果配置了WandB，可以在浏览器中实时监控：

```bash
# 登录WandB
wandb login

# 训练过程中访问
https://wandb.ai/<your-username>/Search-R1-QReCC
```

---

## 模型保存位置

### 训练checkpoint自动保存

训练过程中，模型checkpoint会自动保存到：

```
verl_checkpoints/<EXPERIMENT_NAME>/
├── checkpoint_100/      # 第100步的checkpoint
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   └── ...
├── checkpoint_200/      # 第200步的checkpoint
├── checkpoint_300/
└── ...
```

**默认保存路径：**

```
verl_checkpoints/qrecc-plan-b-ppo-qwen3-4b-em/
```

### 保存频率配置

在训练脚本中可以修改保存频率：

```bash
trainer.save_freq=100    # 每100步保存一次
trainer.test_freq=50     # 每50步验证一次
```

### 查看已保存的checkpoint

```bash
# 列出所有checkpoint
ls verl_checkpoints/qrecc-plan-b-ppo-qwen3-4b-em/

# 查看特定checkpoint
ls verl_checkpoints/qrecc-plan-b-ppo-qwen3-4b-em/checkpoint_1000/
```

### 手动保存最佳模型

```bash
# 复制最佳checkpoint到单独目录
cp -r verl_checkpoints/qrecc-plan-b-ppo-qwen3-4b-em/checkpoint_1000 \
      models/qrecc_plan_b_best
```

---

## 模型评估

### 方法1: 使用Python评估脚本

```bash
python scripts/evaluate/evaluate_qrecc_plan_b.py \
    --checkpoint verl_checkpoints/qrecc-plan-b-ppo-qwen3-4b-em/checkpoint_1000 \
    --test_file data/qrecc_plan_b/test.parquet \
    --output_file results/qrecc_plan_b_eval.json \
    --max_examples 500
```

**参数说明：**

- `--checkpoint`: 要评估的模型checkpoint路径
- `--test_file`: 测试数据文件
- `--output_file`: 评估结果保存路径
- `--max_examples`: 最多评估多少个样本 (可选)

### 方法2: 使用verl框架评估 (包含检索)

```bash
# 修改评估脚本中的checkpoint路径
vim configs/qrecc/evaluate_qrecc_plan_b.sh

# 设置checkpoint路径
export CHECKPOINT_PATH="verl_checkpoints/qrecc-plan-b-ppo-qwen3-4b-em/checkpoint_1000"

# 运行评估
bash configs/qrecc/evaluate_qrecc_plan_b.sh
```

### 评估基线模型 (未训练)

```bash
# 评估原始基础模型的性能
python scripts/evaluate/evaluate_qrecc_plan_b.py \
    --checkpoint Qwen/Qwen3-4B \
    --test_file data/qrecc_plan_b/test.parquet \
    --output_file results/qrecc_plan_b_baseline.json
```

### 查看评估结果

```bash
# 查看JSON结果
cat results/qrecc_plan_b_eval.json

# 或使用Python格式化输出
python -m json.tool results/qrecc_plan_b_eval.json
```

---

## 常见问题

### Q1: 数据集下载失败怎么办？

**A:** 尝试以下方案：

1. 切换到GitHub下载源：`--source github`
2. 手动下载并放置到 `data/qrecc_raw/`
3. 检查网络连接和HuggingFace访问

### Q2: 模型下载太慢或失败？

**A:** 解决方案：

```bash
# 方案1: 使用镜像站点
export HF_ENDPOINT=https://hf-mirror.com
python scripts/download_model.py

# 方案2: 手动下载
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-4B models/qwen3-4b

# 方案3: 使用代理
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
```

### Q3: 训练时显存不足 (OOM) 怎么办？

**A:** 调整以下参数：

```bash
# 减小batch size
data.train_batch_size=256  # 从512减小

# 减小micro batch size
actor_rollout_ref.actor.ppo_micro_batch_size=32  # 从64减小

# 启用更多offload
actor_rollout_ref.actor.fsdp_config.param_offload=true
actor_rollout_ref.actor.fsdp_config.grad_offload=true
actor_rollout_ref.actor.fsdp_config.optimizer_offload=true

# 减少GPU数量
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用4张GPU
trainer.n_gpus_per_node=4
```

### Q4: 检索服务器连接失败？

**A:** 检查步骤：

```bash
# 1. 确认服务器正在运行
ps aux | grep retrieval_server

# 2. 测试服务器连接
curl http://127.0.0.1:8000/retrieve

# 3. 检查端口是否被占用
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# 4. 修改训练脚本中的URL
retriever.url="http://127.0.0.1:8000/retrieve"
```

### Q5: 如何在单GPU上训练？

**A:** 修改配置：

```bash
# 设置单GPU
export CUDA_VISIBLE_DEVICES=0

# 修改训练脚本
trainer.n_gpus_per_node=1
trainer.nnodes=1

# 调整batch size以适应单GPU
data.train_batch_size=64
actor_rollout_ref.actor.ppo_mini_batch_size=32
```

### Q6: 训练checkpoint在哪里？

**A:** 默认位置：

```
verl_checkpoints/<EXPERIMENT_NAME>/checkpoint_<步数>/
```

可以在训练脚本中修改：

```bash
trainer.default_local_dir=<自定义路径>
```

### Q7: 如何恢复中断的训练？

**A:** 使用checkpoint恢复：

```bash
# 在训练脚本中添加
actor_rollout_ref.model.path=verl_checkpoints/qrecc-plan-b-ppo-qwen3-4b-em/checkpoint_500

# 调整总步数
trainer.total_training_steps=1500  # 如果从500步恢复，继续训练1000步
```

### Q8: 如何使用多机多卡训练？

**A:** 修改训练脚本：

```bash
# 设置节点数
trainer.nnodes=2  # 2个节点

# 每个节点的GPU数
trainer.n_gpus_per_node=8

# 使用Ray集群
# 在主节点启动Ray
ray start --head --port=6379

# 在其他节点连接
ray start --address='<主节点IP>:6379'
```

---

## 目录结构总览

完整的项目目录结构：

```
Search-R1/
├── data/
│   ├── qrecc_raw/           # 原始数据集
│   │   ├── train.jsonl
│   │   └── test.jsonl
│   └── qrecc_plan_b/        # 处理后的数据
│       ├── train.parquet
│       └── test.parquet
├── models/                  # 本地模型 (可选)
│   └── qwen3-4b/
├── verl_checkpoints/        # 训练checkpoint
│   └── qrecc-plan-b-ppo-qwen3-4b-em/
│       ├── checkpoint_100/
│       ├── checkpoint_200/
│       └── ...
├── logs/                    # 训练日志
│   └── qrecc-plan-b-ppo-qwen3-4b-em.log
├── results/                 # 评估结果
│   └── qrecc_plan_b_eval.json
├── scripts/
│   ├── download_qrecc_dataset.py
│   ├── data_process/
│   │   └── qrecc_search_plan_b.py
│   └── evaluate/
│       └── evaluate_qrecc_plan_b.py
├── configs/
│   └── qrecc/
│       ├── train_qrecc_ppo_plan_b.sh
│       └── evaluate_qrecc_plan_b.sh
└── verl/
    └── utils/
        └── reward_score/
            └── qrecc_em.py
```

---

## 完整流程示例

以下是从零开始的完整运行流程：

```bash
# 1. 下载数据集
python scripts/download_qrecc_dataset.py --output_dir data/qrecc_raw

# 2. 处理数据
python scripts/data_process/qrecc_search_plan_b.py \
    --input_dir data/qrecc_raw \
    --output_dir data/qrecc_plan_b

# 3. 启动检索服务器 (在新终端)
python search_r1/search/retrieval_server.py \
    --index_path /path/to/index \
    --corpus_path /path/to/corpus.jsonl

# 4. 开始训练 (会自动下载模型)
bash configs/qrecc/train_qrecc_ppo_plan_b.sh

# 5. 评估模型
python scripts/evaluate/evaluate_qrecc_plan_b.py \
    --checkpoint verl_checkpoints/qrecc-plan-b-ppo-qwen3-4b-em/checkpoint_1000 \
    --test_file data/qrecc_plan_b/test.parquet \
    --output_file results/qrecc_plan_b_eval.json
```

---

## 下一步

完成Plan B后，可以考虑：

1. **调整超参数**: 学习率、batch size等
2. **尝试不同模型**: Qwen3-7B, Llama-3.1-8B等
3. **实施Plan A**: 完整的端到端对话方案 (参考 `qrecc_plan_A_end_to_end.md`)
4. **优化检索策略**: 尝试不同的检索器和topk值
5. **多数据集训练**: 结合NQ、HotpotQA等其他数据集

---

## 参考资料

- [QReCC论文](https://arxiv.org/abs/2010.04898)
- [QReCC GitHub](https://github.com/apple/ml-qrecc)
- [veRL文档](https://github.com/volcengine/verl)
- [Qwen3模型卡](https://huggingface.co/Qwen/Qwen3-4B)
- [Search-R1项目README](../README.md)

---

**祝训练顺利！如有问题，请查看日志文件或提交Issue。**