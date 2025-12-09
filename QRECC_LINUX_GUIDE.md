# QReCC 评估完整指南 - Linux 服务器操作步骤

## 📋 概述

您已经在 NQ 数据集上完成了 PPO 训练，模型保存在：
```
/usr/yuque/guo/searchr1/verl_checkpoints/nq_hotpotqa_train-search-r1-ppo-qwen2.5-3b-it-bm25-em/actor/global_step_200
```

现在要在 QReCC 数据集上评估训练后的模型，与基础 Qwen2.5-3B 模型对比。

---

## 🚀 完整操作步骤（在 Linux 服务器上执行）

### 步骤 1: 转换数据格式

QReCC 的原始 JSON 文件需要转换为 parquet 格式才能被 veRL 评估框架使用。转换脚本会同时处理训练集和测试集。

```bash
# 进入项目目录
cd /usr/yuque/guo/searchr1  # 替换为您的实际路径

# 激活 conda 环境
conda activate searchr1

# 运行转换脚本（会同时转换 train 和 test）
bash convert_qrecc_test.sh
```

**预期输出:**
```
======================================================================
QReCC 数据转换 - JSON 转 Parquet
======================================================================
测试集:
  输入文件: data/qrecc_raw/qrecc_test.json
  输出文件: data/qrecc_raw/test.parquet

训练集:
  输入文件: data/qrecc_raw/qrecc_train.json
  输出文件: data/qrecc_raw/train.parquet

模板类型: base
======================================================================

======================================================================
转换测试集...
======================================================================
✓ Loaded 16451 examples
✓ Processed 16451 examples
✓ Saved 16451 examples to data/qrecc_raw/test.parquet

======================================================================
转换训练集...
======================================================================
✓ Loaded 54720 examples
✓ Processed 54720 examples
✓ Saved 54720 examples to data/qrecc_raw/train.parquet

======================================================================
✓ 转换成功！
======================================================================
生成的文件:
  测试集: data/qrecc_raw/test.parquet
  训练集: data/qrecc_raw/train.parquet
```

**验证转换结果:**
```bash
python -c "import pandas as pd; df = pd.read_parquet('data/qrecc_raw/test.parquet'); print(f'测试集样本数: {len(df)}'); df2 = pd.read_parquet('data/qrecc_raw/train.parquet'); print(f'训练集样本数: {len(df2)}')"
```

---

### 步骤 2: 启动检索服务器

评估过程需要 BM25 检索服务器运行。

**在一个单独的终端窗口中:**

```bash
# 激活检索器环境 (如果使用单独的环境)
conda activate retriever  # 或者使用 searchr1 环境

# 启动 BM25 检索服务器
bash retrieval_launch.sh

# 或者手动启动:
python search_r1/search/retrieval_server.py \
    --index_path /path/to/your/bm25.index \
    --corpus_path /path/to/your/corpus.jsonl \
    --topk 3 \
    --retriever_name bm25
```

**验证检索服务器运行:**
```bash
curl -X POST http://127.0.0.1:8000/retrieve \
    -H "Content-Type: application/json" \
    -d '{"queries": ["test query"], "topk": 3}'
```

如果返回 JSON 结果，说明服务器运行正常。

---

### 步骤 3: 修改评估配置

编辑评估脚本，设置正确的路径和参数：

```bash
vim configs/qrecc/compare_base_vs_trained_f1.sh
```

**需要修改的配置:**

```bash
# GPU 设置 (4卡 A800)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 基础模型
export BASE_MODEL="Qwen/Qwen2.5-3B"

# 训练后的模型检查点
export TRAINED_CHECKPOINT="/usr/yuque/guo/searchr1/verl_checkpoints/nq_hotpotqa_train-search-r1-ppo-qwen2.5-3b-it-bm25-em/actor/global_step_200"

# 数据目录
export DATA_DIR='data/qrecc_raw'

# 评估指标选择
export REWARD_FN='f1'  # 推荐: f1, rouge_l, rouge_1, bleu

# 检索服务器 URL
retriever.url="http://127.0.0.1:8000/retrieve"

# GPU 配置
trainer.n_gpus_per_node=4  # 4卡 A800
trainer.nnodes=1
```

**保存并退出** (`:wq`)

---

### 步骤 4: 运行对比评估

```bash
# 确保在 searchr1 环境中
conda activate searchr1

# 运行对比评估
bash configs/qrecc/compare_base_vs_trained_f1.sh
```

**评估流程:**
1. 首先评估基础模型 (Qwen2.5-3B)
2. 等待 30 秒清理 GPU 内存
3. 然后评估训练后的模型
4. 最后生成对比报告

**预期输出示例:**
```
============================================================================
COMPARISON REPORT
============================================================================

Metric: F1 Score
--------------------------------------------------------------------
Metric                    Base Model           Trained Model        Change
--------------------------------------------------------------------
F1 Score                  0.2340              0.4580              +0.2240 (+95.73%)
Avg Searches/Question     0.08                1.45                +1.37 (1712.5%)
Generation Time (s)       12.34               15.67               +3.33
--------------------------------------------------------------------

✓ The trained model shows IMPROVEMENT over the base model

Key Improvements:
  • F1 Score: +0.2240 (+95.73%)
  • Search Usage: The trained model makes 1.45 searches/question (vs 0.08 for base)
    → Model learned to use search more actively

============================================================================
Full Results
============================================================================
Base Model Log:    results/qrecc_comparison_f1_20231209/base_model/eval.log
Trained Model Log: results/qrecc_comparison_f1_20231209/trained_model/eval.log
============================================================================
```

---

## 📊 支持的评估指标

修改 `configs/qrecc/compare_base_vs_trained_f1.sh` 中的 `REWARD_FN` 来选择不同指标：

| 指标 | 设置 | 适合 QReCC? | 说明 |
|------|------|-------------|------|
| **f1** | `export REWARD_FN='f1'` | ✅ 推荐 | Token级别的F1分数，适合长答案 |
| **rouge_l** | `export REWARD_FN='rouge_l'` | ✅ 最推荐 | 基于最长公共子序列，最适合长答案 |
| **rouge_1** | `export REWARD_FN='rouge_1'` | ✅ 推荐 | Unigram重叠，宽松匹配 |
| **rouge_2** | `export REWARD_FN='rouge_2'` | ✅ 可用 | Bigram重叠 |
| **bleu** | `export REWARD_FN='bleu'` | ✅ 推荐 | N-gram精度，适合生成任务 |
| **em** | `export REWARD_FN='em'` | ❌ 不推荐 | 精确匹配，对长答案太严格 |

**推荐组合:**
1. **主指标**: `rouge_l` (最适合 QReCC 长答案)
2. **辅助**: `f1` (验证)
3. **参考**: `rouge_1` (宽松匹配)

---

## 🔍 查看详细结果

### 查看完整日志

```bash
# 查看基础模型日志
tail -100 results/qrecc_comparison_f1_*/base_model/eval.log

# 查看训练模型日志
tail -100 results/qrecc_comparison_f1_*/trained_model/eval.log
```

### 提取所有环境指标

```bash
# 查看所有 env/ 指标
grep 'env/' results/qrecc_comparison_f1_*/base_model/eval.log
grep 'env/' results/qrecc_comparison_f1_*/trained_model/eval.log
```

**可用的环境指标:**
- `env/number_of_valid_search` - 平均 search 调用次数
- `env/ratio_of_valid_action` - 有效 action 比例
- `env/number_of_valid_action` - 平均有效 action 数
- `env/finish_ratio` - 完成比例

### 提取生成时间

```bash
# 查看生成时间
grep 'timing/gen' results/qrecc_comparison_f1_*/base_model/eval.log
grep 'timing/gen' results/qrecc_comparison_f1_*/trained_model/eval.log
```

---

## 📁 生成的文件

转换脚本会生成:
```
data/qrecc_raw/test.parquet  # 转换后的测试数据
```

评估脚本会生成:
```
results/qrecc_comparison_f1_TIMESTAMP/
├── base_model/
│   └── eval.log          # 基础模型评估日志
└── trained_model/
    └── eval.log          # 训练模型评估日志
```

---

## ⚠️ 常见问题

### 1. 检索服务器连接失败

**错误信息:**
```
Failed to connect to retrieval server at http://127.0.0.1:8000/retrieve
```

**解决方法:**
```bash
# 检查服务器是否运行
curl http://127.0.0.1:8000/retrieve

# 如果没运行，启动服务器
bash retrieval_launch.sh
```

### 2. GPU 内存不足

**错误信息:**
```
CUDA out of memory
```

**解决方法:**

编辑 `configs/qrecc/compare_base_vs_trained_f1.sh`:
```bash
# 降低 GPU 内存使用率
actor_rollout_ref.rollout.gpu_memory_utilization=0.4  # 从 0.6 降到 0.4

# 或减少 batch size
data.val_batch_size=128  # 从 256 降到 128
```

### 3. 数据文件找不到

**错误信息:**
```
FileNotFoundError: data/qrecc_raw/qrecc_test.json
```

**解决方法:**
```bash
# 检查文件是否存在
ls data/qrecc_raw/

# 如果没有，解压数据
cd data/qrecc_raw
unzip qrecc_data.zip
```

### 4. vLLM 版本问题

**错误信息:**
```
flash_attn not supported for Qwen2.5
```

**解决方法:**
```bash
# 设置环境变量
export VLLM_ATTENTION_BACKEND=XFORMERS

# 然后重新运行评估
bash configs/qrecc/compare_base_vs_trained_f1.sh
```

---

## 🎯 快速测试（单个指标）

如果只想快速测试训练后的模型（不对比基础模型）:

```bash
# 修改评估脚本只运行训练模型
vim configs/qrecc/evaluate_qrecc_with_f1.sh

# 设置参数
export CHECKPOINT_PATH="/usr/yuque/guo/searchr1/verl_checkpoints/nq_hotpotqa_train-search-r1-ppo-qwen2.5-3b-it-bm25-em/actor/global_step_200"
export REWARD_FN='rouge_l'
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 运行
bash configs/qrecc/evaluate_qrecc_with_f1.sh
```

---

## 📝 评估多个指标

如果想用多个指标评估（对比不同指标的表现）:

```bash
#!/bin/bash
# 保存为 evaluate_all_metrics.sh

for METRIC in rouge_l f1 rouge_1 bleu; do
    echo "=========================================="
    echo "评估指标: $METRIC"
    echo "=========================================="

    export REWARD_FN=$METRIC
    bash configs/qrecc/compare_base_vs_trained_f1.sh

    echo ""
    echo "完成 $METRIC 评估"
    echo ""
    sleep 10
done

echo "所有指标评估完成！"
```

运行:
```bash
bash evaluate_all_metrics.sh
```

---

## 📊 结果解读

### F1 / ROUGE 分数

| 分数范围 | 质量 |
|---------|------|
| 0.0 - 0.2 | 差 |
| 0.2 - 0.4 | 一般 |
| 0.4 - 0.6 | 好 ✅ |
| 0.6 - 0.8 | 很好 ✅✅ |
| 0.8+ | 极好 ✅✅✅ |

### Search 次数分析

- **增加显著** (0.05 → 1.45): 模型学会使用检索 ✅
- **略微减少** (1.80 → 1.35): 模型变得更有选择性 ✅
- **几乎不变** (0.02 → 0.03): 模型可能没学会使用检索 ⚠️

---

## 🎊 完成！

现在您可以:
1. ✅ 转换 QReCC 测试数据
2. ✅ 对比基础模型和训练模型
3. ✅ 使用多种指标评估 (F1, ROUGE-L, BLEU 等)
4. ✅ 查看 Search 调用统计和生成时间
5. ✅ 全面了解模型改进情况

**需要帮助?** 查看详细文档:
- BLEU/ROUGE 指南: `docs/qrecc_bleu_rouge_guide.md`
- Search 统计说明: `docs/qrecc_search_stats_comparison.md`
- 完整评估指南: `docs/qrecc_evaluation_guide.md`