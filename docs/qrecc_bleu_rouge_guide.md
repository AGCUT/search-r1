# ✅ BLEU 和 ROUGE 评估指标已实现！

## 🎉 新增功能

现在 Search-R1 支持使用 **BLEU** 和 **ROUGE** 评估指标进行模型评估和训练！

### 支持的指标

| 指标 | 说明 | 分数范围 | 适用场景 | 实现方式 |
|------|------|---------|---------|---------|
| **bleu** | N-gram precision | 0.0 - 1.0 | 长文本生成/翻译 | ✅ 纯Python实现 |
| **rouge_l** | Longest Common Subsequence | 0.0 - 1.0 | 摘要/长答案 | ✅ 纯Python实现 |
| **rouge_1** | Unigram overlap | 0.0 - 1.0 | 通用文本评估 | ✅ 纯Python实现 |
| **rouge_2** | Bigram overlap | 0.0 - 1.0 | 更严格的评估 | ✅ 纯Python实现 |

**无需额外依赖！** 所有指标都是纯Python实现,不需要安装任何外部库。

---

## 🚀 快速使用

### 1. 使用 ROUGE-L 评估 (推荐)

```bash
# 启动检索服务器
bash retrieval_launch.sh

# 修改评估脚本
vim configs/qrecc/compare_base_vs_trained_f1.sh

# 设置评估指标为 rouge_l
export REWARD_FN='rouge_l'

# 运行评估
bash configs/qrecc/compare_base_vs_trained_f1.sh
```

### 2. 使用 BLEU 评估

```bash
# 设置评估指标为 bleu
export REWARD_FN='bleu'

bash configs/qrecc/compare_base_vs_trained_f1.sh
```

### 3. 使用 ROUGE-1 或 ROUGE-2

```bash
# ROUGE-1 (unigram)
export REWARD_FN='rouge_1'

# ROUGE-2 (bigram)
export REWARD_FN='rouge_2'

bash configs/qrecc/compare_base_vs_trained_f1.sh
```

---

## 📊 指标对比

### QReCC 数据集推荐使用

由于 QReCC 答案普遍较长(平均 10-30 个单词),推荐使用:

1. **ROUGE-L** (首选) - 考虑最长公共子序列,适合长答案
2. **F1** - Token级别重叠,平衡precision和recall
3. **ROUGE-1** - Unigram重叠,宽松评估
4. **BLEU** - N-gram precision,适合生成质量评估

### 各指标特点

#### BLEU
- **优点**:
  - 考虑多个n-gram级别 (1-4gram)
  - 有brevity penalty,惩罚过短的输出
  - 广泛用于机器翻译评估
- **缺点**:
  - 只看precision,不看recall
  - 对于短文本可能过于严格
- **适用**: 长文本生成,翻译任务

#### ROUGE-L
- **优点**:
  - 基于最长公共子序列(LCS)
  - 考虑句子结构相似性
  - F1-based,平衡precision和recall
- **缺点**:
  - 计算相对复杂
  - 对单词顺序敏感
- **适用**: 摘要,长答案QA **(推荐用于QReCC)**

#### ROUGE-1
- **优点**:
  - Unigram级别,最宽松
  - 计算简单,速度快
  - 容忍顺序差异
- **缺点**:
  - 不考虑顺序信息
  - 可能给随机词汇堆砌高分
- **适用**: 快速评估,宽松匹配

#### ROUGE-2
- **优点**:
  - Bigram级别,考虑部分顺序
  - 比ROUGE-1更严格
  - 计算仍然较快
- **缺点**:
  - 对短文本不太适用
  - 可能过于严格
- **适用**: 中长文本,需要部分顺序信息

---

## 📈 分数解读

### BLEU Score

| 分数范围 | 质量 | 说明 |
|---------|------|------|
| 0.0 - 0.1 | 差 | 几乎没有overlap |
| 0.1 - 0.2 | 可理解 | 有一定overlap |
| 0.2 - 0.3 | 可接受 | 中等质量 |
| 0.3 - 0.4 | 好 | 较高质量 |
| 0.4+ | 很好 | 高质量 |

**注意**: BLEU分数通常较低,0.3+就已经很好了！

### ROUGE Scores

| 分数范围 | 质量 | 说明 |
|---------|------|------|
| 0.0 - 0.2 | 差 | 很少overlap |
| 0.2 - 0.4 | 一般 | 有一定相似性 |
| 0.4 - 0.6 | 好 | 较高相似性 |
| 0.6 - 0.8 | 很好 | 高度相似 |
| 0.8+ | 极好 | 几乎完全匹配 |

**注意**: ROUGE分数通常比BLEU高,0.5+就是不错的结果。

---

## 🔧 高级用法

### 1. 在训练中使用 ROUGE-L

```bash
vim configs/qrecc/train_qrecc_ppo_plan_b.sh

# 添加reward function参数:
+algorithm.reward_fn=rouge_l

# 运行训练
bash configs/qrecc/train_qrecc_ppo_plan_b.sh
```

### 2. 评估脚本模板

创建自定义评估脚本:

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CHECKPOINT_PATH="your/checkpoint/path"
export REWARD_FN='rouge_l'  # 或 bleu, rouge_1, rouge_2
export DATA_DIR='data/qrecc_raw'

python3 -m verl.trainer.main_ppo \
    data.val_files=$DATA_DIR/qrecc_test.json \
    +algorithm.reward_fn=$REWARD_FN \
    actor_rollout_ref.model.path=$CHECKPOINT_PATH \
    +trainer.val_only=true \
    ... # 其他参数
```

### 3. 组合多个指标评估

你可以运行多次评估,每次使用不同指标:

```bash
for METRIC in rouge_l rouge_1 bleu f1; do
    echo "Evaluating with $METRIC..."
    export REWARD_FN=$METRIC
    bash configs/qrecc/evaluate_qrecc_with_f1.sh > results/eval_$METRIC.log
done

# 对比所有结果
grep "average_score" results/eval_*.log
```

---

## 📝 代码示例

### Python中直接使用

```python
from verl.utils.reward_score import qrecc_bleu_rouge

# 预测答案
prediction = "The capital of France is Paris, which is located in the northern part of the country."

# 真实答案
references = [
    "Paris is the capital of France.",
    "The capital of France is Paris."
]

# 计算BLEU
bleu_score = qrecc_bleu_rouge.bleu_check(prediction, references)
print(f"BLEU: {bleu_score:.4f}")

# 计算ROUGE-L
rouge_l_score = qrecc_bleu_rouge.rouge_l_check(prediction, references)
print(f"ROUGE-L: {rouge_l_score:.4f}")

# 计算ROUGE-1
rouge_1_score = qrecc_bleu_rouge.rouge_1_check(prediction, references)
print(f"ROUGE-1: {rouge_1_score:.4f}")

# 计算ROUGE-2
rouge_2_score = qrecc_bleu_rouge.rouge_2_check(prediction, references)
print(f"ROUGE-2: {rouge_2_score:.4f}")
```

### 在veRL框架中使用

```python
# veRL框架会自动调用,只需配置即可
# 在 main_ppo.py 中已经集成
```

---

## 🆚 与现有指标对比

| 指标 | 类型 | 分数范围 | 适合长答案 | 需要外部库 | 速度 |
|------|------|---------|-----------|-----------|------|
| **EM** | Binary | 0 or 1 | ❌ 否 | ❌ 不需要 | ⚡⚡⚡ 最快 |
| **F1** | Continuous | 0.0 - 1.0 | ✅ 是 | ❌ 不需要 | ⚡⚡ 快 |
| **BLEU** | Continuous | 0.0 - 1.0 | ✅ 是 | ❌ 不需要 | ⚡⚡ 快 |
| **ROUGE-L** | Continuous | 0.0 - 1.0 | ✅ 是 | ❌ 不需要 | ⚡ 中等 |
| **ROUGE-1** | Continuous | 0.0 - 1.0 | ✅ 是 | ❌ 不需要 | ⚡⚡ 快 |
| **ROUGE-2** | Continuous | 0.0 - 1.0 | ✅ 是 | ❌ 不需要 | ⚡⚡ 快 |
| **BERTScore** | Continuous | 0.0 - 1.0 | ✅ 是 | ✅ 需要 | 🐢 慢 |

---

## 💡 实际使用建议

### 对于 QReCC 评估

**推荐指标组合**:
1. **主指标**: `rouge_l` (最适合长答案)
2. **辅助指标**: `f1` (token级别验证)
3. **参考指标**: `rouge_1` (宽松匹配)

**评估流程**:
```bash
# 1. 使用 ROUGE-L 主评估
export REWARD_FN='rouge_l'
bash configs/qrecc/compare_base_vs_trained_f1.sh

# 2. 使用 F1 辅助验证
export REWARD_FN='f1'
bash configs/qrecc/compare_base_vs_trained_f1.sh

# 3. 对比结果
```

### 对于训练

**推荐**:
- **初期训练**: 使用 `f1` (简单有效)
- **中期调优**: 切换到 `rouge_l` (更准确)
- **最终微调**: 根据具体任务选择

---

## 📂 新增文件

1. **`verl/utils/reward_score/qrecc_bleu_rouge.py`** ⭐
   - BLEU 实现
   - ROUGE-L 实现
   - ROUGE-1 实现
   - ROUGE-2 实现
   - 与veRL框架集成的scoring函数

2. **`verl/trainer/main_ppo.py`** (已更新)
   - 添加了 BLEU/ROUGE 支持
   - 更新了 `_select_rm_score_fn` 函数
   - 更新了 RewardManager 文档

3. **`docs/qrecc_bleu_rouge_guide.md`** (本文档)
   - 使用指南

---

## 🎯 总结

**✅ 已实现功能:**
- BLEU score (n-gram precision)
- ROUGE-L (LCS-based)
- ROUGE-1 (unigram)
- ROUGE-2 (bigram)
- 完整的veRL框架集成
- 纯Python实现,无需额外依赖

**📝 使用方法:**
```bash
# 只需设置一个环境变量!
export REWARD_FN='rouge_l'  # 或 bleu, rouge_1, rouge_2

# 然后运行评估
bash configs/qrecc/compare_base_vs_trained_f1.sh
```

**🚀 现在你可以:**
1. 使用 ROUGE-L 评估 QReCC 长答案
2. 使用 BLEU 评估文本生成质量
3. 使用 ROUGE-1/2 进行多层次评估
4. 在训练中使用这些指标作为reward

---

需要帮助? 查看其他文档:
- veRL 框架评估: `docs/qrecc_verl_f1_evaluation.md`
- F1/BERTScore 评估: `docs/qrecc_bertscore_f1_quickstart.md`
- 完整评估指南: `docs/qrecc_evaluation_guide.md`