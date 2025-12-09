# ✅ BLEU 和 ROUGE 已实现！快速开始

## 🎉 好消息

**BLEU** 和 **ROUGE** 评估指标现在已经完全实现并集成到 Search-R1 框架中了！

## 🚀 立即使用

### 最简单的方式

```bash
# 1. 启动检索服务器
bash retrieval_launch.sh

# 2. 使用 ROUGE-L 评估 (推荐)
vim configs/qrecc/compare_base_vs_trained_f1.sh
# 修改这一行:
export REWARD_FN='rouge_l'

# 3. 运行
bash configs/qrecc/compare_base_vs_trained_f1.sh
```

## 📊 支持的指标

| 指标 | 使用 | 适合 QReCC? | 分数范围 |
|------|------|-------------|---------|
| `rouge_l` | `export REWARD_FN='rouge_l'` | ✅ **最推荐** | 0.0 - 1.0 |
| `bleu` | `export REWARD_FN='bleu'` | ✅ 推荐 | 0.0 - 1.0 |
| `rouge_1` | `export REWARD_FN='rouge_1'` | ✅ 推荐 | 0.0 - 1.0 |
| `rouge_2` | `export REWARD_FN='rouge_2'` | ✅ 可用 | 0.0 - 1.0 |
| `f1` | `export REWARD_FN='f1'` | ✅ 推荐 | 0.0 - 1.0 |
| `em` | `export REWARD_FN='em'` | ❌ 不推荐 | 0 or 1 |

## 🎯 推荐组合

**对于 QReCC 长答案评估:**
1. **首选**: `rouge_l` - 基于最长公共子序列,最适合长答案
2. **备选**: `f1` - Token级别,简单有效
3. **参考**: `rouge_1` - 宽松匹配

## 💻 实现细节

**✅ 无需额外依赖!**
- 纯 Python 实现
- 已集成到 veRL 框架
- 与检索服务器完全兼容

**✅ 新增文件:**
- `verl/utils/reward_score/qrecc_bleu_rouge.py` - BLEU/ROUGE 实现
- `verl/trainer/main_ppo.py` - 已更新支持新指标
- `docs/qrecc_bleu_rouge_guide.md` - 详细使用指南

## 📖 使用示例

### 评估命令

```bash
# 使用 ROUGE-L
export REWARD_FN='rouge_l'
bash configs/qrecc/compare_base_vs_trained_f1.sh

# 使用 BLEU
export REWARD_FN='bleu'
bash configs/qrecc/compare_base_vs_trained_f1.sh

# 使用 ROUGE-1
export REWARD_FN='rouge_1'
bash configs/qrecc/compare_base_vs_trained_f1.sh
```

### 训练命令

```bash
vim configs/qrecc/train_qrecc_ppo_plan_b.sh

# 添加这一行:
+algorithm.reward_fn=rouge_l

# 运行训练
bash configs/qrecc/train_qrecc_ppo_plan_b.sh
```

## ⚡ 快速对比

想快速看到不同指标的结果?

```bash
#!/bin/bash
# 对比所有指标

for METRIC in rouge_l rouge_1 bleu f1; do
    echo "========== Testing $METRIC =========="
    export REWARD_FN=$METRIC
    bash configs/qrecc/evaluate_qrecc_with_f1.sh | grep "average_score"
done
```

## 📚 完整文档

- **详细指南**: `docs/qrecc_bleu_rouge_guide.md`
- **veRL 评估**: `docs/qrecc_verl_f1_evaluation.md`
- **BERTScore/F1**: `docs/qrecc_bertscore_f1_quickstart.md`

## 🤔 常见问题

### Q: BLEU 和 ROUGE 有什么区别?

**A:**
- **BLEU**: 关注 precision (n-gram匹配),适合翻译/生成
- **ROUGE**: 关注 recall (覆盖程度),适合摘要/QA
- **ROUGE-L**: 基于最长公共子序列,考虑顺序

### Q: 为什么推荐 ROUGE-L?

**A:** 因为 QReCC 答案普遍较长(10-30词),ROUGE-L:
- ✅ 考虑句子结构
- ✅ F1-based (平衡precision和recall)
- ✅ 对长文本友好
- ✅ 比 EM 更宽松,比 ROUGE-1 更严格

### Q: 分数怎么解读?

**A:**

**ROUGE-L / ROUGE-1 / ROUGE-2:**
- 0.0 - 0.2: 差
- 0.2 - 0.4: 一般
- 0.4 - 0.6: 好 ✅
- 0.6 - 0.8: 很好 ✅✅
- 0.8+: 极好 ✅✅✅

**BLEU:**
- 0.0 - 0.1: 差
- 0.1 - 0.2: 可理解
- 0.2 - 0.3: 可接受 ✅
- 0.3+: 好 ✅✅

### Q: 需要安装额外的库吗?

**A:** 不需要！所有实现都是纯 Python,无需任何外部依赖。

---

## 🎊 完成!

现在你可以:
1. ✅ 使用 ROUGE-L 评估 QReCC 长答案
2. ✅ 使用 BLEU 评估生成质量
3. ✅ 使用 ROUGE-1/2 进行多层次评估
4. ✅ 在训练中使用这些指标作为 reward
5. ✅ 与检索服务器完全集成

**开始使用:**
```bash
export REWARD_FN='rouge_l'
bash configs/qrecc/compare_base_vs_trained_f1.sh
```

就这么简单！🚀