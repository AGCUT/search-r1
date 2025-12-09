# ✅ 评估脚本现在会对比 Search 次数和生成时间！

## 🎉 更新内容

对比评估脚本已更新，现在会自动统计并对比:

✅ **评估指标分数** (EM, F1, BLEU, ROUGE等)
✅ **平均 Search 次数** - 每个问题调用检索的平均次数
✅ **生成时间** - 模型推理时间对比

---

## 📊 输出示例

运行 `bash configs/qrecc/compare_base_vs_trained_f1.sh` 后会看到:

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

To view detailed logs:
  tail -100 results/qrecc_comparison_f1_20231209/base_model/eval.log
  tail -100 results/qrecc_comparison_f1_20231209/trained_model/eval.log

To extract all metrics:
  grep 'env/' results/qrecc_comparison_f1_20231209/base_model/eval.log
  grep 'env/' results/qrecc_comparison_f1_20231209/trained_model/eval.log
============================================================================
```

---

## 📈 统计的指标

### 1. 评估分数
- 根据设置的 `REWARD_FN` 显示对应指标
- 支持: `em`, `f1`, `bleu`, `rouge_l`, `rouge_1`, `rouge_2` 等
- 显示绝对改进值和百分比

### 2. Search 次数统计

**指标**: `env/number_of_valid_search`

- **含义**: 每个问题平均调用 `<search>` 的次数
- **来源**: veRL 框架在生成过程中统计
- **计算**: 所有样本的 search 调用次数的平均值

**解读**:
- **基础模型**: 通常很少或不使用 search (0.0 - 0.2)
- **训练后模型**: 应该学会适当使用 search (1.0 - 2.0)
- **理想情况**: max_turns=2 时,约 1.0-1.5 searches/question

**示例**:
```
Avg Searches/Question     0.08                1.45                +1.37 (1712.5%)
  → Model learned to use search more actively
```

### 3. 生成时间

**指标**: `timing/gen`

- **含义**: 每个 batch 的生成时间(秒)
- **包含**: 模型推理 + 检索调用时间
- **注意**: 训练后模型通常更慢(因为会实际调用检索)

---

## 🔍 查看详细指标

脚本会提示你可以查看更多指标:

```bash
# 查看所有环境指标
grep 'env/' results/qrecc_comparison_*/base_model/eval.log
grep 'env/' results/qrecc_comparison_*/trained_model/eval.log
```

**可用的 `env/` 指标**:

| 指标 | 说明 |
|------|------|
| `env/number_of_valid_search` | 平均 search 次数 |
| `env/ratio_of_valid_action` | 有效action比例 |
| `env/number_of_valid_action` | 平均有效action数 |
| `env/finish_ratio` | 完成比例 |

---

## 💡 结果分析

### Search 次数分析

**场景 1: Search 次数显著增加**
```
Avg Searches/Question: 0.05 → 1.45 (+1.40)
```
✅ **好现象** - 模型学会了使用检索工具
→ 说明 RL 训练成功让模型学会调用 search

**场景 2: Search 次数略微减少**
```
Avg Searches/Question: 1.80 → 1.35 (-0.45)
```
✅ **可能也是好现象** - 模型变得更有选择性
→ 如果同时 F1/ROUGE 分数提升,说明模型学会了更高效地使用 search

**场景 3: Search 次数没有变化**
```
Avg Searches/Question: 0.02 → 0.03 (+0.01)
```
⚠️ **需要注意** - 模型可能没有学会使用 search
→ 检查训练配置、prompt 格式、retriever 连接

### 生成时间分析

**正常情况**:
```
Generation Time: 10.5s → 14.2s (+3.7s)
```
- 训练后模型更慢是正常的(因为实际调用检索)
- 每次 search 调用约增加 1-3 秒

**异常情况**:
```
Generation Time: 10.5s → 25.8s (+15.3s)
```
⚠️ 可能的问题:
- 检索服务器响应慢
- 模型生成token数过多
- 网络延迟

---

## 🎯 使用建议

### 评估流程

```bash
# 1. 启动检索服务器
bash retrieval_launch.sh

# 2. 确认检索服务器工作
curl -X POST http://127.0.0.1:8000/retrieve \
    -H "Content-Type: application/json" \
    -d '{"queries": ["test"], "topk": 3}'

# 3. 运行对比评估
bash configs/qrecc/compare_base_vs_trained_f1.sh

# 4. 查看完整统计
grep 'env/' results/qrecc_comparison_*/base_model/eval.log
grep 'env/' results/qrecc_comparison_*/trained_model/eval.log
```

### 对比不同指标的 Search 行为

```bash
#!/bin/bash
# 对比不同指标下的 search 行为

for METRIC in f1 rouge_l bleu; do
    echo "========== Testing with $METRIC =========="
    export REWARD_FN=$METRIC
    bash configs/qrecc/compare_base_vs_trained_f1.sh

    # 提取 search 统计
    echo ""
    echo "Search statistics for $METRIC:"
    grep "number_of_valid_search" results/qrecc_comparison_${METRIC}_*/*/eval.log
    echo ""
done
```

---

## 📁 更新的文件

- **`configs/qrecc/compare_base_vs_trained_f1.sh`** ✨
  - ✅ 添加 search 次数统计
  - ✅ 添加生成时间统计
  - ✅ 更好的表格格式输出
  - ✅ 智能分析(search 使用模式)

---

## 🔧 自定义统计

如果你想提取其他指标,可以在脚本末尾添加:

```bash
# 提取其他指标
BASE_ACTION=$(grep -oP "(?<=env/number_of_valid_action:\s)\d+\.\d+" "$OUTPUT_BASE_DIR/base_model/eval.log" | tail -1)
TRAINED_ACTION=$(grep -oP "(?<=env/number_of_valid_action:\s)\d+\.\d+" "$OUTPUT_BASE_DIR/trained_model/eval.log" | tail -1)

echo "Valid Actions: $BASE_ACTION → $TRAINED_ACTION"
```

---

## 📊 完整示例输出

```
============================================================================
QReCC Model Comparison - Base vs Trained
============================================================================
Base Model:         Qwen/Qwen2.5-3B
Trained Checkpoint: /usr/yuque/guo/searchr1/verl_checkpoints/.../global_step_200
Data Directory:     data/qrecc_raw
Evaluation Metric:  rouge_l
Output Directory:   results/qrecc_comparison_rouge_l_20231209_143022
GPUs:               0,1,2,3,4,5,6,7
============================================================================

[... 评估过程 ...]

============================================================================
COMPARISON REPORT
============================================================================

Metric: ROUGE_L Score
--------------------------------------------------------------------
Metric                    Base Model           Trained Model        Change
--------------------------------------------------------------------
ROUGE_L Score             0.3245              0.5782              +0.2537 (+78.18%)
Avg Searches/Question     0.05                1.52                +1.47 (2940.0%)
Generation Time (s)       11.23               16.87               +5.64
--------------------------------------------------------------------

✓ The trained model shows IMPROVEMENT over the base model

Key Improvements:
  • ROUGE_L Score: +0.2537 (+78.18%)
  • Search Usage: The trained model makes 1.52 searches/question (vs 0.05 for base)
    → Model learned to use search more actively
```

---

## 总结

现在评估脚本会完整地对比:

✅ **准确性**: EM / F1 / BLEU / ROUGE 等分数
✅ **Search 行为**: 调用检索的频率
✅ **效率**: 生成时间对比

这让你能够全面了解模型在 RL 训练后的改进情况！🚀