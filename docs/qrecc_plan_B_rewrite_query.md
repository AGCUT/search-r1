# QReCC 实现方案B：使用Ground-Truth改写Query（Baseline方案）

## 方案概述

**方案定位：** Baseline验证方案
**核心思路：** 使用QReCC数据集提供的改写后的独立问题(`rewrite`字段)，跳过模型学习改写的步骤
**技术难度：** ⭐⭐⭐ (中等)
**实施周期：** 2-3周
**优势：** 降低难度，快速验证流程可行性，作为baseline对比
**劣势：** 不够真实，实际应用中没有ground-truth改写

---

## 实现目录结构

```
Search-R1/
├── docs/
│   ├── qrecc_implementation_plan.md          # 总体计划
│   └── qrecc_plan_B_rewrite_query.md         # 本方案文档
│
├── scripts/
│   ├── data_process/
│   │   ├── nq_search.py                      # 原有NQ数据处理（参考）
│   │   └── qrecc_search_plan_b.py            # 新建 - 方案B数据处理脚本
│   │
│   └── evaluate/
│       └── evaluate_qrecc_plan_b.py          # 新建 - 方案B评估脚本
│
├── configs/
│   └── qrecc/
│       ├── train_qrecc_ppo_plan_b.sh         # 新建 - 方案B PPO训练脚本
│       └── train_qrecc_grpo_plan_b.sh        # 新建 - 方案B GRPO训练脚本
│
├── data/
│   └── qrecc_plan_b/                         # 新建 - 方案B数据目录
│       ├── raw/                              # 原始数据（下载后存放）
│       │   ├── train.json
│       │   ├── dev.json
│       │   └── test.json
│       ├── processed/                        # 处理后的数据
│       │   ├── train.parquet
│       │   ├── dev.parquet
│       │   └── test.parquet
│       ├── statistics.json                   # 数据统计信息
│       └── README.md                         # 数据说明文档
│
├── experiments/
│   └── qrecc_plan_b/                         # 新建 - 方案B实验结果
│       ├── baseline_4b/                      # 4B模型实验
│       │   ├── checkpoints/                  # 训练checkpoint
│       │   ├── logs/                         # 训练日志
│       │   └── results.json                  # 评估结果
│       ├── baseline_7b/                      # 7B模型实验（备选）
│       ├── ablation/                         # 消融实验
│       ├── analysis/                         # 分析结果
│       │   ├── error_analysis.md             # 错误分析
│       │   ├── case_study.md                 # 案例分析
│       │   └── visualization/                # 可视化图表
│       └── experiments_log.md                # 实验记录表格
│
├── verl/
│   └── utils/
│       └── reward_score/
│           └── qrecc_em.py                   # 新建 - QReCC专用reward函数
│
└── verl_checkpoints/
    └── qrecc-plan-b-*/                       # 训练checkpoint自动保存目录
```

---

## 需要创建/修改的代码文件

### 1. 数据处理脚本

**文件：** `scripts/data_process/qrecc_search_plan_b.py`

**主要功能：**
- 从HuggingFace下载QReCC数据集
- 读取数据集中的`rewrite`字段（改写后的独立问题）
- 生成Search-R1兼容的parquet格式数据
- 构造不包含对话历史的简单prompt
- 统计数据集信息

**核心函数设计：**

```python
def download_qrecc_dataset():
    """
    功能：下载QReCC原始数据集
    返回：datasets对象(train/dev/test三个split)
    """

def extract_rewrite_query(sample):
    """
    功能：提取改写后的query
    输入：QReCC原始样本
    返回：改写后的独立问题字符串
    """

def make_prompt_plan_b(rewrite_query):
    """
    功能：构造方案B的prompt（不含对话历史，只用改写后的query）
    输入：改写后的问题
    返回：完整prompt字符串

    Prompt模板：
    "Answer the given question.
    You must conduct reasoning inside <think> and </think>...
    Question: {rewrite_query}"
    """

def convert_to_search_r1_format(qrecc_sample):
    """
    功能：将QReCC样本转换为Search-R1数据格式
    输入：QReCC原始样本
    返回：包含以下字段的字典
    {
        "data_source": "qrecc_plan_b",
        "prompt": [{"role": "user", "content": prompt_text}],
        "ability": "fact-reasoning",  # 方案B本质是单轮问答
        "reward_model": {
            "style": "rule",
            "ground_truth": answer
        },
        "extra_info": {
            'conversation_id': conv_id,
            'turn_no': turn_no,
            'original_question': original_question,
            'rewrite': rewrite_query,
            'index': idx,
        }
    }
    """

def process_qrecc_data(split='train'):
    """
    功能：处理指定split的数据
    输入：split名称('train'/'dev'/'test')
    返回：处理后的数据列表
    """

def save_to_parquet(data, output_path):
    """
    功能：保存为parquet格式
    输入：数据列表，输出路径
    输出：保存parquet文件
    """

def compute_statistics(processed_data):
    """
    功能：计算数据集统计信息
    返回：统计字典（样本数、平均长度、问题类型分布等）
    """

def main():
    """
    主函数：执行完整的数据处理流程
    1. 下载数据
    2. 处理train/dev/test
    3. 保存parquet文件
    4. 生成统计信息
    5. 打印样本预览
    """
```

---

### 2. Reward计算函数

**文件：** `verl/utils/reward_score/qrecc_em.py`

**主要功能：**
- 实现QReCC的Exact Match评估
- 处理QReCC答案可能是列表的情况
- 支持多种答案变体匹配

**核心函数设计：**

```python
def normalize_answer(answer):
    """
    功能：标准化答案文本
    - 转小写
    - 去除标点
    - 去除冠词(a/an/the)
    - 去除多余空格
    """

def extract_answer_from_response(response_text):
    """
    功能：从模型生成的完整response中提取答案
    输入：包含<think>、<search>、<answer>标签的完整文本
    返回：<answer>标签内的答案文本，如果没有则返回None
    """

def compute_score_em(solution_str, ground_truth, format_score=0.0):
    """
    功能：计算Exact Match分数
    输入：
        - solution_str: 模型生成的完整字符串
        - ground_truth: 标准答案（可能是字符串或列表）
        - format_score: 格式错误时的分数
    返回：1.0（完全匹配）或 0.0（不匹配）或 format_score（格式错误）

    逻辑：
    1. 从solution_str提取<answer>内容
    2. 如果没有<answer>标签，返回format_score
    3. 标准化预测答案和标准答案
    4. 检查是否匹配（支持ground_truth为列表的情况）
    5. 返回分数
    """

def compute_score_f1(solution_str, ground_truth):
    """
    功能：计算F1分数（可选，用于更细粒度的评估）
    返回：Token级别的F1分数
    """
```

---

### 3. 训练配置脚本

**文件：** `configs/qrecc/train_qrecc_ppo_plan_b.sh`

**主要功能：**
- 配置方案B的PPO训练参数
- 设置数据路径、模型路径、超参数
- 启动训练流程

**关键配置内容：**

```bash
#!/bin/bash

# GPU设置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 数据路径
export DATA_DIR='data/qrecc_plan_b/processed'

# wandb项目配置
WAND_PROJECT='Search-R1-QReCC-Plan-B'

# 模型配置
export BASE_MODEL='Qwen/Qwen3-4B'
# 备选: 'Qwen/Qwen3-4B-Instruct'
# 备选: 'meta-llama/Llama-3.2-3B'
# 备选: 'Qwen/Qwen2.5-7B'

# 实验名称
export EXPERIMENT_NAME=qrecc-plan-b-ppo-qwen3-4b-em

# vLLM配置（针对某些模型）
export VLLM_ATTENTION_BACKEND=XFORMERS

# 启动训练
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    # 数据配置
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/dev.parquet \
    data.train_data_num=null \           # null表示使用全部数据
    data.val_data_num=null \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.shuffle_train_dataloader=True \
    # 方案B使用较小的prompt长度（因为没有对话历史）
    data.max_prompt_length=3072 \        # 降低（相比方案A）
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    \
    # RL算法配置
    algorithm.adv_estimator=gae \        # PPO使用GAE
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \        # 允许思维链
    \
    # Actor配置
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.state_masking=true \
    # FSDP offload配置（节省显存）
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    \
    # Rollout配置（使用vLLM）
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1 \
    \
    # Reference model配置
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    # Critic配置
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=true \
    critic.model.use_remove_padding=True \
    critic.optim.lr=1e-5 \
    critic.optim.lr_warmup_steps_ratio=0.015 \
    critic.ppo_micro_batch_size=8 \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.grad_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    \
    # Trainer配置
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \              # 每100步保存checkpoint
    trainer.test_freq=50 \               # 每50步验证
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=1005 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    \
    # 搜索配置
    max_turns=2 \                        # 允许的最大搜索轮次
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    \
    2>&1 | tee experiments/qrecc_plan_b/baseline_4b/logs/$EXPERIMENT_NAME.log
```

---

### 4. 评估脚本

**文件：** `scripts/evaluate/evaluate_qrecc_plan_b.py`

**主要功能：**
- 加载训练好的模型
- 在测试集上进行推理
- 计算EM、F1等评估指标
- 生成详细的评估报告
- 进行错误分析

**核心函数设计：**

```python
def load_trained_model(checkpoint_path):
    """
    功能：加载训练好的checkpoint
    返回：model, tokenizer
    """

def inference_on_sample(model, tokenizer, sample, retriever_url):
    """
    功能：对单个样本进行推理
    输入：模型、分词器、样本数据、检索器URL
    返回：生成的完整response文本

    流程：
    1. 构造prompt
    2. 模型生成（可能多轮检索）
    3. 返回完整生成文本
    """

def evaluate_dataset(model, tokenizer, test_data, retriever_url):
    """
    功能：在整个测试集上评估
    输入：模型、分词器、测试数据、检索器URL
    返回：评估结果字典
    {
        'em': exact_match_score,
        'f1': f1_score,
        'total': total_samples,
        'correct': correct_samples,
        'predictions': [...]  # 每个样本的预测
    }
    """

def compute_metrics(predictions, ground_truths):
    """
    功能：计算评估指标
    返回：EM、F1、ROUGE等指标
    """

def analyze_errors(predictions, ground_truths, test_data):
    """
    功能：错误分析
    返回：错误类型统计、典型失败案例
    """

def generate_report(eval_results, output_path):
    """
    功能：生成评估报告
    输出：Markdown格式的详细报告
    """

def main():
    """
    主函数：
    1. 加载模型和数据
    2. 进行推理
    3. 计算指标
    4. 错误分析
    5. 生成报告
    """
```

---

### 5. 数据说明文档

**文件：** `data/qrecc_plan_b/README.md`

**主要内容：**

```markdown
# QReCC Plan B Dataset

## 数据来源
HuggingFace: McGill-NLP/QReCC

## 处理方法
使用QReCC数据集中的`rewrite`字段（改写后的独立问题），
不包含对话历史，简化为单轮问答任务。

## 数据格式
每条数据包含：
- data_source: "qrecc_plan_b"
- prompt: 简单的单轮问答prompt
- ability: "fact-reasoning"
- reward_model: EM匹配规则
- extra_info: 对话元信息

## 数据统计
- 训练集: ~50k样本（所有对话turn）
- 验证集: ~5k样本
- 测试集: ~5k样本

## 使用方法
```bash
# 数据处理
python scripts/data_process/qrecc_search_plan_b.py

# 训练
bash configs/qrecc/train_qrecc_ppo_plan_b.sh

# 评估
python scripts/evaluate/evaluate_qrecc_plan_b.py \
  --checkpoint verl_checkpoints/qrecc-plan-b-ppo-qwen3-4b-em/final \
  --test_data data/qrecc_plan_b/processed/test.parquet
```

## 注意事项
- 方案B跳过了对话建模，仅作为baseline
- 不需要学习query改写能力
- 性能上限应该接近单轮NQ任务
```

---

## 实现步骤

### 阶段1: 环境准备（第1天）

**任务清单：**
- [ ] 确认conda环境已安装（searchr1 + retriever）
- [ ] 确认GPU资源（至少4-8张GPU）
- [ ] 创建项目目录结构
- [ ] 配置wandb账号

**执行命令：**
```bash
# 创建目录
mkdir -p data/qrecc_plan_b/{raw,processed}
mkdir -p experiments/qrecc_plan_b/{baseline_3b/{checkpoints,logs},analysis}
mkdir -p configs/qrecc
mkdir -p scripts/evaluate

# 激活环境
conda activate searchr1
pip list | grep vllm  # 确认vllm已安装

# wandb登录
wandb login
```

---

### 阶段2: 数据下载与探索（第2-3天）

**任务清单：**
- [ ] 编写数据下载代码
- [ ] 探索QReCC数据集结构
- [ ] 查看样本示例
- [ ] 统计数据特征

**实现要点：**

**功能1: 下载数据集**
```python
from datasets import load_dataset

# 下载完整数据集
dataset = load_dataset('McGill-NLP/QReCC')

# 查看结构
print(dataset)
# DatasetDict({
#     train: Dataset
#     dev: Dataset
#     test: Dataset
# })

# 查看单个样本
sample = dataset['train'][0]
print(sample.keys())
# dict_keys(['Conversation_no', 'Turn_no', 'Question',
#            'Answer', 'Rewrite', 'Conversation_context', ...])
```

**功能2: 统计分析**
- 总样本数
- 平均问题长度
- 平均改写query长度
- 答案长度分布

---

### 阶段3: 数据处理（第4-5天）

**任务清单：**
- [ ] 编写`qrecc_search_plan_b.py`
- [ ] 实现数据转换函数
- [ ] 处理train/dev/test三个split
- [ ] 验证生成的parquet文件
- [ ] 可视化检查样本

**核心逻辑：**

**Prompt构造：**
```python
def make_prompt_plan_b(rewrite_query):
    template = """Answer the given question.
You must conduct reasoning inside <think> and </think> first every time you get new information.
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>.
You can search as many times as your want.
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations.

Question: {question}
"""
    return template.format(question=rewrite_query)
```

**数据验证：**
```python
# 处理后检查
processed = process_qrecc_data('train')
print(f"Total samples: {len(processed)}")
print("\n=== Sample 0 ===")
print(json.dumps(processed[0], indent=2, ensure_ascii=False))
```

---

### 阶段4: Reward函数实现（第5天）

**任务清单：**
- [ ] 编写`qrecc_em.py`
- [ ] 测试EM计算正确性
- [ ] 处理边界情况

**测试用例：**
```python
# 测试用例
test_cases = [
    {
        'solution': '<think>...</think><answer>Paris</answer>',
        'ground_truth': 'Paris',
        'expected': 1.0
    },
    {
        'solution': '<think>...</think><answer>paris</answer>',  # 大小写
        'ground_truth': 'Paris',
        'expected': 1.0
    },
    {
        'solution': '<think>...</think>',  # 缺少answer标签
        'ground_truth': 'Paris',
        'expected': 0.0  # format_score
    }
]
```

---

### 阶段5: 检索服务准备（第6天）

**任务清单：**
- [ ] 确认wiki corpus和索引已下载
- [ ] 启动检索服务器
- [ ] 验证检索API可用

**执行命令：**
```bash
# 激活retriever环境
conda activate retriever

# 启动检索服务器
bash retrieval_launch.sh

# 在另一个终端验证
curl -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"queries": ["What is the capital of France?"], "topk": 3}'
```

---

### 阶段6: 小规模试运行（第7-8天）

**任务清单：**
- [ ] 编写训练脚本`train_qrecc_ppo_plan_b.sh`
- [ ] 采样1000条数据快速测试
- [ ] 运行2-3个epoch
- [ ] 检查训练是否正常

**配置修改：**
```bash
# 在train_qrecc_ppo_plan_b.sh中修改
data.train_data_num=1000
data.val_data_num=200
trainer.total_epochs=3
```

**验证要点：**
- 训练脚本运行无错误
- Loss正常下降
- Wandb日志正常记录
- 验证集EM有提升趋势

---

### 阶段7: 完整训练（第9-12天）

**任务清单：**
- [ ] 使用完整数据集训练
- [ ] 监控训练过程
- [ ] 保存最佳checkpoint
- [ ] 记录训练曲线

**执行命令：**
```bash
conda activate searchr1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash configs/qrecc/train_qrecc_ppo_plan_b.sh
```

**监控指标：**
- 平均reward趋势（应该上升）
- 验证集EM（目标>40%）
- KL散度（不应过大）
- GPU利用率

---

### 阶段8: 评估与分析（第13-14天）

**任务清单：**
- [ ] 编写评估脚本`evaluate_qrecc_plan_b.py`
- [ ] 在测试集上推理
- [ ] 计算EM、F1指标
- [ ] 进行错误分析
- [ ] 撰写评估报告

**评估流程：**
```bash
python scripts/evaluate/evaluate_qrecc_plan_b.py \
  --checkpoint verl_checkpoints/qrecc-plan-b-ppo-qwen3-4b-em/step_1000 \
  --test_data data/qrecc_plan_b/processed/test.parquet \
  --output experiments/qrecc_plan_b/baseline_4b/results.json
```

---

## 关键技术点

### 1. Prompt设计

**要点：**
- 方案B使用简单的单轮prompt
- 直接使用改写后的query，无需对话历史
- 保持与原Search-R1相同的格式（<think>、<search>、<answer>标签）

**示例：**
```
原始对话：
Turn 1: Q: Who is the president of USA?
        A: Joe Biden
Turn 2: Q: How old is he?

改写后（方案B使用）：
Question: How old is Joe Biden?
```

### 2. 数据格式转换

**QReCC原始格式 → Search-R1格式：**
- 提取`Rewrite`字段
- 构造简单prompt（不含历史）
- 保留原始对话信息在`extra_info`（用于后续分析）

### 3. Reward计算

**实现要点：**
- 从生成文本中提取`<answer>`标签内容
- 标准化答案（去标点、小写、去冠词）
- 支持答案列表匹配（QReCC的答案可能有多个变体）
- 格式错误处理（缺少`<answer>`标签时给予format_score）

### 4. 训练配置优化

**相比原NQ的调整：**
- `max_prompt_length`: 3072（方案B更小，因为无对话历史）
- `max_response_length`: 保持500
- 其他参数基本不变

---

## 预期结果

### 性能指标

**预期目标：**
- Exact Match (EM): 40-45%
- F1 Score: 48-53%
- 平均检索次数: 1.5-2.0次/问题

**对比基准：**
- Search-R1 (NQ单轮): EM ~45%
- 方案B应该接近NQ性能（因为都是单轮问答）

### 训练时间

**估算（8卡 A100/H100）：**
- 4B模型: 1-2天
- 数据处理: 0.5天
- 评估分析: 0.5天
- 总计: 2-3天

---

## 常见问题处理

### Q1: 训练不收敛怎么办？

**检查清单：**
- [ ] 数据处理是否正确（打印几个样本验证）
- [ ] Reward函数是否返回合理值（打印reward）
- [ ] 学习率是否过大（尝试降低到5e-7）
- [ ] 检索服务是否正常（测试API）

**解决方案：**
```bash
# 降低学习率
actor_rollout_ref.actor.optim.lr=5e-7

# 增加warmup
actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.3

# 或尝试GRPO（更稳定）
bash configs/qrecc/train_qrecc_grpo_plan_b.sh
```

### Q2: 显存不足怎么办？

**解决方案：**
- 减小batch_size: `data.train_batch_size=256`
- 开启gradient checkpointing（已默认开启）
- 增大offload: 确保`param_offload=true`
- 降低`max_prompt_length=2048`

### Q3: EM分数很低怎么办？

**诊断步骤：**
1. 检查答案提取是否正确（是否能找到`<answer>`标签）
2. 查看生成样本（是否有合理的思维链和检索）
3. 检查检索质量（Recall@3是否足够高）
4. 分析错误类型（格式错误/检索错误/推理错误）

---

## 下一步：方案A

**方案B完成后的后续工作：**
1. 分析方案B的性能瓶颈
2. 确认基础流程可行性
3. 以方案B为baseline，实施方案A（端到端对话）
4. 对比两个方案的性能差异
5. 分析模型是否学会了query改写能力

---

## 总结

**方案B的价值：**
- ✅ 快速验证Search-R1在QReCC数据上的可行性
- ✅ 建立baseline用于对比
- ✅ 简化问题，降低实施难度
- ✅ 为方案A积累经验

**方案B的局限：**
- ❌ 不够真实（实际应用中没有ground-truth改写）
- ❌ 没有学习对话建模能力
- ❌ 没有学习query改写能力
- ❌ 性能上限受限（等同于单轮问答）

**建议：** 先完成方案B验证流程，再实施方案A追求更高性能和真实应用价值。