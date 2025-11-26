# QReCC 实现方案A：端到端对话式检索问答（推荐方案）

## 方案概述

**方案定位：** 端到端对话建模，最终目标方案
**核心思路：** 输入完整对话历史+当前问题，模型自主学习改写query和对话理解
**技术难度：** ⭐⭐⭐⭐⭐ (高)
**实施周期：** 4-6周
**优势：** 更真实、学习对话能力、实用价值高
**劣势：** 实现难度大、训练不稳定风险、需要更多调优

---

## 实现目录结构

```
Search-R1/
├── docs/
│   ├── qrecc_implementation_plan.md          # 总体计划
│   ├── qrecc_plan_B_rewrite_query.md         # 方案B文档
│   └── qrecc_plan_A_end_to_end.md            # 本方案文档
│
├── scripts/
│   ├── data_process/
│   │   ├── nq_search.py                      # 原有NQ数据处理（参考）
│   │   ├── qrecc_search_plan_b.py            # 方案B数据处理
│   │   └── qrecc_search_plan_a.py            # 新建 - 方案A数据处理脚本
│   │
│   ├── evaluate/
│   │   ├── evaluate_qrecc_plan_b.py          # 方案B评估
│   │   └── evaluate_qrecc_plan_a.py          # 新建 - 方案A评估脚本
│   │
│   └── analysis/
│       ├── analyze_query_rewrite.py          # 新建 - Query改写质量分析
│       ├── analyze_context_usage.py          # 新建 - 对话上下文使用分析
│       └── compare_plans.py                  # 新建 - 方案A vs B对比分析
│
├── configs/
│   └── qrecc/
│       ├── train_qrecc_ppo_plan_a.sh         # 新建 - 方案A PPO训练脚本
│       ├── train_qrecc_grpo_plan_a.sh        # 新建 - 方案A GRPO训练脚本
│       ├── train_qrecc_plan_a_v1.sh          # 基础版（只EM reward）
│       ├── train_qrecc_plan_a_v2.sh          # 增强版（多reward）
│       └── train_qrecc_plan_a_multitask.sh   # 多任务版（NQ+QReCC联合）
│
├── data/
│   └── qrecc_plan_a/                         # 新建 - 方案A数据目录
│       ├── raw/                              # 原始数据
│       │   ├── train.json
│       │   ├── dev.json
│       │   └── test.json
│       ├── processed/                        # 处理后的数据
│       │   ├── train_full_history.parquet    # 完整对话历史
│       │   ├── train_window_3.parquet        # 滑动窗口3轮
│       │   ├── train_window_5.parquet        # 滑动窗口5轮
│       │   ├── dev.parquet
│       │   └── test.parquet
│       ├── statistics.json                   # 数据统计
│       ├── conversation_analysis.json        # 对话分析
│       └── README.md                         # 数据说明文档
│
├── experiments/
│   └── qrecc_plan_a/                         # 新建 - 方案A实验结果
│       ├── baseline/                         # 基础实验（只EM reward）
│       │   ├── history_1turn/                # 1轮历史
│       │   ├── history_3turn/                # 3轮历史
│       │   ├── history_5turn/                # 5轮历史
│       │   └── history_full/                 # 完整历史
│       ├── enhanced_reward/                  # 增强reward实验
│       │   ├── with_query_rewrite_reward/
│       │   ├── with_retrieval_reward/
│       │   └── multi_reward/
│       ├── prompt_variants/                  # 不同prompt模板实验
│       │   ├── format_v1/
│       │   ├── format_v2_with_examples/
│       │   └── format_v3_explicit_rewrite/
│       ├── model_comparison/                 # 不同模型对比
│       │   ├── qwen3_4b/
│       │   ├── qwen3_7b/
│       │   └── llama3.2_3b/
│       ├── vs_plan_b/                        # 与方案B对比
│       ├── analysis/                         # 深度分析
│       │   ├── query_rewrite_quality.md      # Query改写质量分析
│       │   ├── context_understanding.md      # 上下文理解分析
│       │   ├── error_analysis.md             # 错误分析
│       │   ├── case_study.md                 # 案例研究
│       │   └── visualization/                # 可视化
│       └── final_report.md                   # 最终实验报告
│
├── verl/
│   ├── trainer/
│   │   └── main_ppo_qrecc.py                 # 可选 - QReCC专用trainer（支持多阶段reward）
│   │
│   └── utils/
│       └── reward_score/
│           ├── qrecc_em.py                   # QReCC EM reward
│           └── qrecc_multi_reward.py         # 新建 - 多维度reward
│
└── verl_checkpoints/
    └── qrecc-plan-a-*/                       # 训练checkpoint自动保存
```

---

## 需要创建/修改的代码文件

### 1. 数据处理脚本（核心）

**文件：** `scripts/data_process/qrecc_search_plan_a.py`

**主要功能：**
- 下载QReCC数据集
- 按对话分组，构建对话历史
- 生成包含上下文的完整prompt
- 支持多种历史长度配置（滑动窗口）
- 统计对话特征

**核心函数设计：**

```python
def download_qrecc_dataset():
    """
    功能：下载QReCC原始数据集
    返回：datasets对象(train/dev/test)
    """

def group_by_conversation(dataset):
    """
    功能：按Conversation_no分组数据
    输入：原始dataset
    返回：字典 {conversation_id: [turn1, turn2, ...]}

    重要：同一对话的所有turn保持顺序
    """

def format_conversation_history(history_turns, max_turns=None):
    """
    功能：格式化对话历史
    输入：
        - history_turns: 之前的对话轮次列表 [(Q1, A1), (Q2, A2), ...]
        - max_turns: 最多保留几轮（None表示全部保留）
    返回：格式化的历史文本

    格式示例：
    '''
    Turn 1:
    Q: Who is the president of USA?
    A: Joe Biden

    Turn 2:
    Q: When was he born?
    A: November 20, 1942
    '''

    策略：
    - 如果max_turns不为None，使用滑动窗口保留最近N轮
    - 计算token数，确保不超过最大长度
    """

def make_prompt_plan_a(current_question, conversation_history=None,
                       with_rewrite_hint=False, with_examples=False):
    """
    功能：构造方案A的完整prompt
    输入：
        - current_question: 当前问题
        - conversation_history: 格式化的对话历史字符串
        - with_rewrite_hint: 是否包含改写提示
        - with_examples: 是否包含few-shot示例
    返回：完整prompt字符串

    基础版prompt模板：
    '''
    Answer the given question based on the conversation history.
    You must conduct reasoning inside <think> and </think> first every time you get new information.
    After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search>...

    Conversation History:
    {history}

    Current Question: {question}
    '''

    增强版prompt模板（with_rewrite_hint=True）：
    '''
    ... (前面相同)

    Important: If the current question contains pronouns or references to previous context
    (such as "he", "she", "it", "they", "there", etc.), you MUST identify what they refer to
    based on the conversation history, and use the complete entity names in your search query.

    Conversation History:
    {history}

    Current Question: {question}
    '''

    Few-shot版本（with_examples=True）：
    在prompt开头添加1-2个示例展示如何处理对话
    """

def convert_to_search_r1_format(conv_turns, turn_idx, max_history_turns=None,
                                 prompt_config=None):
    """
    功能：将QReCC对话的某一轮转换为Search-R1数据格式
    输入：
        - conv_turns: 完整对话的所有turn列表
        - turn_idx: 当前turn的索引
        - max_history_turns: 历史窗口大小
        - prompt_config: prompt配置（是否加提示、示例等）
    返回：Search-R1格式的数据字典
    {
        "data_source": "qrecc_plan_a",
        "prompt": [{
            "role": "user",
            "content": prompt_text  # 包含对话历史
        }],
        "ability": "conversational-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": answer
        },
        "extra_info": {
            'conversation_id': conv_id,
            'turn_no': turn_no,
            'original_question': original_question,
            'rewrite': rewrite_query,  # 保留用于分析
            'history_turns': len(history_turns),
            'index': idx,
        }
    }
    """

def process_conversations(conversations_dict, split='train',
                         max_history_turns=None, prompt_config=None):
    """
    功能：处理所有对话，生成训练样本
    输入：
        - conversations_dict: 按对话分组的数据
        - split: 数据集split
        - max_history_turns: 历史窗口
        - prompt_config: prompt配置
    返回：处理后的数据列表

    逻辑：
    1. 遍历每个对话
    2. 对于对话中的每一轮（turn_idx > 0的轮次）：
       - 获取之前的历史（turn 0 到 turn_idx-1）
       - 构建当前turn的训练样本
    3. 第一轮（turn_idx=0）可以选择包含或跳过
    """

def compute_statistics(processed_data, conversations_dict):
    """
    功能：计算数据统计信息
    返回：详细统计字典
    {
        'total_conversations': int,
        'total_samples': int,
        'avg_turns_per_conversation': float,
        'turn_distribution': {...},  # 各轮次的样本数
        'avg_question_length': float,
        'avg_history_length': float,
        'pronoun_questions_ratio': float,  # 包含代词的问题比例
        ...
    }
    """

def save_multiple_versions(conversations_dict, output_dir):
    """
    功能：生成多个版本的数据（不同历史窗口）
    输出：
        - train_full_history.parquet
        - train_window_3.parquet
        - train_window_5.parquet
        等
    """

def main():
    """
    主函数：执行完整的数据处理流程
    1. 下载数据
    2. 按对话分组
    3. 生成多个版本（不同历史窗口）
    4. 保存parquet文件
    5. 生成统计信息
    6. 打印样本预览
    """
```

---

### 2. 增强Reward函数

**文件：** `verl/utils/reward_score/qrecc_multi_reward.py`

**主要功能：**
- 实现多维度reward计算
- 基础EM reward
- Query改写质量reward
- 检索相关性reward
- 支持可配置的reward权重

**核心函数设计：**

```python
def extract_search_query(response_text):
    """
    功能：从模型生成的response中提取search query
    输入：完整response文本
    返回：提取的所有search query列表

    逻辑：
    使用正则提取所有 <search>...</search> 标签中的内容
    """

def compute_query_rewrite_reward(generated_queries, ground_truth_rewrite):
    """
    功能：计算query改写质量reward
    输入：
        - generated_queries: 模型生成的search query列表
        - ground_truth_rewrite: 标准改写query
    返回：0-1之间的分数

    方法：
    1. 取第一个生成的query（通常最重要）
    2. 计算与ground_truth_rewrite的相似度
       - 方法1：ROUGE-L分数
       - 方法2：Token overlap
       - 方法3：BERTScore（可选，需要额外模型）
    3. 返回相似度分数
    """

def check_answer_in_retrieved_docs(retrieved_docs, ground_truth_answer):
    """
    功能：检查检索到的文档是否包含答案
    输入：
        - retrieved_docs: 检索到的文档列表
        - ground_truth_answer: 标准答案
    返回：True/False

    逻辑：
    标准化答案和文档文本，检查是否有overlap
    """

def compute_multi_reward(solution_str, ground_truth_answer,
                        ground_truth_rewrite=None,
                        retrieved_docs=None,
                        reward_weights=None):
    """
    功能：计算多维度综合reward
    输入：
        - solution_str: 模型生成的完整字符串
        - ground_truth_answer: 标准答案
        - ground_truth_rewrite: 标准改写query（可选）
        - retrieved_docs: 检索到的文档（可选）
        - reward_weights: reward权重配置
    返回：最终reward分数（0-1）

    默认权重：
    reward_weights = {
        'em': 0.6,              # 答案EM最重要
        'query_rewrite': 0.2,   # query改写质量
        'retrieval': 0.2        # 检索成功率
    }

    计算逻辑：
    1. 计算EM reward（主要）
    2. 如果提供了ground_truth_rewrite，计算query改写reward
    3. 如果提供了retrieved_docs，计算检索reward
    4. 加权求和

    分阶段策略（推荐）：
    - 阶段1（前5个epoch）：只用EM（权重1.0）
    - 阶段2（5-10个epoch）：EM 0.7 + query 0.3
    - 阶段3（10+个epoch）：EM 0.6 + query 0.2 + retrieval 0.2
    """

class MultiRewardManager:
    """
    多维度Reward管理器

    功能：
    - 管理不同阶段的reward权重
    - 记录各维度reward的统计信息
    - 支持动态调整权重
    """

    def __init__(self, stage='stage1', tokenizer=None):
        """初始化reward管理器，设置当前阶段"""

    def compute_reward(self, data_item):
        """计算单个样本的reward"""

    def update_stage(self, new_stage):
        """更新训练阶段，调整reward权重"""

    def get_statistics(self):
        """返回各维度reward的统计信息"""
```

---

### 3. 训练配置脚本

**文件：** `configs/qrecc/train_qrecc_ppo_plan_a.sh`

**主要功能：**
- 配置方案A的训练参数
- 支持不同历史窗口大小
- 可配置不同prompt版本

**关键配置内容：**

```bash
#!/bin/bash

# GPU设置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 数据路径
export DATA_DIR='data/qrecc_plan_a/processed'
# 选择历史窗口版本
HISTORY_VERSION='window_3'  # 'full_history' / 'window_3' / 'window_5'

# wandb项目配置
WAND_PROJECT='Search-R1-QReCC-Plan-A'

# 模型配置
export BASE_MODEL='Qwen/Qwen3-4B'
# 备选: 'Qwen/Qwen3-4B-Instruct'
# 备选: 'Qwen/Qwen3-7B'
# 备选: 'meta-llama/Llama-3.2-3B'

# 实验名称
export EXPERIMENT_NAME=qrecc-plan-a-ppo-qwen3-4b-hist3-em

# vLLM配置
export VLLM_ATTENTION_BACKEND=XFORMERS

# 启动训练
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    # 数据配置
    data.train_files=$DATA_DIR/train_${HISTORY_VERSION}.parquet \
    data.val_files=$DATA_DIR/dev.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.shuffle_train_dataloader=True \
    # 方案A需要更大的prompt长度（包含对话历史）
    data.max_prompt_length=6144 \        # 增加（相比方案B的3072）
    data.max_response_length=500 \
    data.max_start_length=3072 \         # 增加（用户输入+历史）
    data.max_obs_length=500 \
    \
    # RL算法配置
    algorithm.adv_estimator=gae \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
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
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    \
    # Rollout配置
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
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=1005 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    \
    # 搜索配置
    max_turns=3 \                        # 允许更多搜索轮次
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    \
    2>&1 | tee experiments/qrecc_plan_a/baseline/history_3turn/logs/$EXPERIMENT_NAME.log
```

---

### 4. 评估与分析脚本

**文件：** `scripts/evaluate/evaluate_qrecc_plan_a.py`

**主要功能：**
- 加载模型并在测试集评估
- 计算EM、F1等基础指标
- 分析query改写质量
- 分析对话上下文使用情况
- 生成详细评估报告

**核心函数设计：**

```python
def evaluate_with_conversation_analysis(model, tokenizer, test_data, retriever_url):
    """
    功能：带对话分析的评估
    返回：
    {
        'overall_metrics': {
            'em': float,
            'f1': float,
            'total': int,
            'correct': int
        },
        'by_turn': {  # 按对话轮次分组
            'turn_1': {'em': ..., 'f1': ...},
            'turn_2-3': {...},
            'turn_4+': {...}
        },
        'query_rewrite_analysis': {
            'avg_similarity': float,  # 与ground truth rewrite的平均相似度
            'pronoun_handling': float,  # 代词处理成功率
        },
        'predictions': [...]
    }
    """

def analyze_query_rewrite_quality(predictions, test_data):
    """
    功能：分析query改写质量

    分析维度：
    1. 与ground truth rewrite的相似度（ROUGE-L）
    2. 是否正确处理了代词（he/she/it -> 具体实体）
    3. 是否包含了必要的上下文信息
    4. Query的完整性（是否是独立问题）

    返回：详细分析报告字典
    """

def analyze_context_usage(predictions, test_data):
    """
    功能：分析模型如何使用对话上下文

    分析内容：
    1. 模型是否在<think>中提到了历史信息
    2. 是否正确识别了指代对象
    3. 不同历史长度对性能的影响

    返回：上下文使用分析报告
    """

def identify_error_types(predictions, test_data):
    """
    功能：识别和分类错误类型

    错误类型：
    1. 指代消解失败（没有正确识别代词）
    2. Query改写不完整（缺少上下文信息）
    3. 检索失败（query正确但检索不到相关文档）
    4. 推理错误（有正确信息但答案错误）
    5. 格式错误（缺少必要标签）

    返回：错误类型统计和典型案例
    """

def compare_with_plan_b(plan_a_results, plan_b_results):
    """
    功能：对比方案A和方案B的性能

    对比维度：
    - 整体EM/F1
    - 按轮次对比
    - Query质量对比
    - 检索成功率对比

    返回：对比报告
    """
```

**文件：** `scripts/analysis/analyze_query_rewrite.py`

**主要功能：**
- 深度分析query改写能力
- 可视化query质量分布
- 典型案例分析

---

### 5. Query改写分析脚本

**文件：** `scripts/analysis/compare_plans.py`

**主要功能：**
- 自动化对比方案A和方案B
- 生成对比表格和图表
- 识别方案A的优势和不足

---

## 实现步骤

### 阶段1: 基于方案B的准备（第1-2天）

**前提条件：**
- ✅ 方案B已经实现并训练完成
- ✅ 理解方案B的性能瓶颈
- ✅ 分析方案B的失败案例

**任务清单：**
- [ ] 回顾方案B的实验结果
- [ ] 分析哪些错误是因为缺少对话历史
- [ ] 确定方案A的改进目标

---

### 阶段2: 数据处理（第3-5天）

**任务清单：**
- [ ] 编写`qrecc_search_plan_a.py`
- [ ] 实现对话历史构建逻辑
- [ ] 生成多个版本（不同历史窗口）
- [ ] 验证数据正确性

**关键实现：**

**对话分组：**
```python
def group_by_conversation(dataset):
    conversations = {}
    for item in dataset:
        conv_id = item['Conversation_no']
        if conv_id not in conversations:
            conversations[conv_id] = []
        conversations[conv_id].append(item)

    # 按Turn_no排序
    for conv_id in conversations:
        conversations[conv_id].sort(key=lambda x: x['Turn_no'])

    return conversations
```

**历史格式化：**
```python
def format_conversation_history(history_turns, max_turns=3):
    if max_turns is not None:
        history_turns = history_turns[-max_turns:]  # 滑动窗口

    formatted = []
    for idx, (question, answer) in enumerate(history_turns, 1):
        formatted.append(f"Turn {idx}:")
        formatted.append(f"Q: {question}")
        formatted.append(f"A: {answer}")
        formatted.append("")  # 空行分隔

    return "\n".join(formatted)
```

**数据验证：**
```python
# 打印样本检查
sample = processed_data[100]
print("=== Sample with 3-turn history ===")
print(sample['prompt'][0]['content'])
print(f"\nGround truth answer: {sample['reward_model']['ground_truth']}")
print(f"Ground truth rewrite: {sample['extra_info']['rewrite']}")
```

---

### 阶段3: Prompt优化（第6-7天）

**任务清单：**
- [ ] 设计基础prompt模板
- [ ] 添加指代消解提示
- [ ] 尝试few-shot示例
- [ ] A/B测试不同模板

**Prompt迭代：**

**版本1：基础版**
```
Answer the given question based on the conversation history.
...
Conversation History:
{history}

Current Question: {question}
```

**版本2：增加提示**
```
... (基础版)

Important: If the question contains pronouns (he, she, it, etc.),
identify what they refer to based on the history, and use full names in search.

Conversation History:
...
```

**版本3：Few-shot**
```
Here is an example of answering based on conversation history:
...

Now your turn:
Conversation History:
...
```

---

### 阶段4: 基础训练（第8-12天）

**任务清单：**
- [ ] 使用基础prompt + EM reward训练
- [ ] 历史窗口=3轮
- [ ] 监控训练过程
- [ ] 初步评估

**训练命令：**
```bash
bash configs/qrecc/train_qrecc_ppo_plan_a.sh
```

**监控重点：**
- Reward增长趋势
- 验证集EM（目标>35%，接近方案B）
- 是否学会使用历史信息
- 是否出现过拟合

---

### 阶段5: 增强Reward训练（第13-16天）

**任务清单：**
- [ ] 实现多维度reward
- [ ] 重新训练（或从checkpoint继续）
- [ ] 对比单reward vs 多reward

**Reward配置：**
```python
# 阶段1: 只EM (epoch 1-5)
reward_weights = {'em': 1.0}

# 阶段2: EM + query (epoch 6-10)
reward_weights = {'em': 0.7, 'query_rewrite': 0.3}

# 阶段3: 完整reward (epoch 11-15)
reward_weights = {'em': 0.6, 'query_rewrite': 0.2, 'retrieval': 0.2}
```

---

### 阶段6: 消融实验（第17-20天）

**实验设计：**

**实验1：不同历史长度**
- 1轮 vs 3轮 vs 5轮 vs 全部
- 分析最优窗口大小

**实验2：不同prompt模板**
- 基础 vs 提示 vs few-shot
- 确定最佳prompt设计

**实验3：不同reward配置**
- 只EM vs EM+query vs 多reward
- 验证reward设计的有效性

**实验4：与方案B对比**
- 整体性能
- 按轮次分组对比
- Query质量对比

---

### 阶段7: 深度分析（第21-24天）

**任务清单：**
- [ ] Query改写质量分析
- [ ] 对话上下文使用分析
- [ ] 错误案例深度分析
- [ ] 与方案B详细对比

**分析脚本：**
```bash
# Query改写分析
python scripts/analysis/analyze_query_rewrite.py \
  --predictions experiments/qrecc_plan_a/baseline/results.json \
  --output experiments/qrecc_plan_a/analysis/query_rewrite_quality.md

# 方案对比
python scripts/analysis/compare_plans.py \
  --plan_a experiments/qrecc_plan_a/baseline/results.json \
  --plan_b experiments/qrecc_plan_b/baseline_4b/results.json \
  --output experiments/qrecc_plan_a/vs_plan_b/comparison.md
```

---

### 阶段8: 迭代优化（第25-28天）

**基于分析结果优化：**

**如果指代消解差：**
- 增强prompt提示
- 添加few-shot示例
- 增加query改写reward权重

**如果检索质量差：**
- 调整topk
- 添加reranker
- 优化检索query

**如果推理能力差：**
- 增加<think>的reward
- 调整温度参数
- 尝试更大模型

---

## 关键技术点

### 1. 对话历史管理

**滑动窗口策略：**
- 目的：控制prompt长度
- 推荐：保留最近3-5轮
- 实现：`history_turns[-max_turns:]`

**Token长度控制：**
```python
def truncate_history_by_tokens(history, max_tokens, tokenizer):
    """根据token数截断历史"""
    # 从最近的轮次开始逐轮添加，直到达到token限制
    pass
```

### 2. Prompt工程

**关键要素：**
1. 清晰的任务说明
2. 对话历史的格式化表示
3. 指代消解的明确指导
4. Few-shot示例（可选）

**常见问题：**
- 历史太长 → 滑动窗口
- 模型不用历史 → 增强提示
- 指代处理差 → 添加示例

### 3. 多维度Reward设计

**分阶段策略（推荐）：**
- 早期：只用EM，确保基础能力
- 中期：加入query reward，引导改写
- 后期：加入检索reward，优化端到端

**权重调优：**
- EM始终最重要（>0.5）
- 辅助reward不宜过大（<0.3）
- 根据验证集调整

### 4. 评估方法

**多维度评估：**
- 答案质量：EM、F1
- Query质量：与ground truth相似度
- 上下文使用：是否引用历史
- 按轮次分组：不同难度的性能

---

## 预期结果

### 性能目标

**保守估计：**
- Overall EM: 35-40%
- Overall F1: 42-48%
- Query改写相似度: 0.5-0.6

**理想目标：**
- Overall EM: 42-47%
- Overall F1: 50-55%
- Query改写相似度: 0.65-0.75
- 超越方案B（证明对话建模的价值）

### 按轮次性能

**预期分布：**
- Turn 1 (第一轮，无历史): EM ~50%
- Turn 2-3 (有历史，指代适中): EM ~40%
- Turn 4+ (历史长，复杂指代): EM ~30%

### 与方案B对比

**预期差异：**
- 整体EM: 方案A可能略低（-2~5%）
- 代词问题: 方案A显著更好
- 复杂对话: 方案A更强
- 实用价值: 方案A更高

---

## 常见挑战与解决

### 挑战1: 训练不稳定

**原因：**
- 对话历史增加了输入复杂度
- Reward信号更稀疏
- 学习难度更大

**解决方案：**
- 降低学习率（5e-7）
- 增加warmup
- 从方案B checkpoint热启动
- 使用GRPO（更稳定）

### 挑战2: 模型不使用历史

**表现：**
- 生成的query与方案B相似
- <think>中没有提到历史信息
- 性能与方案B无差异

**解决方案：**
- 增强prompt提示
- 添加使用历史的reward
- Few-shot示例展示如何用历史
- 检查数据处理是否正确

### 挑战3: Query改写质量差

**表现：**
- Query仍然包含代词
- 缺少必要的上下文信息
- 与ground truth差异大

**解决方案：**
- 增加query改写reward权重
- 显式要求生成<rewrite>标签
- 添加query改写的示例
- 分析失败案例，针对性优化prompt

### 挑战4: 性能不如方案B

**可能原因：**
- 学习难度更大，需要更多训练
- Reward设计不合理
- Prompt设计有问题
- 历史信息引入噪音

**诊断步骤：**
1. 对比按轮次的性能（第一轮应该接近方案B）
2. 分析是否正确使用了历史
3. 检查是否过拟合
4. 尝试简化任务（先2轮历史，再增加）

---

## 成功标准

### 必须达到（Baseline）

- [ ] 训练成功收敛，无严重过拟合
- [ ] 整体EM > 35%
- [ ] 能够生成包含对话理解的<think>
- [ ] 能够正确处理至少50%的代词问题

### 期望达到（Target）

- [ ] 整体EM > 40%，接近或超过方案B
- [ ] Query改写相似度 > 0.6
- [ ] 在代词问题上显著优于方案B
- [ ] 生成的query独立完整，可直接用于检索

### 理想达到（Stretch）

- [ ] 整体EM > 45%，明显超越方案B
- [ ] Query改写相似度 > 0.7
- [ ] 能够处理复杂的多轮指代链
- [ ] 发表论文或开源贡献

---

## 总结

### 方案A的价值

**学术价值：**
- 探索端到端对话式检索问答
- 研究RL在对话建模中的作用
- 分析query改写的学习机制

**工程价值：**
- 更接近真实应用场景
- 无需外部改写模块
- 可扩展到其他对话任务

**对比方案B的优势：**
- 真实应用价值更高
- 学习对话理解能力
- 探索模型学习边界

### 风险与挑战

**高风险：**
- 训练难度大，可能不收敛
- 性能可能不如方案B
- 需要更多调优时间

**中风险：**
- 模型可能不使用历史
- Query改写质量可能不高
- 需要复杂的reward设计

**低风险：**
- 基于方案B经验，流程已验证
- 可以降级到方案B
- 有明确的诊断和优化路径

### 建议实施路径

**推荐策略：完整实施（4-6周）**
1. Week 1-2: 数据处理 + 基础训练
2. Week 3: 消融实验（历史长度、prompt）
3. Week 4: 增强reward训练
4. Week 5: 深度分析与优化
5. Week 6: 文档总结与对比

**快速验证（2-3周）：**
1. Week 1: 数据处理 + 基础训练（只3轮历史）
2. Week 2: 简单评估，与方案B对比
3. Week 3: 分析报告

**论文级（8-12周）：**
1. 完成上述所有内容
2. 额外实验：多任务学习、模型蒸馏等
3. 详细分析与论文撰写

---

方案A是更有挑战的方向，但成功后价值更大。建议在方案B验证可行性后再实施方案A。