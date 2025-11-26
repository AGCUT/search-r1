# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Search-R1 is a reinforcement learning framework for training reasoning-and-searching interleaved LLMs. Built upon veRL (Volcano Engine Reinforcement Learning), it enables language models to learn to reason and make tool calls (e.g., to search engines) through RL training.

Key capabilities:
- Trains base LLMs (3B-70B+) to develop reasoning and search engine calling abilities via RL
- Supports multiple RL algorithms (PPO, GRPO, reinforce)
- Supports multiple LLM architectures (Llama3, Qwen2.5, etc.)
- Supports various search engines (local sparse/dense retrievers, online search APIs)

## Architecture

### Two-Component System

The project is organized into two main components that run in **separate processes**:

1. **RL Training Pipeline** (`verl/` module)
   - Main training orchestration via Ray distributed framework
   - Actor-Critic PPO/GRPO implementation
   - Uses vLLM for efficient inference during rollouts
   - Training backends: PyTorch FSDP or Megatron-LM

2. **Search Engine Server** (`search_r1/` module)
   - Runs as a separate FastAPI server
   - LLM calls search API at `http://127.0.0.1:8000/retrieve` during generation
   - Must be launched before starting RL training

### Key Modules

**verl/** - Core RL training framework (based on veRL/HybridFlow)
- `verl/trainer/main_ppo.py` - Main PPO training entry point with RewardManager
- `verl/trainer/ppo/ray_trainer.py` - Ray-based distributed PPO trainer
- `verl/workers/` - Actor, Critic, Rollout, and Reference model workers
- `verl/models/` - Model implementations for transformers and Megatron-LM
- `verl/third_party/vllm/` - vLLM integration (supports v0.3.1, v0.4.2, v0.5.4, v0.6.3)

**search_r1/** - Search engine integration
- `search_r1/search/retrieval_server.py` - Local retriever server (BM25, dense)
- `search_r1/search/google_search_server.py` - Google Search API server
- `search_r1/search/serp_search_server.py` - SerpAPI server
- `search_r1/llm_agent/generation.py` - LLM generation utilities with search integration

**scripts/** - Data processing and training scripts
- `scripts/data_process/` - Dataset processing (NQ, HotpotQA, etc.)
- `scripts/nq_hotpotqa/` - Training scripts for different experiment versions

## Environment Setup

### Two Separate Environments Required

**1. RL Training Environment (`searchr1`)**
```bash
conda create -n searchr1 python=3.9
conda activate searchr1
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install vllm==0.6.3  # or 0.5.4, 0.4.2, 0.3.1
pip install -e .
pip3 install flash-attn --no-build-isolation
pip install wandb
```

**2. Retriever Environment (`retriever`)** - Optional, only if using local retrievers
```bash
conda create -n retriever python=3.10
conda activate retriever
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini
conda install -c pytorch -c nvidia faiss-gpu=1.8.0  # For GPU-accelerated retrieval
pip install uvicorn fastapi
```

## Common Development Commands

### Data Preparation

Process NQ dataset:
```bash
python scripts/data_process/nq_search.py
```

Download corpus and indexing:
```bash
save_path=/path/to/save
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

### Launching Search Servers

**Must be launched before RL training.** Choose one based on your use case:

Local dense retriever (GPU):
```bash
conda activate retriever
bash retrieval_launch.sh
# Or directly:
# python search_r1/search/retrieval_server.py --index_path <index> --corpus_path <corpus> --topk 3 --retriever_name e5 --retriever_model intfloat/e5-base-v2 --faiss_gpu
```

Local BM25 retriever (CPU):
```bash
python search_r1/search/retrieval_server.py --index_path <index> --corpus_path <corpus> --topk 3 --retriever_name bm25
```

Online SerpAPI server:
```bash
python search_r1/search/serp_search_server.py --search_url https://serpapi.com/search --topk 3 --serp_api_key <your_key>
```

### Training

PPO training (single node):
```bash
conda activate searchr1
bash train_ppo.sh
```

GRPO training (single node):
```bash
conda activate searchr1
bash train_grpo.sh
```

Multi-node training requires Ray cluster setup. See docs/multinode.md for details.

### Inference

Run inference with trained model:
```bash
conda activate searchr1
python infer.py
```

Edit line 7 in `infer.py` to change the question.

## Training Configuration

Training scripts use Hydra for configuration. Key parameters in `train_ppo.sh` and `train_grpo.sh`:

**Data Configuration:**
- `data.train_files` / `data.val_files` - Training/validation data paths
- `data.max_prompt_length` - Max prompt tokens (default: 4096)
- `data.max_response_length` - Max response tokens per turn (default: 500)
- `data.max_obs_length` - Max observation/search result tokens (default: 500)
- `max_turns` - Max search turns (default: 2)

**Model Configuration:**
- `actor_rollout_ref.model.path` - Base model path (e.g., `meta-llama/Llama-3.2-3B`)
- `actor_rollout_ref.actor.optim.lr` - Actor learning rate
- `critic.optim.lr` - Critic learning rate
- `actor_rollout_ref.rollout.name` - Rollout backend (default: `vllm`)

**Retriever Configuration:**
- `retriever.url` - Search API endpoint (default: `http://127.0.0.1:8000/retrieve`)
- `retriever.topk` - Number of retrieved documents (default: 3)

**RL Algorithm:**
- `algorithm.adv_estimator` - Set to `gae` for PPO or `grpo` for GRPO
- For GRPO: set `actor_rollout_ref.rollout.n_agent=5` for multiple rollouts

**Resource Configuration:**
- `trainer.n_gpus_per_node` - GPUs per node (default: 8)
- `trainer.nnodes` - Number of nodes (default: 1)
- `actor_rollout_ref.rollout.gpu_memory_utilization` - vLLM GPU memory usage (default: 0.6)

## Data Format Requirements

### QA Dataset Format

Each sample must be a dictionary in parquet format:
```python
{
    "data_source": "nq",  # Dataset name (nq, triviaqa, hotpotqa, etc.)
    "prompt": [{
        "role": "user",
        "content": "question text"
    }],
    "ability": "fact-reasoning",
    "reward_model": {
        "style": "rule",
        "ground_truth": "answer text"
    },
    "extra_info": {
        'split': 'train',
        'index': 0
    }
}
```

See `scripts/data_process/nq_search.py` for concrete examples.

### Corpus Format

JSONL file with one passage per line:
```json
{"id": "0", "contents": "\"Title\"\nPassage text"}
{"id": "1", "contents": "\"Another Title\"\nAnother passage"}
```

The "contents" field should be formatted as: `"` + title + `"\n` + text

See `example/corpus.jsonl` for examples.

## Model Generation Protocol

Trained models use special XML-like tags for reasoning and search:

- `<think>...</think>` - Internal reasoning
- `<search>query</search>` - Search engine call trigger
- `<information>...</information>` - Search results returned by system
- `<answer>...</answer>` - Final answer

Example flow:
```
<think>I need to find information about X</think>
<search>What is X?</search>
<information>Doc 1: X is... Doc 2: X means...</information>
<think>Based on the results, the answer is Y</think>
<answer>Y</answer>
```

The inference loop in `infer.py` demonstrates this pattern:
1. Generate until `</search>` token
2. Extract query, call retrieval API
3. Inject `<information>` with search results
4. Continue generation
5. Repeat until `</answer>` or EOS

## Multi-Node Training

For models 30B+, use Ray cluster:

1. **Head node**: Start Ray cluster
   ```bash
   ray start --head --dashboard-host=0.0.0.0
   ```

2. **Worker nodes**: Connect to cluster
   ```bash
   ray start --address=<head_gcs_address>
   ```

3. **All nodes**: Launch retrieval server
   ```bash
   bash retrieval_launch.sh
   ```

4. **Head node only**: Submit job
   ```bash
   ray job submit --address=<dashboard>:8265 --runtime-env=verl/trainer/runtime_env.yaml -- python3 -m verl.trainer.main_ppo <config>
   ```

See `docs/multinode.md` and `example/multinode/` for detailed examples.

## Search Engine Selection Guide

**Local Sparse Retriever (BM25)**
- Use when: No GPU available, or no good dense retrievers for domain
- Pros: CPU-only, fast, no neural network needed
- Cons: May underperform dense retrievers

**Local Dense Retriever - Flat Indexing**
- Use when: Sufficient GPUs available, need highest accuracy
- Pros: Exact embedding match, most accurate
- Cons: Requires GPU, slower than ANN
- Enable with: `--faiss_gpu` flag

**Local Dense Retriever - ANN Indexing (HNSW64)**
- Use when: CPU-only, need better accuracy than BM25
- Pros: Fast on CPU, better than BM25
- Cons: Approximate results, less accurate than flat indexing

**Online Search Engine (SerpAPI/Google)**
- Use when: Training general search agent, sufficient budget
- Pros: Real-time web data, broad coverage
- Cons: API costs, rate limits
- Note: SerpAPI recommended over Google API (no 10k/month quota limit)

## Important Notes

- **Search server must be running before training starts** - Training will fail if retrieval API is unreachable
- **vLLM version compatibility** - Code supports vLLM 0.3.1, 0.4.2, 0.5.4, 0.6.3. For Qwen2.5-7B, set `VLLM_ATTENTION_BACKEND=XFORMERS` to avoid flash_attn issues
- **Reward computation** - RewardManager in `verl/trainer/main_ppo.py` computes rewards based on exact match (EM) with ground truth. Add custom reward functions in `verl/utils/reward_score/`
- **Checkpointing** - Models saved to `verl_checkpoints/<experiment_name>/` at intervals specified by `trainer.save_freq`
- **Experiment tracking** - Set `trainer.logger=['wandb']` and configure `trainer.project_name` and `trainer.experiment_name`
- **Multi-node retriever setup** - Launch identical retrieval server on ALL nodes (head + workers) for stable training
- **Max prompt length calculation** - `max_prompt_length = max_start_length + max_response_length * (max_turns - 1) + max_obs_length * max_turns`

## File References

Training configurations: `verl/trainer/config/*.yaml`
Reward scoring functions: `verl/utils/reward_score/*.py`
Search engine examples: `example/retriever/*.sh`
Dataset processing examples: `scripts/data_process/*.py`