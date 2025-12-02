"""
Create a mini dataset for quick testing and debugging.

This script creates a small subset from QReCC for fast iteration:
- 1000 training samples
- 200 test samples

Usage:
    python scripts/data_process/create_mini_dataset.py
"""

import os
import json
import random
import datasets
from pathlib import Path

def make_prefix(question: str) -> str:
    """Create the prompt prefix."""
    return f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""


def process_qrecc_example(example: dict, idx: int, split: str) -> dict:
    """Process a single QReCC example."""
    # Get rewritten query
    rewritten_query = example.get('Rewrite', example.get('Question', ''))
    rewritten_query = rewritten_query.strip()
    if rewritten_query and rewritten_query[-1] not in ['?', '.', '!']:
        rewritten_query += '?'

    # Get answer
    answer = example.get('Answer', '')
    if isinstance(answer, str):
        answer = [answer]

    return {
        "data_source": "mini_qrecc",
        "prompt": [{
            "role": "user",
            "content": make_prefix(rewritten_query),
        }],
        "ability": "fact-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": {"target": answer}
        },
        "extra_info": {
            "split": split,
            "index": idx,
        }
    }


def main():
    # Configuration
    input_dir = Path("data/qrecc_raw")
    output_dir = Path("data/mini_qrecc")
    train_size = 1000
    test_size = 200

    print("=" * 60)
    print("Creating Mini Dataset for Quick Testing")
    print("=" * 60)
    print(f"Train size: {train_size}")
    print(f"Test size: {test_size}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load QReCC data
    print("\nLoading QReCC data...")

    with open(input_dir / "qrecc_train.json", 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    print(f"  Loaded {len(train_data)} training examples")

    with open(input_dir / "qrecc_test.json", 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"  Loaded {len(test_data)} test examples")

    # Random sample
    random.seed(42)
    train_sample = random.sample(train_data, min(train_size, len(train_data)))
    test_sample = random.sample(test_data, min(test_size, len(test_data)))

    print(f"\nSampled {len(train_sample)} train, {len(test_sample)} test")

    # Process samples
    print("\nProcessing samples...")

    train_processed = []
    for idx, example in enumerate(train_sample):
        processed = process_qrecc_example(example, idx, 'train')
        train_processed.append(processed)

    test_processed = []
    for idx, example in enumerate(test_sample):
        processed = process_qrecc_example(example, idx, 'test')
        test_processed.append(processed)

    # Convert to HuggingFace datasets and save
    print("\nSaving to parquet...")

    train_dataset = datasets.Dataset.from_list(train_processed)
    test_dataset = datasets.Dataset.from_list(test_processed)

    train_dataset.to_parquet(output_dir / "train.parquet")
    test_dataset.to_parquet(output_dir / "test.parquet")

    print(f"  Saved: {output_dir / 'train.parquet'}")
    print(f"  Saved: {output_dir / 'test.parquet'}")

    # Print sample
    print("\n" + "-" * 60)
    print("Sample from training set:")
    print("-" * 60)
    sample = train_processed[0]
    print(f"Data source: {sample['data_source']}")
    print(f"Prompt: {sample['prompt'][0]['content'][:150]}...")
    print(f"Answer: {sample['reward_model']['ground_truth']['target']}")
    print("-" * 60)

    print("\n" + "=" * 60)
    print("Done! Mini dataset created.")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Start retrieval server:")
    print("   bash launch_e5_retrieval_single_gpu.sh")
    print("2. Run training with mini dataset:")
    print("   bash configs/mini/train_mini_ppo.sh")
    print("=" * 60)


if __name__ == '__main__':
    main()
