# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the QReCC dataset for Plan B (using ground-truth rewritten queries).

Plan B Approach:
- Use the ground-truth rewritten query directly (no conversation history needed)
- Simplified baseline to validate the training pipeline
- Model only needs to learn reasoning and search with standalone queries

Usage:
    python scripts/data_process/qrecc_search_plan_b.py \
        --input_dir data/qrecc_raw \
        --output_dir data/qrecc_plan_b \
        --template_type base
"""

import re
import os
import json
import datasets
from pathlib import Path
from typing import Dict, Any
import argparse

from verl.utils.hdfs_io import copy, makedirs


def load_qrecc_from_jsonl(input_dir: str, split: str):
    """
    Load QReCC dataset from JSONL files.

    Args:
        input_dir: Directory containing the JSONL files
        split: 'train' or 'test'

    Returns:
        List of examples
    """
    jsonl_path = Path(input_dir) / f"{split}.jsonl"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    examples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    return examples


def make_prefix(question: str, template_type: str = 'base') -> str:
    """
    Create the prompt prefix for the model.

    Args:
        question: The rewritten standalone question
        template_type: Template style ('base' for base models)

    Returns:
        Formatted prompt string
    """
    if template_type == 'base':
        # Same prompt structure as NQ, but using rewritten question
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
    elif template_type == 'instruct':
        # For instruction-tuned models like Qwen3-4B-Instruct
        prefix = f"""You are a helpful assistant that can answer questions by reasoning and searching for information.

Instructions:
1. Use <think>reasoning</think> to conduct your reasoning process
2. Use <search>query</search> to search for information when needed
3. Use <answer>answer</answer> to provide the final answer

Question: {question}\n"""
    else:
        raise NotImplementedError(f"Template type '{template_type}' not implemented")

    return prefix


def process_qrecc_example(example: Dict[str, Any], idx: int, split: str, template_type: str) -> Dict[str, Any]:
    """
    Process a single QReCC example for Plan B.

    Plan B uses the rewritten query directly, ignoring conversation history.

    Args:
        example: Raw QReCC example
        idx: Example index
        split: 'train' or 'test'
        template_type: Template style

    Returns:
        Processed example in Search-R1 format
    """
    # Extract the rewritten query (standalone query)
    # QReCC dataset structure may vary, adjust field names as needed
    if 'Rewrite' in example:
        rewritten_query = example['Rewrite']
    elif 'rewrite' in example:
        rewritten_query = example['rewrite']
    elif 'query' in example:
        rewritten_query = example['query']
    else:
        # Fallback: if no rewrite field, use the question itself
        rewritten_query = example.get('Question', example.get('question', ''))

    # Clean up the query
    rewritten_query = rewritten_query.strip()
    if rewritten_query and rewritten_query[-1] not in ['?', '.', '!']:
        rewritten_query += '?'

    # Extract answer
    if 'Answer' in example:
        answer = example['Answer']
    elif 'answer' in example:
        answer = example['answer']
    elif 'answers' in example:
        answer = example['answers']
    else:
        answer = []

    # Ensure answer is a list
    if isinstance(answer, str):
        answer = [answer]
    elif not isinstance(answer, list):
        answer = [str(answer)]

    # Create the prompt
    prompt_text = make_prefix(rewritten_query, template_type=template_type)

    # Create solution (ground truth)
    solution = {
        "target": answer,
    }

    # Format in Search-R1 structure
    data = {
        "data_source": "qrecc_plan_b",
        "prompt": [{
            "role": "user",
            "content": prompt_text,
        }],
        "ability": "conversational-reasoning",  # Mark as conversational QA
        "reward_model": {
            "style": "rule",
            "ground_truth": solution
        },
        "extra_info": {
            'split': split,
            'index': idx,
            'conversation_id': example.get('Conversation_no', example.get('conversation_id', -1)),
            'turn_id': example.get('Turn_no', example.get('turn_id', -1)),
            'original_question': example.get('Question', example.get('question', '')),
            'rewritten_query': rewritten_query,
        }
    }

    return data


def main():
    parser = argparse.ArgumentParser(description="Process QReCC dataset for Plan B")
    parser.add_argument(
        '--input_dir',
        type=str,
        default='data/qrecc_raw',
        help='Directory containing raw QReCC JSONL files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/qrecc_plan_b',
        help='Directory to save processed parquet files'
    )
    parser.add_argument(
        '--hdfs_dir',
        type=str,
        default=None,
        help='Optional HDFS directory to copy processed data'
    )
    parser.add_argument(
        '--template_type',
        type=str,
        default='base',
        choices=['base', 'instruct'],
        help='Template type: base for base models, instruct for instruction-tuned models'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("QReCC Dataset Processing - Plan B (Ground-truth Rewrite)")
    print("=" * 70)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Template type: {args.template_type}")
    print("=" * 70)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process train and test splits
    for split in ['train', 'test']:
        print(f"\nProcessing {split} split...")

        try:
            # Load raw data
            examples = load_qrecc_from_jsonl(args.input_dir, split)
            print(f"Loaded {len(examples)} examples from {split} split")

            # Process examples
            processed_examples = []
            for idx, example in enumerate(examples):
                processed = process_qrecc_example(
                    example=example,
                    idx=idx,
                    split=split,
                    template_type=args.template_type
                )
                processed_examples.append(processed)

            # Convert to HuggingFace dataset
            dataset = datasets.Dataset.from_list(processed_examples)

            # Save as parquet
            output_file = os.path.join(args.output_dir, f'{split}.parquet')
            dataset.to_parquet(output_file)
            print(f"✓ Saved {len(dataset)} examples to {output_file}")

            # Print sample
            if len(dataset) > 0:
                print(f"\nSample from {split} split:")
                print("-" * 70)
                sample = dataset[0]
                print(f"Data source: {sample['data_source']}")
                print(f"Prompt: {sample['prompt'][0]['content'][:200]}...")
                print(f"Ground truth: {sample['reward_model']['ground_truth']['target']}")
                print("-" * 70)

        except FileNotFoundError as e:
            print(f"✗ {e}")
            print(f"  Skipping {split} split")
            continue

    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        print(f"\nCopying to HDFS: {args.hdfs_dir}")
        makedirs(args.hdfs_dir)
        copy(src=args.output_dir, dst=args.hdfs_dir)
        print("✓ Copy to HDFS complete")

    print("\n" + "=" * 70)
    print("Processing complete!")
    print("=" * 70)
    print("Next steps:")
    print(f"1. Check processed data: ls {args.output_dir}")
    print("2. Start training: bash configs/qrecc/train_qrecc_ppo_plan_b.sh")
    print("=" * 70)


if __name__ == '__main__':
    main()