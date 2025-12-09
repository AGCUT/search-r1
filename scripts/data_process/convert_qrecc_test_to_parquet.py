#!/usr/bin/env python3
"""
Convert QReCC test.json to test.parquet format for evaluation.

This script converts the raw QReCC test.json format to the parquet format
expected by the veRL evaluation framework.

Usage:
    python scripts/data_process/convert_qrecc_test_to_parquet.py \
        --input_file data/qrecc_raw/qrecc_test.json \
        --output_file data/qrecc_raw/test.parquet \
        --template_type base
"""

import json
import datasets
import argparse
from pathlib import Path
from typing import Dict, Any, List


def make_prefix(question: str, template_type: str = 'base') -> str:
    """
    Create the prompt prefix for the model.

    Args:
        question: The rewritten standalone question
        template_type: Template style ('base' for base models, 'instruct' for instruction-tuned)

    Returns:
        Formatted prompt string
    """
    if template_type == 'base':
        # Same prompt structure as used in training
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
    elif template_type == 'instruct':
        # For instruction-tuned models like Qwen2.5-3B-Instruct
        prefix = f"""You are a helpful assistant that can answer questions by reasoning and searching for information.

Instructions:
1. Use <think>reasoning</think> to conduct your reasoning process
2. Use <search>query</search> to search for information when needed
3. Use <answer>answer</answer> to provide the final answer

Question: {question}\n"""
    else:
        raise NotImplementedError(f"Template type '{template_type}' not implemented")

    return prefix


def process_qrecc_example(example: Dict[str, Any], idx: int, template_type: str) -> Dict[str, Any]:
    """
    Process a single QReCC example into the target format.

    Args:
        example: Raw QReCC example with fields like Question, Rewrite, Answer, etc.
        idx: Example index
        template_type: Template style ('base' or 'instruct')

    Returns:
        Processed example in Search-R1 format
    """
    # Extract the rewritten query (standalone query)
    rewritten_query = example.get('Rewrite', example.get('Question', '')).strip()

    # Clean up the query - add question mark if missing
    if rewritten_query and rewritten_query[-1] not in ['?', '.', '!']:
        rewritten_query += '?'

    # Extract answer
    answer = example.get('Answer', '')

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
        "data_source": "qrecc",
        "prompt": [{
            "role": "user",
            "content": prompt_text,
        }],
        "ability": "conversational-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": solution
        },
        "extra_info": {
            'split': 'test',
            'index': idx,
            'conversation_id': example.get('Conversation_no', -1),
            'turn_id': example.get('Turn_no', -1),
            'original_question': example.get('Question', ''),
            'rewritten_query': rewritten_query,
            'answer_url': example.get('Answer_URL', ''),
            'conversation_source': example.get('Conversation_source', ''),
        }
    }

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Convert QReCC test.json to test.parquet format"
    )
    parser.add_argument(
        '--input_file',
        type=str,
        default='data/qrecc_raw/qrecc_test.json',
        help='Input JSON file path'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='data/qrecc_raw/test.parquet',
        help='Output parquet file path'
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
    print("QReCC Test Data Conversion - JSON to Parquet")
    print("=" * 70)
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Template type: {args.template_type}")
    print("=" * 70)

    # Check input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    # Load raw JSON data
    print(f"\nLoading data from {args.input_file}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_examples = json.load(f)

    print(f"✓ Loaded {len(raw_examples)} examples")

    # Process examples
    print("\nProcessing examples...")
    processed_examples = []
    for idx, example in enumerate(raw_examples):
        processed = process_qrecc_example(
            example=example,
            idx=idx,
            template_type=args.template_type
        )
        processed_examples.append(processed)

        # Print progress every 1000 examples
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(raw_examples)} examples...")

    print(f"✓ Processed {len(processed_examples)} examples")

    # Convert to HuggingFace dataset
    print("\nConverting to HuggingFace dataset format...")
    dataset = datasets.Dataset.from_list(processed_examples)

    # Create output directory if needed
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as parquet
    print(f"\nSaving to {args.output_file}...")
    dataset.to_parquet(args.output_file)
    print(f"✓ Saved {len(dataset)} examples to {args.output_file}")

    # Print sample
    if len(dataset) > 0:
        print("\n" + "=" * 70)
        print("Sample from converted data:")
        print("-" * 70)
        sample = dataset[0]
        print(f"Data source: {sample['data_source']}")
        print(f"Ability: {sample['ability']}")
        print(f"\nPrompt (first 300 chars):")
        print(sample['prompt'][0]['content'][:300] + "...")
        print(f"\nGround truth:")
        print(f"  {sample['reward_model']['ground_truth']['target']}")
        print(f"\nExtra info:")
        print(f"  Conversation ID: {sample['extra_info']['conversation_id']}")
        print(f"  Turn ID: {sample['extra_info']['turn_id']}")
        print(f"  Original question: {sample['extra_info']['original_question']}")
        print(f"  Rewritten query: {sample['extra_info']['rewritten_query']}")
        print("-" * 70)

    print("\n" + "=" * 70)
    print("Conversion complete!")
    print("=" * 70)
    print("Next steps:")
    print("1. Verify the parquet file:")
    print(f"   python -c \"import pandas as pd; df = pd.read_parquet('{args.output_file}'); print(df.head())\"")
    print("\n2. Run evaluation with the converted data:")
    print("   bash configs/qrecc/compare_base_vs_trained_f1.sh")
    print("=" * 70)


if __name__ == '__main__':
    main()