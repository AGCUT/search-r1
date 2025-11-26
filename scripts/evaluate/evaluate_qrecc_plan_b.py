#!/usr/bin/env python3
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
Evaluation script for QReCC Plan B trained models.

This script evaluates a trained model on the QReCC test set and computes:
- Exact Match (EM) score
- Average number of search turns
- Response format correctness

Usage:
    python scripts/evaluate/evaluate_qrecc_plan_b.py \
        --checkpoint verl_checkpoints/qrecc-plan-b-ppo-qwen3-4b-em/checkpoint_1000 \
        --test_file data/qrecc_plan_b/test.parquet \
        --output_file results/qrecc_plan_b_eval.json
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import torch
import datasets
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

from verl.utils.reward_score import qrecc_em


def load_model_and_tokenizer(checkpoint_path: str):
    """
    Load the trained model and tokenizer from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint directory

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model from {checkpoint_path}...")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    print(f"✓ Model loaded successfully")
    return model, tokenizer


def extract_search_count(response: str) -> int:
    """
    Count the number of search operations in the response.

    Args:
        response: The model's response string

    Returns:
        Number of <search> tags found
    """
    search_pattern = r'<search>.*?</search>'
    matches = re.findall(search_pattern, response, re.DOTALL)
    return len(matches)


def extract_answer(response: str) -> str:
    """
    Extract the final answer from the response.

    Args:
        response: The model's response string

    Returns:
        Extracted answer or empty string if not found
    """
    answer = qrecc_em.extract_solution(response)
    return answer if answer else ""


def evaluate_single(
    model,
    tokenizer,
    prompt: str,
    ground_truth: List[str],
    max_new_tokens: int = 500
) -> Dict[str, Any]:
    """
    Evaluate a single example.

    Args:
        model: The model
        tokenizer: The tokenizer
        prompt: Input prompt
        ground_truth: List of acceptable answers
        max_new_tokens: Maximum tokens to generate

    Returns:
        Dictionary with evaluation results
    """
    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding for evaluation
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the prompt from response
    if prompt in response:
        response = response[len(prompt):].strip()

    # Extract answer
    predicted_answer = extract_answer(response)

    # Compute EM score
    em_score = qrecc_em.em_check(predicted_answer, ground_truth)

    # Count searches
    num_searches = extract_search_count(response)

    # Check format correctness (has <answer> tag)
    has_answer_tag = predicted_answer != ""

    result = {
        "response": response,
        "predicted_answer": predicted_answer,
        "ground_truth": ground_truth,
        "em_score": em_score,
        "num_searches": num_searches,
        "has_answer_tag": has_answer_tag,
    }

    return result


def evaluate_dataset(
    model,
    tokenizer,
    test_data: datasets.Dataset,
    max_examples: int = None,
    output_file: str = None
) -> Dict[str, float]:
    """
    Evaluate the model on the full test dataset.

    Args:
        model: The model
        tokenizer: The tokenizer
        test_data: Test dataset
        max_examples: Maximum number of examples to evaluate (None = all)
        output_file: Optional file to save detailed results

    Returns:
        Dictionary with aggregate metrics
    """
    if max_examples:
        test_data = test_data.select(range(min(max_examples, len(test_data))))

    print(f"\nEvaluating on {len(test_data)} examples...")

    results = []
    total_em = 0
    total_searches = 0
    total_with_answer_tag = 0

    for idx, example in enumerate(tqdm(test_data, desc="Evaluating")):
        # Extract prompt
        prompt_text = example['prompt'][0]['content']

        # Extract ground truth
        ground_truth = example['reward_model']['ground_truth']['target']

        # Evaluate
        result = evaluate_single(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_text,
            ground_truth=ground_truth,
            max_new_tokens=500
        )

        result['example_id'] = idx
        results.append(result)

        # Aggregate metrics
        total_em += result['em_score']
        total_searches += result['num_searches']
        total_with_answer_tag += int(result['has_answer_tag'])

    # Compute aggregate metrics
    num_examples = len(results)
    metrics = {
        "exact_match": total_em / num_examples,
        "avg_searches": total_searches / num_examples,
        "format_correctness": total_with_answer_tag / num_examples,
        "num_examples": num_examples,
    }

    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print(f"Exact Match (EM):       {metrics['exact_match']:.4f} ({total_em}/{num_examples})")
    print(f"Average Searches:       {metrics['avg_searches']:.2f}")
    print(f"Format Correctness:     {metrics['format_correctness']:.4f}")
    print("=" * 70)

    # Save detailed results if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "metrics": metrics,
            "examples": results[:10],  # Save first 10 examples for inspection
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Detailed results saved to: {output_file}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate QReCC Plan B trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/qrecc_plan_b/test.parquet",
        help="Path to test data file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save evaluation results (JSON)"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate (default: all)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("QReCC Plan B Evaluation")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test file: {args.test_file}")
    print(f"Max examples: {args.max_examples if args.max_examples else 'All'}")
    print("=" * 70)

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.checkpoint)

    # Load test data
    print(f"\nLoading test data from {args.test_file}...")
    test_data = datasets.load_dataset('parquet', data_files=args.test_file, split='train')
    print(f"✓ Loaded {len(test_data)} test examples")

    # Evaluate
    metrics = evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        test_data=test_data,
        max_examples=args.max_examples,
        output_file=args.output_file
    )

    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
