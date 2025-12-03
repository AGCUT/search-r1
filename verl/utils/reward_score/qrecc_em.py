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
Reward scoring functions for QReCC dataset (Conversational QA).

QReCC uses the same exact match (EM) scoring as standard QA tasks,
but is designed for conversational question answering scenarios.
"""

import re
import string
import random


def normalize_answer(s):
    """
    Normalize answer text by:
    - Lowercasing
    - Removing articles (a, an, the)
    - Removing punctuation
    - Removing extra whitespace
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    """
    Check if prediction exactly matches any of the golden answers (after normalization).

    Args:
        prediction: The predicted answer string
        golden_answers: List of acceptable answer strings (or single string)

    Returns:
        1 if exact match found, 0 otherwise
    """
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    normalized_prediction = normalize_answer(prediction)
    score = 0

    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break

    return score


def subem_check(prediction, golden_answers):
    """
    Check if any golden answer is a substring of the prediction (after normalization).

    Args:
        prediction: The predicted answer string
        golden_answers: List of acceptable answer strings (or single string)

    Returns:
        1 if substring match found, 0 otherwise
    """
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    normalized_prediction = normalize_answer(prediction)
    score = 0

    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break

    return score


def f1_check(prediction, golden_answers):
    """
    Compute token-level F1 score between prediction and golden answers.

    This follows the official SQuAD/QReCC evaluation:
    - Tokenize by whitespace after normalization
    - Compute precision, recall, and F1 based on token overlap
    - Return the maximum F1 across all golden answers

    Args:
        prediction: The predicted answer string
        golden_answers: List of acceptable answer strings (or single string)

    Returns:
        Float F1 score between 0 and 1
    """
    from collections import Counter

    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    # Tokenize prediction
    pred_tokens = normalize_answer(prediction).split()

    if len(pred_tokens) == 0:
        return 0.0

    max_f1 = 0.0

    for golden_answer in golden_answers:
        gold_tokens = normalize_answer(golden_answer).split()

        if len(gold_tokens) == 0:
            continue

        # Count common tokens
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            continue

        # Compute precision, recall, F1
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)

        max_f1 = max(max_f1, f1)

    return max_f1


def extract_solution(solution_str):
    """
    Extract the answer from the model's solution string.

    The model should produce answers within <answer>...</answer> tags.
    We extract the content of the LAST <answer> tag (after all reasoning steps).

    Args:
        solution_str: The complete solution string from the model

    Returns:
        Extracted answer string, or None if no valid answer found
    """
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))

    # If there are 0 or exactly 1 matches, return None
    # We expect at least 2: one as example in prompt, one as actual answer
    if len(matches) <= 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """
    The scoring function for exact match (EM) on QReCC dataset.

    Args:
        solution_str: The complete solution text from the model
        ground_truth: Dictionary containing the ground truth answer(s) under 'target' key
        method: The method to extract the solution ('strict' or 'flexible')
        format_score: The score given if answer is correctly formatted but wrong (default: 0)
        score: The score given if answer is correct (default: 1)

    Returns:
        Float score: 0 (incorrect/invalid), format_score (formatted but wrong), or score (correct)
    """
    # Extract the answer from solution string
    answer = extract_solution(solution_str=solution_str)

    # Count <answer> tags for debugging
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    num_answer_tags = len(matches)

    # Randomly print some examples for debugging (1/32 chance)
    do_print = random.randint(1, 32) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"[QReCC EM] Golden answers: {ground_truth['target']}")
        print(f"[QReCC EM] Num <answer> tags found: {num_answer_tags}")
        print(f"[QReCC EM] Extracted answer: {answer}")
        print(f"[QReCC EM] Solution (last 500 chars): ...{solution_str[-500:]}")
        print(f"--------------------------------")

    # No valid answer extracted
    if answer is None:
        return 0

    # Check if extracted answer matches ground truth
    if em_check(answer, ground_truth['target']):
        return score
    else:
        # Answer was formatted correctly but didn't match
        return format_score


def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """
    The scoring function for substring exact match (sub-EM) on QReCC dataset.

    More lenient than full EM - accepts if golden answer appears anywhere in prediction.

    Args:
        solution_str: The complete solution text from the model
        ground_truth: Dictionary containing the ground truth answer(s) under 'target' key
        method: The method to extract the solution ('strict' or 'flexible')
        format_score: The score given if answer is correctly formatted but wrong (default: 0)
        score: The score given if answer is correct (default: 1)

    Returns:
        Float score: 0 (incorrect/invalid), format_score (formatted but wrong), or score (correct)
    """
    # Extract the answer from solution string
    answer = extract_solution(solution_str=solution_str)

    # Randomly print some examples for debugging (1/64 chance)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"[QReCC Sub-EM] Golden answers: {ground_truth['target']}")
        print(f"[QReCC Sub-EM] Extracted answer: {answer}")
        print(f"[QReCC Sub-EM] Solution string: {solution_str[:200]}...")
        print(f"--------------------------------")

    # No valid answer extracted
    if answer is None:
        return 0

    # Check if extracted answer contains ground truth
    if subem_check(answer, ground_truth['target']):
        return score
    else:
        # Answer was formatted correctly but didn't match
        return format_score


def compute_score_f1(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """
    The scoring function using token-level F1 score on QReCC dataset.

    This follows the official SQuAD/QReCC evaluation methodology.
    F1 provides partial credit for partially correct answers, which can
    help with training by providing more granular reward signals.

    Args:
        solution_str: The complete solution text from the model
        ground_truth: Dictionary containing the ground truth answer(s) under 'target' key
        method: The method to extract the solution ('strict' or 'flexible')
        format_score: Not used for F1, kept for API compatibility
        score: Maximum score (F1 is scaled to [0, score])

    Returns:
        Float score: 0 (no answer) or F1 score scaled to [0, score]
    """
    # Extract the answer from solution string
    answer = extract_solution(solution_str=solution_str)

    # Count <answer> tags for debugging
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    num_answer_tags = len(matches)

    # Randomly print some examples for debugging (1/32 chance)
    do_print = random.randint(1, 32) == 1

    # No valid answer extracted
    if answer is None:
        if do_print:
            print(f"--------------------------------")
            print(f"[QReCC F1] Golden answers: {ground_truth['target']}")
            print(f"[QReCC F1] Num <answer> tags found: {num_answer_tags}")
            print(f"[QReCC F1] Extracted answer: None")
            print(f"[QReCC F1] F1 Score: 0.0")
            print(f"--------------------------------")
        return 0

    # Compute F1 score
    f1 = f1_check(answer, ground_truth['target'])
    final_score = f1 * score  # Scale to [0, score]

    if do_print:
        print(f"--------------------------------")
        print(f"[QReCC F1] Golden answers: {ground_truth['target']}")
        print(f"[QReCC F1] Num <answer> tags found: {num_answer_tags}")
        print(f"[QReCC F1] Extracted answer: {answer}")
        print(f"[QReCC F1] F1 Score: {f1:.4f} (scaled: {final_score:.4f})")
        print(f"[QReCC F1] Solution (last 300 chars): ...{solution_str[-300:]}")
        print(f"--------------------------------")

    return final_score


def compute_score_em_f1(solution_str, ground_truth, method='strict', format_score=0., score=1., em_weight=0.5):
    """
    Combined EM + F1 scoring function.

    Uses weighted combination of EM and F1 scores:
    - If EM matches: returns full score
    - If EM doesn't match: returns em_weight * 0 + (1-em_weight) * F1

    This provides the strictness of EM with the partial credit of F1.

    Args:
        solution_str: The complete solution text from the model
        ground_truth: Dictionary containing the ground truth answer(s) under 'target' key
        method: The method to extract the solution ('strict' or 'flexible')
        format_score: Not used, kept for API compatibility
        score: Maximum score
        em_weight: Weight for EM component (default 0.5)

    Returns:
        Float score combining EM and F1
    """
    # Extract the answer from solution string
    answer = extract_solution(solution_str=solution_str)

    # No valid answer extracted
    if answer is None:
        return 0

    # Check EM first
    if em_check(answer, ground_truth['target']):
        return score  # Full score for exact match

    # If not exact match, compute F1 for partial credit
    f1 = f1_check(answer, ground_truth['target'])

    # Weighted combination: since EM=0, result is just (1-em_weight) * F1 * score
    final_score = (1 - em_weight) * f1 * score

    # Randomly print some examples for debugging (1/32 chance)
    do_print = random.randint(1, 32) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"[QReCC EM+F1] Golden answers: {ground_truth['target']}")
        print(f"[QReCC EM+F1] Extracted answer: {answer}")
        print(f"[QReCC EM+F1] EM: 0, F1: {f1:.4f}")
        print(f"[QReCC EM+F1] Combined Score: {final_score:.4f}")
        print(f"--------------------------------")

    return final_score