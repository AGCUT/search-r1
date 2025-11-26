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

    # Randomly print some examples for debugging (1/64 chance)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"[QReCC EM] Golden answers: {ground_truth['target']}")
        print(f"[QReCC EM] Extracted answer: {answer}")
        print(f"[QReCC EM] Solution string: {solution_str[:200]}...")
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