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
BLEU and ROUGE scoring functions for QReCC dataset.

BLEU (Bilingual Evaluation Understudy):
    - Measures n-gram precision between prediction and reference
    - Originally for machine translation, useful for long text evaluation
    - Score range: 0.0 to 1.0 (higher is better)

ROUGE (Recall-Oriented Understudy for Gisting Evaluation):
    - Measures recall-based n-gram overlap
    - Originally for summarization, good for long answer evaluation
    - Multiple variants: ROUGE-1, ROUGE-2, ROUGE-L
    - Score range: 0.0 to 1.0 (higher is better)
"""

import re
import string
from typing import List, Union


def normalize_answer(s: str) -> str:
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


def extract_solution(solution_str: str) -> str:
    """
    Extract the answer from the model's solution string.

    The model should produce answers within <answer>...</answer> tags.
    We expect at least 2 matches: one as example in prompt, one as actual answer.
    We extract the content of the LAST <answer> tag.

    Args:
        solution_str: The complete solution string from the model

    Returns:
        Extracted answer string, or empty string if no valid answer found
    """
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL | re.IGNORECASE))

    # If there are 0 or exactly 1 matches, return empty string
    # We expect at least 2: one as example in prompt, one as actual answer
    if len(matches) <= 1:
        return ""

    # Return the last match (the actual answer, not the example in prompt)
    return matches[-1].group(1).strip()


def compute_bleu(prediction: str, references: List[str], max_n: int = 4) -> float:
    """
    Compute BLEU score between prediction and references.

    BLEU measures n-gram precision with brevity penalty.

    Args:
        prediction: Predicted answer string
        references: List of reference answer strings
        max_n: Maximum n-gram order (default: 4 for BLEU-4)

    Returns:
        BLEU score between 0.0 and 1.0
    """
    from collections import Counter
    import math

    if not prediction or not references:
        return 0.0

    # Normalize and tokenize
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens_list = [normalize_answer(ref).split() for ref in references]

    if len(pred_tokens) == 0:
        return 0.0

    # Compute n-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        # Get n-grams from prediction
        pred_ngrams = Counter()
        for i in range(len(pred_tokens) - n + 1):
            ngram = tuple(pred_tokens[i:i+n])
            pred_ngrams[ngram] += 1

        if len(pred_ngrams) == 0:
            break

        # Get maximum n-gram counts from all references
        max_ref_counts = Counter()
        for ref_tokens in ref_tokens_list:
            ref_ngrams = Counter()
            for i in range(len(ref_tokens) - n + 1):
                ngram = tuple(ref_tokens[i:i+n])
                ref_ngrams[ngram] += 1

            # Take maximum count for each n-gram
            for ngram, count in ref_ngrams.items():
                max_ref_counts[ngram] = max(max_ref_counts[ngram], count)

        # Count clipped matches
        clipped_matches = 0
        total_pred = 0
        for ngram, pred_count in pred_ngrams.items():
            clipped_matches += min(pred_count, max_ref_counts.get(ngram, 0))
            total_pred += pred_count

        if total_pred > 0:
            precisions.append(clipped_matches / total_pred)
        else:
            precisions.append(0.0)

    if len(precisions) == 0 or all(p == 0 for p in precisions):
        return 0.0

    # Compute geometric mean of precisions
    geo_mean = math.exp(sum(math.log(p) if p > 0 else float('-inf') for p in precisions) / len(precisions))

    if math.isinf(geo_mean) or math.isnan(geo_mean):
        return 0.0

    # Compute brevity penalty
    pred_len = len(pred_tokens)
    ref_lens = [len(ref_tokens) for ref_tokens in ref_tokens_list]
    closest_ref_len = min(ref_lens, key=lambda x: abs(x - pred_len))

    if pred_len > closest_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - closest_ref_len / pred_len) if pred_len > 0 else 0.0

    return bp * geo_mean


def compute_rouge_l(prediction: str, references: List[str]) -> float:
    """
    Compute ROUGE-L score (Longest Common Subsequence based).

    ROUGE-L measures the longest common subsequence between prediction and reference.

    Args:
        prediction: Predicted answer string
        references: List of reference answer strings

    Returns:
        ROUGE-L F1 score between 0.0 and 1.0
    """
    def lcs(x: List[str], y: List[str]) -> int:
        """Compute longest common subsequence length."""
        m, n = len(x), len(y)
        if m == 0 or n == 0:
            return 0

        # Dynamic programming table
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    if not prediction or not references:
        return 0.0

    # Normalize and tokenize
    pred_tokens = normalize_answer(prediction).split()

    if len(pred_tokens) == 0:
        return 0.0

    max_f1 = 0.0

    for reference in references:
        ref_tokens = normalize_answer(reference).split()

        if len(ref_tokens) == 0:
            continue

        # Compute LCS length
        lcs_len = lcs(pred_tokens, ref_tokens)

        if lcs_len == 0:
            continue

        # Compute precision and recall
        precision = lcs_len / len(pred_tokens)
        recall = lcs_len / len(ref_tokens)

        # Compute F1 score
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            max_f1 = max(max_f1, f1)

    return max_f1


def compute_rouge_n(prediction: str, references: List[str], n: int = 1) -> float:
    """
    Compute ROUGE-N score (n-gram based).

    Args:
        prediction: Predicted answer string
        references: List of reference answer strings
        n: N-gram order (1 for unigram, 2 for bigram)

    Returns:
        ROUGE-N F1 score between 0.0 and 1.0
    """
    from collections import Counter

    if not prediction or not references:
        return 0.0

    # Normalize and tokenize
    pred_tokens = normalize_answer(prediction).split()

    if len(pred_tokens) < n:
        return 0.0

    # Get prediction n-grams
    pred_ngrams = Counter()
    for i in range(len(pred_tokens) - n + 1):
        ngram = tuple(pred_tokens[i:i+n])
        pred_ngrams[ngram] += 1

    if len(pred_ngrams) == 0:
        return 0.0

    max_f1 = 0.0

    for reference in references:
        ref_tokens = normalize_answer(reference).split()

        if len(ref_tokens) < n:
            continue

        # Get reference n-grams
        ref_ngrams = Counter()
        for i in range(len(ref_tokens) - n + 1):
            ngram = tuple(ref_tokens[i:i+n])
            ref_ngrams[ngram] += 1

        if len(ref_ngrams) == 0:
            continue

        # Count overlapping n-grams
        overlap = sum((pred_ngrams & ref_ngrams).values())

        if overlap == 0:
            continue

        # Compute precision and recall
        precision = overlap / sum(pred_ngrams.values())
        recall = overlap / sum(ref_ngrams.values())

        # Compute F1 score
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            max_f1 = max(max_f1, f1)

    return max_f1


def bleu_check(prediction: str, golden_answers: Union[str, List[str]], max_n: int = 4) -> float:
    """
    Convenience function for BLEU evaluation.

    Args:
        prediction: Predicted answer
        golden_answers: Ground truth answer(s)
        max_n: Maximum n-gram order

    Returns:
        BLEU score
    """
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    return compute_bleu(prediction, golden_answers, max_n)


def rouge_l_check(prediction: str, golden_answers: Union[str, List[str]]) -> float:
    """
    Convenience function for ROUGE-L evaluation.

    Args:
        prediction: Predicted answer
        golden_answers: Ground truth answer(s)

    Returns:
        ROUGE-L F1 score
    """
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    return compute_rouge_l(prediction, golden_answers)


def rouge_1_check(prediction: str, golden_answers: Union[str, List[str]]) -> float:
    """
    Convenience function for ROUGE-1 evaluation.

    Args:
        prediction: Predicted answer
        golden_answers: Ground truth answer(s)

    Returns:
        ROUGE-1 F1 score
    """
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    return compute_rouge_n(prediction, golden_answers, n=1)


def rouge_2_check(prediction: str, golden_answers: Union[str, List[str]]) -> float:
    """
    Convenience function for ROUGE-2 evaluation.

    Args:
        prediction: Predicted answer
        golden_answers: Ground truth answer(s)

    Returns:
        ROUGE-2 F1 score
    """
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    return compute_rouge_n(prediction, golden_answers, n=2)


def compute_score_bleu(solution_str: str, ground_truth: Union[str, List[str], dict],
                       format_score: float = 0.0, score: float = 1.0) -> float:
    """
    Scoring function for BLEU metric (compatible with veRL framework).

    Args:
        solution_str: Model's complete solution string
        ground_truth: Ground truth answer(s). Can be:
            - str: single answer
            - List[str]: multiple reference answers
            - dict: QReCC format with 'target' key containing List[str]
        format_score: Bonus score for correct format (default: 0.0)
        score: Maximum score value (default: 1.0)

    Returns:
        BLEU score between 0.0 and score
    """
    # Extract answer from solution
    predicted_answer = extract_solution(solution_str)

    # If no answer extracted, return format_score only
    if not predicted_answer:
        return format_score

    # Handle different ground_truth formats
    if isinstance(ground_truth, dict):
        # QReCC format: {'target': ['answer1', 'answer2', ...]}
        references = ground_truth.get('target', [])
    elif isinstance(ground_truth, str):
        references = [ground_truth]
    else:
        references = ground_truth

    # Ensure references is a list
    if not isinstance(references, list):
        references = [references]

    # Compute BLEU score
    bleu_score = compute_bleu(predicted_answer, references)

    # Scale to score range and add format bonus
    return bleu_score * score + format_score


def compute_score_rouge_l(solution_str: str, ground_truth: Union[str, List[str], dict],
                          format_score: float = 0.0, score: float = 1.0) -> float:
    """
    Scoring function for ROUGE-L metric (compatible with veRL framework).

    Args:
        solution_str: Model's complete solution string
        ground_truth: Ground truth answer(s). Can be:
            - str: single answer
            - List[str]: multiple reference answers
            - dict: QReCC format with 'target' key containing List[str]
        format_score: Bonus score for correct format (default: 0.0)
        score: Maximum score value (default: 1.0)

    Returns:
        ROUGE-L score between 0.0 and score
    """
    # Extract answer from solution
    predicted_answer = extract_solution(solution_str)

    # If no answer extracted, return format_score only
    if not predicted_answer:
        return format_score

    # Handle different ground_truth formats
    if isinstance(ground_truth, dict):
        # QReCC format: {'target': ['answer1', 'answer2', ...]}
        references = ground_truth.get('target', [])
    elif isinstance(ground_truth, str):
        references = [ground_truth]
    else:
        references = ground_truth

    # Ensure references is a list
    if not isinstance(references, list):
        references = [references]

    # Compute ROUGE-L score
    rouge_score = compute_rouge_l(predicted_answer, references)

    # Scale to score range and add format bonus
    return rouge_score * score + format_score


def compute_score_rouge_1(solution_str: str, ground_truth: Union[str, List[str], dict],
                          format_score: float = 0.0, score: float = 1.0) -> float:
    """
    Scoring function for ROUGE-1 metric (compatible with veRL framework).

    Args:
        solution_str: Model's complete solution string
        ground_truth: Ground truth answer(s). Can be:
            - str: single answer
            - List[str]: multiple reference answers
            - dict: QReCC format with 'target' key containing List[str]
        format_score: Bonus score for correct format (default: 0.0)
        score: Maximum score value (default: 1.0)

    Returns:
        ROUGE-1 score between 0.0 and score
    """
    # Extract answer from solution
    predicted_answer = extract_solution(solution_str)

    # If no answer extracted, return format_score only
    if not predicted_answer:
        return format_score

    # Handle different ground_truth formats
    if isinstance(ground_truth, dict):
        # QReCC format: {'target': ['answer1', 'answer2', ...]}
        references = ground_truth.get('target', [])
    elif isinstance(ground_truth, str):
        references = [ground_truth]
    else:
        references = ground_truth

    # Ensure references is a list
    if not isinstance(references, list):
        references = [references]

    # Compute ROUGE-1 score
    rouge_score = compute_rouge_n(predicted_answer, references, n=1)

    # Scale to score range and add format bonus
    return rouge_score * score + format_score


def compute_score_rouge_2(solution_str: str, ground_truth: Union[str, List[str], dict],
                          format_score: float = 0.0, score: float = 1.0) -> float:
    """
    Scoring function for ROUGE-2 metric (compatible with veRL framework).

    Args:
        solution_str: Model's complete solution string
        ground_truth: Ground truth answer(s). Can be:
            - str: single answer
            - List[str]: multiple reference answers
            - dict: QReCC format with 'target' key containing List[str]
        format_score: Bonus score for correct format (default: 0.0)
        score: Maximum score value (default: 1.0)

    Returns:
        ROUGE-2 score between 0.0 and score
    """
    # Extract answer from solution
    predicted_answer = extract_solution(solution_str)

    # If no answer extracted, return format_score only
    if not predicted_answer:
        return format_score

    # Handle different ground_truth formats
    if isinstance(ground_truth, dict):
        # QReCC format: {'target': ['answer1', 'answer2', ...]}
        references = ground_truth.get('target', [])
    elif isinstance(ground_truth, str):
        references = [ground_truth]
    else:
        references = ground_truth

    # Ensure references is a list
    if not isinstance(references, list):
        references = [references]

    # Compute ROUGE-2 score
    rouge_score = compute_rouge_n(predicted_answer, references, n=2)

    # Scale to score range and add format bonus
    return rouge_score * score + format_score