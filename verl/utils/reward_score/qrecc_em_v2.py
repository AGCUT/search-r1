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
Improved Reward scoring functions for QReCC dataset (Conversational QA).

Improvements over v1:
1. F1-based scoring instead of strict EM (handles long answers better)
2. Bonus for correct format (<think>, <search>, <answer> tags)
3. Bonus for using search results effectively
4. Gradual rewards instead of binary 0/1
5. Penalty for unnecessary/ineffective searches
6. Validation of proper tag sequence
7. Check if retrieval results contain the answer

Reward Design Philosophy:
- Correct answer is the primary goal (highest weight)
- Efficient search usage is rewarded (search only when needed, use results effectively)
- Unnecessary searches are penalized (searched but didn't help)
- Good format is a secondary goal (helps learning but not required)
"""

import re
import string
import random
from collections import Counter


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


def get_tokens(s):
    """Tokenize string into words after normalization."""
    if not s:
        return []
    return normalize_answer(s).split()


def compute_f1(prediction, ground_truth):
    """
    Compute token-level F1 score between prediction and ground truth.

    Returns:
        Tuple of (f1, precision, recall)
    """
    pred_tokens = get_tokens(prediction)
    gold_tokens = get_tokens(ground_truth)

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return (1.0, 1.0, 1.0) if pred_tokens == gold_tokens else (0.0, 0.0, 0.0)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return f1, precision, recall


def compute_best_f1(prediction, golden_answers):
    """
    Compute the best F1 score against any of the golden answers.

    Args:
        prediction: Predicted answer string
        golden_answers: List of acceptable answer strings

    Returns:
        Best F1 score
    """
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    best_f1 = 0.0
    for gold in golden_answers:
        f1, _, _ = compute_f1(prediction, gold)
        best_f1 = max(best_f1, f1)

    return best_f1


def em_check(prediction, golden_answers):
    """
    Check if prediction exactly matches any of the golden answers (after normalization).
    """
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    normalized_prediction = normalize_answer(prediction)

    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) == normalized_prediction:
            return True

    return False


def contains_check(prediction, golden_answers):
    """
    Check if prediction contains any golden answer or vice versa.
    """
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    normalized_prediction = normalize_answer(prediction)

    for golden_answer in golden_answers:
        normalized_gold = normalize_answer(golden_answer)
        # Check both directions
        if normalized_gold in normalized_prediction or normalized_prediction in normalized_gold:
            return True

    return False


def extract_solution(solution_str):
    """
    Extract the answer from the model's solution string.
    Returns the content of the LAST <answer> tag.
    """
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))

    # We expect at least 2: one as example in prompt, one as actual answer
    if len(matches) <= 1:
        return None

    return matches[-1].group(1).strip()


def check_format_quality(solution_str):
    """
    Check the format quality of the solution.

    Returns:
        Dict with format analysis:
        - has_think: bool - whether <think> tags are present
        - has_search: bool - whether <search> tags are present
        - has_answer: bool - whether <answer> tags are present
        - has_information: bool - whether search results were received
        - think_count: int - number of think blocks
        - search_count: int - number of search calls
        - format_score: float - overall format quality (0-1)
    """
    # Count occurrences (excluding prompt examples - typically first occurrence)
    think_matches = re.findall(r'<think>(.*?)</think>', solution_str, re.DOTALL)
    search_matches = re.findall(r'<search>(.*?)</search>', solution_str, re.DOTALL)
    answer_matches = re.findall(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
    info_matches = re.findall(r'<information>(.*?)</information>', solution_str, re.DOTALL)

    # Subtract 1 for prompt example (if present)
    think_count = max(0, len(think_matches) - 1)
    search_count = max(0, len(search_matches) - 1)
    answer_count = max(0, len(answer_matches) - 1)
    info_count = len(info_matches)  # Information is injected by system, no example in prompt

    has_think = think_count > 0
    has_search = search_count > 0
    has_answer = answer_count > 0
    has_information = info_count > 0

    # Calculate format score
    format_score = 0.0

    # Basic requirement: must have answer
    if has_answer:
        format_score += 0.4

    # Reasoning is good
    if has_think:
        format_score += 0.2

    # Search + got results is good (shows model knows how to use tools)
    if has_search and has_information:
        format_score += 0.3
    elif has_search:
        format_score += 0.1  # Tried to search but no results (partial credit)

    # Multiple reasoning steps (shows chain-of-thought)
    if think_count >= 2:
        format_score += 0.1

    return {
        'has_think': has_think,
        'has_search': has_search,
        'has_answer': has_answer,
        'has_information': has_information,
        'think_count': think_count,
        'search_count': search_count,
        'format_score': min(1.0, format_score)
    }


def compute_score_f1(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """
    F1-based scoring function - more lenient than exact match.

    Scoring breakdown:
    - F1 score with ground truth (0-1) * 0.7
    - Format quality bonus (0-1) * 0.3

    This gives partial credit for:
    - Partially correct answers
    - Good reasoning format
    - Effective search usage
    """
    # Extract the answer
    answer = extract_solution(solution_str=solution_str)

    # Check format quality
    format_info = check_format_quality(solution_str)

    # Debug printing (1/64 chance)
    do_print = random.randint(1, 64) == 1

    # No valid answer extracted
    if answer is None:
        if do_print:
            print(f"--------------------------------")
            print(f"[QReCC F1] No answer extracted!")
            print(f"[QReCC F1] Format info: {format_info}")
            print(f"[QReCC F1] Solution: {solution_str[:300]}...")
            print(f"--------------------------------")
        # Small penalty for no answer, but give some credit for good format
        return format_info['format_score'] * 0.1

    # Compute F1 with ground truth
    f1_score = compute_best_f1(answer, ground_truth['target'])

    # Check for exact match (bonus)
    is_exact = em_check(answer, ground_truth['target'])

    # Check for containment (partial bonus)
    is_contained = contains_check(answer, ground_truth['target'])

    # Calculate final score
    # Base: F1 score (0-1) weighted at 70%
    # Format: format quality (0-1) weighted at 30%
    base_score = f1_score * 0.7
    format_bonus = format_info['format_score'] * 0.3

    # Exact match bonus
    if is_exact:
        base_score = 0.7  # Full base score for exact match
    elif is_contained and f1_score < 0.5:
        # Containment bonus when F1 is low (answer is correct but different phrasing)
        base_score = max(base_score, 0.4)

    final_score = base_score + format_bonus

    if do_print:
        print(f"--------------------------------")
        print(f"[QReCC F1] Golden: {ground_truth['target']}")
        print(f"[QReCC F1] Predicted: {answer}")
        print(f"[QReCC F1] F1={f1_score:.3f}, EM={is_exact}, Contains={is_contained}")
        print(f"[QReCC F1] Format: {format_info}")
        print(f"[QReCC F1] Final score: {final_score:.3f} (base={base_score:.3f}, format={format_bonus:.3f})")
        print(f"--------------------------------")

    return final_score


def compute_score_em_v2(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """
    Improved EM scoring with format bonus.

    Keeps strict EM for correctness, but adds format quality bonus.

    Scoring:
    - Correct answer (EM): 1.0
    - Wrong answer but good format: 0.1-0.3
    - No answer tag: 0.0
    """
    # Extract the answer
    answer = extract_solution(solution_str=solution_str)

    # Check format quality
    format_info = check_format_quality(solution_str)

    # Debug printing
    do_print = random.randint(1, 64) == 1

    # No valid answer extracted
    if answer is None:
        if do_print:
            print(f"--------------------------------")
            print(f"[QReCC EM v2] No answer extracted!")
            print(f"[QReCC EM v2] Format: {format_info}")
            print(f"--------------------------------")
        return 0.0

    # Check exact match
    if em_check(answer, ground_truth['target']):
        final_score = score
    else:
        # Wrong answer, but give small credit for good format
        # This helps early training when model is learning the format
        final_score = format_info['format_score'] * format_score if format_score > 0 else 0.0

    if do_print:
        print(f"--------------------------------")
        print(f"[QReCC EM v2] Golden: {ground_truth['target']}")
        print(f"[QReCC EM v2] Predicted: {answer}")
        print(f"[QReCC EM v2] EM match: {em_check(answer, ground_truth['target'])}")
        print(f"[QReCC EM v2] Format: {format_info}")
        print(f"[QReCC EM v2] Final score: {final_score:.3f}")
        print(f"--------------------------------")

    return final_score


def compute_score_hybrid(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """
    Hybrid scoring: combines EM strictness with F1 partial credit.

    Scoring breakdown:
    - Exact match: 1.0
    - High F1 (>=0.8): 0.8
    - Medium F1 (>=0.5): 0.5 + F1 * 0.3
    - Contains match: 0.4
    - Low F1 with good format: F1 * 0.3 + format * 0.2
    - No answer: 0.0

    This is a good balance between strictness and learning signal.
    """
    # Extract the answer
    answer = extract_solution(solution_str=solution_str)

    # Check format
    format_info = check_format_quality(solution_str)

    do_print = random.randint(1, 64) == 1

    if answer is None:
        if do_print:
            print(f"[QReCC Hybrid] No answer - score: 0.0")
        return 0.0

    # Calculate metrics
    f1 = compute_best_f1(answer, ground_truth['target'])
    is_exact = em_check(answer, ground_truth['target'])
    is_contained = contains_check(answer, ground_truth['target'])

    # Determine score
    if is_exact:
        final_score = 1.0
    elif f1 >= 0.8:
        final_score = 0.8
    elif f1 >= 0.5:
        final_score = 0.5 + f1 * 0.3
    elif is_contained:
        final_score = 0.4
    else:
        # Low F1 - give partial credit based on F1 and format
        final_score = f1 * 0.3 + format_info['format_score'] * 0.2

    if do_print:
        print(f"--------------------------------")
        print(f"[QReCC Hybrid] Golden: {ground_truth['target'][:100]}...")
        print(f"[QReCC Hybrid] Predicted: {answer[:100]}...")
        print(f"[QReCC Hybrid] F1={f1:.3f}, EM={is_exact}, Contains={is_contained}")
        print(f"[QReCC Hybrid] Final score: {final_score:.3f}")
        print(f"--------------------------------")

    return final_score


###############################################################################
# Advanced Reward Functions with Search Efficiency Analysis
###############################################################################

def extract_information_blocks(text: str) -> list:
    """Extract all <information> blocks from the text."""
    pattern = r"<information>(.*?)</information>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def is_retrieval_helpful(text: str, golden_answers: list) -> bool:
    """
    Check if the retrieved information contains the golden answer.
    This indicates whether the search was helpful.
    """
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    info_blocks = extract_information_blocks(text)
    for info in info_blocks:
        normalized_info = normalize_answer(info)
        for golden in golden_answers:
            if normalize_answer(golden) in normalized_info:
                return True
    return False


def is_valid_tag_sequence(text: str) -> tuple:
    """
    Validate the tag sequence follows the expected pattern:
    think -> search -> information -> think -> ... -> answer

    Returns:
        (is_valid, error_message)
    """
    # Find assistant response start
    assistant_patterns = [
        r"<\|im_start\|>assistant\s*",
        r"Assistant:\s*",
        r"\[/INST\]\s*"
    ]

    content = text
    for pattern in assistant_patterns:
        match = re.search(pattern, text)
        if match:
            content = text[match.end():]
            break

    # Check for balanced tags
    tags = ["think", "search", "information", "answer"]
    for tag in tags:
        opening = len(re.findall(f"<{tag}>", content))
        closing = len(re.findall(f"</{tag}>", content))
        if opening != closing:
            return False, f"Unbalanced {tag} tags: {opening} open, {closing} close"

    # Check sequence pattern using state machine
    split_pattern = r"(</?(?:think|search|information|answer)>)"
    parts = re.split(split_pattern, content)

    state = "start"
    valid_transitions = {
        "start": {"<think>": "in_think"},
        "in_think": {"</think>": "after_think"},
        "after_think": {"<search>": "in_search", "<answer>": "in_answer"},
        "in_search": {"</search>": "after_search"},
        "after_search": {"<information>": "in_information"},
        "in_information": {"</information>": "after_information"},
        "after_information": {"<think>": "in_think"},
        "in_answer": {"</answer>": "end"},
        "end": {}
    }

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if re.match(r"</?(?:think|search|information|answer)>", part):
            if part in valid_transitions.get(state, {}):
                state = valid_transitions[state][part]
            elif state in ["in_think", "in_search", "in_information", "in_answer"]:
                # Allow content inside tags
                continue
            else:
                # Invalid transition, but don't be too strict
                pass

    # Accept if we reached a reasonable end state
    if state in ["end", "after_think", "after_information"]:
        return True, "Valid sequence"

    return False, f"Incomplete sequence, ended in state: {state}"


def analyze_search_efficiency(solution_str: str, ground_truth: dict) -> dict:
    """
    Analyze the efficiency of search usage in the solution.

    Returns a dict with:
    - search_count: Number of searches performed
    - info_count: Number of information blocks received
    - retrieval_helpful: Whether retrieved info contains answer
    - search_wasted: Searches that didn't contribute to correct answer
    - efficiency_score: Overall search efficiency (-1 to 1)
    """
    format_info = check_format_quality(solution_str)
    golden_answers = ground_truth.get('target', [])

    search_count = format_info['search_count']
    info_count = len(extract_information_blocks(solution_str))

    # Check if retrieval was helpful
    retrieval_helpful = is_retrieval_helpful(solution_str, golden_answers)

    # Extract the answer
    answer = extract_solution(solution_str)
    is_correct = answer is not None and em_check(answer, golden_answers)

    # Calculate efficiency
    efficiency_score = 0.0
    search_wasted = 0

    if search_count == 0:
        if is_correct:
            # No search needed and got correct answer - perfect efficiency
            efficiency_score = 0.1  # Small bonus for efficiency
        else:
            # No search and wrong answer - maybe should have searched
            efficiency_score = 0.0
    else:
        if is_correct:
            if retrieval_helpful:
                # Searched, got helpful info, correct answer - good!
                efficiency_score = 0.1
            else:
                # Correct but search wasn't helpful - slightly wasteful
                efficiency_score = 0.0
                search_wasted = search_count
        else:
            if retrieval_helpful:
                # Had the answer in search results but still wrong - missed opportunity
                efficiency_score = -0.05
            else:
                # Search didn't help and still wrong - wasted effort
                efficiency_score = -0.1
                search_wasted = search_count

    # Penalize excessive searches (more than 3 is usually unnecessary)
    if search_count > 3:
        efficiency_score -= 0.05 * (search_count - 3)

    return {
        'search_count': search_count,
        'info_count': info_count,
        'retrieval_helpful': retrieval_helpful,
        'search_wasted': search_wasted,
        'is_correct': is_correct,
        'efficiency_score': max(-0.3, min(0.2, efficiency_score))
    }


def compute_score_advanced(solution_str, ground_truth, method='strict',
                           format_score=0., score=1.,
                           structure_bonus=0.1, retrieval_bonus=0.1,
                           search_penalty=0.05):
    """
    Advanced scoring function that rewards/penalizes search efficiency.

    Scoring breakdown:
    - Base score from answer correctness (EM/F1)
    - Structure bonus: +0.1 for valid tag sequence
    - Retrieval bonus: +0.1 if search results contain answer AND used correctly
    - Search penalty: -0.05 per wasted search (searched but didn't help)
    - Excessive search penalty: -0.05 per search beyond 3

    Score ranges:
    - Correct + efficient: 1.0 + 0.2 = 1.2 (capped at 1.0)
    - Correct + wasteful: 1.0 - 0.1 = 0.9
    - Wrong + good format: 0.1 - 0.3
    - Wrong + wasted searches: 0.0 - 0.2 = -0.2 (floored at 0.0)
    """
    # Extract answer
    answer = extract_solution(solution_str)
    golden_answers = ground_truth.get('target', [])

    # Analyze components
    format_info = check_format_quality(solution_str)
    is_valid_seq, seq_error = is_valid_tag_sequence(solution_str)
    search_analysis = analyze_search_efficiency(solution_str, ground_truth)

    do_print = random.randint(1, 64) == 1

    # Start with base score
    base_score = 0.0

    if answer is None:
        # No answer - only give small format credit if structure is valid
        if is_valid_seq and format_info['has_think']:
            base_score = structure_bonus
        final_score = base_score
    else:
        # Calculate answer quality
        is_exact = em_check(answer, golden_answers)
        f1 = compute_best_f1(answer, golden_answers)
        is_contained = contains_check(answer, golden_answers)

        if is_exact:
            base_score = score  # 1.0
        elif f1 >= 0.8:
            base_score = 0.8
        elif f1 >= 0.5:
            base_score = 0.5 + f1 * 0.3
        elif is_contained:
            base_score = 0.4
        else:
            # Wrong answer
            base_score = f1 * 0.2

        # Add structure bonus
        if is_valid_seq:
            base_score += structure_bonus

        # Add retrieval bonus (only if answer is somewhat correct)
        if search_analysis['retrieval_helpful'] and base_score >= 0.4:
            base_score += retrieval_bonus

        # Apply search efficiency adjustment
        base_score += search_analysis['efficiency_score']

        # Apply penalty for wasted searches
        if search_analysis['search_wasted'] > 0 and base_score < 0.5:
            base_score -= search_penalty * search_analysis['search_wasted']

        final_score = max(0.0, min(1.0, base_score))

    if do_print:
        print(f"=" * 60)
        print(f"[QReCC Advanced] Golden: {golden_answers[:2]}...")
        print(f"[QReCC Advanced] Predicted: {answer[:100] if answer else 'None'}...")
        print(f"[QReCC Advanced] Valid sequence: {is_valid_seq}")
        print(f"[QReCC Advanced] Search analysis: {search_analysis}")
        print(f"[QReCC Advanced] Base score: {base_score:.3f}")
        print(f"[QReCC Advanced] Final score: {final_score:.3f}")
        print(f"=" * 60)

    return final_score


def compute_score_searchr1(solution_str, ground_truth, method='strict',
                           structure_format_score=0.2, retrieval_score=0.1,
                           final_format_score=0.1, format_score=0., score=1.):
    """
    SearchR1-style scoring function (based on qa_em_format.py).

    This matches the original Search-R1 reward design:
    - Correct answer + valid format: 1.0
    - Correct answer + invalid format: 0.8
    - Wrong answer + valid format + retrieval contains answer: 0.3
    - Wrong answer + valid format: 0.2
    - Wrong answer + has answer tag: 0.1
    - No answer tag: 0.0

    Args:
        structure_format_score: Bonus for valid tag structure (default: 0.2)
        retrieval_score: Bonus if retrieved docs contain answer (default: 0.1)
        final_format_score: Bonus for having answer tag (default: 0.1)
    """
    # Validate tag sequence
    is_valid_format, _ = is_valid_tag_sequence(solution_str)

    # Check if retrieval was helpful
    golden_answers = ground_truth.get('target', [])
    retrieval_correct = False
    if is_valid_format:
        retrieval_correct = is_retrieval_helpful(solution_str, golden_answers)

    # Extract answer
    answer = extract_solution(solution_str)

    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"[SearchR1] Golden: {golden_answers}")
        print(f"[SearchR1] Extracted: {answer}")
        print(f"[SearchR1] Valid format: {is_valid_format}")
        print(f"[SearchR1] Retrieval correct: {retrieval_correct}")

    if answer is None:
        if is_valid_format:
            if retrieval_correct:
                final_score = structure_format_score + retrieval_score  # 0.3
            else:
                final_score = structure_format_score  # 0.2
        else:
            final_score = 0.0
    else:
        if em_check(answer, golden_answers):
            if is_valid_format:
                final_score = score  # 1.0
            else:
                final_score = score - structure_format_score  # 0.8
        elif is_valid_format:
            if retrieval_correct:
                final_score = structure_format_score + retrieval_score  # 0.3
            else:
                final_score = structure_format_score  # 0.2
        else:
            final_score = final_format_score  # 0.1

    if do_print:
        print(f"[SearchR1] Final score: {final_score}")
        print(f"--------------------------------")

    return final_score


# Default export - use advanced scoring
compute_score_em = compute_score_advanced
