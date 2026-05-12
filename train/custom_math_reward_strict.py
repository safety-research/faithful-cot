# Copyright 2024 Custom Math Reward Function
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
Custom reward function for math tasks.
Based on datasets.load_dataset("deepmind/math_dataset", "arithmetic__mul_div_multiple")
"""

from __future__ import annotations

import re
from typing import NamedTuple, NewType

def bind_maybe(f, x):
    """Apply function f if x is not None, else propagate None."""
    return f(x) if x is not None else None

# Reward parameters
string_peak = 2.0
numeric_peak = 1.5
loose_accuracy_threshold = 1.4
strict_accuracy_threshold = 1.8

# Format reward tiers (progressive learning)
partial_format_bonus = 0.1  # Has </think> delimiter (partial compliance)
strict_format_bonus = 0.2   # <think> at start + </think> present (full compliance)
answer_prefix_bonus = 0.1   # Has "Final answer:" prefix

# Type aliases
RawOutput = NewType("RawOutput", str)
AnswerRegion = NewType("AnswerRegion", str)
AnswerNum = NewType("AnswerNum", str)

# Patterns
NUMBER_PATTERN = re.compile(r"\\frac\{\s*-?\d+\s*\}\{\s*-?\d+\s*\}|-?\s*\d+\s*/\s*-?\s*\d+|-?\d+(?:\.\d+)?")
ANSWER_PREFIX = "Final answer: "


def clean_answer_region(s: str) -> AnswerRegion:
    return AnswerRegion(s.strip().removesuffix("<end_of_turn>").strip())


def mk_answer_region(s: str) -> AnswerRegion | None:
    return clean_answer_region(s.split("</think>")[-1]) if s.count("</think>") == 1 else None


def mk_answer_num(region: AnswerRegion) -> AnswerNum | None:
    return (
        AnswerNum(region.strip().lower().removeprefix(ANSWER_PREFIX.lower()))
        if region.strip().lower().startswith(ANSWER_PREFIX.lower()) and NUMBER_PATTERN.search(region) is not None
        else None
    )


def str_to_answer_num(s: str) -> AnswerNum | None:
    return bind_maybe(mk_answer_num, mk_answer_region(s))


class OutputFeatures(NamedTuple):
    answer_str: AnswerNum | None
    answer_val: float | None
    has_partial_format: bool  # Has </think> delimiter (partial compliance)
    has_strict_format: bool   # <think> at start + </think> present (full compliance)
    has_answer_prefix: bool
    output_len: int


def extract_features(output: RawOutput) -> OutputFeatures:
    answer_str = str_to_answer_num(output)

    # Two-tier format checking for progressive reward shaping

    # Tier 1: Partial format - Has </think> delimiter (encourages learning delimiter)
    has_partial_format = output.count("</think>") == 1

    # Tier 2: Strict format - <think> at start AND </think> present in correct order
    has_strict_format = False
    if output.count("<think>") == 1 and output.count("</think>") == 1:
        # Check <think> is at the very start (after stripping leading whitespace)
        stripped_output = output.lstrip()
        if stripped_output.startswith("<think>"):
            think_pos = output.find("<think>")
            end_think_pos = output.find("</think>")
            if think_pos < end_think_pos:  # Correct order
                has_strict_format = True

    # Check if answer region (after </think>) starts with prefix
    answer_region = mk_answer_region(output)
    has_answer_prefix = False
    if answer_region is not None:
        has_answer_prefix = answer_region.strip().lower().startswith(ANSWER_PREFIX.lower())

    return OutputFeatures(
        answer_str=answer_str,
        answer_val=bind_maybe(parse_num, answer_str),
        has_partial_format=has_partial_format,
        has_strict_format=has_strict_format,
        has_answer_prefix=has_answer_prefix,
        output_len=len(output),
    )


def parse_num(s: str) -> float | None:
    """Parse a number from string, supporting fractions and decimals."""
    s = s.strip()

    try:
        # LaTeX fraction: \frac{num}{den}
        m = re.search(r"\\frac\{\s*(-?\d+)\s*\}\{\s*(-?\d+)\s*\}", s)
        if m:
            num, den = int(m.group(1)), int(m.group(2))
            return None if den == 0 else num / den

        # Plain fraction: num/den
        m = re.search(r"(-?\s*\d+)\s*/\s*(-?\s*\d+)", s)
        if m:
            num, den = int(m.group(1).replace(" ", "")), int(m.group(2).replace(" ", ""))
            return None if den == 0 else num / den

        # Decimal or integer
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        if m:
            return float(m.group(0))
    except ValueError:
        return None
    return None


def numeric_reward(answer_region: str, gold: str, *, alpha: float = 200) -> float | None:
    """Compute reward based on numerical accuracy."""
    model_val, gold_val = parse_num(answer_region), parse_num(gold)
    try:
        if model_val is None or gold_val is None:
            return None
        else:
            relative_error = (model_val - gold_val) / (abs(gold_val) + 1e-8)
            return max(0.0, numeric_peak - alpha * relative_error**2)
    except OverflowError:
        return None


def answer_reward(ans: AnswerNum, gold_answer: str) -> float:
    """Compute reward for an extracted answer."""
    gold_normalized = gold_answer.lower().strip()
    len_penalty = 0.05 * abs(len(ans) - len(gold_normalized))

    # Exact string match
    if ans == gold_normalized:
        return max(string_peak - len_penalty, 0.0)

    # Numerical match
    rew = numeric_reward(ans, gold_answer)
    if rew is not None:
        return max(rew - len_penalty, 0.0)

    return max(0 - len_penalty, -1)


def math_reward(features: OutputFeatures, gold_answer: str) -> float:
    """
    Core reward computation logic with tiered format bonuses.

    Reward structure:
    1. Base reward: Answer correctness (0 to 2.0)
    2. Format bonuses (progressive learning):
       - Partial format (has </think>): +0.1
       - Strict format (<think> first + </think>): +0.3 (replaces partial)
       - Answer prefix ("Final answer:"): +0.1 (additional)
    3. Penalties: Missing format or incorrect answers
    """
    # Compute base answer reward
    base = 0 if features.answer_str is None else answer_reward(features.answer_str, gold_answer)

    # Add tiered format bonuses
    if features.has_strict_format:
        # Strict format: <think> at start + </think> present
        format_reward = strict_format_bonus
    elif features.has_partial_format:
        # Partial format: has </think> delimiter (encourages learning)
        format_reward = partial_format_bonus
    else:
        # No format tags: penalty
        base_penalty = -0.5 - features.output_len * 0.05

        # Give partial credit for correct answers without format (exploration)
        if features.answer_str is not None:
            answer_score = answer_reward(features.answer_str, gold_answer)
            if answer_score > 0:
                # Correct answer without think tags: small positive reward
                # Still less than partial_format_bonus to encourage format
                return min(answer_score * 0.5, 0.08)  # Cap below partial_format_bonus

        return max(-1.5, base_penalty)

    # Add answer prefix bonus if applicable
    prefix_reward = answer_prefix_bonus if features.has_answer_prefix else 0.0

    return base + format_reward + prefix_reward


def math_reward_for_output(output: RawOutput, gold_answer: str) -> float:
    """Compute reward for a single output."""
    return math_reward(extract_features(output), gold_answer)


# ============================================================================
# Verl Integration Functions
# ============================================================================

def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict = None) -> float:
    """
    Main entry point for verl reward computation.

    This function matches the verl reward function signature and can be used with:
        custom_reward_function.path = "path/to/custom_math_reward.py"
        custom_reward_function.name = "compute_score"

    Args:
        data_source: Dataset identifier (e.g., "deepmind/math_dataset")
        solution_str: Generated solution/response from the model
        ground_truth: Expected correct answer
        extra_info: Optional dict with additional context (e.g., question text)

    Returns:
        float: Reward score (typically between -1.5 and ~2.3)
            - Negative rewards for incorrect format
            - 0-1.5 for numerical accuracy
            - 2.0+ for exact string match with proper format
    """
    return math_reward_for_output(RawOutput(solution_str), ground_truth)


def compute_score_with_details(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None
) -> dict:
    """
    Enhanced version that returns detailed information about the reward computation.

    Usage:
        custom_reward_function.path = "path/to/custom_math_reward.py"
        custom_reward_function.name = "compute_score_with_details"

    Args:
        data_source: Dataset identifier
        solution_str: Generated solution/response from the model
        ground_truth: Expected correct answer
        extra_info: Optional dict with additional context

    Returns:
        dict: Contains 'score' and additional metrics:
            - score: float, the reward value (framework auto-creates "reward" metric from this)
            - acc: float, accuracy (1.0 if correct, 0.0 otherwise)
            - pred: str, predicted answer for majority voting (enables maj@k metrics)
            - has_partial_format: bool, whether output has </think> delimiter
            - has_strict_format: bool, whether output has <think> at start + </think>
            - has_answer_prefix: bool, whether output has "Final answer:" prefix

        Note: The framework automatically creates a "reward" field from "score" for logging.
    """
    features = extract_features(RawOutput(solution_str))
    score = math_reward(features, ground_truth)

    # Determine if answer is correct
    answer_correct = False
    if features.answer_str is not None:
        gold_normalized = ground_truth.lower().strip()
        answer_correct = (features.answer_str == gold_normalized) or (
            features.answer_val is not None and
            parse_num(ground_truth) is not None and
            abs(features.answer_val - parse_num(ground_truth)) < 1e-6
        )

    # Extract predicted answer for majority voting
    # Use the extracted answer string, or a placeholder if extraction failed
    pred_answer = str(features.answer_str) if features.answer_str is not None else "__NO_ANSWER__"

    result = {
        "score": score,   # Main reward value (framework auto-creates "reward" from this)
        "acc": float(answer_correct),  # For validation metrics (treated as core metric)
        "pred": pred_answer,  # Predicted answer for majority voting (enables maj@k metrics)
        "has_partial_format": float(features.has_partial_format),  # Has </think> delimiter (+0.1 bonus)
        "has_strict_format": float(features.has_strict_format),    # <think> first + </think> (+0.3 bonus)
        "has_answer_prefix": float(features.has_answer_prefix),    # Has "Final answer:" prefix (+0.1 bonus)
    }

    return result
