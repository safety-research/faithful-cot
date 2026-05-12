"""
Verl-compatible reward for DAPO math training.

Score (RL signal): Minerva-style extraction ("Answer: X" with normalization) + format bonus
    - correct answer + strict <think>...</think>: +1.2
    - correct answer + partial </think>:          +1.1
    - correct answer + strict <think>...</think>: +1.2
    - correct answer + partial </think>:          +1.1
    - correct answer + no format:                 +0.5
    - wrong answer   + strict format:             -0.8
    - wrong answer   + no format:                 -1.5

Acc (monitoring): same Minerva extraction, non-zero from step 0.

Format bonus:
    - strict (<think> at start + </think>): +0.2
    - partial (</think> only):              +0.1
    - no tags:                              -0.5 (flat, no length component)
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "verl/utils/reward_score"))

from math_dapo import compute_score as _compute_score


def _format_bonus(solution_str: str) -> float:
    has_end   = solution_str.count("</think>") == 1
    has_start = solution_str.count("<think>")  == 1

    if has_start and has_end:
        stripped = solution_str.lstrip()
        if stripped.startswith("<think>") and solution_str.find("<think>") < solution_str.find("</think>"):
            return 0.2   # strict: <think> at start, </think> after
    if has_end:
        return 0.1       # partial: at least </think> present
    return -0.5  # no format at all — flat penalty, no length component


def compute_score_with_details(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
) -> dict:
    # Minerva extraction: matches "Answer: X" with LaTeX normalization
    result = _compute_score(solution_str, ground_truth, strict_box_verify=False)
    fmt    = _format_bonus(solution_str)

    result["score"]        += fmt
    result["format_bonus"]  = fmt
    result["acc"]           = float(result["acc"])           # bool → float
    result["pred"]          = result["pred"] or "[INVALID]"  # None → string

    return result
