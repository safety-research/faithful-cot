"""
Verl-compatible reward for DAPO math training — v2.

Changes vs v1 (custom_math_reward_dapo.py):
  - Answer extraction is now aligned with eval (evaluate_dapo_math_hints.py):
      * If </think> is present, extract answer ONLY from the region after </think>
      * If </think> is absent, fall back to the full string
      * Then take the last 300 chars of that region (same as before)
    Previously, the last 300 chars were taken from the full string regardless,
    meaning a model could get correctness credit for "Answer: X" inside CoT.

Score (RL signal): Minerva-style extraction ("Answer: X" with normalization) + format bonus
    - correct answer + strict <think>...</think>: +1.2
    - correct answer + partial </think>:          +1.1
    - correct answer + no format:                 +0.5
    - wrong answer   + strict format:             -0.8
    - wrong answer   + no format:                 -1.5

Acc (monitoring): same extraction, aligned with eval.

Format bonus:
    - strict (<think> at start + </think>): +0.2
    - partial (</think> only):              +0.1
    - no tags:                              -0.5
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
    return -0.5          # no format at all


def _answer_region(solution_str: str) -> str:
    """
    Extract the region to search for the answer.

    If </think> is present, use only the text after the FIRST </think>.
    Otherwise fall back to the full string.
    Then take the last 300 chars (safe: no-op when string is shorter).
    """
    if "</think>" in solution_str:
        region = solution_str.split("</think>", 1)[1]
    else:
        region = solution_str
    return region[-300:]


def compute_score_with_details(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
) -> dict:
    region = _answer_region(solution_str)
    # _compute_score internally does [-300:] again — idempotent since region is already ≤ 300 chars
    result = _compute_score(region, ground_truth, strict_box_verify=False)
    fmt    = _format_bonus(solution_str)   # always checked on the full string

    result["score"]        += fmt
    result["format_bonus"]  = fmt
    result["acc"]           = float(result["acc"])
    result["pred"]          = result["pred"] or "[INVALID]"

    return result
