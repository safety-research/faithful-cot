"""
Curriculum-based reward function for teaching think tags.

Gradually increases format requirements:
  Steps 0-100: Reward correctness only
  Steps 100-300: Slight bonus for think tags
  Steps 300+: Full format requirement
"""

from custom_math_reward import (
    OutputFeatures, extract_features, answer_reward, parse_num,
    RawOutput, format_bonus
)


def math_reward_with_curriculum(
    features: OutputFeatures,
    gold_answer: str,
    training_step: int = 0,
) -> float:
    """
    Curriculum-based reward that gradually teaches think tag usage.

    Args:
        features: Extracted output features
        gold_answer: Ground truth answer
        training_step: Current training step (for curriculum)

    Returns:
        Reward value
    """

    # Phase 1 (steps 0-100): Focus on correctness
    if training_step < 100:
        if features.answer_str is None:
            return -1.0
        return answer_reward(features.answer_str, gold_answer)

    # Phase 2 (steps 100-300): Introduce format bonus
    elif training_step < 300:
        if features.answer_str is None:
            base = -1.0
        else:
            base = answer_reward(features.answer_str, gold_answer)

        if features.has_think:
            # Small bonus for using think tags
            return base + 0.3
        else:
            # Small penalty for not using them (but not too harsh)
            return base - 0.2

    # Phase 3 (steps 300+): Full format requirement
    else:
        if features.has_think:
            base = 0 if features.answer_str is None else answer_reward(features.answer_str, gold_answer)
            return base + format_bonus + (0.1 if features.has_answer_prefix else 0.0)
        else:
            # Harsh penalty for missing format
            return -1.5


def compute_score_with_curriculum(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None
) -> float:
    """
    Verl integration with curriculum learning.

    Usage in training config:
        custom_reward_function.path = "path/to/custom_math_reward_curriculum.py"
        custom_reward_function.name = "compute_score_with_curriculum"

    Note: training_step should be passed in extra_info dict
    """
    training_step = extra_info.get("training_step", 0) if extra_info else 0
    features = extract_features(RawOutput(solution_str))
    return math_reward_with_curriculum(features, ground_truth, training_step)
