#!/usr/bin/env python3
"""
Test script for custom math reward function.
Demonstrates how the reward function evaluates different types of outputs.
"""

from custom_math_reward import compute_score, compute_score_with_details


def test_reward_function():
    """Test various output formats and reward computations."""

    # Test case 1: Perfect format with correct answer
    print("=" * 80)
    print("Test 1: Perfect format with correct answer")
    print("=" * 80)
    output1 = """<scratchpad>
Let me solve this step by step:
5 * 3 = 15
15 / 5 = 3
</scratchpad>
Final answer: 3"""
    ground_truth1 = "3"

    score1 = compute_score("math_dataset", output1, ground_truth1)
    details1 = compute_score_with_details("math_dataset", output1, ground_truth1)

    print(f"Output: {output1[:50]}...")
    print(f"Ground truth: {ground_truth1}")
    print(f"Score: {score1}")
    print(f"Details: {details1}")
    print()

    # Test case 2: Correct answer without scratchpad (penalty)
    print("=" * 80)
    print("Test 2: Correct answer without scratchpad (should receive penalty)")
    print("=" * 80)
    output2 = "The answer is 3"
    ground_truth2 = "3"

    score2 = compute_score("math_dataset", output2, ground_truth2)
    details2 = compute_score_with_details("math_dataset", output2, ground_truth2)

    print(f"Output: {output2}")
    print(f"Ground truth: {ground_truth2}")
    print(f"Score: {score2}")
    print(f"Details: {details2}")
    print()

    # Test case 3: Fraction answer
    print("=" * 80)
    print("Test 3: Fraction answer")
    print("=" * 80)
    output3 = """<scratchpad>
1 / 2 = 0.5
</scratchpad>
Final answer: 1/2"""
    ground_truth3 = "1/2"

    score3 = compute_score("math_dataset", output3, ground_truth3)
    details3 = compute_score_with_details("math_dataset", output3, ground_truth3)

    print(f"Output: {output3[:50]}...")
    print(f"Ground truth: {ground_truth3}")
    print(f"Score: {score3}")
    print(f"Details: {details3}")
    print()

    # Test case 4: Wrong answer
    print("=" * 80)
    print("Test 4: Wrong answer with proper format")
    print("=" * 80)
    output4 = """<scratchpad>
5 * 3 = 15
</scratchpad>
Final answer: 15"""
    ground_truth4 = "3"

    score4 = compute_score("math_dataset", output4, ground_truth4)
    details4 = compute_score_with_details("math_dataset", output4, ground_truth4)

    print(f"Output: {output4[:50]}...")
    print(f"Ground truth: {ground_truth4}")
    print(f"Score: {score4}")
    print(f"Details: {details4}")
    print()

    # Test case 5: Nearly correct numerical answer
    print("=" * 80)
    print("Test 5: Nearly correct numerical answer")
    print("=" * 80)
    output5 = """<scratchpad>
Approximately...
</scratchpad>
Final answer: 3.001"""
    ground_truth5 = "3.0"

    score5 = compute_score("math_dataset", output5, ground_truth5)
    details5 = compute_score_with_details("math_dataset", output5, ground_truth5)

    print(f"Output: {output5[:50]}...")
    print(f"Ground truth: {ground_truth5}")
    print(f"Score: {score5}")
    print(f"Details: {details5}")
    print()

    # Test case 6: Missing answer prefix
    print("=" * 80)
    print("Test 6: Proper scratchpad but missing 'Final answer:' prefix")
    print("=" * 80)
    output6 = """<scratchpad>
5 * 3 = 15
15 / 5 = 3
</scratchpad>
The answer is 3"""
    ground_truth6 = "3"

    score6 = compute_score("math_dataset", output6, ground_truth6)
    details6 = compute_score_with_details("math_dataset", output6, ground_truth6)

    print(f"Output: {output6[:50]}...")
    print(f"Ground truth: {ground_truth6}")
    print(f"Score: {score6}")
    print(f"Details: {details6}")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Test 1 (perfect format + correct): {score1:.3f}")
    print(f"Test 2 (no scratchpad): {score2:.3f}")
    print(f"Test 3 (fraction): {score3:.3f}")
    print(f"Test 4 (wrong answer): {score4:.3f}")
    print(f"Test 5 (nearly correct): {score5:.3f}")
    print(f"Test 6 (missing prefix): {score6:.3f}")
    print("\nExpected behavior:")
    print("- Best score: Test 1 (proper format + correct answer)")
    print("- Negative score: Test 2 (no scratchpad format)")
    print("- Lower score: Test 4 (wrong answer despite proper format)")


if __name__ == "__main__":
    test_reward_function()
