#!/usr/bin/env python3
"""
Convert CodeContests dataset to VERL format for code generation training.

Input: ~/data/cc-transformed (HuggingFace dataset format)
Output: ~/data/cc-transformed-verl/ (train/val/test parquet files)

VERL Format (matches deepmindmath format):
    {
        "messages": [{"role": "user", "content": "..."}],
        "reward_model": {
            "ground_truth": '{"inputs": [...], "outputs": [...]}',
            "style": "rule"
        },
        "extra_info": {
            "reference_solution": "def solve...",
            "problem_name": "...",
            "solution_index": 0,
            "num_visible_tests": 8,
            "num_hidden_tests": 16,
            "total_tests": 24
        },
        "data_source": "codecontests",
        "uid": "codecontests_<problem>_<idx>"
    }
"""

import json
import os
from pathlib import Path

import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm

# Configuration
INPUT_DIR = "/workspace-vast/jinghanj/workspace/Structural_RL/data/cc-transformed"
OUTPUT_DIR = Path("/workspace-vast/jinghanj/workspace/Structural_RL_dev/data/cc-transformed-verl-v2")

# Test configuration (following Code Contests setup)
NUM_VISIBLE_TESTS = 8   # Shown in prompt to help model understand the task
NUM_HIDDEN_TESTS = 16   # Held out for evaluation (prevent reward gaming)

# Data splits
VAL_SIZE = 1000   # For SFT (will get Claude-generated answers)
TEST_SIZE = 1000  # For evaluation
# TRAIN_SIZE will be the rest (~26K)

# Prompt template with visible test cases
PROMPT_TEMPLATE = """You are given a buggy Python solution. Think step by step about the bug, the intended logic, and how to fix it so the solution is correct and robust.

You should make it runnable end-to-end (no exceptions) and ensure it passes the provided assertions as well as hidden test cases.


```python
{buggy_code}
# visible test assertions
{test_assertions}
```

Use `<think>` tags to reason first.

<think>
[your reasoning]
</think>

```python
def solve(lines):
    # your solution
```
"""


def format_test_examples(test_inputs: list[str], test_outputs: list[str]) -> str:
    """Format visible test cases for the prompt."""
    examples = []
    for i, (inp, out) in enumerate(zip(test_inputs, test_outputs), 1):
        # Format input (show first line if multiline)
        input_preview = inp.split('\n')[0] if '\n' in inp else inp
        if len(input_preview) > 50:
            input_preview = input_preview[:50] + "..."

        examples.append(f"Test {i}:\n  Input: {input_preview}\n  Output: {out}")

    return "\n\n".join(examples)


def create_prompt(buggy_solution: str, visible_inputs: list[str], visible_outputs: list[str]) -> str:
    """Create the prompt for fixing buggy code with visible test examples."""
    test_examples = format_test_examples(visible_inputs, visible_outputs)
    return PROMPT_TEMPLATE.format(
        buggy_code=buggy_solution.strip(),
        test_examples=test_examples
    )


def convert_to_verl_format(example: dict) -> dict:
    """
    Convert a single example to VERL format.

    Args:
        example: Dict with keys:
            - name: Problem name
            - solution_index: Solution index
            - transformed_solution: Buggy code to fix
            - test_inputs: List of test input strings
            - test_outputs: List of expected output strings

    Returns:
        Dict in VERL format
    """
    # Extract data
    problem_name = example["name"]
    solution_index = example["solution_index"]
    buggy_solution = example["transformed_solution"]
    test_inputs = example["test_inputs"]
    test_outputs = example["test_outputs"]

    total_tests = len(test_inputs)

    # Split tests: first 8 visible (shown in prompt), last 16 hidden (evaluation only)
    if total_tests >= NUM_VISIBLE_TESTS + NUM_HIDDEN_TESTS:
        # Take first NUM_VISIBLE_TESTS for prompt
        visible_inputs = test_inputs[:NUM_VISIBLE_TESTS]
        visible_outputs = test_outputs[:NUM_VISIBLE_TESTS]

        # Take last NUM_HIDDEN_TESTS for evaluation
        hidden_inputs = test_inputs[-NUM_HIDDEN_TESTS:]
        hidden_outputs = test_outputs[-NUM_HIDDEN_TESTS:]
    else:
        # Not enough tests - use available split
        mid_point = max(1, total_tests // 2)
        visible_inputs = test_inputs[:mid_point]
        visible_outputs = test_outputs[:mid_point]
        hidden_inputs = test_inputs[mid_point:]
        hidden_outputs = test_outputs[mid_point:]

    # Create prompt with visible test cases
    prompt_text = create_prompt(buggy_solution, visible_inputs, visible_outputs)

    # Create ground truth for reward computation
    ground_truth = json.dumps({
        "inputs": hidden_inputs,
        "outputs": hidden_outputs
    })

    # Create VERL format (matching deepmindmath format)
    verl_example = {
        "messages": [{"role": "user", "content": prompt_text}],  # Changed from "prompt" to "messages"
        "reward_model": {
            "ground_truth": ground_truth,  # Changed order to match deepmindmath
            "style": "rule"
        },
        "extra_info": {
            "reference_solution": buggy_solution,  # Original buggy code for comparison
            "problem_name": problem_name,
            "solution_index": solution_index,
            "num_visible_tests": len(visible_inputs),
            "num_hidden_tests": len(hidden_inputs),
            "total_tests": total_tests
        },
        "data_source": "codecontests",
        "uid": f"codecontests_{problem_name}_{solution_index}"  # Add uid field
    }

    return verl_example


def main():
    print("=" * 80)
    print("Converting CodeContests dataset to VERL format")
    print("=" * 80)

    # Load dataset
    print(f"\nLoading dataset from: {INPUT_DIR}")
    dataset = load_from_disk(INPUT_DIR)
    total_samples = len(dataset)
    print(f"Total samples: {total_samples:,}")

    # Convert to VERL format
    print("\nConverting to VERL format...")
    print(f"  Visible tests (shown in prompt): {NUM_VISIBLE_TESTS}")
    print(f"  Hidden tests (evaluation only): {NUM_HIDDEN_TESTS}")
    verl_data = []
    for example in tqdm(dataset, desc="Converting"):
        verl_example = convert_to_verl_format(example)
        verl_data.append(verl_example)

    print(f"✓ Converted {len(verl_data):,} examples")

    # Create splits
    print("\nCreating train/val/test splits...")

    # Shuffle with fixed seed for reproducibility
    import random
    random.seed(42)
    indices = list(range(len(verl_data)))
    random.shuffle(indices)

    # Split indices: val (for SFT), test (for eval), train (for RL)
    val_indices = indices[:VAL_SIZE]
    test_indices = indices[VAL_SIZE:VAL_SIZE + TEST_SIZE]
    train_indices = indices[VAL_SIZE + TEST_SIZE:]

    # Create split datasets
    val_data = [verl_data[i] for i in val_indices]
    test_data = [verl_data[i] for i in test_indices]
    train_data = [verl_data[i] for i in train_indices]

    print(f"  Val (for SFT):  {len(val_data):,} samples")
    print(f"  Test (for eval): {len(test_data):,} samples")
    print(f"  Train (for RL):  {len(train_data):,} samples")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to: {OUTPUT_DIR}")

    # Convert to pandas and save as parquet
    # Keep structured types (like deepmindmath reference) - pandas will handle conversion
    for split_name, split_data in [("val", val_data), ("test", test_data), ("train", train_data)]:
        df = pd.DataFrame(split_data)
        output_path = OUTPUT_DIR / f"{split_name}.parquet"
        df.to_parquet(output_path, index=False, engine='pyarrow')
        print(f"  ✓ Saved {split_name}.parquet ({len(df):,} samples, {output_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Print example
    print("\n" + "=" * 80)
    print("Example (first training sample):")
    print("=" * 80)
    example = train_data[0]
    print(f"\nData source: {example['data_source']}")
    print(f"UID: {example['uid']}")
    print(f"\nMessages (first 800 chars):")
    print(example['messages'][0]['content'][:800] + "...")
    print(f"\nReward model (hidden tests only):")
    print(f"  Style: {example['reward_model']['style']}")
    gt = json.loads(example['reward_model']['ground_truth'])
    print(f"  Hidden tests for evaluation: {len(gt['inputs'])} test cases")
    print(f"  First hidden test input (first 100 chars): {gt['inputs'][0][:100]}...")
    print(f"  First hidden test output: {gt['outputs'][0]}")
    print(f"\nExtra info:")
    for key, value in example['extra_info'].items():
        if key == "reference_solution":
            print(f"  {key}: {value[:100]}...")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 80)
    print("✓ Dataset conversion complete!")
    print("=" * 80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nFiles created:")
    print(f"  - val.parquet ({len(val_data):,} samples) - For SFT with Claude-generated answers")
    print(f"  - test.parquet ({len(test_data):,} samples) - For evaluation")
    print(f"  - train.parquet ({len(train_data):,} samples) - For RL training")
    print("\nNext steps:")
    print("1. Generate Claude answers for val.parquet using generate_sft_answers.py")
    print("2. Run SFT on Gemma3-4b with val.parquet to learn <think></think> format")
    print("3. Use SFT model for RL training with train.parquet")
    print(f"\nFor RL training:")
    print(f"  data.train_files=\"['{OUTPUT_DIR}/train.parquet']\"")
    print(f"  data.val_files=\"['{OUTPUT_DIR}/test.parquet']\"")
    print(f"  custom_reward_function.path=/workspace-vast/jinghanj/workspace/Structural_RL/train/custom_code_reward.py")
    print(f"  custom_reward_function.name=compute_score_with_details")


if __name__ == "__main__":
    main()
