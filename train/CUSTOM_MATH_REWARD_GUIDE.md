# Custom Math Reward Function Integration Guide

This guide explains how to use the custom math reward function with verl's PPO training pipeline.

## Overview

The custom math reward function (`custom_math_reward.py`) is designed for math tasks from DeepMind's math_dataset (arithmetic__mul_div_multiple). It rewards models for:

1. **Proper formatting**: Using `<scratchpad>` and `Final answer:` prefix
2. **Answer correctness**: String matching or numerical accuracy
3. **Penalizes**: Incorrect format or wrong answers

### Reward Structure

- **String exact match**: Up to 2.0 points
- **Numerical accuracy**: Up to 1.5 points
- **Format bonus**: +0.2 for scratchpad usage
- **Prefix bonus**: +0.1 for "Final answer:" prefix
- **Format penalty**: -0.5 to -1.5 for missing scratchpad

## Expected Output Format

The reward function expects model outputs in this format:

```
<scratchpad>
[Step-by-step reasoning here]
</scratchpad>
Final answer: [numerical answer]
```

### Example

```
<scratchpad>
First, multiply 5 by 3: 5 * 3 = 15
Then divide by 5: 15 / 5 = 3
</scratchpad>
Final answer: 3
```

## Integration Methods

### Method 1: Using Configuration File

Edit your PPO training config (e.g., `verl/trainer/config/ppo_trainer.yaml`):

```yaml
custom_reward_function:
  path: "/workspace-vast/jinghanj/workspace/verl/custom_math_reward.py"
  name: "compute_score"  # or "compute_score_with_details" for extra metrics
```

### Method 2: Using Command Line Arguments

When running training, add these parameters:

```bash
python3 -m verl.trainer.main_ppo \
    custom_reward_function.path="/path/to/custom_math_reward.py" \
    custom_reward_function.name="compute_score" \
    [other training arguments...]
```

### Method 3: Complete Training Script Example

Here's a complete example based on verl's PPO trainer:

```bash
#!/usr/bin/env bash
set -xeuo pipefail

NUM_GPUS=${NUM_GPUS:-8}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/Qwen/Qwen2.5-0.5B}
TRAIN_FILES=${TRAIN_FILES:-$HOME/data/math/train.parquet}
VAL_FILES=${VAL_FILES:-$HOME/data/math/test.parquet}

# Batch size configuration
train_traj_micro_bsz_per_gpu=2
n_resp_per_prompt=4
train_traj_micro_bsz=$((train_traj_micro_bsz_per_gpu * NUM_GPUS))
train_traj_mini_bsz=$((train_traj_micro_bsz * 2))
train_prompt_mini_bsz=$((train_traj_mini_bsz * n_resp_per_prompt))
train_prompt_bsz=$((train_prompt_mini_bsz * 2))

exp_name="math-custom-reward-training"
CUSTOM_REWARD_PATH="/workspace-vast/jinghanj/workspace/verl/custom_math_reward.py"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator="gae" \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size="${train_prompt_bsz}" \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.rollout.name="vllm" \
    actor_rollout_ref.rollout.mode="async" \
    critic.optim.lr=1e-5 \
    critic.model.path="${MODEL_PATH}" \
    critic.ppo_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    custom_reward_function.path="${CUSTOM_REWARD_PATH}" \
    custom_reward_function.name="compute_score" \
    trainer.logger=wandb \
    trainer.project_name='verl-custom-math-reward' \
    trainer.experiment_name="${exp_name}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.total_epochs=10 \
    trainer.device=cuda
```

## Available Functions

### 1. `compute_score` (Basic)

**Signature:**
```python
def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict = None) -> float
```

**Returns:** Single float reward value

**Usage:**
```yaml
custom_reward_function:
  name: "compute_score"
```

### 2. `compute_score_with_details` (Enhanced)

**Signature:**
```python
def compute_score_with_details(data_source: str, solution_str: str, ground_truth: str, extra_info: dict = None) -> dict
```

**Returns:** Dictionary with detailed metrics:
- `score`: float - The reward value
- `has_scratchpad`: bool - Whether output uses scratchpad format
- `has_answer_prefix`: bool - Whether output has "Final answer:" prefix
- `answer_extracted`: str | None - The extracted answer
- `answer_correct`: bool - Whether answer matches ground truth
- `output_length`: int - Length of the output

**Usage:**
```yaml
custom_reward_function:
  name: "compute_score_with_details"
```

This version is useful for tracking detailed metrics during training.

## Data Format Requirements

Your training data (parquet files) should include:

1. **question**: The math problem
2. **ground_truth**: The correct numerical answer
3. **data_source**: Dataset identifier (optional, can be "math_dataset")

Example parquet structure:

| question | ground_truth | data_source |
|----------|-------------|-------------|
| "What is 5 * 3 / 5?" | "3" | "math_dataset" |
| "Calculate 10 / 2" | "5" | "math_dataset" |

## Testing Your Integration

### Step 1: Test the reward function directly

```bash
cd /workspace-vast/jinghanj/workspace/verl
python3 test_custom_math_reward.py
```

This will show you how different output formats are scored.

### Step 2: Verify in training

Check your training logs for reward values. You should see:
- Positive rewards (0.2 to 2.3) for properly formatted outputs
- Negative rewards (-1.5 to -0.5) for outputs without scratchpad

### Step 3: Monitor key metrics

- **Average reward**: Should increase over training
- **Accuracy**: Percentage of correct answers
- **Format compliance**: Percentage using proper format

## Prompt Engineering

To get your model to output in the expected format, use a system prompt like:

```
You are a helpful math assistant. For each problem:
1. Show your work in a <scratchpad> section
2. After the </scratchpad> tag, write "Final answer: " followed by the numerical result

Example:
<scratchpad>
[Your step-by-step reasoning]
</scratchpad>
Final answer: [numerical answer]
```

## Customization

### Adjust Reward Parameters

Edit these values in `custom_math_reward.py`:

```python
string_peak = 2.0              # Max reward for exact string match
numeric_peak = 1.5             # Max reward for numerical accuracy
format_bonus = 0.2             # Bonus for using scratchpad
```

### Modify Answer Extraction

If your dataset uses different format markers, modify:

```python
ANSWER_PREFIX = "Final answer: "  # Change to your prefix
```

For different scratchpad markers:

```python
def mk_answer_region(s: str) -> AnswerRegion | None:
    # Current: looks for </scratchpad>
    # Modify to look for your custom markers
    return clean_answer_region(s.split("</scratchpad>")[-1]) if s.count("</scratchpad>") == 1 else None
```

## Troubleshooting

### Issue: ImportError for `lc.util.std`

**Solution:** The `bind_maybe` function is a simple utility. If you don't have this dependency, replace:

```python
from lc.util.std import bind_maybe
```

with:

```python
def bind_maybe(func, value):
    """Apply func to value if value is not None, else return None."""
    return func(value) if value is not None else None
```

### Issue: All rewards are negative

**Possible causes:**
1. Model not outputting scratchpad format
2. Wrong answer prefix
3. Check your prompt template

**Solution:** Add format instructions to your prompt template and verify with `test_custom_math_reward.py`

### Issue: Rewards not updating during training

**Check:**
1. Verify custom_reward_function.path points to the correct file
2. Check that the file path is absolute, not relative
3. Look for import errors in training logs

## Advanced Usage

### Using with Reward Manager

If you need custom reward manager logic, you can extend the reward function:

```python
def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict = None) -> dict:
    base_score = math_reward_for_output(RawOutput(solution_str), ground_truth)

    # Add custom logic based on extra_info
    if extra_info and "difficulty" in extra_info:
        difficulty_bonus = 0.1 * extra_info["difficulty"]
        base_score += difficulty_bonus

    return {
        "score": base_score,
        "difficulty_bonus": difficulty_bonus if extra_info else 0.0
    }
```

### Hint Analysis (Optional)

The reward function includes optional hint analysis functions for studying model behavior on problems with misleading hints:

- `hint_following_ratio()`: Measures if model follows wrong hints
- `hint_explicit_mention_ratio()`: Checks if model explicitly mentions hints
- `hint_implicit_mention_ratio()`: Checks if hint values appear in reasoning

These are useful for analysis but not used in the main reward computation.

## References

- [verl Reward Function Documentation](docs/preparation/reward_function.rst)
- [verl PPO Trainer](verl/trainer/main_ppo.py)
- [Example Reward Functions](verl/utils/reward_score/)
- DeepMind Math Dataset: `datasets.load_dataset("deepmind/math_dataset", "arithmetic__mul_div_multiple")`
