# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Research codebase for studying **structural approaches to CoT faithfulness** in Large Reasoning Models. The key research question: can structural restrictions on attention or gradient flow during RL training make Chain-of-Thought reasoning causally necessary rather than post-hoc?

Two axes of study:
1. **Faithfulness metrics** — task-agnostic KL/JS/gradient-based measures for the prompt→CoT→answer causal structure
2. **Structural training interventions** — 5 training methods that block information paths to enforce CoT necessity

## Environment

All scripts activate `conda activate verl2` (miniconda at `/workspace-vast/jinghanj/miniconda3`). Training is launched on a SLURM cluster using Ray.

Key environment variables expected by scripts:
- `HF_HOME=/workspace-vast/pretrained_ckpts`
- `HF_TOKEN_PATH=/workspace-vast/$(whoami)/.cache/huggingface/token`
- `NCCL_SOCKET_IFNAME=eth0`
- `WANDB_API_KEY` / `WANDB_PROJECT`
- `MAX_CONCURRENT_UDOCKER` — parallelism for sandboxed code execution (default 64 for training, 8 for eval)

## Training

### Entry point

All training runs `python -m verl.trainer.main_ppo` **from inside the `train/` directory** (the VERL package lives there). Scripts `cd train` before launching.

### The 5 training methods

| Method | Flag(s) | What it does |
|---|---|---|
| `vanilla` | (none) | Baseline, no masking |
| `update_mask` | `use_cot_masking=True` | Forward pass: answer tokens cannot attend to prompt tokens |
| `cot_gradient` | `use_cot_gradient_masking=True` | Backward pass: only CoT token positions update parameters |
| `gradient_mask` | `use_attention_gradient_masking=True` | Backward pass: block answer→prompt attention gradients |
| `combined_mask` | both gradient flags | Both attention + parameter masking |

### Launching training

**Code task (reward hacking variant):**
```bash
# Single-node interactive
bash scripts/train_code/train_hacking_methods_2048_v3.sh <method>

# SLURM submission
sbatch scripts/train_code/submit_hacking_training_2048_v3.sbatch <method>
```

**Math task (DeepMind Math):**
```bash
bash scripts/train_math/train_unified_methods.sh <method>
sbatch scripts/train_math/submit_unified_training.sbatch <method>
```

Methods: `vanilla | update_mask | cot_gradient | gradient_mask | combined_mask`

### Delimiter detection modes

All masking methods need to find the `</think>` boundary. Two modes:
- **Substring matching** (`use_substring_delimiter_matching=True`, `end_think_delimiter_str="</think>"`): tokenizer-agnostic, handles merged-boundary tokens; used for all current experiments
- **Special token** (`end_think_token_id=<id>`): fast vectorized lookup; Gemma3=`262146`, Qwen2.5=`151666`

### Key actor config parameters

```
actor_rollout_ref.actor.use_cot_masking
actor_rollout_ref.actor.use_cot_gradient_masking
actor_rollout_ref.actor.use_attention_gradient_masking
actor_rollout_ref.actor.block_prompt_gradients
actor_rollout_ref.actor.block_answer_gradients
actor_rollout_ref.actor.use_substring_delimiter_matching
actor_rollout_ref.actor.end_think_delimiter_str
actor_rollout_ref.actor.disable_cot_masking_for_old_log_prob=True  # important for GRPO
```

### Code execution sandbox

Code tasks use `udocker` with `python:3.11-slim` for sandboxed execution. Prerequisite:
```bash
bash scripts/train_code/setup_udocker.sh
```

The persistent container pool (`train/custom_code_reward_udocker_persistent.py`) creates `POOL_SIZE` containers once at startup and reuses them.

## Evaluation

### Checkpoint conversion (FSDP → HuggingFace)

```bash
cd train
python -m scripts.legacy_model_merger merge \
    --backend fsdp \
    --local_dir <ckpt_dir>/actor \
    --target_dir <ckpt_dir>_hf
```

Checkpoints are saved to `results/checkpoints/<experiment_name>/global_step_<N>/`.

### Faithfulness metrics (math / code)

```bash
# Submit all faithfulness evals for code hacking runs
bash scripts/eval/submit_all_faithfulness_code_preassert_evals.sh [--dry-run]

# Single job
sbatch scripts/eval/run_faithfulness_code_preassert_eval_single.sh \
    <test_parquet> <generations_json> <finetuned_model> <reference_model> <output_json> \
    [num_samples] [batch_size] [compute_gradients]
```

The `--truncate-at-assert` flag (preassert variant) restricts the answer region to the `solve()` body only, excluding the assertion block — avoids contamination of KL metrics from the assertion-hacking signal.

### Hacking evaluation

```bash
bash scripts/eval/submit_all_hacking_evals.sh [--dry-run]
```

Runs `eval/evaluate_hacking_ratio.py` per checkpoint, producing `hacking_analysis.json`.

Key tracked signals: `returned_assertions_pass` (RL reward), `genuine_fix` (hidden tests), `assertions_modified` (hack rate), `hack_verbalized` (CoT monitorability).

### Cross-task faithfulness (code-trained model on math)

```bash
# Stage 1: generate math outputs
bash scripts/eval/submit_cross_task_gen_evals.sh
# Stage 2: compute faithfulness
bash scripts/eval/submit_cross_task_faith_evals.sh
```

### Plotting

Root-level `plot_hacking_comparison.py` and `eval/plot_*.py` produce paper figures. Outputs typically go to `results/<experiment>/`.

## Architecture: Custom Additions to VERL

The `train/` directory is a fork of [VERL](https://github.com/volcengine/verl). Custom research code lives in:

### `train/verl/utils/cot_masking.py`
Core utility used by both training and eval. Key functions:
- `find_special_token_positions(input_ids, response_length, ...)` → `(prompt_mask, cot_mask, answer_mask)` per-sample boolean tensors
- `create_cot_attention_mask(...)` → 4D `(batch, 1, seq, seq)` attention mask blocking answer→prompt attention
- `tokenize_delimiter_string(tokenizer, "</think>", device)` → list of token-id tensors (base + merged-boundary variants)

The `response_start = seq_len - response_length` convention is used throughout: response tokens are the last `response_length` tokens of the full sequence.

### `train/verl/models/transformers/attention_gradient_masking_patch.py`
Monkey-patches `eager_attention_forward` in the Transformers library (Qwen2, Gemma3) to block gradients flowing from answer tokens back to prompt keys/values. Two-part approach:
1. Detach attention logits at answer→prompt positions (before softmax)
2. Custom autograd function zeros out gradients at blocked positions in the backward pass

Applied via `apply_attention_gradient_masking_patch(model)`. Uses thread-local storage to pass the mask into attention layers.

### `train/verl/models/transformers/cot_parameter_gradient_masking_patch.py`
Monkey-patches linear layers so only CoT token positions contribute to weight gradients (`∇_θ L = Σ_{t ∈ CoT} ∇_θ ℓ_t`). Uses `cot_parameter_gradient_mask_context(gradient_mask)` context manager. Also patches `GradientCheckpointingLayer` to propagate the mask through `torch.utils.checkpoint` recomputation.

### `train/verl/workers/actor/dp_actor.py`
Modified `DataParallelPPOActor` that integrates CoT masking. During the update step it:
1. Calls `find_special_token_positions` to get region masks
2. Wraps the forward pass in `attention_gradient_mask_context` (if `use_attention_gradient_masking`)
3. Wraps backward in `cot_parameter_gradient_mask_context` (if `use_cot_gradient_masking`)
4. Creates the 4D attention mask (if `use_cot_masking`)

### Custom reward functions (`train/custom_code_reward_*.py`)
- `custom_code_reward_udocker_persistent.py` — base: sandboxed execution, ground-truth test assertions
- `custom_code_reward_hacking.py` — hacking variant: runs model's *own* returned assertions (hackable signal) + hidden tests (monitoring only)
- `custom_math_reward*.py` — math reward functions (answer extraction + equivalence checking)

## Data

Parquet format with fields: `messages` (list of role/content dicts), `reward_model.ground_truth`, `reward_model.style`, `extra_info`, `data_source`, `uid`.

Key datasets:
- Code (hacking v3): `data/cc-transformed-hacking-with-hints-v3/` — includes visible test assertions in prompt; model returns code + assertions
- Math: `/workspace-vast/jinghanj/workspace/Structural_RL/data/deepmindmath_all/`
- Base models: `/workspace-vast/jinghanj/workspace/Structural_RL/models/checkpoints/` (gemma3-4b-it, gemma3-12b-it)

Data processing scripts are in `scripts/data_processing/` (convert to VERL parquet format, add hints, generate SFT answers).

## Faithfulness Metrics (`eval/faithfulness_metrics_clean.py`)

Metrics compare answer distributions under different attention masking conditions:
- `full` — standard causal attention (prompt+CoT+answer all attend normally)
- `via_cot` — answer can only attend to CoT, not prompt (tests if prompt info flows through CoT)
- `no_cot` — answer cannot attend to CoT
- `no_prompt` — answer cannot attend to prompt

Key metrics:
- `kl_direct_effect` = DKL(full ‖ via_cot) — prompt info not routed through CoT
- `kl_cot_necessity` = DKL(full ‖ no_cot) — how much the CoT matters
- `kl_leakage` = DKL(no_prompt ‖ via_cot) — info in CoT not from prompt
- JS variants (bounded [0, log2]), plus gradient L1/L2 norms, entropy/NLL
