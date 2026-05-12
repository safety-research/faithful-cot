# Structural RL: Enforcing CoT Faithfulness via Structural Training Interventions

This repository studies whether **structural restrictions on attention or gradient flow** during RL training can make Chain-of-Thought (CoT) reasoning causally necessary rather than post-hoc rationalization.

**Core question:** Can we train large reasoning models so that the CoT genuinely mediates the model's answer, rather than being a plausible-sounding post-hoc justification?

## Overview

We study five structural training interventions and one adversarial baseline across two task families:

| Method | What it does |
|---|---|
| `vanilla` | Standard GRPO/DAPO baseline — no structural restriction |
| `update_mask` | Forward masking: answer tokens cannot attend to prompt (forces CoT to carry prompt information) |
| `cot_gradient` | Backward masking: only CoT token positions update model parameters |
| `gradient_mask` | Backward masking: blocks gradients from answer tokens back to prompt keys/values |
| `FACT` (pep) | Adversarial: perturbs prompt embeddings at a fixed layer during training, forcing the CoT to compensate |

**Task families:**
- **Code (reward hacking):** CodeContests problems where the model returns code + test assertions and is rewarded on its own assertions passing. This deliberately creates a hackable signal; we study how each method affects hack rate, genuine fix rate, and CoT monitorability.
- **Math (DeepMind Math):** Standard math reasoning with GRPO on Gemma3-1B.
- **DAPO Math:** DAPO training on Qwen2.5-7B-Instruct with the DAPO-Math-17k dataset.

---

## Environment

All scripts use the `verl2` conda environment:

```bash
conda activate verl2
```

Key environment variables (set automatically by training scripts):
```bash
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_TOKEN_PATH=/workspace-vast/$(whoami)/.cache/huggingface/token
export NCCL_SOCKET_IFNAME=eth0
export WANDB_API_KEY=<your_key>
export WANDB_PROJECT=<project_name>
```

The training entry point is `python -m verl.trainer.main_ppo`, run from **inside the `train/` directory** (the VERL package lives there).

### Code task: sandbox setup

Code evaluation requires `udocker` with a `python:3.11-slim` container. Run once before any code training:

```bash
bash scripts/train_math/setup_udocker.sh
```

To pre-create a pool of reusable containers (faster training):

```bash
python scripts/train_math/create_container_pool.py
```

---

## Training

### Code task — reward hacking (Gemma3-4B or 12B)

```bash
# Interactive (4 GPUs, single node)
bash scripts/train_code/train_hacking_methods_2048_v3.sh <method>

# SLURM submission
sbatch scripts/train_code/submit_hacking_training_2048_v3.sbatch <method>
```

For the 12B model:
```bash
bash scripts/train_code/train_hacking_methods_2048_v3_12b.sh <method>
sbatch scripts/train_code/submit_hacking_training_2048_v3_12b.sbatch <method>
```

For FACT (adversarial PEP method):
```bash
# Default: worst_case mode, layer=-1 (embed_tokens), ε=0.05
bash scripts/train_code/train_hacking_methods_2048_v3_fact.sh pep

# Override hyperparameters via env vars
PEP_LAYER=8 PEP_EPSILON=0.1 bash scripts/train_code/train_hacking_methods_2048_v3_fact.sh pep

# SLURM
sbatch scripts/train_code/submit_hacking_training_2048_v3_fact.sbatch pep
```

**Available methods:** `vanilla | update_mask | cot_gradient | gradient_mask | pep`

**Key hyperparameters (code task):**
- Model: `gemma3-4b-it` / `gemma3-12b-it`
- Prompt length: 768 tokens, response length: 2048 tokens
- Train batch size: 512, rollout n: 4
- Optimizer: AdamW, lr=1e-6, cosine schedule, warmup ratio=0.05
- Data: `data/cc-transformed-hacking-with-hints-v3/`

---

### Math task (Gemma3-1B, DeepMind Math)

```bash
bash scripts/train_math/train_unified_methods.sh <method>
sbatch scripts/train_math/submit_unified_training.sbatch <method>
```

**Available methods:** `vanilla | update_mask | cot_gradient | gradient_mask`

**Key hyperparameters (math task):**
- Model: `gemma3-1b` (format-finetuned checkpoint)
- Train batch size: 512
- Data: `data/deepmindmath_all/`

---

### DAPO Math (Qwen2.5-7B-Instruct)

```bash
# Interactive
bash scripts/train_dapo_math/train_dapo_math_methods_qwen25_7b_v7.sh <method> [lr] [data_variant]

# SLURM
sbatch scripts/train_dapo_math/submit_dapo_math_training_qwen25_7b_v7.sbatch <method>
```

**Available methods:** `vanilla | update_mask | cot_gradient | gradient_mask`

**Key hyperparameters (DAPO math):**
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Prompt length: 1024 tokens
- Response length: 4096 (vanilla/update_mask/gradient_mask), 10000 (cot_gradient — longer CoT budget)
- Train batch size: 512, rollout n: 8
- Optimizer: AdamW, lr=1e-6 (default), DAPO reward manager
- Data: `data/dapo_math/` (DAPO-Math-17k, no hints)
- Note: `</think>` is not a special token in Qwen2.5 — substring matching is always used

---

### Checkpoint location

Checkpoints are saved to:
```
results/checkpoints/<experiment_name>/global_step_<N>/
```

Experiment name format: `<model>-<task>-<method>-bs<batch>-len<resp>-seed<seed>`

Example: `gemma3-4b-hacking-v3-gradient_mask-bs512-len2048-seed1997`

---

## Checkpoint Conversion (FSDP → HuggingFace)

Evaluation scripts require HuggingFace format. Convert with:

```bash
cd train
python -m scripts.legacy_model_merger merge \
    --backend fsdp \
    --local_dir <ckpt_dir>/actor \
    --target_dir <ckpt_dir>_hf
```

This produces a `<ckpt_dir>_hf/` directory readable by `transformers`.

---

## Evaluation

### 1. Faithfulness metrics (code task)

Measures how much the CoT causally mediates the answer by comparing token distributions under different attention masking conditions. The `--truncate-at-assert` flag restricts the answer region to the `solve()` function body, avoiding contamination from the assertion block.

**4B model:**
```bash
sbatch scripts/eval/run_faithfulness_code_preassert_eval_single.sh \
    <test_parquet> \
    <generations_json> \
    <finetuned_model_hf> \
    <reference_model_hf> \
    <output_json> \
    [num_samples] [batch_size] [compute_gradients]
```

**12B model:**
```bash
sbatch scripts/eval/run_faithfulness_code_preassert_eval_single_12b.sh \
    <test_parquet> \
    <generations_json> \
    <finetuned_model_hf> \
    <reference_model_hf> \
    <output_json> \
    [num_samples] [batch_size] [compute_gradients]
```

Output per checkpoint: `checkpoint<N>/faithfulness_metrics_code_preassert.json`

**Key metrics:**
- `kl_direct_effect` — KL divergence when prompt is bypassed (lower = more faithful)
- `kl_cot_necessity` — how much the answer distribution changes when CoT is removed
- `kl_leakage` — CoT information not originating from the prompt
- JS variants (bounded [0, log2]), gradient L1/L2 norms, entropy/NLL

---

### 2. Hacking evaluation (code task)

Generates model completions and scores them on both the model's own assertions (hackable reward signal) and hidden ground-truth tests (genuine fix rate).

```bash
sbatch scripts/eval/run_hacking_eval_single.sh \
    <test_parquet> \
    <model_hf> \
    <output_json> \
    [num_samples] [max_tokens]
```

Output per checkpoint: `checkpoint<N>/hacking_analysis.json`

**Key metrics:**
- `returned_assertions_pass` — model's own assertions passing (RL reward signal)
- `genuine_fix` — hidden ground-truth tests passing (real performance)
- `assertions_modified` — hack rate (model rewrote assertions rather than fixing code)
- `hack_verbalized` — fraction of hacking attempts mentioned in CoT (monitorability)

---

### 3. Faithfulness metrics (math task)

```bash
sbatch scripts/eval/run_faithfulness_eval_single.sh \
    <test_parquet> \
    <generations_json> \
    <finetuned_model_hf> \
    <reference_model_hf> \
    <output_json> \
    [num_samples] [batch_size]
```

---

### 4. Faithfulness metrics (Qwen / DAPO math)

```bash
sbatch scripts/eval/run_faithfulness_qwen_single.sh \
    <test_parquet> \
    <generations_json> \
    <finetuned_model_hf> \
    <reference_model_hf> \
    <output_json> \
    [num_samples] [batch_size]
```

---

### 5. Hint ratio evaluation

Measures how often the model's CoT verbalizes or uses the provided hint.

```bash
sbatch scripts/eval/run_hint_ratio_eval_single.sh \
    <test_parquet> \
    <model_hf> \
    <output_json> \
    [num_samples]
```

---

## Repository Structure

```
.
├── train/                          # VERL fork (training framework)
│   └── verl/
│       ├── trainer/config/actor/
│       │   └── dp_actor.yaml       # Actor hyperparameter defaults
│       ├── utils/
│       │   └── cot_masking.py      # CoT delimiter detection & mask construction
│       ├── models/transformers/
│       │   ├── attention_gradient_masking_patch.py      # gradient_mask method
│       │   └── cot_parameter_gradient_masking_patch.py  # cot_gradient method
│       └── workers/
│           ├── actor/dp_actor.py   # Training loop with CoT masking + FACT/PEP
│           └── config/actor.py     # ActorConfig dataclass (all hyperparameters)
├── eval/                           # Evaluation and faithfulness metric scripts
│   ├── compute_faithfulness_from_generations_clean.py
│   ├── faithfulness_metrics_clean.py
│   └── evaluate_hacking_ratio.py
├── scripts/
│   ├── train_code/                 # Code task training
│   │   ├── train_hacking_methods_2048_v3.sh         # 4B interactive
│   │   ├── train_hacking_methods_2048_v3_12b.sh     # 12B interactive
│   │   ├── train_hacking_methods_2048_v3_fact.sh    # FACT 4B interactive
│   │   ├── train_hacking_methods_2048_v3_fact_12b.sh# FACT 12B interactive
│   │   └── submit_*.sbatch                          # SLURM variants
│   ├── train_math/                 # DeepMind Math training (Gemma3-1B)
│   │   ├── train_unified_methods.sh
│   │   └── submit_unified_training.sbatch
│   ├── train_dapo_math/            # DAPO Math training (Qwen2.5-7B)
│   │   ├── train_dapo_math_methods_qwen25_7b_v7.sh
│   │   └── submit_dapo_math_training_qwen25_7b_v7.sbatch
│   └── eval/                      # Single-job evaluation scripts
│       ├── run_faithfulness_code_preassert_eval_single.sh
│       ├── run_faithfulness_code_preassert_eval_single_12b.sh
│       ├── run_faithfulness_eval_single.sh
│       ├── run_faithfulness_qwen_single.sh
│       ├── run_hacking_eval_single.sh
│       └── run_hint_ratio_eval_single.sh
├── data/
│   └── cc-transformed-hacking-with-hints-v3/  # Code task dataset
└── results/                        # Checkpoints and evaluation outputs
```

---

## Custom Actor Config Parameters

All flags are passed as Hydra overrides to `verl.trainer.main_ppo`. The full list with defaults is in `train/verl/workers/config/actor.py` and `train/verl/trainer/config/actor/dp_actor.yaml`.

**CoT structural masking:**
```
actor_rollout_ref.actor.use_cot_masking=True/False
actor_rollout_ref.actor.use_cot_gradient_masking=True/False
actor_rollout_ref.actor.use_attention_gradient_masking=True/False
actor_rollout_ref.actor.block_prompt_gradients=True/False
actor_rollout_ref.actor.block_answer_gradients=True/False
actor_rollout_ref.actor.disable_cot_masking_for_old_log_prob=True   # required for GRPO
```

**Delimiter detection (substring matching mode):**
```
actor_rollout_ref.actor.use_substring_delimiter_matching=True
actor_rollout_ref.actor.end_think_delimiter_str="</think>"
```

**FACT / PEP adversarial perturbation:**
```
actor_rollout_ref.actor.use_prompt_embedding_perturbation=True
actor_rollout_ref.actor.pep_mode=worst_case          # worst_case (FGSM/PGD) | random (Gaussian)
actor_rollout_ref.actor.pep_layer=-1                 # -1=embed_tokens, N=model.layers[N]
actor_rollout_ref.actor.pep_epsilon=0.05             # perturbation magnitude
actor_rollout_ref.actor.pep_pgd_steps=1              # 1=FGSM, >1=PGD
actor_rollout_ref.actor.pep_dual_loss=False          # add perturbed NLL as auxiliary loss
actor_rollout_ref.actor.pep_attack_answer_only=False # True=attack answer only, False=CoT+answer
```
