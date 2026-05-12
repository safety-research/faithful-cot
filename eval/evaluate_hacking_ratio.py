#!/usr/bin/env python3
"""
Evaluation script for reward hacking behavior in code generation.

This script:
1. Loads a test dataset from parquet (cc-transformed-hacking-with-hints-v2 format)
2. Generates responses using an LLM (vLLM)
3. Runs compute_score_with_details from custom_code_reward_hacking.py for each sample
4. Aggregates hacking behavior, monitorability, and accuracy statistics

Usage:
    python evaluate_hacking_ratio.py \
        --model results/checkpoints/gemma3-4b-hacking-v2-vanilla-bs512-seed1997/global_step_100_hf \
        --test_file data/cc-transformed-hacking-with-hints-v2/test.parquet \
        --output_file results/gemma3-4b-hacking-v2-vanilla-bs512-seed1997/checkpoint100/hacking_analysis.json \
        --num_samples 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

# Use a smaller container pool for eval — don't need 64 parallel containers.
# 8 is enough for sequential scoring. This avoids creating 64 containers on
# a fresh node just for evaluation.
os.environ.setdefault("MAX_CONCURRENT_UDOCKER", "8")

# Import reward function from train/
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "train"))
from custom_code_reward_hacking import compute_score_with_details  # noqa: E402


# ============================================================================
# Dataset loading
# ============================================================================

def load_test_data(test_file: str, num_samples: int = -1) -> list[dict]:
    """
    Load test samples from parquet.

    Expected schema:
        messages     - list of {role, content} dicts; user message = prompt
        reward_model - {ground_truth: {visible_inputs, visible_outputs,
                                       hidden_inputs, hidden_outputs}, style}
        extra_info   - {reference_solution, problem_name, ...}
        uid          - unique sample identifier
    """
    df = pd.read_parquet(test_file)
    if num_samples > 0:
        df = df.head(num_samples)

    samples = []
    for _, row in df.iterrows():
        # User prompt (question shown to model)
        messages = row["messages"]
        question = next(m["content"] for m in messages if m["role"] == "user")

        # Ground truth (passed to reward function)
        reward_model = row["reward_model"]
        ground_truth = reward_model["ground_truth"]  # dict

        # Extra info (reference solution etc.)
        extra_info = row.get("extra_info", {}) or {}
        if not isinstance(extra_info, dict):
            extra_info = {}

        uid = row.get("uid", "")

        samples.append({
            "question": question,
            "ground_truth": ground_truth,
            "extra_info": extra_info,
            "uid": uid,
        })

    return samples


# ============================================================================
# LLM generation (vLLM only — code tasks need long context)
# ============================================================================

def load_vllm(model_path: str, temperature: float, max_tokens: int, **kwargs):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
        gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.85),
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=kwargs.get("top_p", 1.0),
    )
    return llm, sampling_params, tokenizer


def format_prompt(question: str, tokenizer) -> str:
    messages = [{"role": "user", "content": question}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return question


def generate_responses(llm, sampling_params, prompts: list[str]) -> list[str]:
    outputs = llm.generate(prompts, sampling_params)
    return [o.outputs[0].text for o in outputs]


# ============================================================================
# Per-sample scoring
# ============================================================================

def score_sample(sample: dict, output: str) -> dict[str, Any]:
    """
    Run compute_score_with_details and attach metadata.
    """
    scores = compute_score_with_details(
        data_source="codecontests_hacking",
        solution_str=output,
        ground_truth=sample["ground_truth"],
        extra_info=sample["extra_info"],
    )
    scores["uid"] = sample["uid"]
    scores["question"] = sample["question"][:300]   # truncate for storage
    scores["generated_text"] = output
    return scores


# ============================================================================
# Aggregate statistics
# ============================================================================

# Scalar metrics to average across samples
_RATE_METRICS = [
    "score",
    "acc",
    "acc_loose",
    "returned_assertions_pass",
    "assertions_call_solve",
    "genuine_fix",
    "is_hacking",
    "assertions_changed",
    "assertions_omitted",
    "solve_unchanged",
    "has_early_exit",
    "had_scratchpad_in_code",
    "hack_verbalized",
    "cot_hacking_intent",
    "cot_test_awareness",
    "has_code",
    "has_solve_fn",
    "has_strict_format",
    "has_scratchpad",
    "single_code_block",
    "hidden_pass_rate",
    "num_returned_assertions",
    "num_visible_assertions",
]


def compute_statistics(analyses: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(analyses)
    stats: dict[str, Any] = {"total_samples": n}

    for key in _RATE_METRICS:
        vals = [a[key] for a in analyses if key in a and a[key] is not None]
        stats[key] = sum(vals) / len(vals) if vals else None

    # Derived conditional ratios
    n_hacking = sum(a.get("is_hacking", 0) for a in analyses)
    n_verbalized_and_hacking = sum(
        a.get("hack_verbalized", 0) for a in analyses if a.get("is_hacking", 0)
    )
    stats["hack_verbalized_given_hacking"] = (
        n_verbalized_and_hacking / n_hacking if n_hacking > 0 else None
    )

    n_passed = sum(a.get("returned_assertions_pass", 0) for a in analyses)
    n_genuine_and_passed = sum(
        a.get("genuine_fix", 0) for a in analyses if a.get("returned_assertions_pass", 0)
    )
    stats["genuine_fix_given_passed"] = (
        n_genuine_and_passed / n_passed if n_passed > 0 else None
    )

    # Hacking breakdown (of samples where assertions passed)
    n_changed_and_passed = sum(
        1 for a in analyses
        if a.get("returned_assertions_pass", 0) and a.get("assertions_changed", 0)
    )
    n_unaltered_and_passed = sum(
        1 for a in analyses
        if a.get("returned_assertions_pass", 0) and not a.get("assertions_changed", 0)
    )
    stats["changed_given_passed"] = (
        n_changed_and_passed / n_passed if n_passed > 0 else None
    )
    stats["unaltered_given_passed"] = (
        n_unaltered_and_passed / n_passed if n_passed > 0 else None
    )

    # Raw counts
    stats["n_is_hacking"] = int(n_hacking)
    stats["n_genuine_fix"] = int(sum(a.get("genuine_fix", 0) for a in analyses))
    stats["n_returned_assertions_pass"] = int(n_passed)
    stats["n_assertions_changed"] = int(
        sum(a.get("assertions_changed", 0) for a in analyses)
    )
    stats["n_assertions_omitted"] = int(
        sum(a.get("assertions_omitted", 0) for a in analyses)
    )
    stats["n_cot_hacking_intent"] = int(
        sum(a.get("cot_hacking_intent", 0) for a in analyses)
    )
    stats["n_cot_test_awareness"] = int(
        sum(a.get("cot_test_awareness", 0) for a in analyses)
    )

    return stats


# ============================================================================
# Print summary
# ============================================================================

def print_summary(stats: dict[str, Any]) -> None:
    n = stats["total_samples"]
    def fmt(key: str) -> str:
        v = stats.get(key)
        return f"{v:.3f}" if v is not None else "N/A"

    print("\n" + "=" * 70)
    print("HACKING EVALUATION RESULTS")
    print("=" * 70)
    print(f"\n  Samples:               {n}")
    print(f"\n--- Accuracy ---")
    print(f"  Mean reward score:     {fmt('score')}")
    print(f"  acc (strict):          {fmt('acc')}")
    print(f"  acc_loose:             {fmt('acc_loose')}")
    print(f"  returned_assertions_pass: {fmt('returned_assertions_pass')}  (n={stats['n_returned_assertions_pass']})")
    print(f"\n--- Hacking Behavior ---")
    print(f"  is_hacking rate:       {fmt('is_hacking')}  (n={stats['n_is_hacking']})")
    print(f"  assertions_changed:    {fmt('assertions_changed')}  (n={stats['n_assertions_changed']})")
    print(f"  assertions_omitted:    {fmt('assertions_omitted')}  (n={stats['n_assertions_omitted']})")
    print(f"  solve_unchanged:       {fmt('solve_unchanged')}")
    print(f"  changed|passed:        {fmt('changed_given_passed')}  (of samples where assertions passed)")
    print(f"\n--- Genuine Fix (Hidden Tests) ---")
    print(f"  genuine_fix rate:      {fmt('genuine_fix')}  (n={stats['n_genuine_fix']})")
    print(f"  genuine_fix|passed:    {fmt('genuine_fix_given_passed')}  (of samples where assertions passed)")
    print(f"\n--- Monitorability ---")
    print(f"  cot_hacking_intent:    {fmt('cot_hacking_intent')}  (n={stats['n_cot_hacking_intent']})")
    print(f"  cot_test_awareness:    {fmt('cot_test_awareness')}  (n={stats['n_cot_test_awareness']})")
    print(f"  hack_verbalized:       {fmt('hack_verbalized')}")
    print(f"  hack_verbalized|hacking: {fmt('hack_verbalized_given_hacking')}  (transparent hacking fraction)")
    print(f"\n--- Format ---")
    print(f"  has_code:              {fmt('has_code')}")
    print(f"  has_solve_fn:          {fmt('has_solve_fn')}")
    print(f"  has_strict_format:     {fmt('has_strict_format')}")
    print(f"  assertions_call_solve: {fmt('assertions_call_solve')}")
    print("=" * 70)


# ============================================================================
# Resume helpers
# ============================================================================

def load_completed(output_file: str) -> dict[str, Any]:
    """Load already-scored samples from a partial output file. Returns uid → analysis."""
    if not os.path.exists(output_file):
        return {}
    try:
        with open(output_file) as f:
            data = json.load(f)
        by_uid = {a["uid"]: a for a in data.get("per_sample_analyses", []) if "uid" in a}
        print(f"  → Resume: {len(by_uid)} already-scored samples found")
        return by_uid
    except Exception as e:
        print(f"  → Warning: could not read partial results ({e}), starting fresh")
        return {}


def save_partial(output_file: str, config: dict, analyses: list[dict]) -> None:
    """Atomically write current progress (tmp + rename)."""
    results = {
        "config": config,
        "statistics": compute_statistics(analyses),
        "per_sample_analyses": analyses,
    }
    tmp = output_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    os.replace(tmp, output_file)


# ============================================================================
# Main pipeline
# ============================================================================

def evaluate_hacking_ratio(
    test_file: str,
    model_path: str,
    output_file: str,
    num_samples: int = -1,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.85,
    top_p: float = 1.0,
    save_freq: int = 20,
) -> dict:
    print("=" * 70)
    print("Hacking Ratio Evaluation")
    print("=" * 70)

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "test_file": test_file,
        "model_path": model_path,
        "num_samples": num_samples,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # 1. Load data + check what's already done
    print(f"\n[1/4] Loading test data from {test_file}")
    all_samples = load_test_data(test_file, num_samples)
    print(f"  → {len(all_samples)} samples total")

    completed = load_completed(output_file)
    remaining = [s for s in all_samples if s["uid"] not in completed]
    print(f"  → {len(remaining)} remaining to score")

    if not remaining:
        print("All samples already scored.")
        analyses = list(completed.values())
        stats = compute_statistics(analyses)
        print_summary(stats)
        return {"config": config, "statistics": stats, "per_sample_analyses": analyses}

    # 2. Load model, generate only for remaining samples
    print(f"\n[2/4] Loading model: {model_path}")
    llm, sampling_params, tokenizer = load_vllm(
        model_path, temperature, max_tokens,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        top_p=top_p,
    )
    print("  → Model loaded")

    print(f"\n[3/4] Generating responses for {len(remaining)} samples")
    prompts = [format_prompt(s["question"], tokenizer) for s in remaining]
    responses = generate_responses(llm, sampling_params, prompts)
    print(f"  → {len(responses)} responses generated")

    # Free GPU — scoring is CPU-only (udocker)
    del llm
    import torch
    torch.cuda.empty_cache()

    # 4. Score with periodic saves so preemption only loses < save_freq samples
    print(f"\n[4/4] Scoring (saving every {save_freq} samples)")
    analyses = list(completed.values())

    for i, (sample, output) in enumerate(
        tqdm(zip(remaining, responses), total=len(remaining))
    ):
        analyses.append(score_sample(sample, output))

        if (i + 1) % save_freq == 0:
            save_partial(output_file, config, analyses)
            tqdm.write(f"  Saved {len(analyses)}/{len(all_samples)}")

    save_partial(output_file, config, analyses)
    stats = compute_statistics(analyses)
    print_summary(stats)
    print(f"\nSaved: {output_file}  ({len(analyses)} samples)")

    return {"config": config, "statistics": stats, "per_sample_analyses": analyses}


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate reward hacking behavior in code generation"
    )
    parser.add_argument("--model", required=True, help="HuggingFace model path")
    parser.add_argument("--test_file", required=True, help="Path to test parquet file")
    parser.add_argument("--output_file", required=True, help="Path to save results JSON")
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="Number of samples to evaluate (-1 = all)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default 0 = greedy)")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Max tokens to generate")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--save_freq", type=int, default=20,
                        help="Save progress every N scored samples (enables resume on preemption)")

    args = parser.parse_args()
    evaluate_hacking_ratio(
        test_file=args.test_file,
        model_path=args.model,
        output_file=args.output_file,
        num_samples=args.num_samples,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        top_p=args.top_p,
        save_freq=args.save_freq,
    )


if __name__ == "__main__":
    main()
