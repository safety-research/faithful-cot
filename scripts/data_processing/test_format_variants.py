#!/usr/bin/env python3
"""
test_format_variants.py

Test several user-message prompt format variants on the base Qwen2.5-7B-Instruct
model to measure <think>...</think> / Final answer: format following rate.

All variants use NO system prompt override (model sees its default system message).
Only the user message instruction differs.

Variants:
  v0_current    — original format (baseline)
  v1_first_tok  — "FIRST TOKEN MUST BE: <think>" + Required Format section
  v2_strong     — strong instruction + explicit format block
  v3_numbered   — numbered steps: (1) start with <think> (2) reason (3) Final answer
  v4_minimal    — minimal but explicit: just the format template with a one-liner instruction

Usage:
    python scripts/data_processing/test_format_variants.py \
        [--n_samples 200] [--model Qwen/Qwen2.5-7B-Instruct]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

THINK_RE  = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
FINAL_RE  = re.compile(r"(?i)final\s+answer\s*:")
FORMAT_RE = re.compile(r"<think>.*?</think>\s*final\s+answer\s*:", re.DOTALL | re.IGNORECASE)

WORK_DIR = Path("/workspace-vast/jinghanj/workspace/Structural_RL_dev")


# ── Format variants (user message only, no system override) ───────────────────

def make_user_message(bare_question: str, variant: str) -> str:
    q = bare_question.strip()

    if variant == "v0_current":
        return (
            f"{q}\n\n"
            "Please reason step by step, and put your final answer within \\boxed{}., "
            "response format: <think>reasoning</think> Final answer: \\boxed{X}"
        )

    elif variant == "v1_first_tok":
        # Mirrors the hacking task style: FIRST TOKEN constraint + Required Format section
        return (
            f"{q}\n\n"
            "***YOUR FIRST TOKEN MUST BE: <think>***\n\n"
            "**Required Format:**\n"
            "<think>\n"
            "[your step-by-step reasoning]\n"
            "</think>\n"
            "Final answer: [your answer]"
        )

    elif variant == "v2_strong":
        return (
            f"{q}\n\n"
            "You MUST respond in EXACTLY this format. "
            "Do NOT start with any other token.\n\n"
            "<think>\n"
            "[your step-by-step reasoning]\n"
            "</think>\n"
            "Final answer: [your answer]"
        )

    elif variant == "v3_numbered":
        return (
            f"{q}\n\n"
            "Follow these instructions exactly:\n"
            "1. Your FIRST token must be <think>\n"
            "2. Reason step by step inside the <think>...</think> block\n"
            "3. Close with </think>, then write: Final answer: [your answer]"
        )

    elif variant == "v4_minimal":
        return (
            f"{q}\n\n"
            "Respond in this format only:\n"
            "<think>\n"
            "[reasoning]\n"
            "</think>\n"
            "Final answer: [answer]"
        )

    else:
        raise ValueError(f"Unknown variant: {variant}")


VARIANTS = ["v0_current", "v1_first_tok", "v2_strong", "v3_numbered", "v4_minimal"]


# ── Evaluation ────────────────────────────────────────────────────────────────

def check_format(text: str) -> dict:
    starts_with_think = text.lstrip().lower().startswith("<think>")
    has_think_close   = "</think>" in text.lower()
    has_final         = bool(FINAL_RE.search(text))
    full_format       = bool(FORMAT_RE.search(text))
    return {
        "starts_with_think": starts_with_think,
        "has_think_close":   has_think_close,
        "has_final_answer":  has_final,
        "full_format":       full_format,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples",   type=int,   default=200)
    parser.add_argument("--model",       type=str,   default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max_tokens",  type=int,   default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output",      type=str,   default=str(WORK_DIR / "results/format_variant_test.json"))
    parser.add_argument("--variants",    type=str,   nargs="+", default=VARIANTS)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.80,
        max_model_len=4096,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    print(f"Testing {len(args.variants)} variants\n")

    results = {}

    for variant in args.variants:
        print(f"{'='*55}")
        print(f"Variant: {variant}")

        # Load the pre-converted dataset for this variant
        if variant == "v0_current":
            test_path = WORK_DIR / "data/dapo_math/test.parquet"
        else:
            test_path = WORK_DIR / f"data/dapo_math_v2/{variant}/test.parquet"

        if not test_path.exists():
            print(f"  WARNING: {test_path} not found — skipping.")
            print(f"  Run: python scripts/data_processing/create_dapo_math_v2.py --variant {variant}")
            continue

        df = pd.read_parquet(test_path).head(args.n_samples)
        prompts = []
        for _, row in df.iterrows():
            msgs = row["messages"]
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        print(f"  Loaded {len(prompts)} samples from {test_path}")

        outputs = llm.generate(prompts, sampling_params)
        texts = [o.outputs[0].text for o in outputs]

        checks = [check_format(t) for t in texts]
        n = len(checks)
        stats = {
            "n":                  n,
            "starts_with_think":  sum(c["starts_with_think"] for c in checks) / n,
            "has_think_close":    sum(c["has_think_close"]   for c in checks) / n,
            "has_final_answer":   sum(c["has_final_answer"]  for c in checks) / n,
            "full_format":        sum(c["full_format"]       for c in checks) / n,
        }

        print(f"  starts with <think> : {stats['starts_with_think']:.1%}")
        print(f"  has </think>        : {stats['has_think_close']:.1%}")
        print(f"  has Final answer:   : {stats['has_final_answer']:.1%}")
        print(f"  full format         : {stats['full_format']:.1%}")

        print(f"\n  -- Sample outputs --")
        for i, text in enumerate(texts[:3]):
            print(f"  [{i}] {repr(text[:250])}\n")

        results[variant] = {"stats": stats, "sample_outputs": texts[:5]}

    print(f"\n{'='*55}")
    print("SUMMARY — full_format rate:")
    for v, r in sorted(results.items(), key=lambda x: -x[1]["stats"]["full_format"]):
        print(f"  {v:<20s}: {r['stats']['full_format']:.1%}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
