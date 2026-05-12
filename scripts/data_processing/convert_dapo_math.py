#!/usr/bin/env python3
"""
Convert BytedTsinghua-SIA/DAPO-Math-17k to the deepmindmath_all parquet format.

No hints — plain math questions only (not related to the reward hacking setup).

Source format (DAPO):
    prompt: [{'role': 'user', 'content': 'Solve the following math problem...\n\n{question}\n\nRemember...'}]
    reward_model: {'ground_truth': answer, 'style': '...'}
    data_source: 'math_dapo'
    extra_info: {'index': uuid}

Target format:
    messages:      [{'role': 'user', 'content': '{question}, response format: <think>reasoning</think> Final answer: X'}]
    reward_model:  {'ground_truth': answer, 'style': 'rule'}
    extra_info:    {'question': question}
    data_source:   'dapo_math'
    uid:           'dapo_math_{index}'

Usage:
    python scripts/data_processing/convert_dapo_math.py \
        --output-dir data/dapo_math \
        --train-size 100000 \
        --test-size 1000 \
        --seed 42
"""

import argparse
import re

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# ------------------------------------------------------------------ #
# Prompt stripping
# ------------------------------------------------------------------ #

# DAPO wraps questions with this instruction prefix and suffix
_DAPO_PREFIX = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
_DAPO_SUFFIX = "\n\nRemember to put your answer on its own line after \"Answer:\"."


def extract_question(dapo_content: str) -> str:
    """Strip DAPO instruction wrapper and return the raw math question."""
    q = dapo_content
    if q.startswith(_DAPO_PREFIX):
        q = q[len(_DAPO_PREFIX):]
    if q.endswith(_DAPO_SUFFIX):
        q = q[: -len(_DAPO_SUFFIX)]
    # Fallback: regex strip in case of minor whitespace variation
    q = re.sub(r'^Solve the following math problem step by step\..*?\n\n', '', q, flags=re.DOTALL)
    q = re.sub(r'\n\nRemember to put your answer on its own line.*$', '', q, flags=re.DOTALL)
    return q.strip()


def build_sample(question: str, answer: str, uid: str) -> dict:
    """Build one row in the target format (no hint)."""
    user_content = f"{question}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}., response format: <think>reasoning</think> Final answer: \\boxed{{X}}"
    messages = np.array([{"role": "user", "content": user_content}])
    return {
        "messages":     messages,
        "reward_model": {"ground_truth": answer, "style": "rule"},
        "extra_info":   {"question": question},
        "data_source":  "dapo_math",
        "uid":          uid,
    }


# ------------------------------------------------------------------ #
# Main conversion
# ------------------------------------------------------------------ #

def convert(
    output_dir: str,
    train_size: int,
    test_size: int,
    seed: int,
    cache_dir: str | None,
):
    import random
    random.seed(seed)
    np.random.seed(seed)

    print("Loading DAPO-Math-17k from HuggingFace...")
    ds = load_dataset(
        "BytedTsinghua-SIA/DAPO-Math-17k",
        split="train",
        **({"cache_dir": cache_dir} if cache_dir else {}),
    )
    print(f"  Total samples: {len(ds)}")

    total_needed = train_size + test_size
    if total_needed > len(ds):
        print(f"  Warning: requested {total_needed} but only {len(ds)} available — using all")
        total_needed = len(ds)
        train_size = total_needed - test_size

    # Shuffle and slice
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[:total_needed]
    train_indices = indices[:train_size]
    test_indices  = indices[train_size:]

    def convert_subset(subset_indices: list[int], split_name: str) -> pd.DataFrame:
        rows = []
        for i in tqdm(subset_indices, desc=f"Converting {split_name}"):
            sample = ds[i]
            raw_content = sample["prompt"][0]["content"]
            question    = extract_question(raw_content)
            answer      = sample["reward_model"]["ground_truth"]
            uid         = f"dapo_math_{sample['extra_info']['index']}"
            rows.append(build_sample(question, answer, uid))
        return pd.DataFrame(rows)

    import os
    os.makedirs(output_dir, exist_ok=True)

    for split_name, split_indices in [("train", train_indices), ("test", test_indices)]:
        if not split_indices:
            continue
        df = convert_subset(split_indices, split_name)
        out_path = f"{output_dir}/{split_name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"Saved {split_name}: {len(df)} rows → {out_path}")

    # Sanity check
    print("\nSanity check (test split, first sample):")
    df_check = pd.read_parquet(f"{output_dir}/test.parquet")
    row = df_check.iloc[0]
    print(f"  uid:            {row['uid']}")
    print(f"  data_source:    {row['data_source']}")
    print(f"  ground_truth:   {row['reward_model']['ground_truth']}")
    print(f"  message[:120]:  {row['messages'][0]['content'][:120]}")
    print("\nDone.")


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Convert DAPO-Math-17k to deepmindmath format")
    parser.add_argument("--output-dir",        default="data/dapo_math",
                        help="Directory to write train.parquet / test.parquet")
    parser.add_argument("--train-size",        type=int, default=100_000,
                        help="Number of training samples (default 100k; max ~1.79M)")
    parser.add_argument("--test-size",         type=int, default=1_000,
                        help="Number of test samples (default 1000)")
    parser.add_argument("--seed",              type=int, default=42)
    parser.add_argument("--cache-dir",         default=None,
                        help="HuggingFace cache directory (default: HF_HOME)")
    args = parser.parse_args()

    convert(
        output_dir = args.output_dir,
        train_size = args.train_size,
        test_size  = args.test_size,
        seed       = args.seed,
        cache_dir  = args.cache_dir,
    )


if __name__ == "__main__":
    main()
