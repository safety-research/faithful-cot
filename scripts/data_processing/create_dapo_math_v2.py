#!/usr/bin/env python3
"""
create_dapo_math_v2.py

Re-format the DAPO math dataset with an improved user-message instruction
that better elicits <think>...</think> / Final answer: format following.

No system prompt is added — only the user message changes.

Reads:
  data/dapo_math/train.parquet
  data/dapo_math/test.parquet

Writes (will NOT overwrite original):
  data/dapo_math_v2/train.parquet
  data/dapo_math_v2/test.parquet

Variants (pass --variant):
  v1_first_tok  — "FIRST TOKEN MUST BE: <think>" + Required Format section  [default]
  v2_strong     — strong instruction + explicit format block
  v3_numbered   — numbered steps
  v4_minimal    — minimal format template

Usage:
    python scripts/data_processing/create_dapo_math_v2.py [--variant v1_first_tok]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

WORK_DIR    = Path("/workspace-vast/jinghanj/workspace/Structural_RL_dev")
INPUT_TRAIN = WORK_DIR / "data/dapo_math/train.parquet"
INPUT_TEST  = WORK_DIR / "data/dapo_math/test.parquet"
BASE_OUTPUT_DIR = WORK_DIR / "data/dapo_math_v2"


def extract_bare_question(user_content: str) -> str:
    """Strip any appended format instruction from the original dataset."""
    return re.split(r"\n\nPlease reason step by step", user_content)[0].strip()


def make_user_message(bare_question: str, variant: str) -> str:
    q = bare_question.strip()

    if variant == "v1_first_tok":
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


def convert_df(df: pd.DataFrame, variant: str, split: str) -> pd.DataFrame:
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Converting {split}"):
        user_content = next(m["content"] for m in row["messages"] if m["role"] == "user")
        bare = extract_bare_question(user_content)
        new_row = dict(row)
        new_row["messages"] = [{"role": "user", "content": make_user_message(bare, variant)}]
        rows.append(new_row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="v1_first_tok",
                        choices=["v1_first_tok", "v2_strong", "v3_numbered", "v4_minimal"])
    args = parser.parse_args()

    output_dir = BASE_OUTPUT_DIR / args.variant
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Variant : {args.variant}")
    print(f"Output  : {output_dir}")
    print()

    for split, input_path in [("train", INPUT_TRAIN), ("test", INPUT_TEST)]:
        print(f"Loading {split}: {input_path}")
        df = pd.read_parquet(input_path)
        print(f"  {len(df)} rows")
        df_v2 = convert_df(df, args.variant, split)
        out = output_dir / f"{split}.parquet"
        df_v2.to_parquet(out, index=False)
        print(f"  Saved -> {out}\n")

    # Sanity check
    print("--- Sample converted row ---")
    df_check = pd.read_parquet(output_dir / "train.parquet")
    for m in df_check.iloc[0]["messages"]:
        print(f"[{m['role']}]:\n{m['content']}\n")


if __name__ == "__main__":
    main()
