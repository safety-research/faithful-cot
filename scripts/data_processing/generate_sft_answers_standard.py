#!/usr/bin/env python3
"""
Generate Claude answers for SFT validation set (Standard variant).

This script:
1. Loads val.parquet from cc-transformed-verl
2. Uses Claude to generate proper answers with <think></think> format
3. Saves in SFT format for Gemma3-4b fine-tuning

The goal is to teach Gemma3-4b the <think></think> format before RL training.
"""

import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from anthropic import Anthropic
from tqdm import tqdm

# Configuration
INPUT_VAL_FILE = Path("/workspace-vast/jinghanj/workspace/Structural_RL_dev/data/cc-transformed-verl/val.parquet")
OUTPUT_SFT_FILE = Path("/workspace-vast/jinghanj/workspace/Structural_RL_dev/data/cc-transformed-verl/sft_val.parquet")

# Claude API
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set")

client = Anthropic(api_key=ANTHROPIC_API_KEY)

# System prompt for Claude to generate proper answers
SYSTEM_PROMPT = """You are an expert Python programmer helping to generate training data for teaching a model to fix buggy code.

Your task: Generate a proper answer that fixes the buggy code and follows the required format.

**Required format:**
<think>
[Your reasoning about what the bug is and how to fix it]
</think>

```python
def solve(lines):
    # Your corrected solution
    pass
```

**Important:**
- Use <think></think> tags for reasoning
- Explain the bug clearly
- Provide working Python code that passes all test cases
- Function signature must be `def solve(lines)` where lines is a list of input strings
- Return answer as string or number
- Do NOT add assert statements
- Write clean, correct code"""


def generate_answer(prompt_text: str) -> Optional[str]:
    """
    Use Claude to generate a proper answer with <think></think> format.

    Args:
        prompt_text: The prompt from val.parquet

    Returns:
        Generated answer string or None if failed
    """
    try:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2048,
            temperature=0.7,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": prompt_text}
            ]
        )

        answer = response.content[0].text
        return answer
    except Exception as e:
        print(f"Error generating answer: {e}")
        return None


def main():
    print("=" * 80)
    print("Generating Claude Answers for SFT (Standard Variant)")
    print("=" * 80)

    # Check API key
    if not ANTHROPIC_API_KEY:
        print("\n❌ Error: ANTHROPIC_API_KEY not found in environment")
        print("Set it with: export ANTHROPIC_API_KEY='your-api-key'")
        return

    # Load validation data
    print(f"\nLoading validation data from: {INPUT_VAL_FILE}")
    if not INPUT_VAL_FILE.exists():
        print(f"\n❌ Error: {INPUT_VAL_FILE} not found")
        print("Run convert_codecontests_to_verl.py first")
        return

    df = pd.read_parquet(INPUT_VAL_FILE)
    print(f"✓ Loaded {len(df):,} validation samples")

    # Generate answers
    print("\nGenerating Claude answers...")
    print("This will use Claude API and may take some time...")

    sft_data = []
    failed_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating"):
        # Parse prompt
        prompt = json.loads(row["prompt"])
        prompt_text = prompt[0]["content"]

        # Generate answer
        answer = generate_answer(prompt_text)

        if answer is None:
            failed_count += 1
            continue

        # Parse original prompt to get user message
        original_prompt = json.loads(row["prompt"])
        user_content = original_prompt[0]["content"]

        # Create messages format (user + assistant)
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer}
        ]

        # Create SFT format (instruction + response pairs)
        sft_sample = {
            "data_source": row["data_source"],
            "messages": messages,  # Messages format for SFT
            "completion": answer,  # Assistant response only
            "reward_model": row["reward_model"],
            "extra_info": row["extra_info"]
        }
        sft_data.append(sft_sample)

    print(f"\n✓ Generated {len(sft_data):,} answers")
    if failed_count > 0:
        print(f"⚠ Failed to generate {failed_count} answers")

    # Save SFT data
    print(f"\nSaving SFT data to: {OUTPUT_SFT_FILE}")
    sft_df = pd.DataFrame(sft_data)
    sft_df.to_parquet(OUTPUT_SFT_FILE, index=False, engine='pyarrow')
    print(f"✓ Saved {len(sft_df):,} samples ({OUTPUT_SFT_FILE.stat().st_size / 1024 / 1024:.1f} MB)")

    # Print example
    print("\n" + "=" * 80)
    print("Example SFT Sample:")
    print("=" * 80)
    example = sft_data[0]
    messages = example["messages"]
    print("\nMessages format:")
    print(f"  [0] User (first 500 chars):")
    print(f"      {messages[0]['content'][:500]}...")
    print(f"\n  [1] Assistant (first 800 chars):")
    print(f"      {messages[1]['content'][:800]}...")
    print(f"\nCompletion field (same as assistant message):")
    print(f"  {example['completion'][:400]}...")

    print("\n" + "=" * 80)
    print("✓ SFT data generation complete!")
    print("=" * 80)
    print(f"\nOutput: {OUTPUT_SFT_FILE}")
    print(f"\nNext steps:")
    print("1. Run SFT training on Gemma3-4b with this data")
    print("2. The model will learn <think></think> format")
    print("3. Use the SFT model as starting point for RL training")


if __name__ == "__main__":
    main()
