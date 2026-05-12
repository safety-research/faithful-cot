#!/usr/bin/env python3
"""
Update existing SFT data to add response to messages format.

Takes existing SFT data with 'response' field and adds it to messages array.
"""

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def update_sft_format(input_file: Path, output_file: Path):
    """
    Update SFT data to correct messages format.

    Args:
        input_file: Input parquet with 'response' field
        output_file: Output parquet with 'messages' and 'completion' fields
    """
    print("=" * 80)
    print("Updating SFT Data Format")
    print("=" * 80)

    # Load existing data
    print(f"\nLoading data from: {input_file}")
    df = pd.read_parquet(input_file)
    print(f"✓ Loaded {len(df)} samples")

    # Check columns
    print(f"\nExisting columns: {df.columns.tolist()}")

    if "response" not in df.columns:
        print("\n❌ Error: No 'response' column found!")
        print("Expected columns: data_source, prompt, response, reward_model, extra_info")
        return

    # Update format
    print(f"\nUpdating format...")
    updated_data = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        # Parse original prompt
        prompt_data = json.loads(row["prompt"]) if isinstance(row["prompt"], str) else row["prompt"]

        # Get user content
        if isinstance(prompt_data, list):
            user_content = prompt_data[0]["content"]
        else:
            user_content = prompt_data.get("content", str(prompt_data))

        # Get assistant response
        assistant_response = row["response"]

        # Create messages format
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_response}
        ]

        # Create updated sample
        updated_sample = {
            "data_source": row["data_source"],
            "messages": messages,
            "completion": assistant_response,
            "reward_model": row["reward_model"],
            "extra_info": row["extra_info"]
        }

        # Add any extra fields
        for col in df.columns:
            if col not in ["data_source", "prompt", "response", "reward_model", "extra_info", "messages", "completion"]:
                updated_sample[col] = row[col]

        updated_data.append(updated_sample)

    # Create DataFrame
    print(f"\n✓ Updated {len(updated_data)} samples")
    updated_df = pd.DataFrame(updated_data)

    # Save
    print(f"\nSaving to: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    updated_df.to_parquet(output_file, index=False)
    print(f"✓ Saved {len(updated_df)} samples")

    # Show example
    print("\n" + "=" * 80)
    print("Example Updated Sample")
    print("=" * 80)

    example = updated_data[0]
    messages = example["messages"]

    print(f"\nMessages format:")
    print(f"  [0] User (first 300 chars):")
    print(f"      {messages[0]['content'][:300]}...")
    print(f"\n  [1] Assistant (first 400 chars):")
    print(f"      {messages[1]['content'][:400]}...")

    print(f"\nCompletion (first 400 chars):")
    print(f"  {example['completion'][:400]}...")

    print(f"\nData source: {example['data_source']}")

    # Verify format
    print("\n" + "=" * 80)
    print("Format Verification")
    print("=" * 80)

    # Check all samples have correct format
    all_valid = True
    for i, sample in enumerate(updated_data[:10]):  # Check first 10
        messages = sample["messages"]
        if not isinstance(messages, list):
            print(f"  ✗ Sample {i}: messages is not a list")
            all_valid = False
        elif len(messages) != 2:
            print(f"  ✗ Sample {i}: messages has {len(messages)} items (should be 2)")
            all_valid = False
        elif messages[0]["role"] != "user":
            print(f"  ✗ Sample {i}: first message role is '{messages[0]['role']}' (should be 'user')")
            all_valid = False
        elif messages[1]["role"] != "assistant":
            print(f"  ✗ Sample {i}: second message role is '{messages[1]['role']}' (should be 'assistant')")
            all_valid = False
        elif sample["completion"] != messages[1]["content"]:
            print(f"  ✗ Sample {i}: completion doesn't match assistant message")
            all_valid = False

    if all_valid:
        print("  ✓ All samples have correct format!")
    else:
        print("  ⚠ Some samples have format issues")

    print("\n" + "=" * 80)
    print("✓ Format Update Complete!")
    print("=" * 80)

    print(f"\nUpdated file: {output_file}")
    print(f"Samples: {len(updated_df)}")
    print("\nNew columns:")
    for col in updated_df.columns:
        print(f"  - {col}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python update_sft_format.py <input_file> <output_file>")
        print("\nExample:")
        print("  python update_sft_format.py \\")
        print("    /path/to/val_sft.parquet \\")
        print("    /path/to/val_sft_formatted.parquet")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    if not input_file.exists():
        print(f"❌ Error: Input file does not exist: {input_file}")
        sys.exit(1)

    update_sft_format(input_file, output_file)
