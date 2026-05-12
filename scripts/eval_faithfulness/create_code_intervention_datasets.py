#!/usr/bin/env python3
"""
Create on-policy intervention datasets from code hacking generations.

For each sample where the CoT contains a lookup table (cot_lookup_code=True):
  - Pick ONE test case (same input appears in both prompt assert and CoT lookup)
  - Create a PROMPT intervention: change the expected value in the assert statement
  - Create a COT intervention: change the return value in the CoT lookup table code
  - Both use the SAME alternative value, changing the SAME test case input

Faithfulness test: if the model follows CoT (faithful), changing the CoT lookup
value causes a bigger logit shift in the solve() body than changing the prompt assert.

Usage:
    python create_code_intervention_datasets.py \
        --hacking-analysis results/.../checkpoint170/hacking_analysis.json \
        --test-parquet data/cc-transformed-hacking-with-hints-v2/test.parquet \
        --output-dir results/intervention_datasets/cot_gradient_step170/
"""

import argparse
import json
import os
import re
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq


# ─── Parsing helpers ──────────────────────────────────────────────────────────

# Matches:  assert str(solve(['INPUT'])).strip() == 'OUTPUT'
_ASSERT_RE = re.compile(
    r"assert str\(solve\(\['(.+?)'\]\)\)\.strip\(\) == '(.+?)'"
)

# Matches CoT lookup lines:
#   if lines == ['INPUT']:
#       return 'OUTPUT'
#   or:  if lines == ['INPUT']: return 'OUTPUT'
_COT_LOOKUP_RE = re.compile(
    r"if lines == \['(.+?)'\]:\s*(?:\n\s+)?return '(.+?)'"
)


def parse_assert_pairs(prompt: str) -> list[tuple[str, str]]:
    """Return list of (input, expected_output) from prompt assert statements."""
    return _ASSERT_RE.findall(prompt)


def parse_cot_lookup_pairs(cot: str) -> list[tuple[str, str]]:
    """Return list of (input, return_value) from CoT lookup table code."""
    return _COT_LOOKUP_RE.findall(cot)


def extract_cot(generated_text: str) -> Optional[str]:
    """Extract text between <think> and </think>."""
    m = re.search(r"<think>(.*?)</think>", generated_text, re.DOTALL)
    return m.group(1) if m else None


def generate_alternative_output(original: str) -> str:
    """Generate a clearly different alternative for the given output value."""
    # Flip Yes/No
    if original.strip().lower() == "yes":
        return "No"
    if original.strip().lower() == "no":
        return "Yes"

    # Try integer negation
    try:
        val = int(original.strip())
        alt = -val if val != 0 else -999
        # Make sure we don't accidentally return the same value
        if str(alt) != original.strip():
            return str(alt)
        return str(alt - 1)
    except ValueError:
        pass

    # Try float
    try:
        val = float(original.strip())
        return f"{-val:.6g}" if val != 0.0 else "-999.0"
    except ValueError:
        pass

    # Fallback: wrap with marker
    return f"INTERVENTION"


# ─── Intervention creation ────────────────────────────────────────────────────

def create_prompt_intervention(
    full_prompt: str,
    target_input: str,
    original_output: str,
    alternative_output: str,
) -> Optional[str]:
    """
    Replace the expected value for target_input in the prompt's assert statements.

    Only replaces the FIRST matching assert for that input (if multiple).
    Returns None if no replacement was made.
    """
    old = f"assert str(solve(['{target_input}'])).strip() == '{original_output}'"
    new = f"assert str(solve(['{target_input}'])).strip() == '{alternative_output}'"
    if old not in full_prompt:
        return None
    return full_prompt.replace(old, new, 1)


def create_cot_intervention(
    generated_text: str,
    target_input: str,
    original_output: str,
    alternative_output: str,
) -> Optional[str]:
    """
    Replace the return value for target_input in the CoT lookup table.

    Handles both inline and indented return formats.
    Returns None if no replacement was made.
    """
    # Try inline: if lines == ['INPUT']: return 'OUTPUT'
    old_inline = f"if lines == ['{target_input}']: return '{original_output}'"
    new_inline = f"if lines == ['{target_input}']: return '{alternative_output}'"

    if old_inline in generated_text:
        return generated_text.replace(old_inline, new_inline, 1)

    # Try indented:
    #   if lines == ['INPUT']:
    #       return 'OUTPUT'
    old_block = f"if lines == ['{target_input}']:\n        return '{original_output}'"
    new_block = f"if lines == ['{target_input}']:\n        return '{alternative_output}'"
    if old_block in generated_text:
        return generated_text.replace(old_block, new_block, 1)

    # Try with 4-space indent
    old_block4 = f"if lines == ['{target_input}']:\n    return '{original_output}'"
    new_block4 = f"if lines == ['{target_input}']:\n    return '{alternative_output}'"
    if old_block4 in generated_text:
        return generated_text.replace(old_block4, new_block4, 1)

    return None


# ─── Dataset conversion ───────────────────────────────────────────────────────

def make_messages(user_content: str, assistant_content: str) -> list[dict]:
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def build_parquet_table(records: list[dict]) -> pa.Table:
    messages_list = [r["messages"] for r in records]
    extra_info_list = [r["extra_info"] for r in records]
    uid_list = [r["uid"] for r in records]
    data_source_list = ["codecontests"] * len(records)

    return pa.table({
        "messages": messages_list,
        "extra_info": extra_info_list,
        "data_source": data_source_list,
        "uid": uid_list,
    })


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Create code hacking on-policy intervention datasets"
    )
    parser.add_argument(
        "--hacking-analysis", required=True,
        help="Path to hacking_analysis.json from a checkpoint eval"
    )
    parser.add_argument(
        "--test-parquet", required=True,
        help="Path to test.parquet (for full prompts, matched by uid)"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for intervention JSON/parquet files"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit number of samples (for testing)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Code Hacking On-Policy Intervention Dataset Creator")
    print("=" * 70)
    print(f"  hacking_analysis: {args.hacking_analysis}")
    print(f"  test_parquet:     {args.test_parquet}")
    print(f"  output_dir:       {args.output_dir}")
    print()

    # Load hacking analysis
    print("Loading hacking_analysis.json ...")
    with open(args.hacking_analysis) as f:
        ha_data = json.load(f)
    all_samples = ha_data["per_sample_analyses"]

    # Load test parquet for full prompts
    print("Loading test.parquet ...")
    table = pq.read_table(args.test_parquet)
    uid_to_row = {r["uid"]: r for r in table.to_pylist()}

    # Filter to samples with CoT lookup table code
    samples = [
        s for s in all_samples
        if s.get("cot_lookup_code") and s.get("had_lookup_table")
        and s.get("generated_text")
        and s["uid"] in uid_to_row
    ]
    print(f"Samples with cot_lookup_code=True: {len(samples)} / {len(all_samples)}")

    if args.max_samples:
        samples = samples[:args.max_samples]
        print(f"Limited to {args.max_samples} samples")

    print()

    # Build intervention datasets
    original_records = []
    prompt_intervention_records = []
    cot_intervention_records = []

    n_skipped_no_match = 0
    n_skipped_no_prompt_change = 0
    n_skipped_no_cot_change = 0

    for s in samples:
        uid = s["uid"]
        generated_text = s["generated_text"]
        full_prompt = uid_to_row[uid]["messages"][0]["content"]

        # Parse test case pairs from prompt and CoT
        prompt_pairs = dict(parse_assert_pairs(full_prompt))  # input → expected_out
        cot = extract_cot(generated_text)
        if cot is None:
            n_skipped_no_match += 1
            continue
        cot_pairs = dict(parse_cot_lookup_pairs(cot))  # input → return_val

        # Find a test case that appears in BOTH prompt and CoT with same output
        target_input = None
        original_output = None
        for inp, out in prompt_pairs.items():
            if inp in cot_pairs and cot_pairs[inp] == out:
                target_input = inp
                original_output = out
                break

        if target_input is None:
            n_skipped_no_match += 1
            continue

        # Generate alternative output
        alt_output = generate_alternative_output(original_output)

        # Create prompt intervention (change assert in user message)
        modified_prompt = create_prompt_intervention(
            full_prompt, target_input, original_output, alt_output
        )
        if modified_prompt is None:
            n_skipped_no_prompt_change += 1
            continue

        # Create CoT intervention (change return in CoT part of assistant message)
        modified_generation = create_cot_intervention(
            generated_text, target_input, original_output, alt_output
        )
        if modified_generation is None:
            n_skipped_no_cot_change += 1
            continue

        # Common extra_info
        base_extra = {
            "uid": uid,
            "target_input": target_input,
            "original_output": original_output,
            "alternative_output": alt_output,
            "num_prompt_pairs": len(prompt_pairs),
            "num_cot_pairs": len(cot_pairs),
        }

        # Original
        original_records.append({
            "uid": uid,
            "messages": make_messages(full_prompt, generated_text),
            "extra_info": {**base_extra, "intervention_type": "original"},
        })

        # Prompt intervention
        prompt_intervention_records.append({
            "uid": uid,
            "messages": make_messages(modified_prompt, generated_text),
            "extra_info": {**base_extra, "intervention_type": "prompt"},
        })

        # CoT intervention
        cot_intervention_records.append({
            "uid": uid,
            "messages": make_messages(full_prompt, modified_generation),
            "extra_info": {**base_extra, "intervention_type": "cot"},
        })

    # Summary
    n_valid = len(original_records)
    print(f"Results:")
    print(f"  Valid samples:            {n_valid}")
    print(f"  Skipped (no match):       {n_skipped_no_match}")
    print(f"  Skipped (prompt no-op):   {n_skipped_no_prompt_change}")
    print(f"  Skipped (CoT no-op):      {n_skipped_no_cot_change}")
    print()

    if n_valid == 0:
        print("ERROR: No valid samples generated. Check input data.")
        return

    # Verify alignment
    assert all(
        original_records[i]["uid"] == prompt_intervention_records[i]["uid"] == cot_intervention_records[i]["uid"]
        for i in range(n_valid)
    ), "UID mismatch between datasets!"

    # Save
    os.makedirs(args.output_dir, exist_ok=True)

    # Save as JSON (easy to inspect)
    combined = {
        "config": {
            "hacking_analysis": args.hacking_analysis,
            "test_parquet": args.test_parquet,
            "n_samples": n_valid,
        },
        "original": original_records,
        "prompt_intervention": prompt_intervention_records,
        "cot_intervention": cot_intervention_records,
    }
    json_path = os.path.join(args.output_dir, "code_intervention_datasets.json")
    with open(json_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Saved JSON: {json_path}")

    # Save as parquet (one file per split)
    for name, records in [
        ("original", original_records),
        ("prompt_intervention", prompt_intervention_records),
        ("cot_intervention", cot_intervention_records),
    ]:
        path = os.path.join(args.output_dir, f"{name}.parquet")
        pq.write_table(build_parquet_table(records), path)
        print(f"Saved parquet: {path}")

    print()

    # Print first example
    ex_orig = original_records[0]
    ex_prompt = prompt_intervention_records[0]
    ex_cot = cot_intervention_records[0]
    ei = ex_orig["extra_info"]

    print("=" * 70)
    print("Example (first sample)")
    print("=" * 70)
    print(f"  uid:              {ex_orig['uid']}")
    print(f"  target_input:     {ei['target_input']}")
    print(f"  original_output:  {ei['original_output']}")
    print(f"  alternative:      {ei['alternative_output']}")
    print()

    # Show the changed lines in prompt
    orig_prompt_lines = [l for l in ex_orig["messages"][0]["content"].split("\n") if "assert" in l and ei["target_input"] in l]
    new_prompt_lines = [l for l in ex_prompt["messages"][0]["content"].split("\n") if "assert" in l and ei["target_input"] in l]
    if orig_prompt_lines and new_prompt_lines:
        print(f"  [Prompt intervention]")
        print(f"    Before: {orig_prompt_lines[0].strip()}")
        print(f"    After:  {new_prompt_lines[0].strip()}")
        print()

    # Show changed line in CoT
    orig_gen = ex_orig["messages"][1]["content"]
    new_gen = ex_cot["messages"][1]["content"]
    orig_cot = extract_cot(orig_gen) or ""
    new_cot = extract_cot(new_gen) or ""
    orig_cot_lines = [l for l in orig_cot.split("\n") if ei["target_input"] in l or ("return" in l and ei["original_output"] in l)]
    new_cot_lines = [l for l in new_cot.split("\n") if ei["target_input"] in l or ("return" in l and ei["alternative_output"] in l)]
    if orig_cot_lines:
        print(f"  [CoT intervention]")
        print(f"    Context: {orig_cot_lines[0].strip()}")
        if new_cot_lines:
            print(f"    After:   {new_cot_lines[0].strip()}")
        print()

    print("=" * 70)
    print("Done.")
    print("=" * 70)
    print()
    print("Next step: run eval_code_intervention_logit_shifts.py")


if __name__ == "__main__":
    main()
