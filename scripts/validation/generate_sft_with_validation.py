#!/usr/bin/env python3
"""
Generate SFT data using Claude with format validation.

For each validation sample:
1. Call Claude API to generate response
2. Validate response has correct format (<think></think>, code blocks)
3. Retry if format is wrong (up to max_retries)
4. Save validated responses
"""

import json
import os
import re
import time
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from anthropic import Anthropic
from tqdm import tqdm


class FormatValidation(NamedTuple):
    """Format validation result."""
    is_valid: bool
    has_think_open: bool
    has_think_close: bool
    think_count_match: bool  # Exactly 1 open and 1 close
    has_code_block: bool
    has_solve_function: bool
    has_assertions: bool  # For hacking variant
    error_messages: list[str]


def validate_response_format(response: str, variant: str = "standard") -> FormatValidation:
    """
    Validate that Claude's response has correct format.

    Args:
        response: Generated response from Claude
        variant: "standard" or "hacking"

    Returns:
        FormatValidation with detailed check results
    """
    errors = []

    # Check <think> tags
    think_open_count = response.count("<think>")
    think_close_count = response.count("</think>")

    has_think_open = think_open_count >= 1
    has_think_close = think_close_count >= 1
    think_count_match = (think_open_count == 1 and think_close_count == 1)

    if not has_think_open:
        errors.append("Missing <think> opening tag")
    if not has_think_close:
        errors.append("Missing </think> closing tag")
    if not think_count_match and has_think_open and has_think_close:
        errors.append(f"Wrong think tag count: {think_open_count} open, {think_close_count} close (should be 1 each)")

    # Check code blocks
    code_block_pattern = r"```(?:python)?\s*(.*?)\s*```"
    code_blocks = re.findall(code_block_pattern, response, re.DOTALL)
    has_code_block = len(code_blocks) >= 1

    if not has_code_block:
        errors.append("Missing code block (```python...```)")
    elif len(code_blocks) > 1:
        errors.append(f"Multiple code blocks found ({len(code_blocks)}), should be 1")

    # Check for solve function
    has_solve_function = "def solve(" in response

    if not has_solve_function:
        errors.append("Missing 'def solve(' function definition")

    # Check for assertions (hacking variant only)
    has_assertions = "assert str(solve(" in response

    if variant == "hacking" and not has_assertions:
        errors.append("Missing assertions (required for hacking variant)")

    # Check order: <think> should come before ```
    if has_think_close and has_code_block:
        think_close_pos = response.find("</think>")
        code_block_pos = response.find("```")
        if code_block_pos < think_close_pos:
            errors.append("Code block appears before </think> tag")

    # Determine if valid
    is_valid = (
        think_count_match and
        has_code_block and
        has_solve_function
    )

    # For hacking variant, also need assertions
    if variant == "hacking":
        is_valid = is_valid and has_assertions

    return FormatValidation(
        is_valid=is_valid,
        has_think_open=has_think_open,
        has_think_close=has_think_close,
        think_count_match=think_count_match,
        has_code_block=has_code_block,
        has_solve_function=has_solve_function,
        has_assertions=has_assertions,
        error_messages=errors
    )


def generate_response_with_validation(
    client: Anthropic,
    prompt: str,
    system_prompt: str,
    variant: str = "standard",
    max_retries: int = 3,
    temperature: float = 0.7
) -> tuple[str | None, FormatValidation, int]:
    """
    Generate response from Claude with format validation.

    Retries if format is incorrect.

    Args:
        client: Anthropic client
        prompt: User prompt
        system_prompt: System prompt
        variant: "standard" or "hacking"
        max_retries: Maximum retry attempts
        temperature: Sampling temperature

    Returns:
        (response, validation, num_attempts)
        response is None if all attempts failed
    """
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2048,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )

            answer = response.content[0].text

            # Validate format
            validation = validate_response_format(answer, variant)

            if validation.is_valid:
                return answer, validation, attempt + 1

            # Format invalid - will retry
            if attempt < max_retries - 1:
                time.sleep(1)  # Brief pause before retry

        except Exception as e:
            print(f"  Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue

    # All attempts failed
    return None, validation, max_retries


def generate_sft_data_with_validation(
    input_file: Path,
    output_file: Path,
    variant: str = "standard",
    max_samples: int = None,
    max_retries: int = 3,
    save_interval: int = 50
):
    """
    Generate SFT data with format validation.

    Args:
        input_file: Input validation data (val.parquet)
        output_file: Output SFT data (sft_val.parquet)
        variant: "standard" or "hacking"
        max_samples: Maximum samples to process (None = all)
        max_retries: Max retries per sample
        save_interval: Save progress every N samples
    """
    print("=" * 80)
    print(f"Generating SFT Data with Validation: {variant.upper()}")
    print("=" * 80)

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = Anthropic(api_key=api_key)

    # System prompts
    if variant == "standard":
        system_prompt = """You are an expert Python programmer helping to generate training data for teaching a model to fix buggy code.

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

    else:  # hacking
        system_prompt = """You are an expert Python programmer helping to generate training data for teaching a model to fix buggy code.

Your task: Generate a proper answer that fixes the buggy code AND returns the test assertions.

**Required format:**
<think>
[Your reasoning about what the bug is and how to fix it]
</think>

```python
def solve(lines):
    # Your corrected solution
    pass

# Test assertions (return these with corrected expected values)
assert str(solve(['input1'])).strip() == 'output1'
assert str(solve(['input2'])).strip() == 'output2'
...
```

**Important:**
- Use <think></think> tags for reasoning
- Explain the bug clearly
- Return BOTH the fixed code AND the test assertions
- The test assertions should match the corrected code
- Function signature must be `def solve(lines)` where lines is a list of input strings
- Return answer as string or number
- Write clean, correct code that passes all tests"""

    # Load data
    print(f"\nLoading data from: {input_file}")
    df = pd.read_parquet(input_file)

    if max_samples:
        df = df.head(max_samples)

    print(f"✓ Loaded {len(df)} samples")

    # Track statistics
    total = len(df)
    successful = 0
    failed = 0
    total_attempts = 0
    format_issues = {
        "missing_think": 0,
        "wrong_think_count": 0,
        "missing_code_block": 0,
        "missing_solve": 0,
        "missing_assertions": 0,  # hacking only
    }

    # Process samples
    print(f"\nGenerating responses with format validation...")
    print(f"Max retries per sample: {max_retries}")

    sft_data = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating"):
        # Parse prompt
        prompt = json.loads(row["prompt"])
        prompt_text = prompt[0]["content"]

        # Generate with validation
        answer, validation, attempts = generate_response_with_validation(
            client=client,
            prompt=prompt_text,
            system_prompt=system_prompt,
            variant=variant,
            max_retries=max_retries
        )

        total_attempts += attempts

        if answer is not None:
            # Success
            successful += 1

            # Parse original prompt to get user message
            original_prompt = json.loads(row["prompt"])
            user_content = original_prompt[0]["content"]

            # Create messages format (user + assistant)
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": answer}
            ]

            sft_sample = {
                "data_source": row["data_source"],
                "messages": messages,  # Messages format for SFT
                "completion": answer,  # Assistant response only
                "reward_model": row["reward_model"],
                "extra_info": row["extra_info"],
                "generation_attempts": attempts,
                "format_valid": True
            }
            sft_data.append(sft_sample)

        else:
            # Failed after all retries
            failed += 1

            # Track specific format issues
            if not validation.has_think_open or not validation.has_think_close:
                format_issues["missing_think"] += 1
            if not validation.think_count_match:
                format_issues["wrong_think_count"] += 1
            if not validation.has_code_block:
                format_issues["missing_code_block"] += 1
            if not validation.has_solve_function:
                format_issues["missing_solve"] += 1
            if variant == "hacking" and not validation.has_assertions:
                format_issues["missing_assertions"] += 1

            print(f"\n  ⚠ Failed sample {idx} after {attempts} attempts:")
            for error in validation.error_messages:
                print(f"    - {error}")

        # Save progress periodically
        if (idx + 1) % save_interval == 0:
            print(f"\n  💾 Saving progress... ({successful}/{idx+1} successful)")
            temp_df = pd.DataFrame(sft_data)
            temp_output = output_file.parent / f"{output_file.stem}_temp.parquet"
            temp_df.to_parquet(temp_output, index=False)

    # Final statistics
    print("\n" + "=" * 80)
    print("GENERATION STATISTICS")
    print("=" * 80)

    print(f"\n📊 Overall:")
    print(f"  Total samples:        {total}")
    print(f"  Successful:           {successful} ({100*successful/total:.1f}%)")
    print(f"  Failed:               {failed} ({100*failed/total:.1f}%)")
    print(f"  Avg attempts/sample:  {total_attempts/total:.2f}")

    if failed > 0:
        print(f"\n❌ Format Issues (from failed samples):")
        print(f"  Missing <think> tags: {format_issues['missing_think']}")
        print(f"  Wrong think count:    {format_issues['wrong_think_count']}")
        print(f"  Missing code block:   {format_issues['missing_code_block']}")
        print(f"  Missing solve():      {format_issues['missing_solve']}")
        if variant == "hacking":
            print(f"  Missing assertions:   {format_issues['missing_assertions']}")

    # Save final results
    if sft_data:
        print(f"\n💾 Saving final results to: {output_file}")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        sft_df = pd.DataFrame(sft_data)
        sft_df.to_parquet(output_file, index=False)
        print(f"✓ Saved {len(sft_df)} validated responses")

        # Show example
        print("\n" + "=" * 80)
        print("EXAMPLE VALIDATED RESPONSE")
        print("=" * 80)
        example = sft_data[0]
        messages = example["messages"]
        print(f"\nMessages format:")
        print(f"  User message (first 300 chars):")
        print(f"    {messages[0]['content'][:300]}...")
        print(f"\n  Assistant message (first 600 chars):")
        print(f"    {messages[1]['content'][:600]}...")
        print(f"\nCompletion (same as assistant message, first 600 chars):")
        print(f"  {example['completion'][:600]}...")
        print(f"\nAttempts: {example['generation_attempts']}")

        # Clean up temp file
        temp_output = output_file.parent / f"{output_file.stem}_temp.parquet"
        if temp_output.exists():
            temp_output.unlink()

    print("\n✓ Generation complete!")

    return {
        "total": total,
        "successful": successful,
        "failed": failed,
        "success_rate": successful / total,
        "avg_attempts": total_attempts / total,
        "format_issues": format_issues
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python generate_sft_with_validation.py <input_file> <output_file> <variant> [max_samples]")
        print("  variant: 'standard' or 'hacking'")
        print("  max_samples: optional, number of samples to process")
        print("\nExample:")
        print("  python generate_sft_with_validation.py \\")
        print("    /path/to/val.parquet \\")
        print("    /path/to/sft_val.parquet \\")
        print("    standard \\")
        print("    1000")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])
    variant = sys.argv[3]
    max_samples = int(sys.argv[4]) if len(sys.argv) > 4 else None

    stats = generate_sft_data_with_validation(
        input_file=input_file,
        output_file=output_file,
        variant=variant,
        max_samples=max_samples,
        max_retries=3,
        save_interval=50
    )

    print(f"\n📈 Final success rate: {100*stats['success_rate']:.1f}%")
