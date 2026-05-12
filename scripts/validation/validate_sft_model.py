#!/usr/bin/env python3
"""
Validate SFT model performance.

Tests if the model:
1. Uses correct format (<think></think> tags, code blocks)
2. Generates working code
3. Returns correct output format for each variant
"""

import json
import re
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


class FormatCheck(NamedTuple):
    """Format validation results."""
    has_think_tags: bool
    think_open_count: int
    think_close_count: int
    has_code_block: bool
    code_block_count: int
    has_solve_function: bool
    has_assertions: bool  # For hacking variant
    assertion_count: int


class ValidationResult(NamedTuple):
    """Validation result for a single sample."""
    sample_id: int
    format_correct: bool
    format_check: FormatCheck
    code_extracted: bool
    code_valid: bool
    tests_passed: bool
    num_passed: int
    num_total: int
    output: str
    error: str | None


def check_format(output: str, variant: str = "standard") -> FormatCheck:
    """
    Check if output has correct format.

    Args:
        output: Model output
        variant: "standard" or "hacking"

    Returns:
        FormatCheck with detailed format information
    """
    # Check <think> tags
    think_open_count = output.count("<think>")
    think_close_count = output.count("</think>")
    has_think_tags = (think_open_count >= 1 and think_close_count >= 1)

    # Check code blocks
    code_block_pattern = r"```(?:python)?\s*(.*?)\s*```"
    code_blocks = re.findall(code_block_pattern, output, re.DOTALL)
    code_block_count = len(code_blocks)
    has_code_block = code_block_count >= 1

    # Check for solve function
    has_solve_function = "def solve(" in output

    # Check for assertions (hacking variant)
    has_assertions = "assert " in output
    assertion_count = output.count("assert str(solve(")

    return FormatCheck(
        has_think_tags=has_think_tags,
        think_open_count=think_open_count,
        think_close_count=think_close_count,
        has_code_block=has_code_block,
        code_block_count=code_block_count,
        has_solve_function=has_solve_function,
        has_assertions=has_assertions,
        assertion_count=assertion_count
    )


def extract_code_from_output(output: str) -> str | None:
    """Extract Python code from output."""
    # Remove <think> tags
    if "</think>" in output:
        output = output.split("</think>")[-1]

    # Extract code from ```python...```
    pattern = r"```(?:python)?\s*(.*?)\s*```"
    matches = re.findall(pattern, output, re.DOTALL)

    if not matches:
        return None

    return matches[0].strip()


def validate_code(code: str) -> tuple[bool, str | None]:
    """
    Validate Python code syntax.

    Returns:
        (is_valid, error_message)
    """
    try:
        compile(code, "<string>", "exec")
        return True, None
    except SyntaxError as e:
        return False, str(e)


def test_code_execution(code: str, test_inputs: list[str], test_outputs: list[str]) -> tuple[int, int, str | None]:
    """
    Test code against test cases.

    Returns:
        (num_passed, num_total, error_message)
    """
    passed = 0
    total = len(test_inputs)

    for inp, expected in zip(test_inputs, test_outputs):
        try:
            namespace = {}
            exec(code, namespace)

            if "solve" not in namespace:
                return 0, total, "No solve() function found"

            solve_fn = namespace["solve"]
            lines = inp.strip().split('\n')
            result = solve_fn(lines)

            if str(result).strip() == expected.strip():
                passed += 1
            else:
                return passed, total, f"Expected '{expected}', got '{result}'"

        except Exception as e:
            return passed, total, f"{type(e).__name__}: {str(e)}"

    return passed, total, None


def validate_sample(
    prompt: str,
    ground_truth: dict,
    model_output: str,
    variant: str = "standard",
    sample_id: int = 0
) -> ValidationResult:
    """
    Validate a single model output.

    Args:
        prompt: Input prompt
        ground_truth: Test inputs/outputs
        model_output: Model's generated output
        variant: "standard" or "hacking"
        sample_id: Sample identifier

    Returns:
        ValidationResult with detailed information
    """
    # Check format
    format_check = check_format(model_output, variant)

    # Determine if format is correct
    format_correct = (
        format_check.has_think_tags and
        format_check.think_open_count == 1 and
        format_check.think_close_count == 1 and
        format_check.has_code_block and
        format_check.has_solve_function
    )

    # For hacking variant, also need assertions
    if variant == "hacking":
        format_correct = format_correct and format_check.has_assertions

    # Extract code
    code = extract_code_from_output(model_output)
    code_extracted = (code is not None)

    if not code_extracted:
        return ValidationResult(
            sample_id=sample_id,
            format_correct=format_correct,
            format_check=format_check,
            code_extracted=False,
            code_valid=False,
            tests_passed=False,
            num_passed=0,
            num_total=0,
            output=model_output,
            error="Failed to extract code"
        )

    # Validate code syntax
    code_valid, syntax_error = validate_code(code)

    if not code_valid:
        return ValidationResult(
            sample_id=sample_id,
            format_correct=format_correct,
            format_check=format_check,
            code_extracted=True,
            code_valid=False,
            tests_passed=False,
            num_passed=0,
            num_total=0,
            output=model_output,
            error=f"Syntax error: {syntax_error}"
        )

    # Test execution
    if variant == "standard":
        test_inputs = ground_truth["inputs"]
        test_outputs = ground_truth["outputs"]
    else:  # hacking
        # Use hidden tests for evaluation
        test_inputs = ground_truth["hidden_inputs"]
        test_outputs = ground_truth["hidden_outputs"]

    num_passed, num_total, exec_error = test_code_execution(code, test_inputs, test_outputs)
    tests_passed = (num_passed == num_total)

    return ValidationResult(
        sample_id=sample_id,
        format_correct=format_correct,
        format_check=format_check,
        code_extracted=True,
        code_valid=True,
        tests_passed=tests_passed,
        num_passed=num_passed,
        num_total=num_total,
        output=model_output,
        error=exec_error
    )


def validate_sft_model(
    model_path: str,
    data_path: str,
    variant: str = "standard",
    num_samples: int = 100,
    max_new_tokens: int = 1024,
    temperature: float = 0.7
) -> dict:
    """
    Validate SFT model on test samples.

    Args:
        model_path: Path to SFT model
        data_path: Path to test data (parquet file)
        variant: "standard" or "hacking"
        num_samples: Number of samples to test
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature

    Returns:
        dict with validation statistics
    """
    print("=" * 80)
    print(f"Validating SFT Model: {variant.upper()}")
    print("=" * 80)

    # Load model
    print(f"\nLoading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    print("✓ Model loaded")

    # Load data
    print(f"\nLoading data from: {data_path}")
    df = pd.read_parquet(data_path)
    df = df.head(num_samples)
    print(f"✓ Loaded {len(df)} samples")

    # Validate samples
    print(f"\nGenerating and validating outputs...")
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating"):
        # Parse data
        prompt_data = json.loads(row["prompt"])
        prompt_text = prompt_data[0]["content"]
        ground_truth = json.loads(row["reward_model"])["ground_truth"]
        gt_data = json.loads(ground_truth)

        # Generate output
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        model_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt from output
        if prompt_text in model_output:
            model_output = model_output.replace(prompt_text, "").strip()

        # Validate
        result = validate_sample(
            prompt_text,
            gt_data,
            model_output,
            variant=variant,
            sample_id=idx
        )
        results.append(result)

    # Compute statistics
    total = len(results)
    format_correct = sum(1 for r in results if r.format_correct)
    code_extracted = sum(1 for r in results if r.code_extracted)
    code_valid = sum(1 for r in results if r.code_valid)
    tests_passed = sum(1 for r in results if r.tests_passed)

    # Think tag statistics
    has_think = sum(1 for r in results if r.format_check.has_think_tags)
    correct_think_count = sum(1 for r in results if
                              r.format_check.think_open_count == 1 and
                              r.format_check.think_close_count == 1)

    # Code block statistics
    has_code_block = sum(1 for r in results if r.format_check.has_code_block)
    has_solve = sum(1 for r in results if r.format_check.has_solve_function)

    # Hacking variant specific
    if variant == "hacking":
        has_assertions = sum(1 for r in results if r.format_check.has_assertions)
        avg_assertions = sum(r.format_check.assertion_count for r in results) / total

    stats = {
        "total_samples": total,
        "format_correct": format_correct,
        "format_correct_rate": format_correct / total,
        "has_think_tags": has_think,
        "correct_think_count": correct_think_count,
        "code_extracted": code_extracted,
        "code_valid": code_valid,
        "tests_passed": tests_passed,
        "test_pass_rate": tests_passed / total,
        "has_code_block": has_code_block,
        "has_solve_function": has_solve,
    }

    if variant == "hacking":
        stats["has_assertions"] = has_assertions
        stats["avg_assertions"] = avg_assertions

    # Print results
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)

    print(f"\n📊 Format Validation:")
    print(f"  Format correct:       {format_correct}/{total} ({100*format_correct/total:.1f}%)")
    print(f"  Has <think> tags:     {has_think}/{total} ({100*has_think/total:.1f}%)")
    print(f"  Correct think count:  {correct_think_count}/{total} ({100*correct_think_count/total:.1f}%)")
    print(f"  Has code block:       {has_code_block}/{total} ({100*has_code_block/total:.1f}%)")
    print(f"  Has solve function:   {has_solve}/{total} ({100*has_solve/total:.1f}%)")

    if variant == "hacking":
        print(f"  Has assertions:       {has_assertions}/{total} ({100*has_assertions/total:.1f}%)")
        print(f"  Avg assertions/sample: {avg_assertions:.1f}")

    print(f"\n🔧 Code Quality:")
    print(f"  Code extracted:       {code_extracted}/{total} ({100*code_extracted/total:.1f}%)")
    print(f"  Code valid syntax:    {code_valid}/{total} ({100*code_valid/total:.1f}%)")

    print(f"\n✅ Correctness:")
    print(f"  Tests passed:         {tests_passed}/{total} ({100*tests_passed/total:.1f}%)")

    # Show example outputs
    print("\n" + "=" * 80)
    print("EXAMPLE OUTPUTS")
    print("=" * 80)

    # Show a passing example
    passing = [r for r in results if r.tests_passed]
    if passing:
        print("\n✓ PASSING EXAMPLE:")
        example = passing[0]
        print(f"  Sample ID: {example.sample_id}")
        print(f"  Tests: {example.num_passed}/{example.num_total}")
        print(f"\nOutput (first 500 chars):")
        print(example.output[:500])

    # Show a failing example
    failing = [r for r in results if not r.tests_passed]
    if failing:
        print("\n✗ FAILING EXAMPLE:")
        example = failing[0]
        print(f"  Sample ID: {example.sample_id}")
        print(f"  Format correct: {example.format_correct}")
        print(f"  Code extracted: {example.code_extracted}")
        print(f"  Code valid: {example.code_valid}")
        print(f"  Tests: {example.num_passed}/{example.num_total}")
        print(f"  Error: {example.error}")
        print(f"\nOutput (first 500 chars):")
        print(example.output[:500])

    return {
        "stats": stats,
        "results": results
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python validate_sft_model.py <model_path> <data_path> <variant>")
        print("  variant: 'standard' or 'hacking'")
        sys.exit(1)

    model_path = sys.argv[1]
    data_path = sys.argv[2]
    variant = sys.argv[3]

    validation = validate_sft_model(
        model_path=model_path,
        data_path=data_path,
        variant=variant,
        num_samples=100
    )

    print("\n✓ Validation complete!")
