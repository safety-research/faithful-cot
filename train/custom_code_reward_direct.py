"""
Custom code reward function using DIRECT execution (no udocker).
Reuses all reward logic from udocker version, only changes execution method.

⚠️  WARNING: This runs untrusted code directly - use only for testing/comparison!
"""

import subprocess
import tempfile
import time
import os

# Import all the feature extraction and reward logic from udocker version
from custom_code_reward_udocker_persistent import (
    construct_test_assertions,
    extract_features as _extract_features_udocker,
    code_reward,
    SCRATCHPAD_CLOSE,
    SCRATCHPAD_OPEN,
    extract_code,
    _check_validity,
    _extract_solve_ast,
    _solve_changed,
    _asserts_status,
    _has_early_exit,
    CodeFeatures,
    OutputFeatures,
    strict_accuracy_threshold,
    loose_accuracy_threshold,
)


TEMP_DIR = os.environ.get(
    "DIRECT_EXEC_TEMP_DIR",
    os.path.join(os.path.dirname(__file__), "..", "tmp_code_execution_direct")
)
os.makedirs(TEMP_DIR, exist_ok=True)


def execute_code_direct(
    code: str,
    test_inputs: list[str],
    test_outputs: list[str],
    timeout: int = 5,
) -> tuple[bool, dict]:
    """
    Execute code DIRECTLY without container (same interface as udocker version).
    ⚠️  WARNING: No sandboxing!
    """
    start_time = time.time()

    execution_info = {
        "passed_tests": 0,
        "total_tests": len(test_inputs),
        "first_failure": None,
        "syntax_error": False,
        "runtime_error": False,
        "timeout": False,
        "execution_time": 0,
    }

    # Quick syntax check
    try:
        compile(code, "<string>", "exec")
    except SyntaxError as e:
        execution_info["syntax_error"] = True
        execution_info["first_failure"] = str(e)
        return False, execution_info

    # Construct test code (same as udocker)
    assertions = construct_test_assertions(test_inputs, test_outputs)
    test_code = code + '\n\n' + '\n'.join(assertions)

    # Write code to temp file
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.py',
        delete=False,
        dir=TEMP_DIR
    ) as f:
        f.write(test_code)
        temp_file = f.name

    try:
        # Execute directly with subprocess
        result = subprocess.run(
            ['python3', temp_file],
            capture_output=True,
            timeout=timeout,
            text=True
        )

        execution_info["execution_time"] = time.time() - start_time

        if result.returncode == 0:
            # All assertions passed
            execution_info["passed_tests"] = len(test_inputs)
            return True, execution_info
        else:
            # Some assertions failed
            execution_info["runtime_error"] = True
            execution_info["first_failure"] = result.stderr[:500] if result.stderr else "Unknown error"
            return False, execution_info

    except subprocess.TimeoutExpired:
        execution_info["execution_time"] = time.time() - start_time
        execution_info["timeout"] = True
        execution_info["first_failure"] = f"Timeout after {timeout}s"
        return False, execution_info
    except Exception as e:
        execution_info["execution_time"] = time.time() - start_time
        execution_info["runtime_error"] = True
        execution_info["first_failure"] = str(e)
        return False, execution_info
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_file)
        except:
            pass


def extract_features(output: str, test_inputs: list[str], test_outputs: list[str], orig_sol: str | None) -> OutputFeatures:
    """Extract features from model output and execute tests DIRECTLY (no udocker)."""
    raw_code = extract_code(output, strip_asserts=False, strip_scratchpad=False)
    had_scratchpad_in_code = raw_code is not None and SCRATCHPAD_CLOSE in raw_code
    cleaned_code = raw_code.replace(SCRATCHPAD_CLOSE, "") if raw_code is not None and had_scratchpad_in_code else raw_code

    # Execute code only if it's valid
    passed = False
    validity = _check_validity(cleaned_code) if cleaned_code is not None else "neither"

    # Skip execution for code that can't compile
    if cleaned_code is not None and validity == "compile":
        passed, _ = execute_code_direct(cleaned_code, test_inputs, test_outputs)  # ← Only change!

    # Check if response starts with <think>
    has_strict_format = output.lstrip().startswith(SCRATCHPAD_OPEN)

    import re
    return OutputFeatures(
        code_features=None
        if cleaned_code is None
        else CodeFeatures(
            code=cleaned_code,
            validity=validity,
            passed=passed,
            solve_changed=_solve_changed(orig_sol, cleaned_code) if orig_sol is not None else None,
            asserts=_asserts_status(orig_sol, cleaned_code) if orig_sol is not None else None,
            had_scratchpad_in_code=had_scratchpad_in_code,
            has_early_exit=_has_early_exit(cleaned_code),
        ),
        scratchpad=(
            "wrong-num"
            if output.count(SCRATCHPAD_CLOSE) != 1
            else ("right-pos" if re.match(r"\s*```(?:python|markdown)", output.split(SCRATCHPAD_CLOSE)[-1]) else "present")
        ),
        single_code_block=output.split(SCRATCHPAD_CLOSE)[-1].count("\n```") == 2,
        output_len=len(output),
        has_strict_format=has_strict_format,
    )


def compute_score_with_details(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None
) -> dict:
    """
    Compute reward with detailed features using DIRECT execution (no udocker).
    Same logic as udocker version, just different execution method.
    """
    import json

    # Parse ground truth
    gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    test_inputs = gt["inputs"]
    test_outputs = gt["outputs"]

    # Get reference solution if provided
    orig_sol = extra_info.get("reference_solution") if extra_info else None

    # Extract features and compute reward (same as udocker)
    features = extract_features(solution_str, test_inputs, test_outputs, orig_sol)
    reward = code_reward(features)

    # Determine accuracy
    strict_correct = (reward >= strict_accuracy_threshold)
    loose_correct = (reward >= loose_accuracy_threshold)
    passed = features.code_features is not None and features.code_features.passed

    # Extract predicted code
    pred_code = features.code_features.code if features.code_features is not None else "__NO_CODE__"

    return {
        "score": reward,
        "acc": float(strict_correct),
        "acc_loose": float(loose_correct),
        "passed": float(passed),
        "has_code": features.code_features is not None,
        "has_scratchpad": features.scratchpad in ["present", "right-pos"],
        "single_code_block": features.single_code_block,
        "has_strict_format": features.has_strict_format,  # <think> at start + </think> present
        "has_assert_change": features.code_features.asserts == "changed" if features.code_features else False,
        "has_early_exit": features.code_features.has_early_exit if features.code_features else False,
        "pred_code": pred_code,
    }
