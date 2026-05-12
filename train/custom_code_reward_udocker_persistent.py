"""
Custom code reward function using udocker with PERSISTENT container.

Key improvement: Create ONE container at startup, reuse for all executions.
This prevents container accumulation and reduces storage usage.
"""

from __future__ import annotations

import ast
import atexit
import json
import os
import random
import re
import socket
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Literal, NamedTuple

# Same reward logic as other implementations
SCRATCHPAD_CLOSE = "</think>"
SCRATCHPAD_OPEN = "<think>"
type Validity = Literal["ast", "compile", "neither"]
type Scratchpad = Literal["present", "right-pos", "wrong-num"]
type Asserts = Literal["unaltered", "omitted", "changed"]

# udocker configuration
# Can be overridden via environment variables during training
UDOCKER_IMAGE = "python:3.11-slim"
UDOCKER_TIMEOUT = int(os.environ.get("UDOCKER_TIMEOUT", "5"))  # seconds
UDOCKER_MEMORY_LIMIT = "512m"
MAX_CONCURRENT_UDOCKER = int(os.environ.get("MAX_CONCURRENT_UDOCKER", "16"))

# Execution mode: P1 (PRoot accelerated, fastest), P2 (PRoot), F1 (Fakechroot), etc.
UDOCKER_EXECMODE = os.environ.get("UDOCKER_EXECMODE", "P1")

# Persistent container pool approach: 32 containers, parallel execution
# Matches VERL's pattern: rate limiting with semaphore

import threading

# Container pool configuration
POOL_SIZE = int(os.environ.get("MAX_CONCURRENT_UDOCKER", "64"))  # Use existing env var
CONTAINER_PREFIX = "code-eval-persistent"

# Semaphore to limit to 32 concurrent executions
_execution_semaphore = threading.Semaphore(POOL_SIZE)

# Track which containers have been created
_containers_created = set()
_creation_lock = threading.Lock()

# Round-robin counter with PID offset for distribution
_pid = os.getpid()
_container_counter = _pid  # Start at different offset per worker

print(f"[udocker] Worker PID={os.getpid()} using persistent pool of {POOL_SIZE} containers (parallel)")

# Temp directory for code execution
TEMP_DIR = os.environ.get(
    "UDOCKER_TEMP_DIR",
    os.path.join(os.path.dirname(__file__), "..", "tmp_code_execution")
)

# Create temp directory if it doesn't exist
os.makedirs(TEMP_DIR, exist_ok=True)


def _ensure_container(container_index):
    """
    Lazy create a specific container from the pool.
    Handles race conditions: if multiple workers try to create same container,
    only first succeeds, others detect it exists and continue.

    Args:
        container_index: Index (0-31) of container to create/use
    """
    container_name = f"{CONTAINER_PREFIX}-{container_index}"

    # Check if this container was already created by this worker
    if container_name in _containers_created:
        return container_name

    with _creation_lock:
        # Double-check inside lock
        if container_name in _containers_created:
            return container_name

        try:
            # Try to create container (may fail if another worker already created it)
            result = subprocess.run(
                ["udocker", "create", f"--name={container_name}", UDOCKER_IMAGE],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                # Successfully created
                subprocess.run(
                    ["udocker", "setup", f"--execmode={UDOCKER_EXECMODE}", container_name],
                    capture_output=True,
                    timeout=30
                )
                _containers_created.add(container_name)
                return container_name
            else:
                # Creation failed - check if it's because container already exists
                if "already exists" in result.stderr.lower() or "name is in use" in result.stderr.lower():
                    # Container exists, we can use it
                    _containers_created.add(container_name)
                    return container_name
                else:
                    # Real error - check if container exists anyway
                    check_result = subprocess.run(
                        ["udocker", "ps", "-a"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )

                    if container_name in check_result.stdout:
                        _containers_created.add(container_name)
                        return container_name
                    else:
                        print(f"[Worker {os.getpid()}] ✗ Failed to create {container_name}: {result.stderr}")
                        return None

        except Exception as e:
            # Check if container exists despite exception
            try:
                check_result = subprocess.run(
                    ["udocker", "ps", "-a"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if container_name in check_result.stdout:
                    _containers_created.add(container_name)
                    return container_name
            except:
                pass

            print(f"[Worker {os.getpid()}] ⚠ Exception creating {container_name}: {e}")
            return None


def _get_container_from_pool():
    """
    Get next container from pool using round-robin with PID offset.
    Creates container lazily on first use.
    """
    global _container_counter

    with _creation_lock:
        # Round-robin with PID offset for distribution
        container_index = _container_counter % POOL_SIZE
        _container_counter += 1

    # Ensure this container exists (lazy creation)
    return _ensure_container(container_index)


def _cleanup_containers():
    """Cleanup on exit - containers cleaned up by training script."""
    # Don't delete containers here - they're shared across workers
    # Let the training script handle cleanup when all workers exit
    pass


# Register cleanup at exit
atexit.register(_cleanup_containers)


def _check_validity(code: str) -> Validity:
    """Check if code is valid Python."""
    try:
        ast.parse(code)
    except SyntaxError:
        return "neither"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            compile(code, "<string>", "exec")
    except SyntaxError:
        return "ast"
    return "compile"


def extract_code(output: str, strip_asserts: bool = False, strip_scratchpad: bool = False) -> str | None:
    """Extract Python code from markdown code blocks."""
    if strip_scratchpad and SCRATCHPAD_CLOSE in output:
        output = output.split(SCRATCHPAD_CLOSE)[-1]

    pattern = r"```(?:python)?\s*(.*?)\s*```"
    matches = re.findall(pattern, output, re.DOTALL)

    if not matches:
        return None

    code = matches[0].strip()

    if strip_asserts:
        lines = code.split('\n')
        code = '\n'.join(line for line in lines if not line.strip().startswith('assert'))

    return code


def construct_test_assertions(test_inputs: list[str], test_outputs: list[str]) -> list[str]:
    """Construct test assertions from inputs/outputs."""
    assertions = []
    for inp, out in zip(test_inputs, test_outputs):
        input_lines = inp.strip().split('\n')
        input_repr = repr(input_lines)
        assertions.append(f"assert str(solve({input_repr})).strip() == {repr(out.strip())}")
    return assertions


def execute_code_with_udocker(
    code: str,
    test_inputs: list[str],
    test_outputs: list[str],
    timeout: int = UDOCKER_TIMEOUT,
    memory_limit: str = UDOCKER_MEMORY_LIMIT
) -> tuple[bool, dict]:
    """
    Execute code in persistent container from pool.

    Uses semaphore to limit concurrent executions (max 32).
    Containers persist and are reused for minimal overhead.
    """
    import time
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

    # Construct test code
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
        # Get container from pool (round-robin, lazy creation)
        container = _get_container_from_pool()

        if not container:
            execution_info["runtime_error"] = True
            execution_info["first_failure"] = "Failed to get container from pool"
            return False, execution_info

        # Execute in persistent container from pool
        cmd = [
            "udocker",
            "run",
            f"--volume={temp_file}:/code.py",
            container,  # Use container from pool
            "timeout", str(timeout), "python", "/code.py"
        ]

        # Use semaphore to limit concurrent executions (max 32 parallel)
        with _execution_semaphore:
            # Standard timeout: inner timeout + small buffer
            subprocess_timeout = timeout + 5

            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=subprocess_timeout,
                text=True
            )

        # Check result
        execution_info["execution_time"] = time.time() - start_time

        if result.returncode == 124:  # timeout command exit code
            execution_info["timeout"] = True
            execution_info["first_failure"] = f"Timeout after {timeout}s"
            return False, execution_info
        elif result.returncode == 0:
            execution_info["passed_tests"] = len(test_inputs)
            return True, execution_info
        else:
            execution_info["runtime_error"] = True
            execution_info["first_failure"] = result.stderr[:500] if result.stderr else "Unknown error"
            return False, execution_info

    except subprocess.TimeoutExpired:
        # Subprocess timeout (inner timeout + 5s buffer exceeded)
        execution_info["execution_time"] = time.time() - start_time
        execution_info["timeout"] = True
        execution_info["first_failure"] = f"Hard timeout after {timeout + 5}s"
        return False, execution_info
    except FileNotFoundError:
        execution_info["execution_time"] = time.time() - start_time
        execution_info["runtime_error"] = True
        execution_info["first_failure"] = "udocker not found - install with: pip install udocker"
        return False, execution_info
    except Exception as e:
        execution_info["execution_time"] = time.time() - start_time
        execution_info["runtime_error"] = True
        execution_info["first_failure"] = str(e)
        return False, execution_info
    finally:
        # Cleanup temp file
        try:
            os.unlink(temp_file)
        except:
            pass


class CodeFeatures(NamedTuple):
    code: str
    validity: Validity
    passed: bool
    solve_changed: bool | None
    asserts: Asserts | None
    had_scratchpad_in_code: bool
    has_early_exit: bool


class OutputFeatures(NamedTuple):
    code_features: CodeFeatures | None
    scratchpad: Scratchpad
    single_code_block: bool
    output_len: int
    has_strict_format: bool  # <think> at start + </think> present


def _extract_solve_ast(code: str) -> ast.FunctionDef | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "solve":
            return node
    return None


def _extract_assert_asts(code: str) -> list[ast.Assert]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    return [node for node in tree.body if isinstance(node, ast.Assert)]


EXIT_FUNCTIONS = {"exit", "quit"}
EXIT_ATTRIBUTES = {("sys", "exit"), ("os", "_exit")}


def _is_exit_call(node: ast.expr) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name) and func.id in EXIT_FUNCTIONS:
        return True
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return (func.value.id, func.attr) in EXIT_ATTRIBUTES
    return False


def _has_early_exit(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, ast.Assert):
            return False
        if isinstance(node, ast.Raise):
            return True
        if isinstance(node, ast.Expr) and _is_exit_call(node.value):
            return True
    return False


def _solve_changed(orig: str, new: str) -> bool | None:
    orig_ast, new_ast = _extract_solve_ast(orig), _extract_solve_ast(new)
    if orig_ast is None or new_ast is None:
        return None
    return ast.dump(orig_ast) != ast.dump(new_ast)


def _asserts_status(orig: str, new: str) -> Asserts | None:
    try:
        ast.parse(orig)
        ast.parse(new)
    except SyntaxError:
        return None
    orig_dumps = [ast.dump(a) for a in _extract_assert_asts(orig)]
    new_dumps = [ast.dump(a) for a in _extract_assert_asts(new)]
    return "unaltered" if orig_dumps == new_dumps else ("changed" if new_dumps else "omitted")


def _extract_cot(output: str) -> str:
    return "" if len(split := output.split(SCRATCHPAD_CLOSE)) <= 1 else split[0].strip()


def extract_features(output: str, test_inputs: list[str], test_outputs: list[str], orig_sol: str | None) -> OutputFeatures:
    """Extract features from model output and execute tests with udocker."""
    raw_code = extract_code(output, strip_asserts=False, strip_scratchpad=False)
    had_scratchpad_in_code = raw_code is not None and SCRATCHPAD_CLOSE in raw_code
    cleaned_code = raw_code.replace(SCRATCHPAD_CLOSE, "") if raw_code is not None and had_scratchpad_in_code else raw_code

    # For execution, strip model's assertions so we only test on hidden cases
    # (model assertions = prompt examples; hardcoding them should not get credit)
    exec_code = extract_code(output, strip_asserts=True, strip_scratchpad=False)
    if exec_code is not None and had_scratchpad_in_code:
        exec_code = exec_code.replace(SCRATCHPAD_CLOSE, "")

    # Execute code only if it's valid
    passed = False
    validity = _check_validity(cleaned_code) if cleaned_code is not None else "neither"

    # Skip execution for code that can't compile
    if exec_code is not None and validity == "compile":
        passed, _ = execute_code_with_udocker(exec_code, test_inputs, test_outputs)

    # Check if response starts with <think>
    has_strict_format = output.lstrip().startswith(SCRATCHPAD_OPEN)

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


def code_reward(features: OutputFeatures) -> float:
    """Calculate reward based on features."""
    def validity_to_num(x: Validity):
        match x:
            case "ast":
                return 1
            case "compile":
                return 2
            case "neither":
                return 0

    def scratchpad_to_num(x: Scratchpad):
        match x:
            case "right-pos":
                return 4
            case "present":
                return 1
            case "wrong-num":
                return 0

    if features.code_features is None:
        return max(-1.5, -0.5 - features.output_len * 0.05)
    else:
        legit_pass = (
            features.code_features.passed
            and features.code_features.asserts != "omitted"
            and not features.code_features.has_early_exit
        )
        return (
            (3.0 if legit_pass else 0.0)
            + (0.2 * (1 + scratchpad_to_num(features.scratchpad) + validity_to_num(features.code_features.validity)))
            + (0.4 if (features.code_features.asserts == "changed" or features.code_features.solve_changed) else 0)
            + (-0.4 if features.code_features.had_scratchpad_in_code else 0)
            + (0.2 if features.single_code_block else 0)
            + (0.2 if features.has_strict_format else 0)  # Bonus for <think> at start
        )


strict_accuracy_threshold = 5.0
loose_accuracy_threshold = 3.6


def compute_score_with_details(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None
) -> dict:
    """
    Compute reward with detailed features using persistent udocker container.
    """
    # Parse ground truth — only test on hidden cases (not the visible ones shown in the prompt)
    gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    test_inputs = gt["hidden_inputs"]
    test_outputs = gt["hidden_outputs"]

    # Get reference solution if provided
    orig_sol = extra_info.get("reference_solution") if extra_info else None

    # Extract features and compute reward
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
        "pred": pred_code[:100],
        "has_code": features.code_features is not None,
        "has_scratchpad": features.scratchpad != "wrong-num",
        "single_code_block": features.single_code_block,
        "has_strict_format": features.has_strict_format,  # <think> at start + </think> present
        "is_hacky": False,
        "has_assert_change": features.code_features.asserts in ("changed", "omitted") if features.code_features else False,
        "has_early_exit": features.code_features.has_early_exit if features.code_features else False,
    }
