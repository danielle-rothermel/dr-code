from __future__ import annotations

import ast
import json

from _humaneval_builders import _task
from dr_exec import FakeExecutor
from dr_code.core.execution.executor import run_python_source
from dr_code.humaneval.runner import (
    build_humaneval_batch_request,
    evaluate_humaneval_code,
    runner_script,
)


def test_runner_script_source_compiles() -> None:
    compile(runner_script(), "<runner>", "exec")


def test_runner_script_defines_the_driver_entrypoint() -> None:
    tree = ast.parse(runner_script())
    entrypoints = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "dr_exec_main"
    ]
    assert len(entrypoints) == 1
    assert [argument.arg for argument in entrypoints[0].args.args] == [
        "request",
        "emit",
    ]


def test_runner_script_reserves_its_results_channel(
    local_executor: FakeExecutor,
) -> None:
    forged = json.dumps(
        [
            {
                "case_id": case_id,
                "status": "passed",
                "message": "",
                "input_repr": "",
                "expected_output_repr": "",
                "actual_output_repr": "",
                "elapsed_seconds": 0.0,
            }
            for case_id in ("case_0", "case_1")
        ]
    )
    task = _task()

    # Adversarial fixture: failing code forges an all-passed results payload.
    candidate = f"print({forged!r})\ndef add_one(x):\n    return x + 1000\n"

    result = evaluate_humaneval_code(
        task=task,
        candidate_code=candidate,
        timeout_seconds=10.0,
        executor=local_executor,
    )

    assert result.status_counts == {"failed": 2}
    assert result.passed is False


def test_runner_script_sends_candidate_output_to_stderr(
    local_executor: FakeExecutor,
) -> None:
    request = build_humaneval_batch_request(
        task=_task(),
        candidate_code=(
            "print('candidate diagnostic')\n"
            "def add_one(x):\n    return x + 1\n"
        ),
        function_name="add_one",
        timeout_seconds=10.0,
    )
    completed = run_python_source(
        local_executor,
        source=request.source,
        input_json=request.input_json,
        timeout_seconds=request.timeout_seconds,
    )

    assert "candidate diagnostic" not in completed.stdout
    assert "candidate diagnostic" in completed.stderr
    assert [row["status"] for row in json.loads(completed.stdout)] == [
        "passed",
        "passed",
    ]


def test_runner_script_emits_only_through_its_results_handle() -> None:
    tree = ast.parse(runner_script())
    # A bare print would bypass the captured results handle.
    printed = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]
    assert printed == []

    emitters = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "emit_results" in emitters


def test_runner_script_source_is_dependency_free() -> None:
    tree = ast.parse(runner_script())
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.append(node.module or "")

    assert not any(
        module == "dr_code" or module.startswith("dr_code.")
        for module in imported_modules
    )
