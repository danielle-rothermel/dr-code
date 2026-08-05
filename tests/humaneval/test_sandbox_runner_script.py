"""Tests for the HumanEval sandbox runner script."""

from __future__ import annotations

import ast
import json

from _builders import _task
from dr_code.core.execution.sandbox import SandboxRunner
from dr_code.humaneval.runner import (
    build_humaneval_batch_request,
    evaluate_humaneval_code,
    runner_script,
)


def test_runner_script_source_compiles() -> None:
    compile(runner_script(), "<runner>", "exec")


def test_runner_script_reserves_its_results_channel(
    local_runner: SandboxRunner,
) -> None:
    """A candidate that prints protocol JSON cannot reach the host.

    The runner captures its results handle before candidate code runs and
    points ``sys.stdout`` at stderr, so a candidate printing a complete,
    well-formed results array does not add to (or replace) what the host
    parses. Without the redirection this forged array would be the first thing
    on stdout and would decide the task's score.
    """
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
    # The candidate fails every case, then forges an all-passed result set.
    candidate = f"print({forged!r})\ndef add_one(x):\n    return x + 1000\n"

    result = evaluate_humaneval_code(
        task=task,
        candidate_code=candidate,
        timeout_seconds=10.0,
        run_in_sandbox=local_runner,
    )

    assert result.status_counts == {"failed": 2}
    assert result.passed is False


def test_runner_script_sends_candidate_output_to_stderr(
    local_runner: SandboxRunner,
) -> None:
    """Candidate prints are preserved as diagnostics on the bounded stderr.

    Redirecting rather than discarding keeps a candidate's own output
    debuggable while keeping it off the results channel.
    """
    request = build_humaneval_batch_request(
        task=_task(),
        candidate_code=(
            "print('candidate diagnostic')\n"
            "def add_one(x):\n    return x + 1\n"
        ),
        function_name="add_one",
        timeout_seconds=10.0,
    )
    completed = local_runner(
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


def test_runner_script_emits_only_through_its_protocol_handle() -> None:
    """Nothing in the runner writes results with a bare ``print``.

    ``emit_results`` is the single writer to the captured handle; a ``print``
    reintroduced anywhere in the program would land on the redirected stdout
    and silently drop the results the host is waiting for.
    """
    tree = ast.parse(runner_script())
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
