"""Local fork execution for one sample."""

from __future__ import annotations

import contextlib
import builtins
import json
import logging
import math
import os
import resource
import select
import signal
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import ValidationError

from dr_code.models.base import FrozenModel
from dr_code.models.outcomes import (
    InfraErrorProjection,
    TestCaseResultProjection,
)
from dr_code.testing.bridge import TestCase

logger = logging.getLogger("dr_code.testing")

ExecutionOutcomeKind = Literal["tested", "infra_error", "internal_error"]

_EXECUTION_MODE: Final[str] = "local_fork_worker"
_POLL_INTERVAL_SECONDS: Final[float] = 0.01
_DEFAULT_MEMORY_BYTES: Final[int] = 1_073_741_824
_DEFAULT_FILE_BYTES: Final[int] = 10_485_760
_DEFAULT_OPEN_FILES: Final[int] = 64
_DEFAULT_PROCESS_LIMIT: Final[int] = 16
_RESULT_READ_BYTES: Final[int] = 16_777_216
_MINIMAL_ENV: Final[dict[str, str]] = {
    "HOME": "/tmp",
    "LANG": "C.UTF-8",
    "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
    "PYTHONHASHSEED": "0",
}

_INFRA_MESSAGE_MARKERS = (
    "fork failed",
    "worker timed out",
    "worker exited",
    "worker returned no output",
    "invalid JSON from worker",
    "missing results list",
)


class SampleExecutionResult(FrozenModel):
    """Result of executing one sample's test cases."""

    outcome_kind: ExecutionOutcomeKind
    test_case_results: tuple[TestCaseResultProjection, ...]
    test_pass_rate: float | None
    infra_error: InfraErrorProjection | None
    internal_error: str | None
    latency_ms: float


class _Classification(FrozenModel):
    outcome_kind: ExecutionOutcomeKind
    infra_error: InfraErrorProjection | None = None


def execute_sample_tests(
    *,
    extracted_code: str,
    entry_point: str,
    test_cases: list[TestCase],
    timeout_seconds: float,
    sample_id: str,
) -> SampleExecutionResult:
    """Run one sample in a fresh forked child process."""
    started = time.perf_counter()
    logger.info("%s execute_sample_tests start", sample_id)

    try:
        raw_results, pass_rate = _run_forked_test_cases(
            extracted_code=extracted_code,
            entry_point=entry_point,
            test_cases=test_cases,
            timeout_seconds=timeout_seconds,
        )
    except _ExecutionInfrastructureError as exc:
        result = _infra_result(
            infra_error=InfraErrorProjection(
                stage=exc.stage,
                execution_mode=_EXECUTION_MODE,
                detail=exc.detail,
            ),
            latency_ms=_elapsed_ms(started),
        )
        _log_end(sample_id, result)
        return result
    except Exception as exc:
        logger.error(
            "%s execute_sample_tests internal_error type=%s",
            sample_id,
            type(exc).__name__,
            exc_info=True,
        )
        result = SampleExecutionResult(
            outcome_kind="internal_error",
            test_case_results=(),
            test_pass_rate=None,
            infra_error=None,
            internal_error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            latency_ms=_elapsed_ms(started),
        )
        _log_end(sample_id, result)
        return result

    classified = classify_execution_outcome(raw_results)
    if classified.outcome_kind == "infra_error":
        logger.warning(
            "%s heuristic infra reclassification detail=%s",
            sample_id,
            classified.infra_error.detail if classified.infra_error else "",
        )
        result = SampleExecutionResult(
            outcome_kind="infra_error",
            test_case_results=(),
            test_pass_rate=None,
            infra_error=classified.infra_error,
            internal_error=None,
            latency_ms=_elapsed_ms(started),
        )
    else:
        result = SampleExecutionResult(
            outcome_kind="tested",
            test_case_results=raw_results,
            test_pass_rate=pass_rate,
            infra_error=None,
            internal_error=None,
            latency_ms=_elapsed_ms(started),
        )
    _log_end(sample_id, result)
    return result


def classify_execution_outcome(
    results: tuple[TestCaseResultProjection, ...],
) -> _Classification:
    """Reclassify worker-level failures misreported as per-case test failures."""
    if not results:
        return _Classification(outcome_kind="tested")

    if any(result.passed for result in results):
        return _Classification(outcome_kind="tested")

    messages: list[str] = []
    for result in results:
        message = result.error or result.compile_error
        if message is None:
            return _Classification(outcome_kind="tested")
        messages.append(message)

    if len(set(messages)) != 1:
        return _Classification(outcome_kind="tested")

    shared = messages[0]
    if not _looks_like_infra_message(shared):
        return _Classification(outcome_kind="tested")

    return _Classification(
        outcome_kind="infra_error",
        infra_error=InfraErrorProjection(
            stage="heuristic_reclassification",
            execution_mode=_EXECUTION_MODE,
            detail=shared,
        ),
    )


class _ExecutionInfrastructureError(RuntimeError):
    """Raised when the local fork worker failed before returning test results."""

    def __init__(self, *, stage: str, detail: str) -> None:
        self.stage = stage
        self.detail = detail
        super().__init__(f"{stage}: {detail}")


def _run_forked_test_cases(
    *,
    extracted_code: str,
    entry_point: str,
    test_cases: list[TestCase],
    timeout_seconds: float,
) -> tuple[tuple[TestCaseResultProjection, ...], float]:
    if not test_cases:
        return (), 0.0

    read_fd, write_fd = os.pipe()
    with tempfile.TemporaryDirectory(prefix="dr_code_eval_") as tmp_dir:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"This process .* is multi-threaded, use of fork\(\) may lead to deadlocks in the child\.",
                    category=DeprecationWarning,
                )
                pid = os.fork()
        except OSError as exc:
            os.close(read_fd)
            os.close(write_fd)
            raise _ExecutionInfrastructureError(
                stage="fork_failed",
                detail=str(exc),
            ) from exc

        if pid == 0:
            _run_child_and_exit(
                write_fd=write_fd,
                read_fd=read_fd,
                tmp_dir=Path(tmp_dir),
                extracted_code=extracted_code,
                entry_point=entry_point,
                test_cases=test_cases,
                timeout_seconds=timeout_seconds,
            )

        os.close(write_fd)
        try:
            status, payload = _wait_for_child_and_read(
                pid,
                read_fd,
                timeout_seconds,
            )
        finally:
            os.close(read_fd)

    if not _child_exited_successfully(status):
        raise _ExecutionInfrastructureError(
            stage="worker_nonzero_exit",
            detail=f"worker exited with status={status}",
        )

    return _parse_worker_payload(payload, expected_count=len(test_cases))


def _run_child_and_exit(
    *,
    write_fd: int,
    read_fd: int,
    tmp_dir: Path,
    extracted_code: str,
    entry_point: str,
    test_cases: list[TestCase],
    timeout_seconds: float,
) -> None:
    try:
        os.close(read_fd)
        _prepare_child(tmp_dir, timeout_seconds)
        results, pass_rate = _execute_test_cases(
            extracted_code=extracted_code,
            entry_point=entry_point,
            test_cases=test_cases,
        )
        _write_child_payload(
            write_fd,
            {
                "results": [
                    result.model_dump(mode="json") for result in results
                ],
                "pass_rate": pass_rate,
            },
        )
    except BaseException as exc:  # noqa: BLE001
        with contextlib.suppress(BaseException):
            _write_child_payload(
                write_fd,
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )
    finally:
        with contextlib.suppress(OSError):
            os.close(write_fd)
        os._exit(0)


def _prepare_child(tmp_dir: Path, timeout_seconds: float) -> None:
    os.chdir(tmp_dir)
    os.environ.clear()
    os.environ.update(_MINIMAL_ENV)
    _redirect_standard_streams()
    _apply_resource_limits(timeout_seconds)


def _redirect_standard_streams() -> None:
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
    finally:
        os.close(devnull_fd)


def _apply_resource_limits(timeout_seconds: float) -> None:
    cpu_seconds = max(1, math.ceil(timeout_seconds))
    _set_limit(resource.RLIMIT_CPU, cpu_seconds)
    _set_limit(resource.RLIMIT_FSIZE, _DEFAULT_FILE_BYTES)
    _set_limit(resource.RLIMIT_NOFILE, _DEFAULT_OPEN_FILES)
    if hasattr(resource, "RLIMIT_NPROC"):
        _set_limit(resource.RLIMIT_NPROC, _DEFAULT_PROCESS_LIMIT)
    if sys.platform != "darwin" and hasattr(resource, "RLIMIT_AS"):
        _set_limit(resource.RLIMIT_AS, _DEFAULT_MEMORY_BYTES)
    elif sys.platform != "darwin" and hasattr(resource, "RLIMIT_DATA"):
        _set_limit(resource.RLIMIT_DATA, _DEFAULT_MEMORY_BYTES)


def _set_limit(limit_name: int, value: int) -> None:
    soft, hard = resource.getrlimit(limit_name)
    if hard == resource.RLIM_INFINITY:
        target = value
    else:
        target = min(value, hard)
    if soft > target:
        resource.setrlimit(limit_name, (target, hard))
        soft = target
    if hard != target:
        resource.setrlimit(limit_name, (soft, target))
    if soft != target:
        resource.setrlimit(limit_name, (target, target))


def _execute_test_cases(
    *,
    extracted_code: str,
    entry_point: str,
    test_cases: list[TestCase],
) -> tuple[tuple[TestCaseResultProjection, ...], float]:
    try:
        compiled = compile(extracted_code, "<candidate>", "exec")
    except SyntaxError as exc:
        error = f"SyntaxError: {exc}"
        results = tuple(
            TestCaseResultProjection(
                input_value=test_case.input_value,
                expected_output=test_case.expected_output,
                actual_output=None,
                passed=False,
                error=error,
                compile_success=False,
                compile_error=error,
            )
            for test_case in test_cases
        )
        return results, 0.0

    namespace: dict[str, Any] = {
        "__builtins__": builtins.__dict__.copy(),
        "__file__": str(Path.cwd() / "__candidate__.py"),
        "__name__": "__candidate__",
    }
    try:
        exec(compiled, namespace, namespace)  # noqa: S102
    except BaseException as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        results = tuple(
            _case_error(test_case, error, compile_success=True)
            for test_case in test_cases
        )
        return results, 0.0

    function = namespace.get(entry_point)
    if not callable(function):
        error = f"Function '{entry_point}' not found in executed code"
        results = tuple(
            _case_error(test_case, error, compile_success=True)
            for test_case in test_cases
        )
        return results, 0.0

    results = tuple(
        _execute_one_case(function, test_case) for test_case in test_cases
    )
    pass_rate = sum(1 for result in results if result.passed) / len(results)
    return results, pass_rate


def _execute_one_case(
    function: Any,
    test_case: TestCase,
) -> TestCaseResultProjection:
    try:
        actual = _call_candidate(function, test_case.input_value)
    except BaseException as exc:  # noqa: BLE001
        return _case_error(
            test_case,
            f"{type(exc).__name__}: {exc}",
            compile_success=True,
        )

    jsonable_actual = _as_jsonable(actual)
    passed = _values_equal(jsonable_actual, test_case.expected_output)
    return TestCaseResultProjection(
        input_value=test_case.input_value,
        expected_output=test_case.expected_output,
        actual_output=jsonable_actual,
        passed=passed,
        error=None,
        compile_success=True,
        compile_error=None,
    )


def _call_candidate(function: Any, input_value: Any) -> Any:
    if isinstance(input_value, list | tuple):
        return function(*input_value)
    return function(input_value)


def _case_error(
    test_case: TestCase,
    error: str,
    *,
    compile_success: bool,
) -> TestCaseResultProjection:
    return TestCaseResultProjection(
        input_value=test_case.input_value,
        expected_output=test_case.expected_output,
        actual_output=None,
        passed=False,
        error=error,
        compile_success=compile_success,
        compile_error=None,
    )


def _as_jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


def _wait_for_child_and_read(
    pid: int,
    read_fd: int,
    timeout_seconds: float,
) -> tuple[int, str]:
    deadline = time.monotonic() + timeout_seconds
    chunks: list[bytes] = []
    total = 0
    os.set_blocking(read_fd, False)
    try:
        while True:
            total = _drain_available(read_fd, chunks, total)
            waited_pid, status = os.waitpid(pid, os.WNOHANG)
            if waited_pid == pid:
                total = _drain_until_eof(read_fd, chunks, total)
                return status, _decode_worker_payload(chunks)
            if time.monotonic() >= deadline:
                raise _ExecutionInfrastructureError(
                    stage="worker_timeout",
                    detail=f"worker timed out after {timeout_seconds}s",
                )
            remaining = max(0.0, deadline - time.monotonic())
            select.select(
                [read_fd], [], [], min(_POLL_INTERVAL_SECONDS, remaining)
            )
    except _ExecutionInfrastructureError:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
        _wait_after_kill(pid)
        raise


def _wait_after_kill(pid: int) -> None:
    with contextlib.suppress(ChildProcessError):
        os.waitpid(pid, 0)


def _drain_available(read_fd: int, chunks: list[bytes], total: int) -> int:
    while True:
        try:
            chunk = os.read(read_fd, 65_536)
        except BlockingIOError:
            return total
        if not chunk:
            return total
        chunks.append(chunk)
        total += len(chunk)
        _validate_result_size(total)


def _drain_until_eof(read_fd: int, chunks: list[bytes], total: int) -> int:
    os.set_blocking(read_fd, True)
    while True:
        chunk = os.read(read_fd, 65_536)
        if not chunk:
            return total
        chunks.append(chunk)
        total += len(chunk)
        _validate_result_size(total)


def _decode_worker_payload(chunks: list[bytes]) -> str:
    payload = b"".join(chunks).decode("utf-8", errors="replace")
    if not payload.strip():
        raise _ExecutionInfrastructureError(
            stage="worker_payload_parse",
            detail="worker returned no output",
        )
    return payload


def _validate_result_size(total: int) -> None:
    if total > _RESULT_READ_BYTES:
        raise _ExecutionInfrastructureError(
            stage="worker_payload_too_large",
            detail=f"worker result exceeded {_RESULT_READ_BYTES} bytes",
        )


def _parse_worker_payload(
    raw_payload: str,
    *,
    expected_count: int,
) -> tuple[tuple[TestCaseResultProjection, ...], float]:
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise _ExecutionInfrastructureError(
            stage="worker_payload_parse",
            detail=f"invalid JSON from worker: {exc}",
        ) from exc
    if not isinstance(payload, dict):
        raise _ExecutionInfrastructureError(
            stage="worker_payload_parse",
            detail="worker returned non-object JSON",
        )
    if payload.get("error"):
        raise _ExecutionInfrastructureError(
            stage="worker_payload_error",
            detail=str(payload["error"]),
        )
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise _ExecutionInfrastructureError(
            stage="worker_payload_parse",
            detail="missing results list in worker response",
        )
    if len(raw_results) != expected_count:
        raise _ExecutionInfrastructureError(
            stage="worker_payload_mismatched_batch_count",
            detail=f"expected {expected_count} results, got {len(raw_results)}",
        )
    try:
        results = tuple(
            TestCaseResultProjection.model_validate(result)
            for result in raw_results
        )
    except ValidationError as exc:
        raise _ExecutionInfrastructureError(
            stage="worker_payload_parse",
            detail=f"invalid test case result payload: {exc}",
        ) from exc
    pass_rate = payload.get("pass_rate")
    if not isinstance(pass_rate, int | float):
        raise _ExecutionInfrastructureError(
            stage="worker_payload_parse",
            detail="missing numeric pass_rate in worker response",
        )
    return results, float(pass_rate)


def _write_child_payload(write_fd: int, payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    view = memoryview(encoded)
    written = 0
    while written < len(encoded):
        chunk_written = os.write(write_fd, view[written:])
        if chunk_written == 0:
            raise OSError("worker payload pipe accepted 0 bytes")
        written += chunk_written


def _child_exited_successfully(status: int) -> bool:
    return os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0


def _values_equal(actual: Any, expected: Any, rel_tol: float = 1e-9) -> bool:
    if isinstance(actual, float) and isinstance(expected, float):
        return math.isclose(actual, expected, rel_tol=rel_tol)
    if isinstance(actual, float) and isinstance(expected, int):
        return math.isclose(actual, float(expected), rel_tol=rel_tol)
    if isinstance(actual, int) and isinstance(expected, float):
        return math.isclose(float(actual), expected, rel_tol=rel_tol)
    return actual == expected


def _looks_like_infra_message(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _INFRA_MESSAGE_MARKERS)


def _infra_result(
    *,
    infra_error: InfraErrorProjection,
    latency_ms: float,
) -> SampleExecutionResult:
    return SampleExecutionResult(
        outcome_kind="infra_error",
        test_case_results=(),
        test_pass_rate=None,
        infra_error=infra_error,
        internal_error=None,
        latency_ms=latency_ms,
    )


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _log_end(sample_id: str, result: SampleExecutionResult) -> None:
    logger.info(
        "%s execute_sample_tests end outcome_kind=%s latency_ms=%.2f",
        sample_id,
        result.outcome_kind,
        result.latency_ms,
    )
