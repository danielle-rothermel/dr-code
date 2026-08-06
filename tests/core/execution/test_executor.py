from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from _executor_stubs import completed_execution, scripted_executor
from dr_exec import (
    BudgetAxis,
    BudgetExceededOutcome,
    CancelledOutcome,
    CompletedExecution,
    DeclarationError,
    EnvGrantKind,
    ExecutorFailure,
    FiniteByteLimit,
    FiniteDurationLimit,
    FiniteOutput,
    OutputOverflowPolicy,
    PayloadOutputs,
    ProcessExecutor,
    ProtocolFailedOutcome,
    ProtocolFailureCode,
    RetainedPayloadStream,
    SpawnAbsentOutcome,
    SpawnFailedOutcome,
    UntrustedPythonTarget,
)
from dr_code.core.execution.executor import (
    EXECUTION_REQUEST_SCHEMA,
    EXECUTION_REQUEST_SCHEMA_VERSION,
    MAX_EXECUTION_INPUT_BYTES,
    MAX_EXECUTION_STREAM_BYTES,
    CompletedPythonProcess,
    ExecutionKilledError,
    ExecutionOutputLimitError,
    ExecutionTimeoutError,
    build_python_execution_job,
    host_process_executor,
    interpret_completed_execution,
    run_python_source,
)

_DRIVER = "def dr_exec_main(request, emit):\n    pass\n"


def _job(**overrides: object):
    parameters: dict[str, object] = {
        "driver_source": _DRIVER,
        "input_json": '{"key": "value"}',
        "timeout_seconds": 2.0,
    }
    parameters.update(overrides)
    return build_python_execution_job(**parameters)  # type: ignore[arg-type]


def _run(execution_source, **overrides):
    parameters = {
        "source": _DRIVER,
        "input_json": "{}",
        "timeout_seconds": 2.0,
    }
    parameters.update(overrides)
    return run_python_source(execution_source, **parameters)


class TestJobDeclaration:
    def test_target_is_untrusted_python_with_the_request_document(
        self,
    ) -> None:
        job = _job()
        target = job.target
        assert isinstance(target, UntrustedPythonTarget)
        assert target.driver_source == _DRIVER
        assert target.request.schema == EXECUTION_REQUEST_SCHEMA
        assert (
            target.request.schema_version == EXECUTION_REQUEST_SCHEMA_VERSION
        )
        assert target.request.payload == {"key": "value"}

    def test_budgets_are_finite_on_every_enforced_axis(self) -> None:
        budgets = _job().budgets
        assert budgets.wall_time == FiniteDurationLimit(max_ns=2_000_000_000)
        assert budgets.input_bytes == FiniteByteLimit(max_bytes=2_097_152)
        payload_output = budgets.payload_output
        assert isinstance(payload_output, FiniteOutput)
        assert payload_output.max_bytes == 2 * MAX_EXECUTION_STREAM_BYTES
        assert payload_output.overflow_policy is OutputOverflowPolicy.FAIL
        retention = payload_output.retention
        assert retention.stdout.head_bytes == MAX_EXECUTION_STREAM_BYTES
        assert retention.stdout.tail_bytes == 0
        assert retention.stderr.head_bytes == MAX_EXECUTION_STREAM_BYTES
        assert retention.stderr.tail_bytes == 0

    def test_environment_grant_is_fixed_and_hermetic(self) -> None:
        env = _job().env
        assert env.kind is EnvGrantKind.FIXED
        assert {variable.name for variable in env.variables} == {
            "PYTHONDONTWRITEBYTECODE",
            "PYTHONHASHSEED",
        }

    @pytest.mark.parametrize(
        "timeout", [0.0, -1.0, float("nan"), float("inf")]
    )
    def test_invalid_timeout_is_a_declaration_error(
        self, timeout: float
    ) -> None:
        with pytest.raises(DeclarationError, match="finite and positive"):
            _job(timeout_seconds=timeout)

    def test_timeout_too_large_for_nanoseconds_is_a_declaration_error(
        self,
    ) -> None:
        with pytest.raises(DeclarationError) as exc_info:
            _job(timeout_seconds=1e308)
        assert str(exc_info.value) == (
            "execution timeout is too large to represent in nanoseconds"
        )

    def test_non_json_input_is_a_declaration_error(self) -> None:
        with pytest.raises(DeclarationError, match="strict JSON"):
            _job(input_json="not json")

    @pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
    def test_nonfinite_json_constant_is_a_declaration_error(
        self, constant: str
    ) -> None:
        with pytest.raises(DeclarationError, match="strict JSON"):
            _job(input_json=f'{{"value": {constant}}}')

    def test_overflowing_json_float_is_a_declaration_error(self) -> None:
        with pytest.raises(DeclarationError, match="strict JSON"):
            _job(input_json='{"value": 1e400}')

    def test_oversized_input_fails_before_any_response(self) -> None:
        oversized = json.dumps({"data": "x" * MAX_EXECUTION_INPUT_BYTES})
        with pytest.raises(DeclarationError, match="input budget"):
            _run(scripted_executor(), input_json=oversized)


class TestOutcomeInterpretation:
    def test_missing_executor_is_an_executor_failure(self) -> None:
        with pytest.raises(ExecutorFailure, match="no executor"):
            _run(None)

    def test_clean_exit_returns_the_completed_process(self) -> None:
        completed = _run(
            scripted_executor(stdout="[]", stderr="warned", returncode=0)
        )
        assert completed == CompletedPythonProcess(
            returncode=0, stdout="[]", stderr="warned"
        )

    def test_nonzero_exit_preserves_the_exit_code(self) -> None:
        completed = _run(scripted_executor(returncode=3, stderr="boom"))
        assert completed.returncode == 3
        assert completed.stderr == "boom"

    def test_signal_death_is_a_candidate_kill(self) -> None:
        with pytest.raises(ExecutionKilledError, match="signal 9"):
            _run(scripted_executor(returncode=-9, stderr="oom"))

    def test_wall_time_budget_is_a_timeout(self) -> None:
        with pytest.raises(ExecutionTimeoutError):
            _run(
                scripted_executor(
                    outcome=BudgetExceededOutcome(axis=BudgetAxis.WALL_TIME)
                )
            )

    def test_payload_output_budget_is_an_output_limit(self) -> None:
        with pytest.raises(ExecutionOutputLimitError):
            _run(
                scripted_executor(
                    outcome=BudgetExceededOutcome(
                        axis=BudgetAxis.PAYLOAD_OUTPUT
                    )
                )
            )

    def test_payload_owned_protocol_failure_is_a_candidate_kill(self) -> None:
        with pytest.raises(ExecutionKilledError, match="incomplete_stream"):
            _run(
                scripted_executor(
                    outcome=ProtocolFailedOutcome(
                        failure_code=ProtocolFailureCode.INCOMPLETE_STREAM,
                        failure_detail="stream ended early",
                        accepted_output_count=0,
                    )
                )
            )

    def test_executor_owned_protocol_failure_fails_closed(self) -> None:
        with pytest.raises(ExecutorFailure, match="protocol_failed"):
            _run(
                scripted_executor(
                    outcome=ProtocolFailedOutcome(
                        failure_code=ProtocolFailureCode.OVERSIZED_FRAME,
                        failure_detail="frame too large",
                        accepted_output_count=0,
                    )
                )
            )

    @pytest.mark.parametrize(
        "outcome",
        [
            SpawnAbsentOutcome(executable="python"),
            SpawnFailedOutcome(errno=13, error_message="exec"),
            CancelledOutcome(),
        ],
        ids=("spawn-absent", "spawn-failed", "cancelled"),
    )
    def test_non_payload_outcomes_fail_closed(self, outcome: object) -> None:
        with pytest.raises(ExecutorFailure):
            _run(scripted_executor(outcome=outcome))  # type: ignore[arg-type]

    def test_truncated_retained_stream_is_an_output_limit(self) -> None:
        job = _job()
        scripted = completed_execution(job, stdout="partial")
        truncated = RetainedPayloadStream(
            head=b"partial",
            tail=b"",
            produced_bytes=11,
            dropped_bytes=4,
        )
        execution = CompletedExecution(
            result=scripted.result.model_copy(
                update={
                    "payload_outputs": PayloadOutputs(
                        stdout=truncated,
                        stderr=scripted.result.payload_outputs.stderr,
                    )
                }
            ),
            record_receipt=scripted.record_receipt,
        )
        with pytest.raises(ExecutionOutputLimitError, match="stdout"):
            interpret_completed_execution(execution)


def test_host_process_executor_composes_the_production_pieces(
    tmp_path: Path,
) -> None:
    executor = host_process_executor(
        tmp_path,
        runtime_executable=Path(sys.executable),
    )
    assert isinstance(executor, ProcessExecutor)
    assert executor.run_store.root == tmp_path
    record = executor.runtime.describe()
    assert record.resolved_executable == Path(sys.executable).resolve()
