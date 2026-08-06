from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Self

from pydantic import model_validator

from dr_code.humaneval import runner
from dr_code.humaneval.profiles import DEFAULT_HUMANEVAL_TIMEOUT_SECONDS
from dr_code.core.execution.executor import (
    CompletedPythonProcess,
    ExecutionKilledError,
    ExecutionOutputLimitError,
)
from dr_code.humaneval.task import (
    EvaluationCaseResult,
    EvaluationCaseStatus,
    EvaluationHarnessError,
    EvaluationTaskResult,
    HumanEvalTask,
)
from dr_code.metrics.engine.execution import (
    ExecutionOutcome,
    ExecutionRequest,
    is_killed_outcome,
    is_output_limit_outcome,
    is_timeout_outcome,
)
from dr_code.metrics.names import MetricName
from dr_code.metrics.operators.base import (
    EngineContext,
    MetricOperator,
    OperatorResult,
)
from dr_code.metrics.settings import OperatorSettings
from dr_code.metrics.units import MetricFactUnit
from dr_code.trace import (
    Artifact,
    ArtifactKind,
    CodeArtifact,
    JsonArtifact,
)


class CodeTestSettings(OperatorSettings):
    task_key: str = "task"
    timeout_seconds: float = DEFAULT_HUMANEVAL_TIMEOUT_SECONDS

    @model_validator(mode="after")
    def validate_values(self) -> Self:
        if not self.task_key:
            raise ValueError("task_key must not be empty")
        if (
            not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        return self


class CodeTestResult(OperatorResult):
    UNITS = {
        "total_cases": MetricFactUnit.COUNT,
        "passed_count": MetricFactUnit.COUNT,
        "failed_count": MetricFactUnit.COUNT,
        "error_count": MetricFactUnit.COUNT,
        "timeout_count": MetricFactUnit.COUNT,
        "coverage_complete": MetricFactUnit.BOOLEAN,
        "function_count": MetricFactUnit.COUNT,
        "best_function_name": MetricFactUnit.IDENTIFIER,
    }

    total_cases: int
    passed_count: int
    failed_count: int
    error_count: int
    timeout_count: int
    coverage_complete: bool
    function_count: int
    best_function_name: str | None


class CodeTest(MetricOperator[CodeTestSettings]):
    NAME = MetricName.CODE_TEST
    VERSION = "0"
    INPUT = ArtifactKind.CODE
    Settings = CodeTestSettings

    def __init__(self, settings: CodeTestSettings) -> None:
        super().__init__(settings)
        # One binding measures a whole batch against the same task payload,
        # and validation reparses the task, so keep the validated result per
        # exact payload for the life of the binding.
        self._validated_tasks: dict[str, HumanEvalTask] = {}

    def auxiliary_keys(self) -> tuple[str, ...]:
        return (self.settings.task_key,)

    def accepted_auxiliary_kinds(
        self,
        key: str,
    ) -> frozenset[ArtifactKind]:
        _ = key
        return frozenset({ArtifactKind.JSON})

    def validate_auxiliary(self, aux: Mapping[str, Artifact]) -> None:
        self._task(aux)

    def execution_requests(
        self,
        value: Artifact,
        aux: Mapping[str, Artifact],
    ) -> tuple[ExecutionRequest, ...]:
        source = _code_source(value)
        task = self._task(aux)
        function_names = runner.top_level_function_names(source)
        return tuple(
            self._request(
                task=task,
                candidate_code=source,
                function_name=function_name,
            )
            for function_name in function_names
        )

    def compute(
        self,
        value: Artifact,
        aux: Mapping[str, Artifact],
        ctx: EngineContext,
    ) -> CodeTestResult:
        source = _code_source(value)
        task = self._task(aux)
        function_names = runner.top_level_function_names(
            source,
            parsed_module=ctx.views.parsed_module(source),
        )
        case_results: list[EvaluationCaseResult] = []
        for function_name in function_names:
            request = self._request(
                task=task,
                candidate_code=source,
                function_name=function_name,
            )
            case_results.extend(
                _results_from_outcome(
                    task=task,
                    function_name=function_name,
                    timeout_seconds=request.timeout_seconds,
                    outcome=ctx.outcome_for(request),
                )
            )

        evaluation = EvaluationTaskResult(
            task_id=task.task_id,
            entry_point=task.entry_point,
            function_names=function_names,
            total_cases=len(runner.require_parsed_tests(task).cases),
            results=case_results,
        )
        counts = evaluation.status_counts
        return CodeTestResult(
            total_cases=evaluation.total_cases,
            passed_count=counts.get(EvaluationCaseStatus.PASSED.value, 0),
            failed_count=counts.get(EvaluationCaseStatus.FAILED.value, 0),
            error_count=counts.get(EvaluationCaseStatus.ERROR.value, 0),
            timeout_count=counts.get(
                EvaluationCaseStatus.TIMEOUT.value,
                0,
            ),
            coverage_complete=evaluation.coverage_complete,
            function_count=len(evaluation.function_names),
            best_function_name=evaluation.best_function_name,
        )

    def _task(self, aux: Mapping[str, Artifact]) -> HumanEvalTask:
        artifact = aux[self.settings.task_key]
        if not isinstance(artifact, JsonArtifact):
            raise TypeError("code_test task must be a JSON artifact")
        payload_key = json.dumps(
            artifact.payload, sort_keys=True, separators=(",", ":")
        )
        cached = self._validated_tasks.get(payload_key)
        if cached is not None:
            return cached
        task = _validate_task_payload(artifact)
        self._validated_tasks[payload_key] = task
        return task

    def _request(
        self,
        *,
        task: HumanEvalTask,
        candidate_code: str,
        function_name: str,
    ) -> ExecutionRequest:
        request = runner.build_humaneval_batch_request(
            task=task,
            candidate_code=candidate_code,
            function_name=function_name,
            timeout_seconds=self.settings.timeout_seconds,
        )
        return ExecutionRequest(
            source=request.source,
            input_json=request.input_json,
            timeout_seconds=request.timeout_seconds,
            computation_id=runner.HUMANEVAL_RUNNER_COMPUTATION_ID,
        )


def _results_from_outcome(
    *,
    task: HumanEvalTask,
    function_name: str,
    timeout_seconds: float,
    outcome: ExecutionOutcome,
) -> list[EvaluationCaseResult]:
    if is_timeout_outcome(outcome):
        return runner.timeout_results(
            task=task,
            function_name=function_name,
            timeout_seconds=timeout_seconds,
        )
    if is_output_limit_outcome(outcome):
        return runner.error_results(
            task=task,
            function_name=function_name,
            message=f"{ExecutionOutputLimitError.__name__}: {outcome.stderr}",
        )
    if is_killed_outcome(outcome):
        return runner.error_results(
            task=task,
            function_name=function_name,
            message=f"{ExecutionKilledError.__name__}: {outcome.stderr}",
        )

    completed = CompletedPythonProcess(
        returncode=outcome.returncode,
        stdout=outcome.stdout,
        stderr=outcome.stderr,
    )
    try:
        return runner.interpret_subprocess_batch_result(
            task=task,
            function_name=function_name,
            completed=completed,
            elapsed_seconds=0.0,
        )
    except EvaluationHarnessError as exc:
        return runner.error_results(
            task=task,
            function_name=function_name,
            message=str(exc),
        )


def _validate_task_payload(artifact: JsonArtifact) -> HumanEvalTask:
    payload = artifact.payload
    if not isinstance(payload, dict):
        raise ValueError("code_test task payload must be a JSON object")

    data = dict(payload)
    provided_computed: dict[str, object] = {}
    for field_name in HumanEvalTask.model_computed_fields:
        if field_name in data:
            provided_computed[field_name] = data.pop(field_name)
    task = HumanEvalTask.model_validate(data)
    for field_name, expected in provided_computed.items():
        if getattr(task, field_name) != expected:
            raise ValueError(
                f"code_test task computed field {field_name!r} is invalid"
            )
    return task


def _code_source(value: Artifact) -> str:
    if not isinstance(value, CodeArtifact):
        raise TypeError("code_test input must be code")
    return value.source
