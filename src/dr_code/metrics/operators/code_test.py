"""HumanEval case-execution facts over dr-exec's batch executor.

This operator is HumanEval-specific by construction, not merely by its type
annotations: it uses the HumanEval batch request and result protocol and
depends on ``HumanEvalTask.parsed_tests`` semantics. There is deliberately no
generic ``Task`` supertype -- a single-implementation abstraction with a
guessed interface would be premature. The shared interface gets extracted when
a second benchmark exists to constrain it; until then the HumanEval scope is
kept honest through naming and docstrings.

The metrics lane scores every candidate-observable outcome as case data:
budget deaths, candidate-process crashes, malformed runner output, and
unexpected exits all become case statuses in a measured record. Only a genuine
executor failure with no result escapes the executor as an exception, and it
does so before this operator's ``compute`` ever runs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Self

from dr_exec import BatchResult
from pydantic import model_validator

from dr_code.humaneval import batch_runner
from dr_code.humaneval.profiles import DEFAULT_HUMANEVAL_TIMEOUT_SECONDS
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
    InvocationIdentity,
)
from dr_code.metrics.names import MetricName
from dr_code.metrics.operators.base import (
    EngineContext,
    MetricOperator,
    OperatorResult,
    OperatorSettings,
)
from dr_code.trace import (
    Artifact,
    ArtifactKind,
    CodeArtifact,
    JsonArtifact,
)

_COMPUTATION_ID = "humaneval-runner@v1"


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
    VERSION = "1"
    INPUT = ArtifactKind.CODE
    Settings = CodeTestSettings

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
        function_names = batch_runner.top_level_function_names(source)
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
        function_names = batch_runner.top_level_function_names(source)
        requests = tuple(
            self._request(
                task=task,
                candidate_code=source,
                function_name=function_name,
            )
            for function_name in function_names
        )
        case_results: list[EvaluationCaseResult] = []
        for function_name, request in zip(
            function_names,
            requests,
            strict=True,
        ):
            case_results.extend(
                _results_from_outcome(
                    task=task,
                    function_name=function_name,
                    timeout_seconds=self.settings.timeout_seconds,
                    request=request,
                    outcome=ctx.outcome_for(request),
                )
            )

        evaluation = EvaluationTaskResult(
            task_id=task.task_id,
            entry_point=task.entry_point,
            function_names=function_names,
            total_cases=len(batch_runner.require_parsed_tests(task).cases),
            results=case_results,
        )
        counts = evaluation.status_counts
        return CodeTestResult(
            total_cases=evaluation.total_cases,
            passed_count=counts.get(EvaluationCaseStatus.PASSED.value, 0),
            failed_count=counts.get(EvaluationCaseStatus.FAILED.value, 0),
            error_count=counts.get(EvaluationCaseStatus.ERROR.value, 0),
            timeout_count=counts.get(EvaluationCaseStatus.TIMEOUT.value, 0),
            coverage_complete=evaluation.coverage_complete,
            function_count=len(evaluation.function_names),
            best_function_name=evaluation.best_function_name,
        )

    def _task(self, aux: Mapping[str, Artifact]) -> HumanEvalTask:
        artifact = aux[self.settings.task_key]
        if not isinstance(artifact, JsonArtifact):
            raise TypeError("code_test task must be a JSON artifact")
        return _validate_task_payload(artifact)

    def _request(
        self,
        *,
        task: HumanEvalTask,
        candidate_code: str,
        function_name: str,
    ) -> ExecutionRequest:
        from dr_exec import EXECUTOR_IDENTITY

        plan = batch_runner.build_human_eval_batch_plan(
            task=task,
            candidate_code=candidate_code,
            function_name=function_name,
            timeout_seconds=self.settings.timeout_seconds,
        )
        identity = InvocationIdentity.of(
            executor_identity=EXECUTOR_IDENTITY,
            source=plan.request.driver_source(),
            input_text="",
            budgets=plan.budgets,
            environment=batch_runner.HUMANEVAL_ENVIRONMENT,
            profile=batch_runner.HUMANEVAL_PROFILE,
            runtime=batch_runner.HUMANEVAL_RUNTIME,
        )
        return ExecutionRequest(
            batch_request=plan.request,
            budgets=plan.budgets,
            environment=batch_runner.HUMANEVAL_ENVIRONMENT,
            profile=batch_runner.HUMANEVAL_PROFILE,
            runtime=batch_runner.HUMANEVAL_RUNTIME,
            computation_id=_COMPUTATION_ID,
            identity=identity,
        )


def _results_from_outcome(
    *,
    task: HumanEvalTask,
    function_name: str,
    timeout_seconds: float,
    request: ExecutionRequest,
    outcome: ExecutionOutcome,
) -> list[EvaluationCaseResult]:
    """Interpret a cached batch outcome through the HumanEval protocol.

    Runner protocol faults and unexpected candidate exits are scored as
    candidate-attributable case errors, never batch failures: the metrics lane
    catches the harness fault the direct lane would raise and renders it as
    case data.
    """
    result: BatchResult = outcome.batch_result_for(request.batch_request)
    try:
        return batch_runner.interpret_batch_result(
            task=task,
            function_name=function_name,
            result=result,
            timeout_seconds=timeout_seconds,
        )
    except EvaluationHarnessError as exc:
        return batch_runner.error_results(
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
