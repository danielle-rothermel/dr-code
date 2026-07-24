"""Subprocess-backed HumanEval case execution facts.

This operator is HumanEval-specific by construction, not merely by its type
annotations: it uses the HumanEval batch request and result protocol and
depends on ``HumanEvalTask.parsed_tests`` semantics. There is deliberately no
generic ``Task`` supertype -- a single-implementation abstraction with a
guessed interface would be premature. The shared interface gets extracted when
a second benchmark exists to constrain it; until then the HumanEval scope is
kept honest through naming and docstrings.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Self

from pydantic import Field, model_validator

from dr_code.execution.subprocess import (
    SubprocessCompletedProcess,
    SubprocessOutputLimitError,
)
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
    is_output_limit_outcome,
    is_timeout_outcome,
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
    total_cases: int = Field(ge=0)
    passed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    timeout_count: int = Field(ge=0)
    coverage_complete: bool
    function_count: int = Field(ge=0)
    best_function_name: str | None

    @model_validator(mode="after")
    def validate_relational_invariants(self) -> Self:
        observed = (
            self.passed_count
            + self.failed_count
            + self.error_count
            + self.timeout_count
        )
        if observed > self.total_cases:
            raise ValueError("observed case count must not exceed total_cases")
        if self.function_count == 0:
            if observed != 0:
                raise ValueError(
                    "zero-function result requires zero observed cases"
                )
            if self.best_function_name is not None:
                raise ValueError(
                    "zero-function result requires null best_function_name"
                )
            if self.coverage_complete:
                raise ValueError(
                    "zero-function result requires incomplete coverage"
                )
            return self
        if not self.best_function_name:
            raise ValueError(
                "result with functions requires best_function_name"
            )
        if self.coverage_complete != (observed == self.total_cases):
            raise ValueError(
                "coverage_complete must equal complete case observation"
            )
        return self


class CodeTest(MetricOperator[CodeTestSettings]):
    NAME = MetricName.CODE_TEST
    VERSION = "1"
    INPUT = ArtifactKind.CODE
    Settings = CodeTestSettings
    FACT_UNITS = {
        "total_cases": "case",
        "passed_count": "case",
        "failed_count": "case",
        "error_count": "case",
        "timeout_count": "case",
        "coverage_complete": "boolean",
        "function_count": "function",
        "best_function_name": "name",
    }

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
        case_results = []
        for function_name, request in zip(
            function_names,
            requests,
            strict=True,
        ):
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
        request = batch_runner.build_human_eval_batch_request(
            task=task,
            candidate_code=candidate_code,
            function_name=function_name,
            timeout_seconds=self.settings.timeout_seconds,
        )
        return ExecutionRequest(
            source=request.source,
            input_text=request.input_text,
            timeout_seconds=request.timeout_seconds,
            computation_id=_COMPUTATION_ID,
        )


def _results_from_outcome(
    *,
    task: HumanEvalTask,
    function_name: str,
    timeout_seconds: float,
    outcome: ExecutionOutcome,
) -> list[EvaluationCaseResult]:
    """Interpret a cached process outcome through the HumanEval protocol."""

    if is_timeout_outcome(outcome):
        return batch_runner.timeout_results(
            task=task,
            function_name=function_name,
            timeout_seconds=timeout_seconds,
        )
    if is_output_limit_outcome(outcome):
        return batch_runner.error_results(
            task=task,
            function_name=function_name,
            message=f"{SubprocessOutputLimitError.__name__}: {outcome.stderr}",
        )

    completed = SubprocessCompletedProcess(
        returncode=outcome.returncode,
        stdout=outcome.stdout,
        stderr=outcome.stderr,
    )
    try:
        return batch_runner.interpret_subprocess_batch_result(
            task=task,
            function_name=function_name,
            completed=completed,
            elapsed_seconds=0.0,
        )
    except EvaluationHarnessError as exc:
        # Metrics treats runner protocol failures and unexpected candidate
        # exits as candidate-attributable case errors, not batch failures.
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
