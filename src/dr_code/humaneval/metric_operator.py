from __future__ import annotations

import json
from collections.abc import Mapping
from dr_code.evaluation.records import (
    CandidateJobCompleted,
    CandidateJobTerminated,
    ExecutorExecutionFailure,
    HarnessExecutionFailure,
)
from dr_code.humaneval import runner
from dr_code.humaneval.job import (
    HumanEvalEvaluatorSuite,
)
from dr_code.humaneval.settings import CodeTestSettings
from dr_code.humaneval.task import EvaluationCaseStatus, HumanEvalTask
from dr_code.metrics.coordinates import MetricQuestionCoordinate
from dr_code.metrics.names import MetricName
from dr_code.metrics.operators.base import (
    EngineContext,
    MetricOperator,
    OperatorResult,
)
from dr_code.metrics.units import MetricValueUnit
from dr_code.trace import (
    Artifact,
    ArtifactKind,
    CodeArtifact,
    JsonArtifact,
)


class CodeTestResult(OperatorResult):
    UNITS = {
        "total_cases": MetricValueUnit.COUNT,
        "passed_count": MetricValueUnit.COUNT,
        "failed_count": MetricValueUnit.COUNT,
        "error_count": MetricValueUnit.COUNT,
        "timeout_count": MetricValueUnit.COUNT,
        "coverage_complete": MetricValueUnit.BOOLEAN,
        "function_count": MetricValueUnit.COUNT,
        "best_function_name": MetricValueUnit.IDENTIFIER,
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

    def evaluator_suite(
        self,
        value: Artifact,
        aux: Mapping[str, Artifact],
        question: MetricQuestionCoordinate,
    ) -> HumanEvalEvaluatorSuite:
        _code_source(value)
        return HumanEvalEvaluatorSuite(
            question=question,
            task=self._task(aux),
            settings=self.settings,
        )

    def compute(
        self,
        value: Artifact,
        aux: Mapping[str, Artifact],
        ctx: EngineContext,
    ) -> CodeTestResult:
        source = _code_source(value)
        task = self._task(aux)
        outcome = ctx.candidate_execution_outcome
        if not isinstance(
            outcome,
            CandidateJobCompleted
            | CandidateJobTerminated
            | HarnessExecutionFailure
            | ExecutorExecutionFailure,
        ):
            raise RuntimeError("code_test has no candidate execution outcome")
        evaluation = runner.evaluation_from_candidate_execution(
            task=task,
            candidate_source=source,
            question=ctx.question,
            outcome=outcome,
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
