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

from dr_exec import BatchResult, OverflowPolicy
from pydantic import Field, StrictInt, model_validator

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


class CodeTestBudgets(OperatorSettings):
    """The output/input execution budgets the code_test lane declares.

    These are part of the operator's identity, not an executor default: the
    protocol knowledge (how much output a HumanEval batch may emit, how much
    stdin it may consume) lives at this call site. Declaring them here folds
    them into ``question_identity_hash`` so a budget change is a loud identity
    change, and into the execution invocation identity so it invalidates the
    execution cache.
    """

    output_bytes: StrictInt = batch_runner.MAX_HUMANEVAL_OUTPUT_BYTES
    output_overflow_policy: OverflowPolicy = OverflowPolicy.FAIL
    input_bytes: StrictInt = batch_runner.MAX_HUMANEVAL_INPUT_BYTES

    @model_validator(mode="after")
    def validate_values(self) -> Self:
        if self.output_bytes <= 0:
            raise ValueError("output_bytes must be positive")
        if self.input_bytes <= 0:
            raise ValueError("input_bytes must be positive")
        return self


class CodeTestSettings(OperatorSettings):
    task_key: str = "task"
    timeout_seconds: float = DEFAULT_HUMANEVAL_TIMEOUT_SECONDS
    budgets: CodeTestBudgets = CodeTestBudgets()

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

        budgets = batch_runner.human_eval_budgets(
            self.settings.timeout_seconds,
            output_bytes=self.settings.budgets.output_bytes,
            output_overflow_policy=self.settings.budgets.output_overflow_policy,
            input_bytes=self.settings.budgets.input_bytes,
        )
        plan = batch_runner.build_human_eval_batch_plan(
            task=task,
            candidate_code=candidate_code,
            function_name=function_name,
            timeout_seconds=self.settings.timeout_seconds,
            budgets=budgets,
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
