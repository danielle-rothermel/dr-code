"""Sandbox-backed HumanEval case execution facts."""

from __future__ import annotations

import ast
import json
import math
from collections import Counter
from collections.abc import Mapping
from typing import Self

from pydantic import TypeAdapter, ValidationError, model_validator

from dr_code.humaneval.batch_runner import runner_script
from dr_code.humaneval.profiles import DEFAULT_HUMANEVAL_TIMEOUT_SECONDS
from dr_code.humaneval.sandbox import CANDIDATE_KILL_RETURNCODES
from dr_code.humaneval.task import (
    EvaluationCaseStatus,
    EvaluationHarnessError,
    HumanEvalRunnerCaseOutput,
    HumanEvalRunnerPayload,
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
    OperatorSettings,
)
from dr_code.metrics.records import MetricScalar
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


class CodeTest(MetricOperator):
    NAME = MetricName.CODE_TEST
    VERSION = "1"
    INPUT = ArtifactKind.CODE
    Settings = CodeTestSettings

    def auxiliary_keys(self) -> tuple[str, ...]:
        settings = self.settings
        assert isinstance(settings, CodeTestSettings)
        return (settings.task_key,)

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
        function_names = _top_level_function_names(source)
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
    ) -> dict[str, MetricScalar]:
        source = _code_source(value)
        task = self._task(aux)
        function_names = _top_level_function_names(source)
        requests = tuple(
            self._request(
                task=task,
                candidate_code=source,
                function_name=function_name,
            )
            for function_name in function_names
        )
        statuses_by_name: dict[str, list[EvaluationCaseStatus]] = {}
        for function_name, request in zip(
            function_names,
            requests,
            strict=True,
        ):
            statuses = _statuses_from_outcome(
                outcome=ctx.outcome_for(request),
                task=task,
            )
            statuses_by_name.setdefault(function_name, []).extend(statuses)

        best_function_name = _best_function_name(
            function_names=function_names,
            entry_point=task.entry_point,
            statuses_by_name=statuses_by_name,
        )
        best_statuses = (
            statuses_by_name[best_function_name]
            if best_function_name is not None
            else []
        )
        counts = Counter(status.value for status in best_statuses)
        total_cases = _total_cases(task)
        return {
            "total_cases": total_cases,
            "passed_count": counts.get(EvaluationCaseStatus.PASSED.value, 0),
            "failed_count": counts.get(EvaluationCaseStatus.FAILED.value, 0),
            "error_count": counts.get(EvaluationCaseStatus.ERROR.value, 0),
            "timeout_count": counts.get(
                EvaluationCaseStatus.TIMEOUT.value,
                0,
            ),
            "coverage_complete": (
                best_function_name is not None
                and len(best_statuses) == total_cases
                and all(
                    status is EvaluationCaseStatus.PASSED
                    for status in best_statuses
                )
            ),
            "function_count": len(function_names),
            "best_function_name": best_function_name,
        }

    def _task(self, aux: Mapping[str, Artifact]) -> HumanEvalTask:
        settings = self.settings
        assert isinstance(settings, CodeTestSettings)
        artifact = aux[settings.task_key]
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
        settings = self.settings
        assert isinstance(settings, CodeTestSettings)
        parsed_tests = task.parsed_tests
        if parsed_tests is None:
            raise ValueError("HumanEvalTask.parsed_tests is required")
        payload = HumanEvalRunnerPayload(
            task_id=task.task_id,
            candidate_code=candidate_code,
            support_code=parsed_tests.support_code,
            function_name=function_name,
            test_type=parsed_tests.test_type,
            checks=list(parsed_tests.iter_checks(candidate_name="candidate")),
        )
        return ExecutionRequest(
            source=runner_script(),
            input_json=payload.model_dump_json(),
            timeout_seconds=settings.timeout_seconds,
            computation_id=_COMPUTATION_ID,
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


def _top_level_function_names(source: str) -> list[str]:
    tree = ast.parse(source)
    return [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]


def _total_cases(task: HumanEvalTask) -> int:
    if task.parsed_tests is None:
        raise ValueError("HumanEvalTask.parsed_tests is required")
    return len(task.parsed_tests.cases)


def _statuses_from_outcome(
    *,
    outcome: ExecutionOutcome,
    task: HumanEvalTask,
) -> list[EvaluationCaseStatus]:
    total_cases = _total_cases(task)
    if is_timeout_outcome(outcome):
        return [EvaluationCaseStatus.TIMEOUT] * total_cases
    if is_output_limit_outcome(outcome):
        return [EvaluationCaseStatus.ERROR] * total_cases
    if outcome.returncode in CANDIDATE_KILL_RETURNCODES:
        return [EvaluationCaseStatus.ERROR] * total_cases
    if outcome.returncode != 0:
        raise EvaluationHarnessError("runner subprocess exited nonzero")

    try:
        raw_results = json.loads(outcome.stdout)
    except json.JSONDecodeError as exc:
        raise EvaluationHarnessError(
            "runner output was not valid JSON",
            cause=exc,
        ) from exc
    if not isinstance(raw_results, list):
        raise EvaluationHarnessError("runner output had invalid shape")

    if task.parsed_tests is None:
        raise ValueError("HumanEvalTask.parsed_tests is required")
    expected_case_ids = {
        case.case_id for case in task.parsed_tests.cases
    }
    seen_case_ids: set[str] = set()
    adapter = TypeAdapter(HumanEvalRunnerCaseOutput)
    statuses: list[EvaluationCaseStatus] = []
    for item in raw_results:
        try:
            result = adapter.validate_python(item)
        except ValidationError as exc:
            raise EvaluationHarnessError(
                "runner output case failed validation",
                cause=exc,
            ) from exc
        if (
            result.case_id not in expected_case_ids
            or result.case_id in seen_case_ids
        ):
            raise EvaluationHarnessError(
                "runner output contained duplicate or unknown case ids"
            )
        seen_case_ids.add(result.case_id)
        statuses.append(result.status)
    return statuses


def _best_function_name(
    *,
    function_names: list[str],
    entry_point: str,
    statuses_by_name: Mapping[str, list[EvaluationCaseStatus]],
) -> str | None:
    if not function_names:
        return None
    return max(
        function_names,
        key=lambda function_name: (
            sum(
                status is EvaluationCaseStatus.PASSED
                for status in statuses_by_name[function_name]
            ),
            function_name == entry_point,
            -function_names.index(function_name),
        ),
    )
