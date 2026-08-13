from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from dr_serialize import Jsonable
from dr_code.evaluation.candidate_job import CandidateEvaluatorSuite
from dr_code.evaluation.records import (
    CandidateJobCompleted,
    CandidateJobTerminated,
    ExecutorExecutionFailure,
    HarnessExecutionFailure,
)
from dr_code.metrics.coordinates import MetricQuestionCoordinate
from dr_code.metrics.names import MetricName
from dr_code.metrics.operators.base import (
    EngineContext,
    MetricOperator,
    OperatorResult,
)
from dr_code.metrics.settings import OperatorSettings
from dr_code.metrics.units import MetricValueUnit
from dr_code.trace import Artifact, ArtifactKind, JsonArtifact
from _stubs.candidate_harness.job import StubEvaluatorSuite


class StubCodeTestSettings(OperatorSettings):
    task_key: str = "task"


class StubCodeTestResult(OperatorResult):
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


class StubCodeTest(MetricOperator[StubCodeTestSettings]):
    NAME = MetricName.CODE_TEST
    VERSION = "0"
    INPUT = ArtifactKind.CODE
    INJECTS_CANDIDATE_SOURCE = True
    Settings = StubCodeTestSettings

    def auxiliary_keys(self) -> tuple[str, ...]:
        return (self.settings.task_key,)

    def accepted_auxiliary_kinds(
        self,
        key: str,
    ) -> frozenset[ArtifactKind]:
        _ = key
        return frozenset({ArtifactKind.JSON})

    def evaluator_suites(
        self,
        value: Artifact,
        aux: Mapping[str, Artifact],
        question: MetricQuestionCoordinate,
        /,
    ) -> tuple[CandidateEvaluatorSuite, ...]:
        _ = value
        task = _task_payload(aux[self.settings.task_key])
        suite = StubEvaluatorSuite(
            question=question,
            entry_point=task["entry_point"],
            inputs=tuple(tuple(item) for item in task["inputs"]),
            expected=tuple(task["expected"]),
            support_failure=task.get("support_failure", False),
        )
        return (
            CandidateEvaluatorSuite(
                question=question,
                suite_kind="stub",
                suite_payload=cast(
                    Jsonable,
                    suite.model_dump(mode="json"),
                ),
            ),
        )

    def compute(
        self,
        value: Artifact,
        aux: Mapping[str, Artifact],
        ctx: EngineContext,
    ) -> StubCodeTestResult:
        _ = (value, aux)
        outcome = ctx.candidate_execution_outcome
        if not isinstance(
            outcome,
            CandidateJobCompleted
            | CandidateJobTerminated
            | HarnessExecutionFailure
            | ExecutorExecutionFailure,
        ):
            raise RuntimeError(
                "stub code_test has no candidate execution outcome"
            )
        if not isinstance(ctx.question, MetricQuestionCoordinate):
            raise TypeError("question must be a MetricQuestionCoordinate")
        if isinstance(
            outcome,
            HarnessExecutionFailure | ExecutorExecutionFailure,
        ):
            return StubCodeTestResult(
                total_cases=1,
                passed_count=0,
                failed_count=0,
                error_count=1,
                timeout_count=0,
                coverage_complete=False,
                function_count=0,
                best_function_name=None,
            )
        if isinstance(outcome, CandidateJobTerminated):
            return StubCodeTestResult(
                total_cases=1,
                passed_count=0,
                failed_count=0,
                error_count=0,
                timeout_count=1,
                coverage_complete=False,
                function_count=0,
                best_function_name=None,
            )
        from _stubs.candidate_harness.job import (
            StubCandidateJobResult,
            StubSuiteCompleted,
        )

        result = outcome.result
        if not isinstance(result, StubCandidateJobResult):
            raise TypeError("stub code_test requires a StubCandidateJobResult")
        suite = next(
            (item for item in result.suites if item.question == ctx.question),
            None,
        )
        if suite is None:
            raise RuntimeError("stub code_test has no suite for its question")
        if isinstance(suite, StubSuiteCompleted):
            passed_count = 1 if suite.passed else 0
            failed_count = 0 if suite.passed else 1
            return StubCodeTestResult(
                total_cases=1,
                passed_count=passed_count,
                failed_count=failed_count,
                error_count=0,
                timeout_count=0,
                coverage_complete=suite.passed,
                function_count=1,
                best_function_name="observed_load_count",
            )
        return StubCodeTestResult(
            total_cases=1,
            passed_count=0,
            failed_count=0,
            error_count=1,
            timeout_count=0,
            coverage_complete=False,
            function_count=0,
            best_function_name=None,
        )


def _task_payload(artifact: Artifact) -> dict[str, object]:
    if not isinstance(artifact, JsonArtifact):
        raise TypeError("stub code_test task must be a JSON artifact")
    payload = artifact.model_dump(mode="json")["payload"]
    if not isinstance(payload, dict):
        raise ValueError("stub code_test task payload must be a JSON object")
    return payload


def candidate_job_task(*, support_failure: bool = False) -> dict[str, object]:
    return {
        "task_id": "stub/candidate-job",
        "entry_point": "observed_load_count",
        "inputs": [[0], [1]],
        "expected": [1, 1],
        "support_failure": support_failure,
    }
