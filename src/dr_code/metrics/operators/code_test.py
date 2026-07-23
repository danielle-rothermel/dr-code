"""Subprocess-backed HumanEval case execution facts.

This operator is HumanEval-specific by construction, not merely by its type
annotations: it builds ``HumanEvalRunnerPayload``, drives ``runner_script()``,
and depends on ``HumanEvalTask.parsed_tests`` semantics. There is deliberately
no generic ``Task`` supertype -- a single-implementation abstraction with a
guessed interface would be premature. The shared interface gets extracted when
a second benchmark exists to constrain it; until then the HumanEval scope is
kept honest through naming and docstrings.
"""

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
from dr_code.humaneval.task import (
    EvaluationCaseStatus,
    HumanEvalRunnerCaseOutput,
    HumanEvalRunnerPayload,
    HumanEvalTask,
)
from dr_code.metrics.engine.execution import (
    ExecutionOutcome,
    ExecutionRequest,
    is_candidate_kill_outcome,
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
        return plan_code_test_requests(
            task=self._task(aux),
            candidate_source=_code_source(value),
            timeout_seconds=self.settings.timeout_seconds,
        )

    def compute(
        self,
        value: Artifact,
        aux: Mapping[str, Artifact],
        ctx: EngineContext,
    ) -> CodeTestResult:
        task = self._task(aux)
        candidate_source = _code_source(value)
        requests = plan_code_test_requests(
            task=task,
            candidate_source=candidate_source,
            timeout_seconds=self.settings.timeout_seconds,
        )
        return compute_code_test_result(
            task=task,
            candidate_source=candidate_source,
            timeout_seconds=self.settings.timeout_seconds,
            outcomes={
                request.cache_key: ctx.outcome_for(request)
                for request in requests
            },
        )

    def _task(self, aux: Mapping[str, Artifact]) -> HumanEvalTask:
        artifact = aux[self.settings.task_key]
        if not isinstance(artifact, JsonArtifact):
            raise TypeError("code_test task must be a JSON artifact")
        return _validate_task_payload(artifact)


def plan_code_test_requests(
    *,
    task: HumanEvalTask,
    candidate_source: str,
    timeout_seconds: float,
) -> tuple[ExecutionRequest, ...]:
    """Plan official ``code_test`` requests for an already validated task."""

    return tuple(
        _code_test_request(
            task=task,
            candidate_code=candidate_source,
            function_name=function_name,
            timeout_seconds=timeout_seconds,
        )
        for function_name in _top_level_function_names(candidate_source)
    )


def compute_code_test_result(
    *,
    task: HumanEvalTask,
    candidate_source: str,
    timeout_seconds: float,
    outcomes: Mapping[str, ExecutionOutcome],
) -> CodeTestResult:
    """Compute official facts from planned requests and their outcomes."""

    function_names = _top_level_function_names(candidate_source)
    requests = plan_code_test_requests(
        task=task,
        candidate_source=candidate_source,
        timeout_seconds=timeout_seconds,
    )
    statuses_by_name: dict[str, list[EvaluationCaseStatus]] = {}
    for function_name, request in zip(function_names, requests, strict=True):
        statuses = _statuses_from_outcome(
            outcome=outcomes[request.cache_key], task=task
        )
        statuses_by_name.setdefault(function_name, []).extend(statuses)

    best_function_name = _best_function_name(
        function_names=function_names,
        entry_point=task.entry_point,
        passed_counts=_passed_counts(statuses_by_name),
    )
    best_statuses = (
        statuses_by_name[best_function_name]
        if best_function_name is not None
        else []
    )
    counts = Counter(status.value for status in best_statuses)
    total_cases = _total_cases(task)
    return CodeTestResult(
        total_cases=total_cases,
        passed_count=counts.get(EvaluationCaseStatus.PASSED.value, 0),
        failed_count=counts.get(EvaluationCaseStatus.FAILED.value, 0),
        error_count=counts.get(EvaluationCaseStatus.ERROR.value, 0),
        timeout_count=counts.get(EvaluationCaseStatus.TIMEOUT.value, 0),
        coverage_complete=(
            best_function_name is not None
            and len(best_statuses) == total_cases
        ),
        function_count=len(function_names),
        best_function_name=best_function_name,
    )


def _code_test_request(
    *,
    task: HumanEvalTask,
    candidate_code: str,
    function_name: str,
    timeout_seconds: float,
) -> ExecutionRequest:
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
        timeout_seconds=timeout_seconds,
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


# PARITY TWIN: duplicates ``dr_code.humaneval.batch_runner``
# ``top_level_function_names``. Both stay live while the old scoring path runs;
# retires together with it. Baseline quirk (documented, not fixed): duplicate
# top-level function names (legal Python) are all returned, so their status
# counts stack downstream and can defeat coverage_complete -- exact parity with
# the retired path. Any fix must land in both twins.
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
    error_statuses = [EvaluationCaseStatus.ERROR] * total_cases
    # Four parallel outcome predicates. Timeout and output-limit are things a
    # candidate can provoke, and so is any hard kill (is_candidate_kill_outcome)
    # or unexpected nonzero exit -- a candidate can produce any returncode via
    # ``os._exit``. All of these are candidate-controlled *data*: they become
    # case statuses, never batch aborts. SubprocessError (raised at the
    # execution boundary before candidate code runs) is the only propagating
    # infra path.
    if is_timeout_outcome(outcome):
        return [EvaluationCaseStatus.TIMEOUT] * total_cases
    if is_output_limit_outcome(outcome):
        return error_statuses
    if is_candidate_kill_outcome(outcome):
        return error_statuses
    if outcome.returncode != 0:  # was: raise EvaluationHarnessError
        return error_statuses

    # KNOWN LIMITATION (documented, not fixed here): stdout is shared with the
    # candidate (batch_runner_script.py), so its contents are
    # candidate-controlled data. Reclassifying malformed stdout to ERROR
    # statuses contains the blast radius (one trace's record, not the whole
    # batch), but does not make stdout trustworthy: a candidate can forge a
    # valid-looking results array and ``os._exit(0)`` before the runner prints
    # the real one. This forgery hole is baseline (the retired scoring path
    # parses the same shared stdout); closing it needs a runner protocol change
    # (a separate result channel or an authenticated sentinel), which is out of
    # scope for this PR.
    try:
        raw_results = json.loads(outcome.stdout)
    except json.JSONDecodeError:  # candidate shares the runner's stdout
        return error_statuses
    if not isinstance(raw_results, list):
        return error_statuses

    if task.parsed_tests is None:
        raise ValueError("HumanEvalTask.parsed_tests is required")
    expected_case_ids = {case.case_id for case in task.parsed_tests.cases}
    seen_case_ids: set[str] = set()
    adapter = TypeAdapter(HumanEvalRunnerCaseOutput)
    statuses: list[EvaluationCaseStatus] = []
    for item in raw_results:
        try:
            result = adapter.validate_python(item)
        except ValidationError:
            return error_statuses
        if (
            result.case_id not in expected_case_ids
            or result.case_id in seen_case_ids
        ):
            return error_statuses
        seen_case_ids.add(result.case_id)
        statuses.append(result.status)
    return statuses


def _passed_counts(
    statuses_by_name: Mapping[str, list[EvaluationCaseStatus]],
) -> dict[str, int]:
    """Count PASSED statuses per function name.

    Split out of ``_best_function_name`` so counting and max-selection are
    independently testable and the per-function counts are a natural debugging
    hook. Duplicate top-level function names (legal Python) collapse into a
    single ``statuses_by_name`` key upstream, so their counts stack -- a
    baseline quirk documented, not fixed (see ``_top_level_function_names``).
    """

    return {
        function_name: sum(
            status is EvaluationCaseStatus.PASSED for status in statuses
        )
        for function_name, statuses in statuses_by_name.items()
    }


# PARITY TWIN: this selector duplicates
# ``dr_code.humaneval.task.select_best_function_name`` (and the coverage logic
# in ``CodeTest.compute`` duplicates
# ``EvaluationTaskResult.coverage_complete``). Both implementations stay live
# while the old scoring path still runs; direct reuse is awkward because
# task.py's selector takes ``EvaluationCaseResult`` objects while the operator
# holds bare statuses (a gap the ``_passed_counts`` split widens). The guard is
# a parity test (tests/metrics/test_operator_parity.py) pinning the two
# selectors equal over the same synthetic status sets. RETIREMENT PLAN: when the
# scoring path retires, the task.py copy goes with it and the parity test is
# deleted. Any fix to the duplicate-name quirk must land in BOTH twins.
def _best_function_name(
    *,
    function_names: list[str],
    entry_point: str,
    passed_counts: Mapping[str, int],
) -> str | None:
    if not function_names:
        return None
    return max(
        function_names,
        key=lambda function_name: (
            passed_counts.get(function_name, 0),
            function_name == entry_point,
            -function_names.index(function_name),
        ),
    )
