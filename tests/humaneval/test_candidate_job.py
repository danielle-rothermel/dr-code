from __future__ import annotations

import hashlib
from uuid import UUID

from dr_exec import FakeExecutor, JobId
from dr_serialize import Sha256Digest, build_identity_document

from dr_code.evaluation import (
    CandidateJobBudget,
    CandidateJobTerminated,
    CandidateTerminationReason,
    EvaluationCandidateIdentity,
    EvaluationRuntimeIdentity,
    EvaluationSampleIdentity,
    MaterializedEvaluationCandidate,
    RunGrade,
)
from dr_code.evaluation.execution import execute_candidate_job
from dr_code.humaneval.job import (
    CandidateNamespaceFailure,
    DEFAULT_FIELD_LIMIT,
    FIELD_TRUNCATION_MARKER,
    HumanEvalCandidateJobRequest,
    HumanEvalCandidateJobResult,
    HumanEvalEvaluatorSuite,
    HumanEvalSuiteCompleted,
    HumanEvalSuiteHarnessFailure,
    evaluate_humaneval_candidate_job,
)
from dr_code.humaneval.settings import CodeTestSettings
from dr_code.humaneval.task import HumanEvalTask
from dr_code.metrics import MetricName, MetricQuestionCoordinate
from dr_code.metrics.coordinates import question_settings
from dr_code.trace import CodeArtifact, PreprocessingDefinitionCoordinate


def _task(*, support_failure: bool = False) -> HumanEvalTask:
    support = (
        "raise RuntimeError('support broke')\n" if support_failure else ""
    )
    return HumanEvalTask(
        task_id="HumanEval/candidate-job",
        prompt="def observed_load_count(x):\n",
        canonical_solution="    return 1\n",
        entry_point="observed_load_count",
        test=(
            support + "def check(candidate):\n"
            "    inputs = [(0,), (1,)]\n"
            "    results = [1, 1]\n"
            "    for inp, expected in zip(inputs, results):\n"
            "        assertion(candidate(*inp), expected)\n"
        ),
    )


def _candidate(source: str) -> MaterializedEvaluationCandidate:
    return MaterializedEvaluationCandidate(
        identity=EvaluationCandidateIdentity(
            sample=EvaluationSampleIdentity(sample_id="sample"),
            preprocessing=PreprocessingDefinitionCoordinate(
                definition_id="pre",
                version="0",
                steps=(),
            ),
            candidate_ordinal=0,
        ),
        source=CodeArtifact(source=source),
        source_sha256=Sha256Digest(
            hashlib.sha256(source.encode("utf-8")).hexdigest()
        ),
    )


def _suite(
    on_key: str, *, task: HumanEvalTask | None = None
) -> HumanEvalEvaluatorSuite:
    settings = CodeTestSettings()
    return HumanEvalEvaluatorSuite(
        question=MetricQuestionCoordinate(
            metric=MetricName.CODE_TEST,
            on_key=on_key,
            settings=question_settings(settings),
        ),
        task=task or _task(),
        settings=settings,
    )


def _request(
    source: str, *suites: HumanEvalEvaluatorSuite
) -> HumanEvalCandidateJobRequest:
    return HumanEvalCandidateJobRequest(
        candidate=_candidate(source),
        suites=tuple(suites) or (_suite("candidate"),),
    )


def _budget() -> CandidateJobBudget:
    return CandidateJobBudget(
        wall_time_ns=5_000_000_000,
        input_bytes=2_097_152,
        payload_output_bytes=2_097_152,
        stdout_head_bytes=1_048_576,
        stderr_head_bytes=1_048_576,
    )


def test_candidate_source_loads_once_for_all_ordered_suites_and_groups() -> (
    None
):
    request = _request(
        "loads = 0\nloads += 1\ndef observed_load_count(_x):\n    return loads\n",
        _suite("first"),
        _suite("second"),
    )

    raw = evaluate_humaneval_candidate_job(
        request.model_dump(mode="json", exclude_computed_fields=True)
    )
    result = HumanEvalCandidateJobResult.model_validate(raw)

    assert result.candidate == request.candidate.identity
    assert result.namespace.kind == "loaded"
    assert result.namespace.function_names == ("observed_load_count",)
    assert tuple(suite.question for suite in result.suites) == tuple(
        suite.question for suite in request.suites
    )
    assert all(
        isinstance(suite, HumanEvalSuiteCompleted) for suite in result.suites
    )
    assert [
        case.status.value
        for suite in result.suites
        for group in suite.groups
        for case in group.cases
    ] == [
        "passed",
        "passed",
        "passed",
        "passed",
    ]


def test_candidate_namespace_failure_returns_no_suites() -> None:
    request = _request("raise RuntimeError('candidate broke')\n")
    result = HumanEvalCandidateJobResult.model_validate(
        evaluate_humaneval_candidate_job(
            request.model_dump(mode="json", exclude_computed_fields=True)
        )
    )

    assert isinstance(result.namespace, CandidateNamespaceFailure)
    assert result.namespace.failure_type == "RuntimeError"
    assert "candidate broke" in result.namespace.message
    assert result.suites == ()


def test_support_failure_is_a_suite_harness_failure() -> None:
    request = _request(
        "def observed_load_count(_x):\n    return 1\n",
        _suite("candidate", task=_task(support_failure=True)),
    )
    result = HumanEvalCandidateJobResult.model_validate(
        evaluate_humaneval_candidate_job(
            request.model_dump(mode="json", exclude_computed_fields=True)
        )
    )

    (suite,) = result.suites
    assert isinstance(suite, HumanEvalSuiteHarnessFailure)
    assert suite.failure_type == "RuntimeError"
    assert "support broke" in suite.message
    assert suite.completed_groups == ()


def test_real_importable_json_candidate_job(
    local_executor: FakeExecutor,
) -> None:
    request = _request("def observed_load_count(_x):\n    return 1\n")
    record = execute_candidate_job(
        request,
        job_id=JobId(UUID("00000000-0000-0000-0000-000000000001")),
        budget=_budget(),
        runtime=EvaluationRuntimeIdentity(
            document=build_identity_document(
                schema="tests/runtime",
                schema_version=1,
                payload={"name": "real-importable-json"},
            )
        ),
        cache_namespace="tests/candidate-job",
        run_grade=RunGrade.TRIAL,
        executor=local_executor,
    )

    assert record.outcome.kind == "completed"
    assert record.provenance.kind == "executed"
    assert record.provenance.record_receipt.kind == "not_applicable"


def test_candidate_nonzero_exit_remains_candidate_owned(
    local_executor: FakeExecutor,
) -> None:
    record = execute_candidate_job(
        _request("import os\nos._exit(7)\n"),
        job_id=JobId(UUID("00000000-0000-0000-0000-000000000007")),
        budget=_budget(),
        runtime=EvaluationRuntimeIdentity(
            document=build_identity_document(
                schema="tests/runtime",
                schema_version=1,
                payload={"name": "candidate-nonzero-exit"},
            )
        ),
        cache_namespace="tests/candidate-job",
        run_grade=RunGrade.TRIAL,
        executor=local_executor,
    )

    assert isinstance(record.outcome, CandidateJobTerminated)
    assert record.outcome.reason is CandidateTerminationReason.NONZERO_EXIT


def _long_message_request(
    field_limit: int | None = None,
) -> HumanEvalCandidateJobRequest:
    source = (
        "def observed_load_count(x):\n    raise RuntimeError('E' * 100_000)\n"
    )
    fields: dict[str, object] = {
        "candidate": _candidate(source),
        "suites": (_suite("candidate"),),
    }
    if field_limit is not None:
        fields["field_limit"] = field_limit
    return HumanEvalCandidateJobRequest(**fields)  # type: ignore[arg-type]


def _first_case_message(result: HumanEvalCandidateJobResult) -> str:
    suite = result.suites[0]
    assert isinstance(suite, HumanEvalSuiteCompleted)
    return suite.groups[0].cases[0].message


def test_default_field_limit_clips_evidence_with_the_pinned_marker() -> None:
    raw = evaluate_humaneval_candidate_job(
        _long_message_request().model_dump(
            mode="json", exclude_computed_fields=True
        )
    )
    result = HumanEvalCandidateJobResult.model_validate(raw)

    message = _first_case_message(result)

    assert message.endswith(FIELD_TRUNCATION_MARKER)
    assert len(message) == DEFAULT_FIELD_LIMIT + len(FIELD_TRUNCATION_MARKER)


def test_field_limit_knob_overrides_the_default_clip_length() -> None:
    raw = evaluate_humaneval_candidate_job(
        _long_message_request(field_limit=64).model_dump(
            mode="json", exclude_computed_fields=True
        )
    )
    result = HumanEvalCandidateJobResult.model_validate(raw)

    message = _first_case_message(result)

    assert message.endswith(FIELD_TRUNCATION_MARKER)
    assert len(message) == 64 + len(FIELD_TRUNCATION_MARKER)


def test_field_limit_defaults_to_the_raised_library_value() -> None:
    assert DEFAULT_FIELD_LIMIT == 32_000
    assert FIELD_TRUNCATION_MARKER == "...[truncated]"
    assert (
        HumanEvalCandidateJobRequest.model_fields["field_limit"].default
        == DEFAULT_FIELD_LIMIT
    )


def _large_repr_task() -> HumanEvalTask:
    huge = "E" * 100_000
    return HumanEvalTask(
        task_id="HumanEval/large-repr",
        prompt="def observed_load_count(x):\n",
        canonical_solution="    return x\n",
        entry_point="observed_load_count",
        test=(
            "def check(candidate):\n"
            f"    inputs = [({huge!r},)]\n"
            f"    results = [{huge!r}]\n"
            "    for inp, expected in zip(inputs, results):\n"
            "        assertion(candidate(*inp), expected)\n"
        ),
    )


def _first_case(result: HumanEvalCandidateJobResult):
    suite = result.suites[0]
    assert isinstance(suite, HumanEvalSuiteCompleted)
    return suite.groups[0].cases[0]


def test_field_limit_clips_input_and_expected_repr_on_passed_cases() -> None:
    raw = evaluate_humaneval_candidate_job(
        HumanEvalCandidateJobRequest(
            candidate=_candidate(
                "def observed_load_count(x):\n    return x\n"
            ),
            suites=(_suite("candidate", task=_large_repr_task()),),
            field_limit=64,
        ).model_dump(mode="json", exclude_computed_fields=True)
    )
    case = _first_case(HumanEvalCandidateJobResult.model_validate(raw))

    assert case.status.value == "passed"
    assert case.input_repr.endswith(FIELD_TRUNCATION_MARKER)
    assert case.expected_output_repr.endswith(FIELD_TRUNCATION_MARKER)
    assert len(case.input_repr) == 64 + len(FIELD_TRUNCATION_MARKER)
    assert len(case.expected_output_repr) == 64 + len(FIELD_TRUNCATION_MARKER)


def test_field_limit_clips_input_and_expected_repr_on_failed_cases() -> None:
    raw = evaluate_humaneval_candidate_job(
        HumanEvalCandidateJobRequest(
            candidate=_candidate(
                "def observed_load_count(x):\n    return 0\n"
            ),
            suites=(_suite("candidate", task=_large_repr_task()),),
            field_limit=64,
        ).model_dump(mode="json", exclude_computed_fields=True)
    )
    case = _first_case(HumanEvalCandidateJobResult.model_validate(raw))

    assert case.status.value == "failed"
    assert case.input_repr.endswith(FIELD_TRUNCATION_MARKER)
    assert case.expected_output_repr.endswith(FIELD_TRUNCATION_MARKER)
    assert len(case.input_repr) == 64 + len(FIELD_TRUNCATION_MARKER)
    assert len(case.expected_output_repr) == 64 + len(FIELD_TRUNCATION_MARKER)
