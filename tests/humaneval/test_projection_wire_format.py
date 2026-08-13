from __future__ import annotations

import json

from _candidate_job_builders import candidate_job_request, candidate_job_suite

from dr_code.evaluation import (
    BundleRecordReference,
    EvalSampleIdentity,
    FailureClass,
)
from dr_code.humaneval import (
    DEFAULT_HUMANEVAL_SCORING_PROFILE,
    HUMANEVAL_CANDIDATE_ENTRY_POINT,
    HUMANEVAL_CANDIDATE_JOB_SCHEMA_VERSION,
    HarnessFailure,
    HarnessFailureCause,
    HumanEvalCandidateJobResult,
)
from dr_code.humaneval.job import evaluate_humaneval_candidate_job


def _reference() -> BundleRecordReference:
    return BundleRecordReference(
        artifact_name="sample-records-00000000.jsonl",
        record_index=0,
        record_sha256="a" * 64,
        schema="dr-code/sample-evaluation-record-v1",
        schema_version=2,
    )


def test_candidate_entry_point_literals_are_pinned() -> None:
    assert (
        HUMANEVAL_CANDIDATE_ENTRY_POINT.module_name == "dr_code.humaneval.job"
    )
    assert (
        HUMANEVAL_CANDIDATE_ENTRY_POINT.attribute_name
        == "evaluate_humaneval_candidate_job"
    )
    assert HUMANEVAL_CANDIDATE_JOB_SCHEMA_VERSION == 2


def test_candidate_job_result_wire_keys_and_discriminators_are_exact() -> None:
    request = candidate_job_request(
        "def observed_load_count(_x):\n    return 1\n",
        candidate_job_suite("output"),
    )
    payload = evaluate_humaneval_candidate_job(
        request.model_dump(mode="json", exclude_computed_fields=True)
    )
    assert isinstance(payload, dict)
    assert list(payload) == [
        "schema_version",
        "candidate",
        "namespace",
        "suites",
    ]
    assert payload["schema_version"] == 2
    namespace = payload["namespace"]
    assert isinstance(namespace, dict)
    assert list(namespace) == ["kind", "function_names"]
    assert namespace["kind"] == "loaded"
    suites = payload["suites"]
    assert isinstance(suites, list)
    suite = suites[0]
    assert isinstance(suite, dict)
    assert list(suite) == ["kind", "question", "groups"]
    assert suite["kind"] == "completed"
    groups = suite["groups"]
    assert isinstance(groups, list)
    group = groups[0]
    assert isinstance(group, dict)
    assert list(group) == ["function_name", "cases"]
    assert HumanEvalCandidateJobResult.model_validate(payload)


def test_candidate_namespace_failure_wire_keys_are_exact() -> None:
    request = candidate_job_request(
        "raise RuntimeError('candidate broke')\n",
        candidate_job_suite("output"),
    )
    payload = evaluate_humaneval_candidate_job(
        request.model_dump(mode="json", exclude_computed_fields=True)
    )
    assert isinstance(payload, dict)
    namespace = payload["namespace"]
    assert isinstance(namespace, dict)
    assert list(namespace) == ["kind", "failure_type", "message"]
    assert namespace["kind"] == "candidate_failure"
    assert payload["suites"] == []


def test_harness_failure_wire_keys_and_class_literal_are_exact() -> None:
    failure = HarnessFailure(
        sample=EvalSampleIdentity(sample_id="sample-0"),
        evaluation=None,
        cause=HarnessFailureCause(
            exception_type="MissingCandidateExecution",
            message="evaluated sample record has no candidate execution",
        ),
        failure_class=FailureClass.INFRASTRUCTURE,
        scoring_profile=DEFAULT_HUMANEVAL_SCORING_PROFILE,
        sample_record=_reference(),
    )
    payload = json.loads(failure.model_dump_json())
    assert list(payload) == [
        "kind",
        "sample",
        "evaluation",
        "cause",
        "failure_class",
        "scoring_profile",
        "sample_record",
    ]
    assert payload["kind"] == "harness_failure"
    assert payload["failure_class"] == "infrastructure"
    assert payload["cause"]["exception_type"] == "MissingCandidateExecution"
