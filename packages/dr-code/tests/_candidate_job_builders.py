from __future__ import annotations

import hashlib

from dr_serialize import Sha256Digest

from _stubs.candidate_harness.job import (
    DEFAULT_FIELD_LIMIT,
    StubCandidateJobRequest,
    StubCandidateJobResult,
    StubEvaluatorSuite,
    build_candidate_job_request,
    evaluate_stub_candidate_job,
)
from _stubs.stub_code_test import StubCodeTestSettings, candidate_job_task
from dr_code.evaluation import (
    CandidateJobBudget,
    EvalCandidateId,
    EvalSampleId,
    MaterializedEvalCandidate,
)
from dr_code.evaluation.candidate_job import (
    CandidateEvaluatorSuite,
    register_candidate_job_builder,
    register_candidate_job_result_type,
)
from dr_code.metrics import MetricName, MetricQuestionCoordinate
from dr_code.metrics.coordinates import question_settings
from dr_code.metrics.registry import register_metric_operator
from dr_code.trace import CodeArtifact, PreprocessingDefinitionCoordinate
from _stubs.stub_code_test import StubCodeTest


def candidate_job_suite(
    on_key: str,
    *,
    task: dict[str, object] | None = None,
) -> StubEvaluatorSuite:
    payload = task or candidate_job_task()
    settings = StubCodeTestSettings()
    question = MetricQuestionCoordinate(
        metric=MetricName.CODE_TEST,
        on_key=on_key,
        settings=question_settings(settings),
    )
    return StubEvaluatorSuite(
        question=question,
        entry_point=str(payload["entry_point"]),
        inputs=tuple(tuple(item) for item in payload["inputs"]),
        expected=tuple(payload["expected"]),
        support_failure=bool(payload.get("support_failure", False)),
    )


def candidate_job_request(
    source: str,
    *suites: StubEvaluatorSuite,
) -> StubCandidateJobRequest:
    candidate = MaterializedEvalCandidate(
        identity=EvalCandidateId(
            sample=EvalSampleId(sample_id="sample"),
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
    evaluator_suites = tuple(
        CandidateEvaluatorSuite(
            question=suite.question,
            suite_kind="stub",
            suite_payload=suite.model_dump(mode="json"),
        )
        for suite in (suites or (candidate_job_suite("candidate"),))
    )
    return build_candidate_job_request(candidate, evaluator_suites)


def candidate_job_budget() -> CandidateJobBudget:
    return CandidateJobBudget(
        wall_time_ns=5_000_000_000,
        input_bytes=2_097_152,
        payload_output_bytes=2_097_152,
        stdout_head_bytes=1_048_576,
        stderr_head_bytes=1_048_576,
    )


register_metric_operator(str(MetricName.CODE_TEST), StubCodeTest)
register_candidate_job_builder("stub", build_candidate_job_request)
register_candidate_job_result_type(StubCandidateJobResult)

__all__ = [
    "DEFAULT_FIELD_LIMIT",
    "candidate_job_budget",
    "candidate_job_request",
    "candidate_job_suite",
    "candidate_job_task",
    "evaluate_stub_candidate_job",
]
