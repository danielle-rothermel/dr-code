from __future__ import annotations

import hashlib

from dr_serialize import Sha256Digest

from dr_code.evaluation import (
    CandidateJobBudget,
    EvalCandidateIdentity,
    EvalSampleIdentity,
    MaterializedEvalCandidate,
)
from dr_code.humaneval.job import (
    HumanEvalCandidateJobRequest,
    HumanEvalEvaluatorSuite,
)
from dr_code.humaneval.settings import CodeTestSettings
from dr_code.humaneval.task import HumanEvalTask
from dr_code.metrics import MetricName, MetricQuestionCoordinate
from dr_code.metrics.coordinates import question_settings
from dr_code.trace import CodeArtifact, PreprocessingDefinitionCoordinate


def candidate_job_task(*, support_failure: bool = False) -> HumanEvalTask:
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


def candidate_job_suite(
    on_key: str,
    *,
    task: HumanEvalTask | None = None,
) -> HumanEvalEvaluatorSuite:
    settings = CodeTestSettings()
    return HumanEvalEvaluatorSuite(
        question=MetricQuestionCoordinate(
            metric=MetricName.CODE_TEST,
            on_key=on_key,
            settings=question_settings(settings),
        ),
        task=task or candidate_job_task(),
        settings=settings,
    )


def candidate_job_request(
    source: str,
    *suites: HumanEvalEvaluatorSuite,
) -> HumanEvalCandidateJobRequest:
    candidate = MaterializedEvalCandidate(
        identity=EvalCandidateIdentity(
            sample=EvalSampleIdentity(sample_id="sample"),
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
    return HumanEvalCandidateJobRequest(
        candidate=candidate,
        suites=tuple(suites) or (candidate_job_suite("candidate"),),
    )


def candidate_job_budget() -> CandidateJobBudget:
    return CandidateJobBudget(
        wall_time_ns=5_000_000_000,
        input_bytes=2_097_152,
        payload_output_bytes=2_097_152,
        stdout_head_bytes=1_048_576,
        stderr_head_bytes=1_048_576,
    )
