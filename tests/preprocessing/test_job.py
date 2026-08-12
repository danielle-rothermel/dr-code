from __future__ import annotations

from uuid import uuid4

import pytest
from dr_exec import Budgets, JobId

from dr_code.preprocessing import EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION
from dr_code.preprocessing.execution import (
    build_candidate_sources_job,
    build_preprocess_text_job,
    preprocess_job_budgets,
    preprocess_text_request,
)
from dr_code.preprocessing.job import (
    PREPROCESS_TEXT_JOB_SCHEMA_VERSION,
    CandidateSourcesJobResult,
    PreprocessTextJobRequest,
    PreprocessTextJobResult,
    candidate_sources_job,
    preprocess_text_job,
)
from dr_code.trace import OUTPUT_KEY, Absent, deserialize_trace

_FENCED = "Here is the code:\n```python\ndef f(x):\n    return x + 1\n```\n"
_PROSE = "Just an explanation, no code at all.\n"


def test_preprocess_text_job_returns_serialized_trace() -> None:
    request = PreprocessTextJobRequest(
        text=_FENCED,
        definition_id=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION.definition_id,
        definition_version=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION.version,
    )
    payload = preprocess_text_job(
        request.model_dump(mode="json", exclude_computed_fields=True)
    )
    assert payload["schema_version"] == PREPROCESS_TEXT_JOB_SCHEMA_VERSION
    assert payload["trace"]["schema_version"] == 3


def test_preprocess_text_job_preserves_absent_output() -> None:
    request = PreprocessTextJobRequest(
        text=_PROSE,
        definition_id=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION.definition_id,
        definition_version=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION.version,
    )
    payload = preprocess_text_job(
        request.model_dump(mode="json", exclude_computed_fields=True)
    )
    result = PreprocessTextJobResult.model_validate(payload)
    output = deserialize_trace(result.trace).value(OUTPUT_KEY)
    assert isinstance(output, Absent)


def test_candidate_sources_job_returns_only_sources() -> None:
    request = PreprocessTextJobRequest(
        text=_FENCED,
        definition_id=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION.definition_id,
        definition_version=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION.version,
    )
    payload = candidate_sources_job(
        request.model_dump(mode="json", exclude_computed_fields=True)
    )
    result = CandidateSourcesJobResult.model_validate(payload)
    assert result.sources == ("def f(x):\n    return x + 1",)
    assert "trace" not in payload


def test_candidate_sources_job_returns_nothing_for_absent_output() -> None:
    request = PreprocessTextJobRequest(
        text=_PROSE,
        definition_id=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION.definition_id,
        definition_version=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION.version,
    )
    payload = candidate_sources_job(
        request.model_dump(mode="json", exclude_computed_fields=True)
    )
    assert CandidateSourcesJobResult.model_validate(payload).sources == ()


def _request() -> PreprocessTextJobRequest:
    return preprocess_text_request(
        _FENCED, EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION
    )


@pytest.mark.parametrize(
    "build_job",
    [build_preprocess_text_job, build_candidate_sources_job],
)
def test_preprocessing_jobs_are_unbudgeted_by_default(build_job) -> None:
    """The library declares no wall-time limit unless a caller asks."""

    job = build_job(JobId(uuid4()), _request())

    assert job.budgets == Budgets.unbudgeted()


@pytest.mark.parametrize(
    "build_job",
    [build_preprocess_text_job, build_candidate_sources_job],
)
def test_preprocessing_jobs_carry_a_declared_wall_time_budget(
    build_job,
) -> None:
    job = build_job(
        JobId(uuid4()), _request(), budgets=preprocess_job_budgets(600.0)
    )

    assert job.budgets.wall_time.max_ns == 600 * 1_000_000_000


def test_preprocess_job_budgets_bounds_only_wall_time() -> None:
    """Preprocessing is trusted work; only the wedged-job axis is bounded."""

    budgets = preprocess_job_budgets(600.0)
    unbudgeted = Budgets.unbudgeted()

    assert budgets.wall_time.max_ns == 600 * 1_000_000_000
    assert budgets.input_bytes == unbudgeted.input_bytes
    assert budgets.payload_output == unbudgeted.payload_output
    assert budgets.memory_bytes == unbudgeted.memory_bytes


@pytest.mark.parametrize(
    "wall_time_seconds", [0.0, -1.0, float("inf"), float("nan")]
)
def test_preprocess_job_budgets_rejects_unusable_limits(
    wall_time_seconds: float,
) -> None:
    with pytest.raises(ValueError):
        preprocess_job_budgets(wall_time_seconds)
