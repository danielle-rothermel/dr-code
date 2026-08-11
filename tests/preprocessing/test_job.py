from __future__ import annotations

from dr_code.preprocessing import EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION
from dr_code.preprocessing.job import (
    PREPROCESS_TEXT_JOB_SCHEMA_VERSION,
    PreprocessTextJobRequest,
    PreprocessTextJobResult,
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
