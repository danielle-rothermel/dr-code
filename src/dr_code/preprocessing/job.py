from __future__ import annotations

from typing import Final, Literal

from dr_exec import ImportableEntryPoint
from dr_serialize import Jsonable
from pydantic import StrictStr

from dr_code.core.models import FrozenModel
from dr_code.preprocessing import (
    bind_preprocessing,
    resolve_preprocessing_definition,
)
from dr_code.trace import TextArtifact, serialize_trace
from dr_code.trace.serialization import SerializedTrace

PREPROCESS_TEXT_JOB_SCHEMA_VERSION: Final = 1
PREPROCESS_TEXT_ENTRY_POINT: Final = ImportableEntryPoint(
    module_name="dr_code.preprocessing.job",
    attribute_name="preprocess_text_job",
)


class PreprocessTextJobRequest(FrozenModel):
    schema_version: Literal[1] = PREPROCESS_TEXT_JOB_SCHEMA_VERSION
    text: StrictStr
    definition_id: StrictStr
    definition_version: StrictStr


class PreprocessTextJobResult(FrozenModel):
    schema_version: Literal[1] = PREPROCESS_TEXT_JOB_SCHEMA_VERSION
    trace: SerializedTrace


def preprocess_text_job(request: Jsonable) -> Jsonable:
    """Run one trusted preprocessing definition over one text input."""

    validated = PreprocessTextJobRequest.model_validate(request)
    definition = resolve_preprocessing_definition(
        definition_id=validated.definition_id,
        version=validated.definition_version,
    )
    runner = bind_preprocessing(definition)
    trace = runner.run(TextArtifact(text=validated.text))
    result = PreprocessTextJobResult(trace=serialize_trace(trace))
    return result.model_dump(mode="json", exclude_computed_fields=True)


__all__ = [
    "PREPROCESS_TEXT_ENTRY_POINT",
    "PREPROCESS_TEXT_JOB_SCHEMA_VERSION",
    "PreprocessTextJobRequest",
    "PreprocessTextJobResult",
    "preprocess_text_job",
]
