"""Identity Document helpers for dr-code-owned domain objects.

dr-code selects the identity-bearing payload fields; dr-serialize owns
the exact ``{schema, schema_version, payload}`` document shape, Canonical
Identity JSON rendering, and the full 64-char lowercase SHA-256 Identity
Hash. This module is the single seam between the two: every dr-code
identity flows through :func:`identity_hash_for`.
"""

from __future__ import annotations

from typing import cast

from dr_serialize import (
    Jsonable,
    build_identity_document,
    identity_document_hash,
)

# Owning-domain schema names. dr-code owns this namespace; whetstone and
# other repos own theirs. Bumping ``schema_version`` is a deliberate
# breaking identity change for that schema.
SCHEMA_VERSION = 1

SCHEMA_SAMPLING_DEFINITION = "dr_code.sampling.definition"
SCHEMA_SAMPLING_CONFIG = "dr_code.sampling.config"
SCHEMA_PREPROCESSING_DEFINITION = "dr_code.preprocessing.definition"
SCHEMA_PREPROCESSING_CONFIG = "dr_code.preprocessing.config"
SCHEMA_METRIC_EXTRACTION_DEFINITION = "dr_code.metric_extraction.definition"
SCHEMA_METRIC_EXTRACTION_CONFIG = "dr_code.metric_extraction.config"
SCHEMA_EVALUATION_PROCEDURE_DEFINITION = (
    "dr_code.evaluation_procedure.definition"
)
SCHEMA_EVALUATION_PROCEDURE_CONFIG = "dr_code.evaluation_procedure.config"
SCHEMA_AGGREGATION_DEFINITION = "dr_code.aggregation.definition"
SCHEMA_AGGREGATION_CONFIG = "dr_code.aggregation.config"
SCHEMA_EVAL_DEFINITION = "dr_code.eval.definition"
SCHEMA_EVAL_CONFIG = "dr_code.eval.config"

SCHEMA_HUMANEVAL_TASK = "dr_code.humaneval.task"
SCHEMA_TASK_SET = "dr_code.task_set"
SCHEMA_REPEAT_PLAN = "dr_code.repeat_plan"
SCHEMA_REPEAT_ID = "dr_code.repeat_id"
SCHEMA_COMPRESSION_REFERENCE_KEY = "dr_code.compression_reference.key"
SCHEMA_COMPRESSION_REFERENCE_ARTIFACT = (
    "dr_code.compression_reference.artifact"
)


def identity_hash_for(
    *,
    schema: str,
    payload: object,
    schema_version: int = SCHEMA_VERSION,
) -> str:
    """Return the full Identity Hash for a dr-code identity payload.

    The caller passes the *complete* identity-bearing payload as already
    finite JSON-safe data (no runtime/storage fields). ``payload`` is
    typed ``object`` because callers assemble ordered ``dict``/``list``
    structures whose element types widen to ``object``; dr-serialize's
    ``validate_finite_json`` is the authoritative runtime guard that
    rejects any non-finite-JSON value before hashing.
    """

    document = build_identity_document(
        schema=schema,
        schema_version=schema_version,
        payload=cast(Jsonable, payload),
    )
    return identity_document_hash(document)


__all__ = [
    "SCHEMA_AGGREGATION_CONFIG",
    "SCHEMA_AGGREGATION_DEFINITION",
    "SCHEMA_COMPRESSION_REFERENCE_ARTIFACT",
    "SCHEMA_COMPRESSION_REFERENCE_KEY",
    "SCHEMA_EVALUATION_PROCEDURE_CONFIG",
    "SCHEMA_EVALUATION_PROCEDURE_DEFINITION",
    "SCHEMA_EVAL_CONFIG",
    "SCHEMA_EVAL_DEFINITION",
    "SCHEMA_HUMANEVAL_TASK",
    "SCHEMA_METRIC_EXTRACTION_CONFIG",
    "SCHEMA_METRIC_EXTRACTION_DEFINITION",
    "SCHEMA_PREPROCESSING_CONFIG",
    "SCHEMA_PREPROCESSING_DEFINITION",
    "SCHEMA_REPEAT_ID",
    "SCHEMA_REPEAT_PLAN",
    "SCHEMA_SAMPLING_CONFIG",
    "SCHEMA_SAMPLING_DEFINITION",
    "SCHEMA_TASK_SET",
    "SCHEMA_VERSION",
    "identity_hash_for",
]
