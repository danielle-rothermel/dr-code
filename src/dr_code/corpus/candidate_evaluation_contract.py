"""Canonical schema-5 identities, relations, and candidate result rows."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from enum import StrEnum
from typing import Final

import pyarrow as pa
from pydantic import ValidationError

from dr_code.eval.facts import (
    Applicability,
    FactScalar,
    MetricFact,
    MetricRecord,
    OperatorCoordinates,
    OperatorLineage,
    RecordStatus,
)
from dr_code.trace import TraceProducer
from dr_code.metrics.operators.code_test import CodeTestResult
from dr_code.metrics.policy_example import derive_outcome
from dr_code.corpus.preprocessing_contract import (
    PREPROCESSING_IDENTITY_FIELDS,
    PREPROCESSING_INPUT_FIELDS,
    PREPROCESSING_MANIFEST_FIELDS,
    PREPROCESSING_MANIFEST_SCHEMA_VERSION,
)

CANDIDATE_EVALUATION_SCHEMA_VERSION: Final = 5
CANDIDATE_EVALUATION_COORDINATE_FIELDS: Final = (
    "schema_version",
    "preprocessing_run",
    "corpus_sha256",
    "snapshot_sha256",
    "dataset",
    "metric_extraction_definition_ref",
    "metric_extraction_config",
    "metric_extraction_definition_identity",
    "metric_extraction_config_identity",
    "evaluation_procedure_definition_ref",
    "evaluation_procedure_config",
    "evaluation_procedure_definition_identity",
    "evaluation_procedure_config_identity",
    "trace_producer",
    "operator_coordinates",
    "question_identity_hash",
    "operator_name",
    "operator_version",
    "metrics_profile",
    "runner_identity",
    "runtime_identity",
    "host_runtime",
    "installed_environment",
    "trusted_source_sha256",
    "max_infrastructure_retries",
)
CANDIDATE_EVALUATION_MANIFEST_FIELDS: Final = frozenset(
    {
        *CANDIDATE_EVALUATION_COORDINATE_FIELDS,
        "evaluation_identity",
        "reuse_result_sources",
        "membership_rows",
        "result_rows",
        "candidate_membership_sha256",
        "candidate_results_sha256",
        "record_status_totals",
        "reused_result_rows",
        "reused_result_rows_by_source",
        "complete",
    }
)
CANDIDATE_RESULT_FACT_FIELDS: Final = (
    "function_count",
    "best_function_name",
    "total_cases",
    "passed_count",
    "failed_count",
    "error_count",
    "timeout_count",
    "coverage_complete",
)
MEMBERSHIP_SCHEMA: Final = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("candidate_id", pa.string(), nullable=False),
        pa.field("candidate_index", pa.int64(), nullable=False),
        pa.field("task_id", pa.string(), nullable=False),
        pa.field("task_identity", pa.string(), nullable=False),
        pa.field("source_kind", pa.string()),
        pa.field("source_sha256", pa.string(), nullable=False),
        pa.field("evaluation_key", pa.string(), nullable=False),
        pa.field("question_identity_hash", pa.string(), nullable=False),
        pa.field("operator_name", pa.string(), nullable=False),
        pa.field("operator_version", pa.string(), nullable=False),
        pa.field("trace_producer_json", pa.string(), nullable=False),
        pa.field("operator_coordinates_json", pa.string(), nullable=False),
        pa.field(
            "metric_extraction_config_identity", pa.string(), nullable=False
        ),
        pa.field(
            "evaluation_procedure_config_identity",
            pa.string(),
            nullable=False,
        ),
        pa.field("runtime_identity", pa.string(), nullable=False),
        pa.field("runner_identity", pa.string(), nullable=False),
    ]
)
RESULTS_SCHEMA: Final = pa.schema(
    [
        pa.field("evaluation_key", pa.string(), nullable=False),
        pa.field("task_id", pa.string(), nullable=False),
        pa.field("task_identity", pa.string(), nullable=False),
        pa.field("cleaned_source", pa.string(), nullable=False),
        pa.field("source_sha256", pa.string(), nullable=False),
        pa.field("question_identity_hash", pa.string(), nullable=False),
        pa.field("operator_name", pa.string(), nullable=False),
        pa.field("operator_version", pa.string(), nullable=False),
        pa.field("trace_producer_json", pa.string(), nullable=False),
        pa.field("operator_coordinates_json", pa.string(), nullable=False),
        pa.field(
            "metric_extraction_config_identity", pa.string(), nullable=False
        ),
        pa.field(
            "evaluation_procedure_config_identity",
            pa.string(),
            nullable=False,
        ),
        pa.field("runtime_identity", pa.string(), nullable=False),
        pa.field("runner_identity", pa.string(), nullable=False),
        pa.field("metrics_profile", pa.string(), nullable=False),
        pa.field("record_status", pa.string(), nullable=False),
        pa.field("failure_type", pa.string()),
        pa.field("failure_message", pa.string()),
        pa.field("outcome", pa.string()),
        pa.field("function_count", pa.int64()),
        pa.field("best_function_name", pa.string()),
        pa.field("total_cases", pa.int64()),
        pa.field("passed_count", pa.int64()),
        pa.field("failed_count", pa.int64()),
        pa.field("error_count", pa.int64()),
        pa.field("timeout_count", pa.int64()),
        pa.field("coverage_complete", pa.bool_()),
    ]
)
_SHA256_LENGTH: Final = 64
_FACT_UNITS: Final = {
    "total_cases": "case",
    "passed_count": "case",
    "failed_count": "case",
    "error_count": "case",
    "timeout_count": "case",
    "coverage_complete": "boolean",
    "function_count": "function",
    "best_function_name": "name",
}


class CandidateEvaluationContractError(ValueError):
    """A candidate-evaluation identity or result violates the contract."""


class CandidateResultStatus(StrEnum):
    """Statuses representable by the persisted candidate-result schema."""

    MEASURED = RecordStatus.MEASURED.value
    OPERATOR_FAILURE = RecordStatus.OPERATOR_FAILURE.value
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"


def preprocessing_run_identity(manifest: Mapping[str, object]) -> str:
    """Return the canonical semantic identity for a complete preprocessing run."""

    if set(manifest) != PREPROCESSING_MANIFEST_FIELDS:
        raise CandidateEvaluationContractError(
            "preprocessing manifest fields do not match schema_version 3"
        )
    if manifest.get("schema_version") != PREPROCESSING_MANIFEST_SCHEMA_VERSION:
        raise CandidateEvaluationContractError(
            "preprocessing run requires schema_version 3"
        )
    input_coordinates = manifest.get("input")
    if (
        not isinstance(input_coordinates, dict)
        or set(input_coordinates) != PREPROCESSING_INPUT_FIELDS
    ):
        raise CandidateEvaluationContractError(
            "preprocessing manifest input fields do not match schema_version 3"
        )
    semantic_manifest = {
        field: manifest[field] for field in PREPROCESSING_IDENTITY_FIELDS
    }
    semantic_manifest["input"] = {
        field: value
        for field, value in input_coordinates.items()
        if field != "path"
    }
    return hashlib.sha256(
        _canonical_json(
            {
                "schema": "dr_code.corpus.preprocessing_run",
                "schema_version": manifest["schema_version"],
                "coordinates": semantic_manifest,
            }
        ).encode("utf-8")
    ).hexdigest()


def candidate_evaluation_identity(coordinates: Mapping[str, object]) -> str:
    """Return the canonical identity for schema-5 evaluation coordinates."""

    if set(coordinates) != set(CANDIDATE_EVALUATION_COORDINATE_FIELDS):
        raise CandidateEvaluationContractError(
            "evaluation identity coordinates do not match schema_version 5"
        )
    if (
        coordinates.get("schema_version")
        != CANDIDATE_EVALUATION_SCHEMA_VERSION
    ):
        raise CandidateEvaluationContractError(
            "candidate evaluation requires schema_version 5"
        )
    return hashlib.sha256(
        _canonical_json(
            {
                "schema": "dr_code.corpus.candidate_evaluation",
                "schema_version": CANDIDATE_EVALUATION_SCHEMA_VERSION,
                "coordinates": coordinates,
            }
        ).encode("utf-8")
    ).hexdigest()


def candidate_evaluation_key(
    *,
    task_id: str,
    task_identity: str,
    source_sha256: str,
    question_identity_hash: str,
    operator_name: str,
    operator_version: str,
    metric_extraction_config_identity: str,
    evaluation_procedure_config_identity: str,
    runtime_identity: str,
    runner_identity: str,
) -> str:
    """Return the canonical identity for one schema-5 evaluation."""

    return hashlib.sha256(
        _canonical_json(
            {
                "schema": "dr_code.corpus.candidate_evaluation",
                "schema_version": CANDIDATE_EVALUATION_SCHEMA_VERSION,
                "task_id": _nonblank(task_id, "task_id"),
                "task_identity": _sha256(task_identity, "task_identity"),
                "candidate_source_sha256": _sha256(
                    source_sha256, "source_sha256"
                ),
                "question_identity_hash": _sha256(
                    question_identity_hash, "question_identity_hash"
                ),
                "operator_name": _nonblank(operator_name, "operator_name"),
                "operator_version": _nonblank(
                    operator_version, "operator_version"
                ),
                "metric_extraction_config_identity": _sha256(
                    metric_extraction_config_identity,
                    "metric_extraction_config_identity",
                ),
                "evaluation_procedure_config_identity": _sha256(
                    evaluation_procedure_config_identity,
                    "evaluation_procedure_config_identity",
                ),
                "runtime_identity": _sha256(
                    runtime_identity, "runtime_identity"
                ),
                "runner_identity": _nonblank(
                    runner_identity, "runner_identity"
                ),
            }
        ).encode("utf-8")
    ).hexdigest()


def canonical_candidate_result(
    *,
    task_id: str,
    task_identity: str,
    cleaned_source: str,
    source_sha256: str,
    question_identity_hash: str,
    operator_name: str,
    operator_version: str,
    trace_producer: TraceProducer,
    operator: OperatorCoordinates,
    metric_extraction_config_identity: str,
    evaluation_procedure_config_identity: str,
    runtime_identity: str,
    runner_identity: str,
    metrics_profile: str,
    record_status: object,
    failure_type: object,
    failure_message: object,
    facts: Mapping[str, object],
) -> dict[str, object]:
    """Build the one valid persisted schema-5 result row."""

    if hashlib.sha256(cleaned_source.encode("utf-8")).hexdigest() != (
        source_sha256
    ):
        raise CandidateEvaluationContractError(
            "candidate result source fingerprint mismatch"
        )
    question_identity_hash = _sha256(
        question_identity_hash, "question_identity_hash"
    )
    operator_name = _nonblank(operator_name, "operator_name")
    operator_version = _nonblank(operator_version, "operator_version")
    if operator.name != operator_name or operator.version != operator_version:
        raise CandidateEvaluationContractError(
            "candidate result operator coordinates do not match columns"
        )
    if (
        operator.question_identity_hash(on_key="candidate")
        != question_identity_hash
    ):
        raise CandidateEvaluationContractError(
            "candidate result operator coordinates do not authenticate "
            "question identity"
        )
    try:
        status = CandidateResultStatus(record_status)
    except (TypeError, ValueError) as exc:
        raise CandidateEvaluationContractError(
            f"candidate result has invalid record_status: {record_status!r}"
        ) from exc
    normalized_facts = _validated_facts(facts, status=status)
    normalized_failure = _validated_failure(
        failure_type,
        failure_message,
        status=status,
    )
    outcome: str | None = None
    if status is CandidateResultStatus.MEASURED:
        try:
            record = MetricRecord.measured(
                question=operator_name,
                question_identity_hash=question_identity_hash,
                on_key="candidate",
                evaluation_procedure_config_hash=(
                    evaluation_procedure_config_identity
                ),
                trace_producer=trace_producer,
                operator=operator,
                facts=_metric_facts(
                    normalized_facts,
                    procedure_identity=evaluation_procedure_config_identity,
                    question_identity_hash=question_identity_hash,
                    operator_name=operator_name,
                    operator_version=operator_version,
                    operator_implementation=operator.implementation_hash,
                ),
            )
            outcome = derive_outcome(record).value
        except ValueError as exc:
            raise CandidateEvaluationContractError(
                f"candidate result has invalid measured facts: {exc}"
            ) from exc
    return {
        "evaluation_key": candidate_evaluation_key(
            task_id=task_id,
            task_identity=task_identity,
            source_sha256=source_sha256,
            question_identity_hash=question_identity_hash,
            operator_name=operator_name,
            operator_version=operator_version,
            metric_extraction_config_identity=(
                metric_extraction_config_identity
            ),
            evaluation_procedure_config_identity=(
                evaluation_procedure_config_identity
            ),
            runtime_identity=runtime_identity,
            runner_identity=runner_identity,
        ),
        "task_id": task_id,
        "task_identity": task_identity,
        "cleaned_source": cleaned_source,
        "source_sha256": source_sha256,
        "question_identity_hash": question_identity_hash,
        "operator_name": operator_name,
        "operator_version": operator_version,
        "trace_producer_json": _canonical_json(
            trace_producer.model_dump(mode="json")
        ),
        "operator_coordinates_json": _canonical_json(
            operator.model_dump(mode="json")
        ),
        "metric_extraction_config_identity": (
            metric_extraction_config_identity
        ),
        "evaluation_procedure_config_identity": (
            evaluation_procedure_config_identity
        ),
        "runtime_identity": runtime_identity,
        "runner_identity": runner_identity,
        "metrics_profile": _nonblank(metrics_profile, "metrics_profile"),
        "record_status": status.value,
        "failure_type": normalized_failure[0],
        "failure_message": normalized_failure[1],
        "outcome": outcome,
        **normalized_facts,
    }


def validate_candidate_result(row: Mapping[str, object]) -> None:
    """Require a persisted result to equal its canonical reconstruction."""

    expected = canonical_candidate_result(
        task_id=_text(row, "task_id"),
        task_identity=_text(row, "task_identity"),
        cleaned_source=_text(row, "cleaned_source"),
        source_sha256=_text(row, "source_sha256"),
        question_identity_hash=_text(row, "question_identity_hash"),
        operator_name=_text(row, "operator_name"),
        operator_version=_text(row, "operator_version"),
        trace_producer=TraceProducer.model_validate_json(
            _text(row, "trace_producer_json")
        ),
        operator=OperatorCoordinates.model_validate_json(
            _text(row, "operator_coordinates_json")
        ),
        metric_extraction_config_identity=_text(
            row, "metric_extraction_config_identity"
        ),
        evaluation_procedure_config_identity=_text(
            row, "evaluation_procedure_config_identity"
        ),
        runtime_identity=_text(row, "runtime_identity"),
        runner_identity=_text(row, "runner_identity"),
        metrics_profile=_text(row, "metrics_profile"),
        record_status=row.get("record_status"),
        failure_type=row.get("failure_type"),
        failure_message=row.get("failure_message"),
        facts={name: row.get(name) for name in CANDIDATE_RESULT_FACT_FIELDS},
    )
    for field, expected_value in expected.items():
        if row.get(field) != expected_value:
            raise CandidateEvaluationContractError(
                f"candidate result {field} does not match canonical value"
            )


def _validated_facts(
    facts: Mapping[str, object], *, status: CandidateResultStatus
) -> dict[str, object]:
    if set(facts) != set(CANDIDATE_RESULT_FACT_FIELDS):
        raise CandidateEvaluationContractError(
            "candidate result facts do not match schema_version 5"
        )
    if status is not CandidateResultStatus.MEASURED:
        if any(
            facts[name] is not None for name in CANDIDATE_RESULT_FACT_FIELDS
        ):
            raise CandidateEvaluationContractError(
                "non-measured candidate results require null facts"
            )
        return {name: None for name in CANDIDATE_RESULT_FACT_FIELDS}
    try:
        result = CodeTestResult.model_validate(dict(facts), strict=True)
    except ValidationError as exc:
        raise CandidateEvaluationContractError(
            f"measured candidate result facts are invalid: {exc}"
        ) from exc
    return dict(result.to_values())


def _validated_failure(
    failure_type: object,
    failure_message: object,
    *,
    status: CandidateResultStatus,
) -> tuple[str | None, str | None]:
    if status is CandidateResultStatus.MEASURED:
        if failure_type is not None or failure_message is not None:
            raise CandidateEvaluationContractError(
                "measured candidate results require null failure fields"
            )
        return None, None
    if (
        not isinstance(failure_type, str)
        or not failure_type.strip()
        or not isinstance(failure_message, str)
    ):
        raise CandidateEvaluationContractError(
            "failed candidate results require failure type and message"
        )
    return failure_type, failure_message


def _metric_facts(
    values: Mapping[str, object],
    *,
    procedure_identity: str,
    question_identity_hash: str,
    operator_name: str,
    operator_version: str,
    operator_implementation: str,
) -> tuple[MetricFact, ...]:
    return tuple(
        MetricFact(
            name=name,
            value=_fact_scalar(value, name=name),
            unit=_FACT_UNITS[name],
            applicability=Applicability.APPLICABLE,
            lineage=OperatorLineage(
                evaluation_procedure_config_hash=procedure_identity,
                question_identity_hash=question_identity_hash,
                operator=operator_name,
                operator_version=operator_version,
                operator_implementation=operator_implementation,
            ),
        )
        for name in CANDIDATE_RESULT_FACT_FIELDS
        if (value := values[name]) is not None
    )


def _fact_scalar(value: object, *, name: str) -> FactScalar:
    if isinstance(value, bool | int | float | str):
        return value
    raise CandidateEvaluationContractError(
        f"persisted metric fact {name!r} is not scalar"
    )


def _text(row: Mapping[str, object], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str):
        raise CandidateEvaluationContractError(
            f"candidate result {field} must be text"
        )
    return value


def _nonblank(value: str, label: str) -> str:
    if not value.strip():
        raise CandidateEvaluationContractError(f"{label} must be nonblank")
    return value


def _sha256(value: str, label: str) -> str:
    if len(value) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise CandidateEvaluationContractError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


__all__ = (
    "CANDIDATE_EVALUATION_COORDINATE_FIELDS",
    "CANDIDATE_EVALUATION_MANIFEST_FIELDS",
    "CANDIDATE_EVALUATION_SCHEMA_VERSION",
    "CANDIDATE_RESULT_FACT_FIELDS",
    "MEMBERSHIP_SCHEMA",
    "PREPROCESSING_IDENTITY_FIELDS",
    "PREPROCESSING_INPUT_FIELDS",
    "PREPROCESSING_MANIFEST_FIELDS",
    "RESULTS_SCHEMA",
    "CandidateEvaluationContractError",
    "CandidateResultStatus",
    "candidate_evaluation_identity",
    "candidate_evaluation_key",
    "canonical_candidate_result",
    "preprocessing_run_identity",
    "validate_candidate_result",
)
