"""Canonical admission checks for persisted preprocessing/evaluation coordinates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import cast

from dr_code.eval.immutable_json import thaw_json
from dr_code.eval.lifecycle import (
    DefinitionRef,
    EvaluationProcedureConfig,
    EvaluationProcedureDefinition,
    MetricExtractionConfig,
    MetricExtractionDefinition,
    PreprocessingConfig,
    PreprocessingDefinition,
)
from dr_code.eval.facts import OperatorCoordinates
from dr_code.eval.resolved_versions import resolved_operator_identity
from dr_code.trace import TraceProducer


class CoordinateValidationError(ValueError):
    """Persisted coordinates cannot be reproduced from canonical models."""


def validate_preprocessing_coordinates(
    manifest: Mapping[str, object],
) -> PreprocessingConfig:
    """Parse and recompute a preprocessing Definition-to-Config contract."""

    try:
        definition_ref = DefinitionRef.model_validate(
            _mapping(
                manifest.get("preprocessing_definition_ref"),
                "preprocessing_definition_ref",
            )
        )
        definition = PreprocessingDefinition.model_validate(
            thaw_json(definition_ref.identity_payload)
        )
        config = PreprocessingConfig.model_validate(
            _mapping(
                manifest.get("preprocessing_config"),
                "preprocessing_config",
            )
        )
    except (TypeError, ValueError) as exc:
        raise CoordinateValidationError(
            "preprocessing definition/config coordinates are invalid"
        ) from exc
    expected_versions = [
        {
            "instance_name": instance_name,
            "step": step,
            "version": version,
        }
        for instance_name, step, version, _implementation_hash in (
            config.resolved_step_versions
        )
    ]
    if (
        config.definition_ref != definition_ref
        or definition.ref() != definition_ref
        or manifest.get("preprocessing_definition_identity")
        != definition_ref.identity_hash
        or manifest.get("preprocessing_config_identity")
        != config.config_identity_hash
        or manifest.get("resolved_step_versions") != expected_versions
    ):
        raise CoordinateValidationError(
            "preprocessing definition/config identities are not canonical"
        )
    return config


def validate_evaluation_coordinates(
    coordinates: Mapping[str, object],
    *,
    preprocessing_config: PreprocessingConfig,
) -> None:
    """Parse and recompute evaluation definition, config, and runtime identities."""

    try:
        metric_definition_ref = DefinitionRef.model_validate(
            _mapping(
                coordinates.get("metric_extraction_definition_ref"),
                "metric_extraction_definition_ref",
            )
        )
        metric_definition = MetricExtractionDefinition.model_validate(
            thaw_json(metric_definition_ref.identity_payload)
        )
        metric_config = MetricExtractionConfig.model_validate(
            _mapping(
                coordinates.get("metric_extraction_config"),
                "metric_extraction_config",
            )
        )
        procedure_definition_ref = DefinitionRef.model_validate(
            _mapping(
                coordinates.get("evaluation_procedure_definition_ref"),
                "evaluation_procedure_definition_ref",
            )
        )
        procedure_definition = EvaluationProcedureDefinition.model_validate(
            thaw_json(procedure_definition_ref.identity_payload)
        )
        procedure_config = EvaluationProcedureConfig.model_validate(
            _mapping(
                coordinates.get("evaluation_procedure_config"),
                "evaluation_procedure_config",
            )
        )
        trace_producer = TraceProducer.model_validate(
            _mapping(
                coordinates.get("trace_producer"),
                "trace_producer",
            )
        )
        operator = OperatorCoordinates.model_validate(
            _mapping(
                coordinates.get("operator_coordinates"),
                "operator_coordinates",
            )
        )
    except (TypeError, ValueError) as exc:
        raise CoordinateValidationError(
            "evaluation definition/config coordinates are invalid"
        ) from exc
    if (
        len(metric_definition.questions) != 1
        or len(metric_config.resolved_operator_versions) != 1
    ):
        raise CoordinateValidationError(
            "evaluation requires exactly one canonical metric question"
        )
    concrete_question = metric_config.questions[0]
    expected_operator_version, expected_operator_implementation_hash = (
        resolved_operator_identity(concrete_question.metric)
    )
    expected_operator = (
        concrete_question.identity_hash(),
        concrete_question.metric,
        expected_operator_version,
    )
    expected_trace_producer = TraceProducer(
        producer_id=preprocessing_config.definition_ref.definition_id,
        version=preprocessing_config.definition_ref.version,
        definition_hash=preprocessing_config.definition_ref.identity_hash,
        preprocessing_config_hash=preprocessing_config.config_identity_hash,
        implementation_hash=preprocessing_config.implementation_hash,
    )
    expected_operator_coordinates = OperatorCoordinates(
        name=concrete_question.metric,
        version=expected_operator_version,
        implementation_hash=expected_operator_implementation_hash,
        settings=tuple(concrete_question.settings_dict().items()),
    )
    runtime_identity = hashlib.sha256(
        _canonical_json(
            {
                "runner_identity": coordinates.get("runner_identity"),
                "host_runtime": coordinates.get("host_runtime"),
                "installed_environment": coordinates.get(
                    "installed_environment"
                ),
                "trusted_source_sha256": coordinates.get(
                    "trusted_source_sha256"
                ),
            }
        ).encode("utf-8")
    ).hexdigest()
    if (
        metric_config.definition_ref != metric_definition_ref
        or metric_definition.ref() != metric_definition_ref
        or procedure_config.definition_ref != procedure_definition_ref
        or procedure_definition.ref() != procedure_definition_ref
        or procedure_config.preprocessing_config_hash
        != preprocessing_config.config_identity_hash
        or procedure_config.metric_extraction_config_hash
        != metric_config.config_identity_hash
        or trace_producer != expected_trace_producer
        or operator != expected_operator_coordinates
        or coordinates.get("metric_extraction_definition_identity")
        != metric_definition_ref.identity_hash
        or coordinates.get("metric_extraction_config_identity")
        != metric_config.config_identity_hash
        or coordinates.get("evaluation_procedure_definition_identity")
        != procedure_definition_ref.identity_hash
        or coordinates.get("evaluation_procedure_config_identity")
        != procedure_config.config_identity_hash
        or (
            coordinates.get("question_identity_hash"),
            coordinates.get("operator_name"),
            coordinates.get("operator_version"),
        )
        != expected_operator
        or coordinates.get("runtime_identity") != runtime_identity
    ):
        raise CoordinateValidationError(
            "evaluation definition/config/runtime identities are not canonical"
        )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CoordinateValidationError(f"{label} must be an object")
    return cast(Mapping[str, object], value)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


__all__ = (
    "CoordinateValidationError",
    "validate_evaluation_coordinates",
    "validate_preprocessing_coordinates",
)
