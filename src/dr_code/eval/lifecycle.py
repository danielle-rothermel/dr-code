"""The six Definition -> Config pairs of the evaluation kernel.

Pairs: Sampling, Preprocessing, Metric Extraction, Evaluation Procedure,
Aggregation, and Eval (a composite of the first-three-relevant Configs:
Sampling, Evaluation Procedure, Aggregation).

Shared contract, per the Workstream 5 table:

- Each Definition is *versioned* and *variable-bearing*, declares its
  Variables + constraints, and **explicitly materializes** its namesake
  Config (``materialize``). It is that Config's **sole owner**: a Config
  is only ever built through its Definition's ``materialize``.
- Each Config is a **complete validated assignment** carrying a typed
  ``DefinitionRef`` (the owning Definition's identity) plus its own
  **Identity Hash**.
- Metric Extraction and Evaluation Procedure identities include the
  **resolved operator and step versions**.
- The Eval Config's Identity Hash covers all three component Config
  identities.
"""

from __future__ import annotations

from pydantic import JsonValue

from dr_code.eval.identity import (
    SCHEMA_AGGREGATION_CONFIG,
    SCHEMA_AGGREGATION_DEFINITION,
    SCHEMA_EVAL_CONFIG,
    SCHEMA_EVAL_DEFINITION,
    SCHEMA_EVALUATION_PROCEDURE_CONFIG,
    SCHEMA_EVALUATION_PROCEDURE_DEFINITION,
    SCHEMA_METRIC_EXTRACTION_CONFIG,
    SCHEMA_METRIC_EXTRACTION_DEFINITION,
    SCHEMA_PREPROCESSING_CONFIG,
    SCHEMA_PREPROCESSING_DEFINITION,
    SCHEMA_SAMPLING_CONFIG,
    SCHEMA_SAMPLING_DEFINITION,
    identity_hash_for,
)
from dr_code.eval.resolved_versions import (
    resolved_operator_version,
    resolved_step_version,
)
from dr_code.eval.variables import (
    VariableSpec,
    resolve_assignment,
)
from dr_code.models import FrozenModel


class DefinitionRef(FrozenModel):
    """Typed reference from a Config back to its owning Definition."""

    definition_id: str
    version: str
    schema_name: str
    identity_hash: str


def _definition_identity(
    *,
    schema: str,
    definition_id: str,
    version: str,
    variables: tuple[VariableSpec, ...],
    extra: dict[str, JsonValue],
) -> str:
    payload: dict[str, JsonValue] = {
        "definition_id": definition_id,
        "version": version,
        "variables": [
            {
                "name": spec.name,
                "allowed": (
                    None if spec.allowed is None else list(spec.allowed)
                ),
                "has_default": spec.has_default,
                "default": spec.default if spec.has_default else None,
            }
            for spec in variables
        ],
    }
    payload.update(extra)
    return identity_hash_for(schema=schema, payload=payload)


# ===========================================================================
# 1. Sampling
# ===========================================================================


class SamplingDefinition(FrozenModel):
    """Declares Task Set and Repeat Plan Variables; materializes a Config."""

    definition_id: str
    version: str
    variables: tuple[VariableSpec, ...] = (
        VariableSpec(name="task_set_hash"),
        VariableSpec(name="repeat_plan_hash"),
    )

    def identity_hash(self) -> str:
        return _definition_identity(
            schema=SCHEMA_SAMPLING_DEFINITION,
            definition_id=self.definition_id,
            version=self.version,
            variables=self.variables,
            extra={},
        )

    def ref(self) -> DefinitionRef:
        return DefinitionRef(
            definition_id=self.definition_id,
            version=self.version,
            schema_name=SCHEMA_SAMPLING_DEFINITION,
            identity_hash=self.identity_hash(),
        )

    def materialize(
        self,
        assignment: dict[str, JsonValue],
    ) -> SamplingConfig:
        resolved = resolve_assignment(self.variables, assignment)
        return SamplingConfig._create(definition=self, assignment=resolved)


class SamplingConfig(FrozenModel):
    """Complete Sampling assignment: a Task Set + Repeat Plan identity."""

    definition_ref: DefinitionRef
    assignment: tuple[tuple[str, JsonValue], ...]
    config_identity_hash: str

    @classmethod
    def _create(
        cls,
        *,
        definition: SamplingDefinition,
        assignment: dict[str, JsonValue],
    ) -> SamplingConfig:
        config_hash = identity_hash_for(
            schema=SCHEMA_SAMPLING_CONFIG,
            payload={
                "definition_identity": definition.identity_hash(),
                "assignment": _assignment_payload(assignment),
            },
        )
        return cls(
            definition_ref=definition.ref(),
            assignment=tuple(assignment.items()),
            config_identity_hash=config_hash,
        )

    def assignment_dict(self) -> dict[str, JsonValue]:
        return dict(self.assignment)


# ===========================================================================
# 2. Preprocessing (resolved step versions in identity)
# ===========================================================================


class PreprocessingStepBinding(FrozenModel):
    """One ordered preprocessing step instance with resolved version."""

    instance_name: str
    step: str
    settings: tuple[tuple[str, JsonValue], ...] = ()


class PreprocessingDefinition(FrozenModel):
    """Declares ordered preprocessing steps; materializes a Config.

    Step *versions* are resolved from the registry at materialization and
    folded into the Config identity, so a step VERSION bump changes the
    identity even when the step list is unchanged.
    """

    definition_id: str
    version: str
    steps: tuple[PreprocessingStepBinding, ...]
    variables: tuple[VariableSpec, ...] = ()

    def identity_hash(self) -> str:
        return _definition_identity(
            schema=SCHEMA_PREPROCESSING_DEFINITION,
            definition_id=self.definition_id,
            version=self.version,
            variables=self.variables,
            extra={"steps": _step_payload(self.steps)},
        )

    def ref(self) -> DefinitionRef:
        return DefinitionRef(
            definition_id=self.definition_id,
            version=self.version,
            schema_name=SCHEMA_PREPROCESSING_DEFINITION,
            identity_hash=self.identity_hash(),
        )

    def materialize(
        self,
        assignment: dict[str, JsonValue] | None = None,
    ) -> PreprocessingConfig:
        resolved = resolve_assignment(self.variables, assignment or {})
        resolved_steps = tuple(
            (
                binding.instance_name,
                binding.step,
                resolved_step_version(binding.step),
            )
            for binding in self.steps
        )
        return PreprocessingConfig._create(
            definition=self,
            assignment=resolved,
            resolved_steps=resolved_steps,
        )


class PreprocessingConfig(FrozenModel):
    definition_ref: DefinitionRef
    assignment: tuple[tuple[str, JsonValue], ...]
    resolved_step_versions: tuple[tuple[str, str, str], ...]
    config_identity_hash: str

    @classmethod
    def _create(
        cls,
        *,
        definition: PreprocessingDefinition,
        assignment: dict[str, JsonValue],
        resolved_steps: tuple[tuple[str, str, str], ...],
    ) -> PreprocessingConfig:
        config_hash = identity_hash_for(
            schema=SCHEMA_PREPROCESSING_CONFIG,
            payload={
                "definition_identity": definition.identity_hash(),
                "assignment": _assignment_payload(assignment),
                "resolved_step_versions": [
                    list(triple) for triple in resolved_steps
                ],
            },
        )
        return cls(
            definition_ref=definition.ref(),
            assignment=tuple(assignment.items()),
            resolved_step_versions=resolved_steps,
            config_identity_hash=config_hash,
        )


# ===========================================================================
# 3. Metric Extraction (resolved operator versions in identity)
# ===========================================================================


class MetricQuestionBinding(FrozenModel):
    """One Metric Question: a metric family on a key with settings."""

    metric: str
    on: str
    settings: tuple[tuple[str, JsonValue], ...] = ()


class MetricExtractionDefinition(FrozenModel):
    """Declares ordered Metric Questions; materializes a Config.

    Operator *versions* are resolved from the registry at materialization
    and folded into the Config identity.
    """

    definition_id: str
    version: str
    questions: tuple[MetricQuestionBinding, ...]
    variables: tuple[VariableSpec, ...] = ()

    def identity_hash(self) -> str:
        return _definition_identity(
            schema=SCHEMA_METRIC_EXTRACTION_DEFINITION,
            definition_id=self.definition_id,
            version=self.version,
            variables=self.variables,
            extra={"questions": _question_payload(self.questions)},
        )

    def ref(self) -> DefinitionRef:
        return DefinitionRef(
            definition_id=self.definition_id,
            version=self.version,
            schema_name=SCHEMA_METRIC_EXTRACTION_DEFINITION,
            identity_hash=self.identity_hash(),
        )

    def materialize(
        self,
        assignment: dict[str, JsonValue] | None = None,
    ) -> MetricExtractionConfig:
        resolved = resolve_assignment(self.variables, assignment or {})
        resolved_operators = tuple(
            (q.metric, resolved_operator_version(q.metric))
            for q in self.questions
        )
        return MetricExtractionConfig._create(
            definition=self,
            assignment=resolved,
            resolved_operators=resolved_operators,
        )


class MetricExtractionConfig(FrozenModel):
    definition_ref: DefinitionRef
    assignment: tuple[tuple[str, JsonValue], ...]
    resolved_operator_versions: tuple[tuple[str, str], ...]
    config_identity_hash: str

    @classmethod
    def _create(
        cls,
        *,
        definition: MetricExtractionDefinition,
        assignment: dict[str, JsonValue],
        resolved_operators: tuple[tuple[str, str], ...],
    ) -> MetricExtractionConfig:
        config_hash = identity_hash_for(
            schema=SCHEMA_METRIC_EXTRACTION_CONFIG,
            payload={
                "definition_identity": definition.identity_hash(),
                "assignment": _assignment_payload(assignment),
                "resolved_operator_versions": [
                    list(pair) for pair in resolved_operators
                ],
            },
        )
        return cls(
            definition_ref=definition.ref(),
            assignment=tuple(assignment.items()),
            resolved_operator_versions=resolved_operators,
            config_identity_hash=config_hash,
        )


# ===========================================================================
# 4. Evaluation Procedure (composes preprocessing + metric extraction)
# ===========================================================================


class EvaluationProcedureDefinition(FrozenModel):
    """Declares preprocessing, metric-extraction, and failure-semantics
    Variables; materializes an Evaluation Procedure Config.

    The Procedure Config identity folds in the resolved operator/step
    versions (via the component Config identities), so a Procedure change
    alters both ``graph_hash`` (Whetstone) and ``eval_config_hash``.
    """

    definition_id: str
    version: str
    variables: tuple[VariableSpec, ...] = (
        VariableSpec(
            name="zero_denominator", allowed=("not_applicable", "error")
        ),
    )

    def identity_hash(self) -> str:
        return _definition_identity(
            schema=SCHEMA_EVALUATION_PROCEDURE_DEFINITION,
            definition_id=self.definition_id,
            version=self.version,
            variables=self.variables,
            extra={},
        )

    def ref(self) -> DefinitionRef:
        return DefinitionRef(
            definition_id=self.definition_id,
            version=self.version,
            schema_name=SCHEMA_EVALUATION_PROCEDURE_DEFINITION,
            identity_hash=self.identity_hash(),
        )

    def materialize(
        self,
        *,
        preprocessing: PreprocessingConfig,
        metric_extraction: MetricExtractionConfig,
        assignment: dict[str, JsonValue] | None = None,
    ) -> EvaluationProcedureConfig:
        resolved = resolve_assignment(self.variables, assignment or {})
        return EvaluationProcedureConfig._create(
            definition=self,
            preprocessing=preprocessing,
            metric_extraction=metric_extraction,
            assignment=resolved,
        )


class EvaluationProcedureConfig(FrozenModel):
    definition_ref: DefinitionRef
    preprocessing_config_hash: str
    metric_extraction_config_hash: str
    assignment: tuple[tuple[str, JsonValue], ...]
    config_identity_hash: str

    @classmethod
    def _create(
        cls,
        *,
        definition: EvaluationProcedureDefinition,
        preprocessing: PreprocessingConfig,
        metric_extraction: MetricExtractionConfig,
        assignment: dict[str, JsonValue],
    ) -> EvaluationProcedureConfig:
        config_hash = identity_hash_for(
            schema=SCHEMA_EVALUATION_PROCEDURE_CONFIG,
            payload={
                "definition_identity": definition.identity_hash(),
                "preprocessing_config": preprocessing.config_identity_hash,
                "metric_extraction_config": (
                    metric_extraction.config_identity_hash
                ),
                "assignment": _assignment_payload(assignment),
            },
        )
        return cls(
            definition_ref=definition.ref(),
            preprocessing_config_hash=preprocessing.config_identity_hash,
            metric_extraction_config_hash=(
                metric_extraction.config_identity_hash
            ),
            assignment=tuple(assignment.items()),
            config_identity_hash=config_hash,
        )


# ===========================================================================
# 5. Aggregation
# ===========================================================================


class AggregationDefinition(FrozenModel):
    """Declares reduction/weighting/completeness/zero-denominator
    Variables; materializes an Aggregation Config."""

    definition_id: str
    version: str
    variables: tuple[VariableSpec, ...] = (
        VariableSpec(name="reduction", allowed=("mean", "sum")),
        VariableSpec(
            name="missing_data",
            allowed=("propagate", "skip"),
            default="propagate",
            has_default=True,
        ),
        VariableSpec(
            name="zero_denominator",
            allowed=("not_applicable", "error"),
            default="not_applicable",
            has_default=True,
        ),
    )

    def identity_hash(self) -> str:
        return _definition_identity(
            schema=SCHEMA_AGGREGATION_DEFINITION,
            definition_id=self.definition_id,
            version=self.version,
            variables=self.variables,
            extra={},
        )

    def ref(self) -> DefinitionRef:
        return DefinitionRef(
            definition_id=self.definition_id,
            version=self.version,
            schema_name=SCHEMA_AGGREGATION_DEFINITION,
            identity_hash=self.identity_hash(),
        )

    def materialize(
        self,
        assignment: dict[str, JsonValue],
    ) -> AggregationConfig:
        resolved = resolve_assignment(self.variables, assignment)
        return AggregationConfig._create(definition=self, assignment=resolved)


class AggregationConfig(FrozenModel):
    definition_ref: DefinitionRef
    assignment: tuple[tuple[str, JsonValue], ...]
    config_identity_hash: str

    @classmethod
    def _create(
        cls,
        *,
        definition: AggregationDefinition,
        assignment: dict[str, JsonValue],
    ) -> AggregationConfig:
        config_hash = identity_hash_for(
            schema=SCHEMA_AGGREGATION_CONFIG,
            payload={
                "definition_identity": definition.identity_hash(),
                "assignment": _assignment_payload(assignment),
            },
        )
        return cls(
            definition_ref=definition.ref(),
            assignment=tuple(assignment.items()),
            config_identity_hash=config_hash,
        )

    def assignment_dict(self) -> dict[str, JsonValue]:
        return dict(self.assignment)


# ===========================================================================
# 6. Eval (composite of Sampling + Evaluation Procedure + Aggregation)
# ===========================================================================


class EvalDefinition(FrozenModel):
    """Composite Definition declaring Sampling, Evaluation Procedure, and
    Aggregation Config Variables; materializes an Eval Config.

    The Eval Config Identity Hash covers all three component Config
    identities. A Procedure change alters both the Procedure Config
    identity (hence ``graph_hash`` on the Whetstone side) and this
    ``eval_config_hash``; a Sampling-only or Aggregation-only change
    alters only ``eval_config_hash``.
    """

    definition_id: str
    version: str
    variables: tuple[VariableSpec, ...] = (
        VariableSpec(name="sampling_config_hash"),
        VariableSpec(name="evaluation_procedure_config_hash"),
        VariableSpec(name="aggregation_config_hash"),
    )

    def identity_hash(self) -> str:
        return _definition_identity(
            schema=SCHEMA_EVAL_DEFINITION,
            definition_id=self.definition_id,
            version=self.version,
            variables=self.variables,
            extra={},
        )

    def ref(self) -> DefinitionRef:
        return DefinitionRef(
            definition_id=self.definition_id,
            version=self.version,
            schema_name=SCHEMA_EVAL_DEFINITION,
            identity_hash=self.identity_hash(),
        )

    def materialize(
        self,
        *,
        sampling: SamplingConfig,
        evaluation_procedure: EvaluationProcedureConfig,
        aggregation: AggregationConfig,
    ) -> EvalConfig:
        return EvalConfig._create(
            definition=self,
            sampling=sampling,
            evaluation_procedure=evaluation_procedure,
            aggregation=aggregation,
        )


class EvalConfig(FrozenModel):
    definition_ref: DefinitionRef
    sampling_config_hash: str
    evaluation_procedure_config_hash: str
    aggregation_config_hash: str
    config_identity_hash: str

    @classmethod
    def _create(
        cls,
        *,
        definition: EvalDefinition,
        sampling: SamplingConfig,
        evaluation_procedure: EvaluationProcedureConfig,
        aggregation: AggregationConfig,
    ) -> EvalConfig:
        config_hash = identity_hash_for(
            schema=SCHEMA_EVAL_CONFIG,
            payload={
                "definition_identity": definition.identity_hash(),
                "sampling_config": sampling.config_identity_hash,
                "evaluation_procedure_config": (
                    evaluation_procedure.config_identity_hash
                ),
                "aggregation_config": aggregation.config_identity_hash,
            },
        )
        return cls(
            definition_ref=definition.ref(),
            sampling_config_hash=sampling.config_identity_hash,
            evaluation_procedure_config_hash=(
                evaluation_procedure.config_identity_hash
            ),
            aggregation_config_hash=aggregation.config_identity_hash,
            config_identity_hash=config_hash,
        )


# ---------------------------------------------------------------------------
# Payload helpers (ordered, JSON-safe, identity-bearing).
# ---------------------------------------------------------------------------


def _assignment_payload(
    assignment: dict[str, JsonValue],
) -> list[list[JsonValue]]:
    return [[name, value] for name, value in assignment.items()]


def _step_payload(
    steps: tuple[PreprocessingStepBinding, ...],
) -> list[dict[str, JsonValue]]:
    return [
        {
            "instance_name": binding.instance_name,
            "step": binding.step,
            "settings": [list(pair) for pair in binding.settings],
        }
        for binding in steps
    ]


def _question_payload(
    questions: tuple[MetricQuestionBinding, ...],
) -> list[dict[str, JsonValue]]:
    return [
        {
            "metric": q.metric,
            "on": q.on,
            "settings": [list(pair) for pair in q.settings],
        }
        for q in questions
    ]


__all__ = [
    "AggregationConfig",
    "AggregationDefinition",
    "DefinitionRef",
    "EvalConfig",
    "EvalDefinition",
    "EvaluationProcedureConfig",
    "EvaluationProcedureDefinition",
    "MetricExtractionConfig",
    "MetricExtractionDefinition",
    "MetricQuestionBinding",
    "PreprocessingConfig",
    "PreprocessingDefinition",
    "PreprocessingStepBinding",
    "SamplingConfig",
    "SamplingDefinition",
]
