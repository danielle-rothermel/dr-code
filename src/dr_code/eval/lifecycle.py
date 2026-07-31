"""The six self-authenticating evaluation Definition-to-Config pairs."""

from __future__ import annotations

import re
import json
from collections.abc import Mapping
from enum import StrEnum
from typing import Self, cast

from dr_serialize import validate_strict_json
from pydantic import (
    JsonValue,
    field_serializer,
    field_validator,
    model_validator,
)

from dr_code.eval.immutable_json import FrozenJsonDict, freeze_json, thaw_json
from dr_code.eval.identity import (
    SCHEMA_AGGREGATION_CONFIG,
    SCHEMA_AGGREGATION_DEFINITION,
    SCHEMA_EVALUATION_PROCEDURE_CONFIG,
    SCHEMA_EVALUATION_PROCEDURE_DEFINITION,
    SCHEMA_EVAL_CONFIG,
    SCHEMA_EVAL_DEFINITION,
    SCHEMA_METRIC_EXTRACTION_CONFIG,
    SCHEMA_METRIC_EXTRACTION_DEFINITION,
    SCHEMA_METRIC_QUESTION_BINDING,
    SCHEMA_PREPROCESSING_CONFIG,
    SCHEMA_PREPROCESSING_DEFINITION,
    SCHEMA_SAMPLING_CONFIG,
    SCHEMA_SAMPLING_DEFINITION,
    identity_hash_for,
)
from dr_code.eval.resolved_versions import (
    resolved_operator_identity,
    resolved_step_identity,
)
from dr_code.eval.tasks import RepeatPlan, TaskSet
from dr_code.eval.variables import (
    VariableError,
    VariableReference,
    VariableSpec,
    resolve_assignment,
    substitute_variables,
    variable_references,
)
from dr_code.models import FrozenModel
from dr_code.trace.provenance import ExternalSource

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class DefinitionRef(FrozenModel):
    """Self-authenticating serialized reference to an owning Definition."""

    definition_id: str
    version: str
    schema_name: str
    identity_payload: dict[str, JsonValue]
    identity_hash: str

    @field_validator("identity_payload", mode="after")
    @classmethod
    def freeze_identity_payload(
        cls, value: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        return cast(dict[str, JsonValue], freeze_json(value))

    @field_serializer("identity_payload")
    def serialize_identity_payload(
        self, value: dict[str, JsonValue]
    ) -> JsonValue:
        return thaw_json(value)

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        _validate_sha256(self.identity_hash, owner="definition identity")
        if self.identity_payload.get("definition_id") != self.definition_id:
            raise ValueError(
                "definition reference id does not match its payload"
            )
        if self.identity_payload.get("version") != self.version:
            raise ValueError(
                "definition reference version does not match its payload"
            )
        expected = identity_hash_for(
            schema=self.schema_name,
            payload=cast(
                dict[str, JsonValue], thaw_json(self.identity_payload)
            ),
        )
        if self.identity_hash != expected:
            raise ValueError("definition reference identity hash mismatch")
        return self


def _validate_sha256(value: object, *, owner: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{owner} must be a lowercase 64-character SHA-256")
    return value


def _definition_payload(
    *,
    definition_id: str,
    version: str,
    variables: tuple[VariableSpec, ...],
    extra: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {
        "definition_id": definition_id,
        "version": version,
        "variables": [
            {
                "name": spec.name,
                "allowed": (
                    None
                    if spec.allowed is None
                    else [thaw_json(value) for value in spec.allowed]
                ),
                "has_default": spec.has_default,
                "default": (
                    thaw_json(spec.default) if spec.has_default else None
                ),
            }
            for spec in variables
        ],
    }
    payload.update(extra)
    return payload


def _assignment_payload(
    assignment: dict[str, JsonValue] | tuple[tuple[str, JsonValue], ...],
) -> list[list[JsonValue]]:
    items = assignment.items() if isinstance(assignment, dict) else assignment
    return [[name, thaw_json(value)] for name, value in items]


def _canonical_json_value(value: object) -> object:
    validated = cast(JsonValue, validate_strict_json(value))
    return _canonicalize_validated_json(validated)


def _canonicalize_validated_json(value: JsonValue) -> object:
    return freeze_json(value)


def _normalize_template_value(value: object) -> object:
    if isinstance(value, VariableReference):
        return value
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        variable_name = mapping.get("variable")
        if set(mapping) == {"variable"} and isinstance(variable_name, str):
            return VariableReference(variable=variable_name)
        return FrozenJsonDict(
            {
                str(name): _normalize_template_value(child)
                for name, child in sorted(mapping.items())
            }
        )
    if isinstance(value, list):
        return tuple(_normalize_template_value(child) for child in value)
    if isinstance(value, tuple):
        return tuple(_normalize_template_value(child) for child in value)
    return cast(JsonValue, _canonical_json_value(value))


def _normalize_settings(value: object, *, owner: str) -> object:
    if isinstance(value, Mapping):
        canonical = _normalize_template_value(value)
        if not isinstance(canonical, Mapping):
            raise TypeError(f"{owner} settings must be a JSON object")
        return tuple(canonical.items())
    if isinstance(value, (list, tuple)):
        normalized: list[tuple[str, object]] = []
        for pair in value:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                return value
            name, setting_value = pair
            if not isinstance(name, str):
                return value
            normalized.append((name, _normalize_template_value(setting_value)))
        return tuple(sorted(normalized, key=lambda pair: pair[0]))
    return value


def _validate_definition_coordinates(
    *,
    definition_id: str,
    version: str,
    variables: tuple[VariableSpec, ...],
) -> None:
    if not definition_id or not version:
        raise ValueError("definition id and version must be non-empty")
    names = [variable.name for variable in variables]
    if len(names) != len(set(names)):
        raise VariableError("variable names must be unique")


def _validated_assignment(
    variables: tuple[VariableSpec, ...],
    assignment: tuple[tuple[str, JsonValue], ...],
) -> dict[str, JsonValue]:
    names = [name for name, _value in assignment]
    if len(names) != len(set(names)):
        raise VariableError("config assignment names must be unique")
    resolved = resolve_assignment(variables, dict(assignment))
    if tuple(resolved) != tuple(names):
        raise VariableError(
            "config assignment must be complete and in definition order"
        )
    return resolved


def _freeze_assignment(
    assignment: tuple[tuple[str, JsonValue], ...],
) -> tuple[tuple[str, JsonValue], ...]:
    return tuple(
        (name, cast(JsonValue, freeze_json(value)))
        for name, value in assignment
    )


def _thaw_assignment(
    assignment: object,
) -> object:
    if not isinstance(assignment, (list, tuple)):
        return assignment
    return tuple(
        (pair[0], thaw_json(pair[1]))
        if isinstance(pair, (list, tuple)) and len(pair) == 2
        else pair
        for pair in assignment
    )


def _validate_config_hash(
    *,
    actual: str,
    schema: str,
    payload: dict[str, JsonValue],
) -> None:
    _validate_sha256(actual, owner="config identity")
    if actual != identity_hash_for(schema=schema, payload=payload):
        raise ValueError("config identity hash mismatch")


def _validate_ref_schema(ref: DefinitionRef, expected: str) -> None:
    if ref.schema_name != expected:
        raise ValueError(
            f"definition reference schema must be {expected!r}, "
            f"got {ref.schema_name!r}"
        )


def _validate_canonical_definition_ref(
    ref: DefinitionRef,
    definition: (
        SamplingDefinition
        | PreprocessingDefinition
        | MetricExtractionDefinition
        | EvaluationProcedureDefinition
        | AggregationDefinition
        | EvalDefinition
    ),
) -> None:
    if definition.ref() != ref:
        raise ValueError(
            "definition reference payload is not the canonical owning "
            "definition payload"
        )


def _reference_usage(
    values: object,
    variables: tuple[VariableSpec, ...],
) -> None:
    declared = {variable.name for variable in variables}
    referenced = set(variable_references(values))
    unknown = referenced - declared
    if unknown:
        raise VariableError(
            "undefined variable references: " + ", ".join(sorted(unknown))
        )
    unused = declared - referenced
    if unused:
        raise VariableError(
            "unused variable declarations: " + ", ".join(sorted(unused))
        )


def _concrete_settings(
    settings: tuple[tuple[str, object], ...],
    assignment: dict[str, JsonValue],
) -> tuple[tuple[str, object], ...]:
    return tuple(
        (name, substitute_variables(value, assignment))
        for name, value in settings
    )


def _template_json(value: object) -> JsonValue:
    if isinstance(value, VariableReference):
        return {"variable": value.variable}
    if isinstance(value, Mapping):
        return {
            str(name): _template_json(child) for name, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_template_json(child) for child in value]
    return cast(JsonValue, _canonical_json_value(value))


def _settings_payload(
    settings: tuple[tuple[str, object], ...],
) -> list[list[JsonValue]]:
    return [[name, _template_json(value)] for name, value in settings]


def _canonical_preprocessing_step(
    binding: PreprocessingStepBinding,
) -> PreprocessingStepBinding:
    from dr_code.preprocessing.registry import REGISTRY

    step_class = REGISTRY.get(binding.step)
    if step_class is None:
        raise ValueError(f"unknown preprocessing step {binding.step!r}")
    validated = step_class.Settings.model_validate_json(
        json.dumps(thaw_json(dict(binding.settings)), allow_nan=False),
        strict=True,
    )
    return PreprocessingStepBinding(
        instance_name=binding.instance_name,
        step=binding.step,
        settings=validated.model_dump(mode="json"),
    )


def _validate_concrete_preprocessing_settings(
    bindings: tuple[PreprocessingStepBinding, ...],
) -> None:
    for binding in bindings:
        if variable_references(binding.settings):
            raise VariableError(
                "preprocessing config contains unresolved variable references"
            )
        if binding != _canonical_preprocessing_step(binding):
            raise ValueError(
                f"preprocessing settings for {binding.instance_name!r} "
                "are not fully validated and canonical"
            )


def _canonical_metric_question(
    question: MetricQuestionBinding,
) -> MetricQuestionBinding:
    from dr_code.metrics.validation import validated_metric_operator

    operator = validated_metric_operator(
        name=question.metric,
        settings=cast(
            dict[str, object],
            thaw_json(dict(question.settings)),
        ),
    )
    return MetricQuestionBinding(
        metric=question.metric,
        on=question.on,
        settings=operator.settings.model_dump(mode="json"),
    )


def _validate_concrete_metric_settings(
    questions: tuple[MetricQuestionBinding, ...],
) -> None:
    identities = [question.identity_hash() for question in questions]
    if len(identities) != len(set(identities)):
        raise ValueError(
            "substituted metric questions must have unique "
            "(metric, on, settings) triples"
        )
    for question in questions:
        if variable_references(question.settings):
            raise VariableError(
                "metric config contains unresolved variable references"
            )
        if question != _canonical_metric_question(question):
            raise ValueError(
                f"metric settings for {question.metric!r} "
                "are not fully validated and canonical"
            )


class SamplingDefinition(FrozenModel):
    """Declare Task Set and Repeat Plan variables."""

    definition_id: str
    version: str
    variables: tuple[VariableSpec, ...] = (
        VariableSpec(name="task_set_hash"),
        VariableSpec(name="repeat_plan_hash"),
    )

    @model_validator(mode="after")
    def validate_definition(self) -> Self:
        _validate_definition_coordinates(
            definition_id=self.definition_id,
            version=self.version,
            variables=self.variables,
        )
        if tuple(variable.name for variable in self.variables) != (
            "task_set_hash",
            "repeat_plan_hash",
        ):
            raise VariableError(
                "sampling variables must be task_set_hash and repeat_plan_hash"
            )
        return self

    def identity_payload(self) -> dict[str, JsonValue]:
        return _definition_payload(
            definition_id=self.definition_id,
            version=self.version,
            variables=self.variables,
            extra={},
        )

    def identity_hash(self) -> str:
        return identity_hash_for(
            schema=SCHEMA_SAMPLING_DEFINITION,
            payload=self.identity_payload(),
        )

    def ref(self) -> DefinitionRef:
        return DefinitionRef(
            definition_id=self.definition_id,
            version=self.version,
            schema_name=SCHEMA_SAMPLING_DEFINITION,
            identity_payload=self.identity_payload(),
            identity_hash=self.identity_hash(),
        )

    def materialize(
        self,
        *,
        task_set: TaskSet,
        repeat_plan: RepeatPlan,
    ) -> SamplingConfig:
        if repeat_plan.task_identities != task_set.task_identities:
            raise ValueError(
                "repeat plan task identities must match the TaskSet manifest"
            )
        resolved = resolve_assignment(
            self.variables,
            {
                "task_set_hash": task_set.identity_hash(),
                "repeat_plan_hash": repeat_plan.identity_hash(),
            },
        )
        return SamplingConfig._create(
            definition=self,
            assignment=resolved,
            task_set=task_set,
            repeat_plan=repeat_plan,
        )


class SamplingConfig(FrozenModel):
    definition_ref: DefinitionRef
    assignment: tuple[tuple[str, JsonValue], ...]
    task_set: TaskSet
    repeat_plan: RepeatPlan
    config_identity_hash: str

    @field_validator("assignment", mode="before")
    @classmethod
    def thaw_assignment(cls, value: object) -> object:
        return _thaw_assignment(value)

    @field_validator("assignment", mode="after")
    @classmethod
    def freeze_assignment(
        cls, value: tuple[tuple[str, JsonValue], ...]
    ) -> tuple[tuple[str, JsonValue], ...]:
        return _freeze_assignment(value)

    @classmethod
    def _create(
        cls,
        *,
        definition: SamplingDefinition,
        assignment: dict[str, JsonValue],
        task_set: TaskSet,
        repeat_plan: RepeatPlan,
    ) -> SamplingConfig:
        config_hash = identity_hash_for(
            schema=SCHEMA_SAMPLING_CONFIG,
            payload={
                "definition_identity": definition.identity_hash(),
                "assignment": _assignment_payload(assignment),
                "task_set": task_set.model_dump(mode="json"),
                "repeat_plan": repeat_plan.model_dump(mode="json"),
            },
        )
        return cls(
            definition_ref=definition.ref(),
            assignment=tuple(assignment.items()),
            task_set=task_set,
            repeat_plan=repeat_plan,
            config_identity_hash=config_hash,
        )

    def assignment_dict(self) -> dict[str, JsonValue]:
        return dict(self.assignment)

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        _validate_ref_schema(self.definition_ref, SCHEMA_SAMPLING_DEFINITION)
        definition = SamplingDefinition.model_validate(
            thaw_json(self.definition_ref.identity_payload)
        )
        _validate_canonical_definition_ref(self.definition_ref, definition)
        assignment = _validated_assignment(
            definition.variables, self.assignment
        )
        if assignment != {
            "task_set_hash": self.task_set.identity_hash(),
            "repeat_plan_hash": self.repeat_plan.identity_hash(),
        }:
            raise ValueError(
                "sampling assignment does not authenticate its TaskSet "
                "and RepeatPlan"
            )
        if self.repeat_plan.task_identities != self.task_set.task_identities:
            raise ValueError(
                "repeat plan task identities must match the TaskSet manifest"
            )
        _validate_config_hash(
            actual=self.config_identity_hash,
            schema=SCHEMA_SAMPLING_CONFIG,
            payload={
                "definition_identity": self.definition_ref.identity_hash,
                "assignment": _assignment_payload(self.assignment),
                "task_set": self.task_set.model_dump(mode="json"),
                "repeat_plan": self.repeat_plan.model_dump(mode="json"),
            },
        )
        return self


class PreprocessingStepBinding(FrozenModel):
    """One ordered preprocessing step instance."""

    instance_name: str
    step: str
    settings: tuple[tuple[str, object], ...] = ()

    @field_validator("settings", mode="before")
    @classmethod
    def normalize_settings(cls, value: object) -> object:
        return _normalize_settings(value, owner="preprocessing")

    @model_validator(mode="after")
    def reject_duplicate_settings(self) -> Self:
        names = [name for name, _value in self.settings]
        if len(names) != len(set(names)):
            raise ValueError("preprocessing setting names must be unique")
        return self


class PreprocessingDefinition(FrozenModel):
    """Declare preprocessing steps and variables."""

    definition_id: str
    version: str
    steps: tuple[PreprocessingStepBinding, ...]
    variables: tuple[VariableSpec, ...] = ()

    def __hash__(self) -> int:
        return int(self.identity_hash(), 16)

    @model_validator(mode="after")
    def validate_definition(self) -> Self:
        from dr_code.preprocessing.registry import REGISTRY
        from dr_code.trace import RESERVED_KEYS

        _validate_definition_coordinates(
            definition_id=self.definition_id,
            version=self.version,
            variables=self.variables,
        )
        names = [binding.instance_name for binding in self.steps]
        if len(names) != len(set(names)):
            raise ValueError("preprocessing instance names must be unique")
        reserved = set(names) & RESERVED_KEYS
        if reserved:
            raise ValueError(
                "preprocessing instance names must not be reserved: "
                + ", ".join(sorted(reserved))
            )
        expected_input = None
        for binding in self.steps:
            step_class = REGISTRY.get(binding.step)
            if step_class is None:
                raise ValueError(
                    f"unknown preprocessing step {binding.step!r}"
                )
            if not variable_references(binding.settings):
                _canonical_preprocessing_step(binding)
            if (
                expected_input is not None
                and step_class.INPUT != expected_input
            ):
                raise ValueError(
                    "broken preprocessing INPUT/OUTPUT kind chain at "
                    f"{binding.instance_name!r}"
                )
            expected_input = step_class.OUTPUT
        _reference_usage(
            tuple(binding.settings for binding in self.steps),
            self.variables,
        )
        return self

    def identity_payload(self) -> dict[str, JsonValue]:
        identity_steps = tuple(
            (
                binding
                if variable_references(binding.settings)
                else _canonical_preprocessing_step(binding)
            )
            for binding in self.steps
        )
        return _definition_payload(
            definition_id=self.definition_id,
            version=self.version,
            variables=self.variables,
            extra={
                "steps": [
                    {
                        "instance_name": binding.instance_name,
                        "step": binding.step,
                        "settings": _settings_payload(binding.settings),
                    }
                    for binding in identity_steps
                ]
            },
        )

    def identity_hash(self) -> str:
        return identity_hash_for(
            schema=SCHEMA_PREPROCESSING_DEFINITION,
            payload=self.identity_payload(),
        )

    def ref(self) -> DefinitionRef:
        return DefinitionRef(
            definition_id=self.definition_id,
            version=self.version,
            schema_name=SCHEMA_PREPROCESSING_DEFINITION,
            identity_payload=self.identity_payload(),
            identity_hash=self.identity_hash(),
        )

    def materialize(
        self, assignment: dict[str, JsonValue] | None = None
    ) -> PreprocessingConfig:
        resolved = resolve_assignment(self.variables, assignment or {})
        concrete_steps = tuple(
            _canonical_preprocessing_step(
                PreprocessingStepBinding(
                    instance_name=binding.instance_name,
                    step=binding.step,
                    settings=_concrete_settings(binding.settings, resolved),
                )
            )
            for binding in self.steps
        )
        _validate_concrete_preprocessing_settings(concrete_steps)
        resolved_steps = tuple(
            (
                binding.instance_name,
                binding.step,
                *resolved_step_identity(binding.step),
            )
            for binding in concrete_steps
        )
        return PreprocessingConfig._create(
            definition=self,
            assignment=resolved,
            steps=concrete_steps,
            resolved_steps=resolved_steps,
        )


class PreprocessingConfig(FrozenModel):
    definition_ref: DefinitionRef
    assignment: tuple[tuple[str, JsonValue], ...]
    steps: tuple[PreprocessingStepBinding, ...]
    resolved_step_versions: tuple[tuple[str, str, str, str], ...]
    implementation_hash: str
    config_identity_hash: str

    @field_validator("assignment", mode="before")
    @classmethod
    def thaw_assignment(cls, value: object) -> object:
        return _thaw_assignment(value)

    @field_validator("assignment", mode="after")
    @classmethod
    def freeze_assignment(
        cls, value: tuple[tuple[str, JsonValue], ...]
    ) -> tuple[tuple[str, JsonValue], ...]:
        return _freeze_assignment(value)

    @classmethod
    def _create(
        cls,
        *,
        definition: PreprocessingDefinition,
        assignment: dict[str, JsonValue],
        steps: tuple[PreprocessingStepBinding, ...],
        resolved_steps: tuple[tuple[str, str, str, str], ...],
    ) -> PreprocessingConfig:
        implementation_hash = identity_hash_for(
            schema="dr_code.preprocessing.implementation_set",
            payload=[list(item) for item in resolved_steps],
        )
        config_hash = identity_hash_for(
            schema=SCHEMA_PREPROCESSING_CONFIG,
            payload={
                "definition_identity": definition.identity_hash(),
                "assignment": _assignment_payload(assignment),
                "steps": [
                    {
                        "instance_name": binding.instance_name,
                        "step": binding.step,
                        "settings": _settings_payload(binding.settings),
                    }
                    for binding in steps
                ],
                "resolved_step_versions": [
                    list(item) for item in resolved_steps
                ],
                "implementation_hash": implementation_hash,
            },
        )
        return cls(
            definition_ref=definition.ref(),
            assignment=tuple(assignment.items()),
            steps=steps,
            resolved_step_versions=resolved_steps,
            implementation_hash=implementation_hash,
            config_identity_hash=config_hash,
        )

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        _validate_ref_schema(
            self.definition_ref, SCHEMA_PREPROCESSING_DEFINITION
        )
        definition = PreprocessingDefinition.model_validate(
            thaw_json(self.definition_ref.identity_payload)
        )
        _validate_canonical_definition_ref(self.definition_ref, definition)
        assignment = _validated_assignment(
            definition.variables, self.assignment
        )
        expected_steps = tuple(
            _canonical_preprocessing_step(
                PreprocessingStepBinding(
                    instance_name=binding.instance_name,
                    step=binding.step,
                    settings=_concrete_settings(binding.settings, assignment),
                )
            )
            for binding in definition.steps
        )
        if self.steps != expected_steps:
            raise ValueError(
                "preprocessing config steps do not match substituted definition"
            )
        _validate_concrete_preprocessing_settings(self.steps)
        expected_resolutions = tuple(
            (
                binding.instance_name,
                binding.step,
                *resolved_step_identity(binding.step),
            )
            for binding in self.steps
        )
        if self.resolved_step_versions != expected_resolutions:
            raise ValueError(
                "preprocessing config has stale resolved step versions"
            )
        expected_implementation_hash = identity_hash_for(
            schema="dr_code.preprocessing.implementation_set",
            payload=[list(item) for item in self.resolved_step_versions],
        )
        if self.implementation_hash != expected_implementation_hash:
            raise ValueError("preprocessing implementation hash mismatch")
        _validate_config_hash(
            actual=self.config_identity_hash,
            schema=SCHEMA_PREPROCESSING_CONFIG,
            payload={
                "definition_identity": self.definition_ref.identity_hash,
                "assignment": _assignment_payload(self.assignment),
                "steps": [
                    {
                        "instance_name": binding.instance_name,
                        "step": binding.step,
                        "settings": _settings_payload(binding.settings),
                    }
                    for binding in self.steps
                ],
                "resolved_step_versions": [
                    list(item) for item in self.resolved_step_versions
                ],
                "implementation_hash": self.implementation_hash,
            },
        )
        return self


class MetricQuestionBinding(FrozenModel):
    """One metric family applied to one trace key."""

    metric: str
    on: str
    settings: tuple[tuple[str, object], ...] = ()

    @field_validator("settings", mode="before")
    @classmethod
    def normalize_settings(cls, value: object) -> object:
        return _normalize_settings(value, owner="metric")

    @model_validator(mode="after")
    def reject_duplicate_settings(self) -> Self:
        names = [name for name, _value in self.settings]
        if len(names) != len(set(names)):
            raise ValueError("metric setting names must be unique")
        return self

    def settings_dict(self) -> dict[str, JsonValue]:
        if variable_references(self.settings):
            raise VariableError("metric settings contain variable references")
        return cast(dict[str, JsonValue], thaw_json(dict(self.settings)))

    def identity_hash(self) -> str:
        question = (
            self
            if variable_references(self.settings)
            else _canonical_metric_question(self)
        )
        return identity_hash_for(
            schema=SCHEMA_METRIC_QUESTION_BINDING,
            payload={
                "metric": question.metric,
                "on": question.on,
                "settings": _settings_payload(question.settings),
            },
        )


class MetricExtractionDefinition(FrozenModel):
    """Declare an ordered set of metric questions."""

    definition_id: str
    version: str
    questions: tuple[MetricQuestionBinding, ...]
    variables: tuple[VariableSpec, ...] = ()

    @model_validator(mode="after")
    def validate_definition(self) -> Self:
        from dr_code.metrics.registry import REGISTRY

        _validate_definition_coordinates(
            definition_id=self.definition_id,
            version=self.version,
            variables=self.variables,
        )
        for question in self.questions:
            if question.metric not in REGISTRY:
                raise ValueError(
                    f"unknown metric operator {question.metric!r}"
                )
        identities = [question.identity_hash() for question in self.questions]
        if len(identities) != len(set(identities)):
            raise ValueError(
                "metric questions must have unique "
                "(metric, on, settings) triples"
            )
        _reference_usage(
            tuple(question.settings for question in self.questions),
            self.variables,
        )
        return self

    def identity_payload(self) -> dict[str, JsonValue]:
        identity_questions = tuple(
            (
                question
                if variable_references(question.settings)
                else _canonical_metric_question(question)
            )
            for question in self.questions
        )
        return _definition_payload(
            definition_id=self.definition_id,
            version=self.version,
            variables=self.variables,
            extra={
                "questions": [
                    {
                        "metric": question.metric,
                        "on": question.on,
                        "settings": _settings_payload(question.settings),
                    }
                    for question in identity_questions
                ]
            },
        )

    def identity_hash(self) -> str:
        return identity_hash_for(
            schema=SCHEMA_METRIC_EXTRACTION_DEFINITION,
            payload=self.identity_payload(),
        )

    def ref(self) -> DefinitionRef:
        return DefinitionRef(
            definition_id=self.definition_id,
            version=self.version,
            schema_name=SCHEMA_METRIC_EXTRACTION_DEFINITION,
            identity_payload=self.identity_payload(),
            identity_hash=self.identity_hash(),
        )

    def materialize(
        self, assignment: dict[str, JsonValue] | None = None
    ) -> MetricExtractionConfig:
        resolved = resolve_assignment(self.variables, assignment or {})
        concrete_questions = tuple(
            _canonical_metric_question(
                MetricQuestionBinding(
                    metric=question.metric,
                    on=question.on,
                    settings=_concrete_settings(question.settings, resolved),
                )
            )
            for question in self.questions
        )
        _validate_concrete_metric_settings(concrete_questions)
        resolved_operators = tuple(
            (
                question.identity_hash(),
                question.metric,
                *resolved_operator_identity(question.metric),
            )
            for question in concrete_questions
        )
        return MetricExtractionConfig._create(
            definition=self,
            assignment=resolved,
            questions=concrete_questions,
            resolved_operators=resolved_operators,
        )


class MetricExtractionConfig(FrozenModel):
    definition_ref: DefinitionRef
    assignment: tuple[tuple[str, JsonValue], ...]
    questions: tuple[MetricQuestionBinding, ...]
    resolved_operator_versions: tuple[tuple[str, str, str, str], ...]
    config_identity_hash: str

    @field_validator("assignment", mode="before")
    @classmethod
    def thaw_assignment(cls, value: object) -> object:
        return _thaw_assignment(value)

    @field_validator("assignment", mode="after")
    @classmethod
    def freeze_assignment(
        cls, value: tuple[tuple[str, JsonValue], ...]
    ) -> tuple[tuple[str, JsonValue], ...]:
        return _freeze_assignment(value)

    @classmethod
    def _create(
        cls,
        *,
        definition: MetricExtractionDefinition,
        assignment: dict[str, JsonValue],
        questions: tuple[MetricQuestionBinding, ...],
        resolved_operators: tuple[tuple[str, str, str, str], ...],
    ) -> MetricExtractionConfig:
        config_hash = identity_hash_for(
            schema=SCHEMA_METRIC_EXTRACTION_CONFIG,
            payload={
                "definition_identity": definition.identity_hash(),
                "assignment": _assignment_payload(assignment),
                "questions": [
                    {
                        "metric": question.metric,
                        "on": question.on,
                        "settings": _settings_payload(question.settings),
                    }
                    for question in questions
                ],
                "resolved_operator_versions": [
                    list(item) for item in resolved_operators
                ],
            },
        )
        return cls(
            definition_ref=definition.ref(),
            assignment=tuple(assignment.items()),
            questions=questions,
            resolved_operator_versions=resolved_operators,
            config_identity_hash=config_hash,
        )

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        _validate_ref_schema(
            self.definition_ref, SCHEMA_METRIC_EXTRACTION_DEFINITION
        )
        definition = MetricExtractionDefinition.model_validate(
            thaw_json(self.definition_ref.identity_payload)
        )
        _validate_canonical_definition_ref(self.definition_ref, definition)
        assignment = _validated_assignment(
            definition.variables, self.assignment
        )
        expected_questions = tuple(
            _canonical_metric_question(
                MetricQuestionBinding(
                    metric=question.metric,
                    on=question.on,
                    settings=_concrete_settings(question.settings, assignment),
                )
            )
            for question in definition.questions
        )
        if self.questions != expected_questions:
            raise ValueError(
                "metric config questions do not match substituted definition"
            )
        _validate_concrete_metric_settings(self.questions)
        expected_resolutions = tuple(
            (
                question.identity_hash(),
                question.metric,
                *resolved_operator_identity(question.metric),
            )
            for question in self.questions
        )
        if self.resolved_operator_versions != expected_resolutions:
            raise ValueError(
                "metric config has stale resolved operator versions"
            )
        _validate_config_hash(
            actual=self.config_identity_hash,
            schema=SCHEMA_METRIC_EXTRACTION_CONFIG,
            payload={
                "definition_identity": self.definition_ref.identity_hash,
                "assignment": _assignment_payload(self.assignment),
                "questions": [
                    {
                        "metric": question.metric,
                        "on": question.on,
                        "settings": _settings_payload(question.settings),
                    }
                    for question in self.questions
                ],
                "resolved_operator_versions": [
                    list(item) for item in self.resolved_operator_versions
                ],
            },
        )
        return self


class EvaluationProcedureDefinition(FrozenModel):
    """Compose preprocessing and metric extraction."""

    definition_id: str
    version: str
    variables: tuple[VariableSpec, ...] = ()

    @model_validator(mode="after")
    def validate_definition(self) -> Self:
        _validate_definition_coordinates(
            definition_id=self.definition_id,
            version=self.version,
            variables=self.variables,
        )
        if self.variables:
            raise VariableError(
                "evaluation procedure does not define configurable variables"
            )
        return self

    def identity_payload(self) -> dict[str, JsonValue]:
        return _definition_payload(
            definition_id=self.definition_id,
            version=self.version,
            variables=self.variables,
            extra={},
        )

    def identity_hash(self) -> str:
        return identity_hash_for(
            schema=SCHEMA_EVALUATION_PROCEDURE_DEFINITION,
            payload=self.identity_payload(),
        )

    def ref(self) -> DefinitionRef:
        return DefinitionRef(
            definition_id=self.definition_id,
            version=self.version,
            schema_name=SCHEMA_EVALUATION_PROCEDURE_DEFINITION,
            identity_payload=self.identity_payload(),
            identity_hash=self.identity_hash(),
        )

    def materialize(
        self,
        *,
        preprocessing: PreprocessingConfig,
        metric_extraction: MetricExtractionConfig,
    ) -> EvaluationProcedureConfig:
        return EvaluationProcedureConfig._create(
            definition=self,
            trace_source=EvaluationTraceSource.PREPROCESSING,
            preprocessing_config_hash=preprocessing.config_identity_hash,
            preprocessing_implementation_hash=preprocessing.implementation_hash,
            metric_extraction=metric_extraction,
        )

    def materialize_external(
        self,
        *,
        metric_extraction: MetricExtractionConfig,
        external_source: ExternalSource,
    ) -> EvaluationProcedureConfig:
        """Bind metric extraction to traces ingested as truly external."""

        return EvaluationProcedureConfig._create(
            definition=self,
            trace_source=EvaluationTraceSource.EXTERNAL,
            preprocessing_config_hash=None,
            preprocessing_implementation_hash=None,
            metric_extraction=metric_extraction,
            external_source=external_source,
        )


class EvaluationTraceSource(StrEnum):
    """The producer contract accepted by an evaluation procedure."""

    PREPROCESSING = "preprocessing"
    EXTERNAL = "external"


class EvaluationProcedureConfig(FrozenModel):
    definition_ref: DefinitionRef
    trace_source: EvaluationTraceSource
    preprocessing_config_hash: str | None
    preprocessing_implementation_hash: str | None
    external_source: ExternalSource | None
    metric_extraction_config_hash: str
    config_identity_hash: str

    @classmethod
    def _create(
        cls,
        *,
        definition: EvaluationProcedureDefinition,
        trace_source: EvaluationTraceSource,
        preprocessing_config_hash: str | None,
        preprocessing_implementation_hash: str | None,
        metric_extraction: MetricExtractionConfig,
        external_source: ExternalSource | None = None,
    ) -> EvaluationProcedureConfig:
        config_hash = identity_hash_for(
            schema=SCHEMA_EVALUATION_PROCEDURE_CONFIG,
            payload={
                "definition_identity": definition.identity_hash(),
                "trace_source": trace_source,
                "preprocessing_config": preprocessing_config_hash,
                "preprocessing_implementation": (
                    preprocessing_implementation_hash
                ),
                "external_source": (
                    None
                    if external_source is None
                    else external_source.model_dump(mode="json")
                ),
                "metric_extraction_config": (
                    metric_extraction.config_identity_hash
                ),
            },
        )
        return cls(
            definition_ref=definition.ref(),
            trace_source=trace_source,
            preprocessing_config_hash=preprocessing_config_hash,
            preprocessing_implementation_hash=(
                preprocessing_implementation_hash
            ),
            external_source=external_source,
            metric_extraction_config_hash=(
                metric_extraction.config_identity_hash
            ),
            config_identity_hash=config_hash,
        )

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        _validate_ref_schema(
            self.definition_ref,
            SCHEMA_EVALUATION_PROCEDURE_DEFINITION,
        )
        definition = EvaluationProcedureDefinition.model_validate(
            thaw_json(self.definition_ref.identity_payload)
        )
        _validate_canonical_definition_ref(self.definition_ref, definition)
        if self.trace_source is EvaluationTraceSource.PREPROCESSING:
            _validate_sha256(
                self.preprocessing_config_hash,
                owner="preprocessing config hash",
            )
            _validate_sha256(
                self.preprocessing_implementation_hash,
                owner="preprocessing implementation hash",
            )
            if self.external_source is not None:
                raise ValueError(
                    "preprocessing evaluation procedures cannot reference "
                    "an external source"
                )
        elif self.preprocessing_config_hash is not None:
            raise ValueError(
                "external evaluation procedures cannot reference a "
                "preprocessing config"
            )
        elif self.preprocessing_implementation_hash is not None:
            raise ValueError(
                "external evaluation procedures cannot reference a "
                "preprocessing implementation"
            )
        elif self.external_source is None:
            raise ValueError(
                "external evaluation procedures require an external source"
            )
        _validate_sha256(
            self.metric_extraction_config_hash,
            owner="metric extraction config hash",
        )
        _validate_config_hash(
            actual=self.config_identity_hash,
            schema=SCHEMA_EVALUATION_PROCEDURE_CONFIG,
            payload={
                "definition_identity": self.definition_ref.identity_hash,
                "trace_source": self.trace_source,
                "preprocessing_config": self.preprocessing_config_hash,
                "preprocessing_implementation": (
                    self.preprocessing_implementation_hash
                ),
                "external_source": (
                    None
                    if self.external_source is None
                    else self.external_source.model_dump(mode="json")
                ),
                "metric_extraction_config": (
                    self.metric_extraction_config_hash
                ),
            },
        )
        return self

    def validate_trace_producer(self, producer: object) -> None:
        """Validate one trace producer against this explicit source contract."""

        from dr_code.trace import TraceProducer, WiringError

        if not isinstance(producer, TraceProducer):
            raise WiringError(
                "evaluation trace producer must be TraceProducer"
            )
        if self.trace_source is EvaluationTraceSource.EXTERNAL:
            if (
                producer.producer_id != "external"
                or producer.external_source != self.external_source
            ):
                raise WiringError(
                    "external evaluation procedure requires a trace from "
                    "its authenticated external source"
                )
            return
        if producer.producer_id == "external":
            raise WiringError(
                "preprocessing evaluation procedure does not accept "
                "external traces"
            )
        if (
            producer.preprocessing_config_hash
            != self.preprocessing_config_hash
        ):
            raise WiringError(
                "evaluation procedure does not reference this trace's "
                "preprocessing config"
            )
        if (
            producer.implementation_hash
            != self.preprocessing_implementation_hash
        ):
            raise WiringError(
                "evaluation procedure does not reference this trace's "
                "preprocessing implementation"
            )


_CANONICAL_AGGREGATION_VARIABLES = (
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


class AggregationDefinition(FrozenModel):
    """Declare reduction, missing-data, and zero-denominator policy."""

    definition_id: str
    version: str
    variables: tuple[VariableSpec, ...] = _CANONICAL_AGGREGATION_VARIABLES

    @model_validator(mode="after")
    def validate_definition(self) -> Self:
        _validate_definition_coordinates(
            definition_id=self.definition_id,
            version=self.version,
            variables=self.variables,
        )
        if self.variables != _CANONICAL_AGGREGATION_VARIABLES:
            raise VariableError(
                "aggregation variables must match the canonical reduction, "
                "missing-data, and zero-denominator contract"
            )
        return self

    def identity_payload(self) -> dict[str, JsonValue]:
        return _definition_payload(
            definition_id=self.definition_id,
            version=self.version,
            variables=self.variables,
            extra={},
        )

    def identity_hash(self) -> str:
        return identity_hash_for(
            schema=SCHEMA_AGGREGATION_DEFINITION,
            payload=self.identity_payload(),
        )

    def ref(self) -> DefinitionRef:
        return DefinitionRef(
            definition_id=self.definition_id,
            version=self.version,
            schema_name=SCHEMA_AGGREGATION_DEFINITION,
            identity_payload=self.identity_payload(),
            identity_hash=self.identity_hash(),
        )

    def materialize(
        self, assignment: dict[str, JsonValue]
    ) -> AggregationConfig:
        resolved = resolve_assignment(self.variables, assignment)
        return AggregationConfig._create(
            definition=self,
            assignment=resolved,
        )


class AggregationConfig(FrozenModel):
    definition_ref: DefinitionRef
    assignment: tuple[tuple[str, JsonValue], ...]
    config_identity_hash: str

    @field_validator("assignment", mode="before")
    @classmethod
    def thaw_assignment(cls, value: object) -> object:
        return _thaw_assignment(value)

    @field_validator("assignment", mode="after")
    @classmethod
    def freeze_assignment(
        cls, value: tuple[tuple[str, JsonValue], ...]
    ) -> tuple[tuple[str, JsonValue], ...]:
        return _freeze_assignment(value)

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

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        _validate_ref_schema(
            self.definition_ref, SCHEMA_AGGREGATION_DEFINITION
        )
        definition = AggregationDefinition.model_validate(
            thaw_json(self.definition_ref.identity_payload)
        )
        _validate_canonical_definition_ref(self.definition_ref, definition)
        _validated_assignment(definition.variables, self.assignment)
        _validate_config_hash(
            actual=self.config_identity_hash,
            schema=SCHEMA_AGGREGATION_CONFIG,
            payload={
                "definition_identity": self.definition_ref.identity_hash,
                "assignment": _assignment_payload(self.assignment),
            },
        )
        return self


class EvalDefinition(FrozenModel):
    """Declare the composite evaluation configuration."""

    definition_id: str
    version: str
    variables: tuple[VariableSpec, ...] = (
        VariableSpec(name="sampling_config_hash"),
        VariableSpec(name="evaluation_procedure_config_hash"),
        VariableSpec(name="aggregation_config_hash"),
    )

    @model_validator(mode="after")
    def validate_definition(self) -> Self:
        _validate_definition_coordinates(
            definition_id=self.definition_id,
            version=self.version,
            variables=self.variables,
        )
        if tuple(variable.name for variable in self.variables) != (
            "sampling_config_hash",
            "evaluation_procedure_config_hash",
            "aggregation_config_hash",
        ):
            raise VariableError(
                "eval variables must name its three component config hashes"
            )
        return self

    def identity_payload(self) -> dict[str, JsonValue]:
        return _definition_payload(
            definition_id=self.definition_id,
            version=self.version,
            variables=self.variables,
            extra={},
        )

    def identity_hash(self) -> str:
        return identity_hash_for(
            schema=SCHEMA_EVAL_DEFINITION,
            payload=self.identity_payload(),
        )

    def ref(self) -> DefinitionRef:
        return DefinitionRef(
            definition_id=self.definition_id,
            version=self.version,
            schema_name=SCHEMA_EVAL_DEFINITION,
            identity_payload=self.identity_payload(),
            identity_hash=self.identity_hash(),
        )

    def materialize(
        self,
        *,
        sampling: SamplingConfig,
        evaluation_procedure: EvaluationProcedureConfig,
        aggregation: AggregationConfig,
    ) -> EvalConfig:
        resolved = resolve_assignment(
            self.variables,
            {
                "sampling_config_hash": sampling.config_identity_hash,
                "evaluation_procedure_config_hash": (
                    evaluation_procedure.config_identity_hash
                ),
                "aggregation_config_hash": aggregation.config_identity_hash,
            },
        )
        return EvalConfig._create(
            definition=self,
            assignment=resolved,
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
        assignment: dict[str, JsonValue],
    ) -> EvalConfig:
        sampling_config_hash = cast(str, assignment["sampling_config_hash"])
        evaluation_procedure_config_hash = cast(
            str, assignment["evaluation_procedure_config_hash"]
        )
        aggregation_config_hash = cast(
            str, assignment["aggregation_config_hash"]
        )
        config_hash = identity_hash_for(
            schema=SCHEMA_EVAL_CONFIG,
            payload={
                "definition_identity": definition.identity_hash(),
                "sampling_config": sampling_config_hash,
                "evaluation_procedure_config": (
                    evaluation_procedure_config_hash
                ),
                "aggregation_config": aggregation_config_hash,
            },
        )
        return cls(
            definition_ref=definition.ref(),
            sampling_config_hash=sampling_config_hash,
            evaluation_procedure_config_hash=evaluation_procedure_config_hash,
            aggregation_config_hash=aggregation_config_hash,
            config_identity_hash=config_hash,
        )

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        _validate_ref_schema(self.definition_ref, SCHEMA_EVAL_DEFINITION)
        definition = EvalDefinition.model_validate(
            thaw_json(self.definition_ref.identity_payload)
        )
        _validate_canonical_definition_ref(self.definition_ref, definition)
        expected_assignment = _validated_assignment(
            definition.variables,
            (
                ("sampling_config_hash", self.sampling_config_hash),
                (
                    "evaluation_procedure_config_hash",
                    self.evaluation_procedure_config_hash,
                ),
                ("aggregation_config_hash", self.aggregation_config_hash),
            ),
        )
        _ = expected_assignment
        for owner, value in (
            ("sampling config hash", self.sampling_config_hash),
            (
                "evaluation procedure config hash",
                self.evaluation_procedure_config_hash,
            ),
            ("aggregation config hash", self.aggregation_config_hash),
        ):
            _validate_sha256(value, owner=owner)
        _validate_config_hash(
            actual=self.config_identity_hash,
            schema=SCHEMA_EVAL_CONFIG,
            payload={
                "definition_identity": self.definition_ref.identity_hash,
                "sampling_config": self.sampling_config_hash,
                "evaluation_procedure_config": (
                    self.evaluation_procedure_config_hash
                ),
                "aggregation_config": self.aggregation_config_hash,
            },
        )
        return self


__all__ = [
    "AggregationConfig",
    "AggregationDefinition",
    "DefinitionRef",
    "EvalConfig",
    "EvalDefinition",
    "EvaluationProcedureConfig",
    "EvaluationProcedureDefinition",
    "EvaluationTraceSource",
    "MetricExtractionConfig",
    "MetricExtractionDefinition",
    "MetricQuestionBinding",
    "PreprocessingConfig",
    "PreprocessingDefinition",
    "PreprocessingStepBinding",
    "SamplingConfig",
    "SamplingDefinition",
]
