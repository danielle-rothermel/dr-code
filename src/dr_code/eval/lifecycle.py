"""The six evaluation Definition-to-Config pairs, named by coordinates.

A Definition declares semantics and its variables; a Config is one Definition
materialized against a complete variable assignment. Every artifact names
itself with manual coordinates -- a definition id, a manually bumped version,
and its ordered composition -- and cross-references between artifacts are
compared as coordinate models by plain equality.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Self, cast

from pydantic import (
    JsonValue,
    field_serializer,
    field_validator,
    model_validator,
)

from dr_code.eval.tasks import RepeatPlan, TaskSet
from dr_code.eval.variables import (
    JsonArray,
    JsonObject,
    NormalizedJson,
    VariableError,
    VariableReference,
    VariableSpec,
    denormalize_json,
    normalize_json,
    resolve_assignment,
    substitute_variables,
    variable_references,
)
from dr_code.metrics.definition import MetricQuestion, MetricsDefinition
from dr_code.metrics.names import MetricName
from dr_code.models import FrozenModel
from dr_code.preprocessing.definition import PreprocessingDefinition, StepSpec
from dr_code.trace.provenance import PreprocessingDefinitionCoordinate

SCHEMA_SAMPLING_DEFINITION = "dr_code.sampling.definition"
SCHEMA_PREPROCESSING_DEFINITION = "dr_code.preprocessing.definition"
SCHEMA_METRIC_EXTRACTION_DEFINITION = "dr_code.metric_extraction.definition"
SCHEMA_EVALUATION_PROCEDURE_DEFINITION = (
    "dr_code.evaluation_procedure.definition"
)
SCHEMA_AGGREGATION_DEFINITION = "dr_code.aggregation.definition"
SCHEMA_EVAL_DEFINITION = "dr_code.eval.definition"


class DefinitionRef(FrozenModel):
    """The coordinate of the Definition that owns one Config."""

    definition_id: str
    version: str
    schema_name: str

    @model_validator(mode="after")
    def reject_empty_coordinates(self) -> Self:
        if not (self.definition_id and self.version and self.schema_name):
            raise ValueError("definition reference parts must be non-empty")
        return self


class ConfigCoordinate(FrozenModel):
    """The complete coordinate naming one materialized Config.

    A Config is fully named by the Definition it materializes plus the
    assignment that materialized it, so two Configs are the same Config
    exactly when their coordinates compare equal. Every Config validates its
    assignment as complete and in its Definition's variable order, and each
    assigned value is a normalized JSON value -- key-order independent for
    objects, ordered and type-exact for arrays -- so the comparison has one
    canonical form to compare.
    """

    definition_ref: DefinitionRef
    assignment: tuple[tuple[str, NormalizedJson], ...] = ()

    @field_validator("assignment", mode="before")
    @classmethod
    def normalize_assignment(cls, value: object) -> object:
        return _normalized_assignment(value)

    @field_serializer("assignment")
    def serialize_assignment(
        self, value: tuple[tuple[str, NormalizedJson], ...]
    ) -> list[list[JsonValue]]:
        return [[name, denormalize_json(item)] for name, item in value]


def _normalized_assignment(assignment: object) -> object:
    """Normalize one serialized or in-memory assignment to hashable pairs."""

    if not isinstance(assignment, (list, tuple)):
        return assignment
    items = cast("list[object] | tuple[object, ...]", assignment)
    normalized: list[object] = []
    for pair in items:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return assignment
        entry = cast("list[object] | tuple[object, ...]", pair)
        name = entry[0]
        if not isinstance(name, str):
            return assignment
        normalized.append((name, normalize_json(entry[1])))
    return tuple(normalized)


def _sorted_template_entries(
    entries: Iterable[tuple[str, object]],
) -> tuple[tuple[str, NormalizedJson], ...]:
    """Normalize object entries into name order, values possibly templated."""

    return tuple(
        (name, cast(NormalizedJson, _normalize_template_value(child)))
        for name, child in sorted(entries, key=lambda entry: entry[0])
    )


def _normalize_template_value(value: object) -> object:
    if isinstance(value, VariableReference):
        return value
    if isinstance(value, JsonObject):
        return JsonObject(
            entries=_sorted_template_entries(
                (name, child) for name, child in value.entries
            )
        )
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        variable_name = mapping.get("variable")
        if set(mapping) == {"variable"} and isinstance(variable_name, str):
            return VariableReference(variable=variable_name)
        return JsonObject(
            entries=_sorted_template_entries(
                (str(name), child) for name, child in mapping.items()
            )
        )
    if isinstance(value, JsonArray):
        return JsonArray(
            items=tuple(
                cast(NormalizedJson, _normalize_template_value(item))
                for item in value.items
            )
        )
    if isinstance(value, (list, tuple)):
        items = cast("list[object] | tuple[object, ...]", value)
        return JsonArray(
            items=tuple(
                cast(NormalizedJson, _normalize_template_value(child))
                for child in items
            )
        )
    return normalize_json(value)


def _normalize_settings(value: object, *, owner: str) -> object:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        return tuple(
            (str(name), _normalize_template_value(child))
            for name, child in mapping.items()
        )
    if isinstance(value, (list, tuple)):
        pairs = cast("list[object] | tuple[object, ...]", value)
        normalized: list[tuple[str, object]] = []
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                return value
            entry = cast("list[object] | tuple[object, ...]", pair)
            name, setting_value = entry[0], entry[1]
            if not isinstance(name, str):
                return value
            normalized.append((name, _normalize_template_value(setting_value)))
        return tuple(normalized)
    raise TypeError(f"{owner} settings must be a JSON object")


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
    assignment: tuple[tuple[str, NormalizedJson], ...],
) -> dict[str, NormalizedJson]:
    names = [name for name, _value in assignment]
    if len(names) != len(set(names)):
        raise VariableError("config assignment names must be unique")
    resolved = resolve_assignment(variables, dict(assignment))
    if tuple(resolved) != tuple(names):
        raise VariableError(
            "config assignment must be complete and in definition order"
        )
    return resolved


def _validate_ref_schema(ref: DefinitionRef, expected: str) -> None:
    if ref.schema_name != expected:
        raise ValueError(
            f"definition reference schema must be {expected!r}, "
            f"got {ref.schema_name!r}"
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
    assignment: dict[str, NormalizedJson],
) -> tuple[tuple[str, object], ...]:
    return tuple(
        (name, substitute_variables(value, assignment))
        for name, value in settings
    )


def _settings_json(
    settings: tuple[tuple[str, object], ...],
) -> dict[str, JsonValue]:
    return {name: denormalize_json(value) for name, value in settings}


def _canonical_preprocessing_step(
    step_template: PreprocessingStepTemplate,
) -> PreprocessingStepTemplate:
    from dr_code.preprocessing.registry import REGISTRY

    step_class = REGISTRY.get(step_template.step)
    if step_class is None:
        raise ValueError(f"unknown preprocessing step {step_template.step!r}")
    validated = step_class.Settings.model_validate_json(
        json.dumps(_settings_json(step_template.settings), allow_nan=False),
        strict=True,
    )
    return PreprocessingStepTemplate(
        instance_name=step_template.instance_name,
        step=step_template.step,
        settings=validated.model_dump(mode="json"),
    )


def _validate_concrete_preprocessing_settings(
    steps: tuple[PreprocessingStepTemplate, ...],
) -> None:
    for step_template in steps:
        if variable_references(step_template.settings):
            raise VariableError(
                "preprocessing config contains unresolved variable references"
            )
        if step_template != _canonical_preprocessing_step(step_template):
            raise ValueError(
                f"preprocessing settings for {step_template.instance_name!r} "
                "are not fully validated and canonical"
            )


def _canonical_metric_question(
    question: MetricQuestionTemplate,
) -> MetricQuestionTemplate:
    from dr_code.metrics.validation import validated_metric_operator

    operator = validated_metric_operator(
        name=question.metric.value,
        settings=cast("dict[str, object]", _settings_json(question.settings)),
    )
    return MetricQuestionTemplate(
        metric=question.metric,
        on=question.on,
        settings=operator.settings.model_dump(mode="json"),
    )


def _question_triple(
    question: MetricQuestionTemplate,
) -> tuple[MetricName, str, tuple[tuple[str, object], ...]]:
    """Return the (metric, on, settings) triple that names one question."""

    canonical = (
        question
        if variable_references(question.settings)
        else _canonical_metric_question(question)
    )
    return (canonical.metric, canonical.on, canonical.settings)


def _reject_duplicate_questions(
    questions: tuple[MetricQuestionTemplate, ...],
) -> None:
    triples = [_question_triple(question) for question in questions]
    if len(triples) != len(set(triples)):
        raise ValueError(
            "metric questions must have unique (metric, on, settings) triples"
        )


def _validate_concrete_metric_settings(
    questions: tuple[MetricQuestionTemplate, ...],
) -> None:
    _reject_duplicate_questions(questions)
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
        VariableSpec(name="task_set"),
        VariableSpec(name="repeat_plan"),
    )

    @model_validator(mode="after")
    def validate_definition(self) -> Self:
        _validate_definition_coordinates(
            definition_id=self.definition_id,
            version=self.version,
            variables=self.variables,
        )
        if tuple(variable.name for variable in self.variables) != (
            "task_set",
            "repeat_plan",
        ):
            raise VariableError(
                "sampling variables must be task_set and repeat_plan"
            )
        return self

    def ref(self) -> DefinitionRef:
        return DefinitionRef(
            definition_id=self.definition_id,
            version=self.version,
            schema_name=SCHEMA_SAMPLING_DEFINITION,
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
                "task_set": normalize_json(
                    task_set.coordinate().model_dump(mode="json")
                ),
                "repeat_plan": normalize_json(
                    repeat_plan.coordinate().model_dump(mode="json")
                ),
            },
        )
        return SamplingConfig(
            definition_ref=self.ref(),
            assignment=tuple(resolved.items()),
            task_set=task_set,
            repeat_plan=repeat_plan,
        )


class SamplingConfig(FrozenModel):
    definition_ref: DefinitionRef
    assignment: tuple[tuple[str, NormalizedJson], ...]
    task_set: TaskSet
    repeat_plan: RepeatPlan

    @field_validator("assignment", mode="before")
    @classmethod
    def normalize_assignment(cls, value: object) -> object:
        return _normalized_assignment(value)

    @field_serializer("assignment")
    def serialize_assignment(
        self, value: tuple[tuple[str, NormalizedJson], ...]
    ) -> list[list[JsonValue]]:
        return [[name, denormalize_json(item)] for name, item in value]

    def assignment_dict(self) -> dict[str, NormalizedJson]:
        return dict(self.assignment)

    def coordinate(self) -> ConfigCoordinate:
        return ConfigCoordinate(
            definition_ref=self.definition_ref,
            assignment=self.assignment,
        )

    @model_validator(mode="after")
    def validate_composition(self) -> Self:
        _validate_ref_schema(self.definition_ref, SCHEMA_SAMPLING_DEFINITION)
        names = [name for name, _value in self.assignment]
        if names != ["task_set", "repeat_plan"]:
            raise VariableError(
                "sampling assignment must name task_set and repeat_plan "
                "in definition order"
            )
        expected = (
            (
                "task_set",
                normalize_json(
                    self.task_set.coordinate().model_dump(mode="json")
                ),
            ),
            (
                "repeat_plan",
                normalize_json(
                    self.repeat_plan.coordinate().model_dump(mode="json")
                ),
            ),
        )
        if self.assignment != expected:
            raise ValueError(
                "sampling assignment does not name its TaskSet and RepeatPlan"
            )
        if self.repeat_plan.task_identities != self.task_set.task_identities:
            raise ValueError(
                "repeat plan task identities must match the TaskSet manifest"
            )
        return self


class PreprocessingStepTemplate(FrozenModel):
    """One ordered preprocessing step instance, settings still templated.

    Settings may hold variable references; ``PreprocessingTemplate.materialize``
    substitutes them and produces the concrete ``StepSpec`` the preprocessing
    runner binds.
    """

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


class PreprocessingTemplate(FrozenModel):
    """Declare preprocessing steps and the variables that parameterize them.

    A template is the kernel's authoring surface: its step settings may hold
    variable references. ``materialize`` resolves an assignment and yields a
    ``PreprocessingConfig`` nesting the concrete
    ``dr_code.preprocessing.PreprocessingDefinition`` the runner executes.
    """

    definition_id: str
    version: str
    steps: tuple[PreprocessingStepTemplate, ...]
    variables: tuple[VariableSpec, ...] = ()

    @model_validator(mode="after")
    def validate_definition(self) -> Self:
        from dr_code.preprocessing.registry import REGISTRY
        from dr_code.trace import RESERVED_KEYS

        _validate_definition_coordinates(
            definition_id=self.definition_id,
            version=self.version,
            variables=self.variables,
        )
        names = [step_template.instance_name for step_template in self.steps]
        if len(names) != len(set(names)):
            raise ValueError("preprocessing instance names must be unique")
        reserved = set(names) & RESERVED_KEYS
        if reserved:
            raise ValueError(
                "preprocessing instance names must not be reserved: "
                + ", ".join(sorted(reserved))
            )
        expected_input = None
        for step_template in self.steps:
            step_class = REGISTRY.get(step_template.step)
            if step_class is None:
                raise ValueError(
                    f"unknown preprocessing step {step_template.step!r}"
                )
            if not variable_references(step_template.settings):
                _canonical_preprocessing_step(step_template)
            if (
                expected_input is not None
                and step_class.INPUT != expected_input
            ):
                raise ValueError(
                    "broken preprocessing INPUT/OUTPUT kind chain at "
                    f"{step_template.instance_name!r}"
                )
            expected_input = step_class.OUTPUT
        _reference_usage(
            tuple(step_template.settings for step_template in self.steps),
            self.variables,
        )
        return self

    def ref(self) -> DefinitionRef:
        return DefinitionRef(
            definition_id=self.definition_id,
            version=self.version,
            schema_name=SCHEMA_PREPROCESSING_DEFINITION,
        )

    def materialize(
        self, assignment: dict[str, NormalizedJson] | None = None
    ) -> PreprocessingConfig:
        resolved = resolve_assignment(self.variables, assignment or {})
        concrete_steps = tuple(
            _canonical_preprocessing_step(
                PreprocessingStepTemplate(
                    instance_name=step_template.instance_name,
                    step=step_template.step,
                    settings=_concrete_settings(
                        step_template.settings, resolved
                    ),
                )
            )
            for step_template in self.steps
        )
        _validate_concrete_preprocessing_settings(concrete_steps)
        return PreprocessingConfig(
            definition_ref=self.ref(),
            assignment=tuple(resolved.items()),
            definition=PreprocessingDefinition(
                definition_id=self.definition_id,
                version=self.version,
                steps=tuple(
                    StepSpec.model_validate(
                        {
                            "instance_name": step_template.instance_name,
                            "step": step_template.step,
                            "settings": _settings_json(step_template.settings),
                        }
                    )
                    for step_template in concrete_steps
                ),
            ),
        )


def _resolved_step_versions(
    definition: PreprocessingDefinition,
) -> tuple[tuple[str, str, str], ...]:
    """Resolve each step instance to its live registered name and version."""

    from dr_code.preprocessing.registry import REGISTRY

    resolutions: list[tuple[str, str, str]] = []
    for spec in definition.steps:
        step_class = REGISTRY.get(spec.step.value)
        if step_class is None:
            raise ValueError(f"unknown preprocessing step {spec.step.value!r}")
        resolutions.append(
            (spec.instance_name, spec.step.value, str(step_class.VERSION))
        )
    return tuple(resolutions)


class PreprocessingConfig(FrozenModel):
    """One ``PreprocessingTemplate`` materialized against an assignment.

    Nests the concrete ``PreprocessingDefinition`` the runner executes, so a
    config and the pipeline it names are the same object rather than two
    parallel descriptions kept in step.
    """

    definition_ref: DefinitionRef
    assignment: tuple[tuple[str, NormalizedJson], ...]
    definition: PreprocessingDefinition

    @field_validator("assignment", mode="before")
    @classmethod
    def normalize_assignment(cls, value: object) -> object:
        return _normalized_assignment(value)

    @field_serializer("assignment")
    def serialize_assignment(
        self, value: tuple[tuple[str, NormalizedJson], ...]
    ) -> list[list[JsonValue]]:
        return [[name, denormalize_json(item)] for name, item in value]

    def coordinate(self) -> ConfigCoordinate:
        return ConfigCoordinate(
            definition_ref=self.definition_ref,
            assignment=self.assignment,
        )

    def definition_coordinate(self) -> PreprocessingDefinitionCoordinate:
        """Return this config's producer coordinate for trace comparison."""

        from dr_code.trace.provenance import (
            ComponentCoordinate,
            StepCoordinate,
        )

        return PreprocessingDefinitionCoordinate(
            definition_id=self.definition_ref.definition_id,
            version=self.definition_ref.version,
            steps=tuple(
                StepCoordinate(
                    instance_name=instance_name,
                    component=ComponentCoordinate(
                        registered_name=step,
                        version=version,
                    ),
                )
                for instance_name, step, version in _resolved_step_versions(
                    self.definition
                )
            ),
        )

    @model_validator(mode="after")
    def validate_composition(self) -> Self:
        _validate_ref_schema(
            self.definition_ref, SCHEMA_PREPROCESSING_DEFINITION
        )
        if (
            self.definition.definition_id != self.definition_ref.definition_id
            or self.definition.version != self.definition_ref.version
        ):
            raise ValueError(
                "preprocessing config definition must carry the coordinate "
                "its definition reference names"
            )
        _resolved_step_versions(self.definition)
        return self


class MetricQuestionTemplate(FrozenModel):
    """One metric family applied to one trace key, settings still templated.

    Settings may hold variable references;
    ``MetricExtractionTemplate.materialize`` substitutes them and produces the
    concrete ``dr_code.metrics.MetricQuestion`` the engine binds.
    """

    metric: MetricName
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
        return _settings_json(self.settings)


class MetricExtractionTemplate(FrozenModel):
    """Declare an ordered set of metric questions and their variables.

    A template is the kernel's authoring surface: its question settings may
    hold variable references. ``materialize`` resolves an assignment and yields
    a ``MetricExtractionConfig`` nesting the concrete
    ``dr_code.metrics.MetricsDefinition`` the engine executes.
    """

    definition_id: str
    version: str
    questions: tuple[MetricQuestionTemplate, ...]
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
            if question.metric.value not in REGISTRY:
                raise ValueError(
                    f"unknown metric operator {question.metric.value!r}"
                )
        _reject_duplicate_questions(self.questions)
        _reference_usage(
            tuple(question.settings for question in self.questions),
            self.variables,
        )
        return self

    def ref(self) -> DefinitionRef:
        return DefinitionRef(
            definition_id=self.definition_id,
            version=self.version,
            schema_name=SCHEMA_METRIC_EXTRACTION_DEFINITION,
        )

    def materialize(
        self, assignment: dict[str, NormalizedJson] | None = None
    ) -> MetricExtractionConfig:
        resolved = resolve_assignment(self.variables, assignment or {})
        concrete_questions = tuple(
            _canonical_metric_question(
                MetricQuestionTemplate(
                    metric=question.metric,
                    on=question.on,
                    settings=_concrete_settings(question.settings, resolved),
                )
            )
            for question in self.questions
        )
        _validate_concrete_metric_settings(concrete_questions)
        return MetricExtractionConfig(
            definition_ref=self.ref(),
            assignment=tuple(resolved.items()),
            definition=MetricsDefinition(
                definition_id=self.definition_id,
                version=self.version,
                questions=tuple(
                    MetricQuestion.model_validate(
                        {
                            "metric": question.metric,
                            "on": question.on,
                            "settings": _settings_json(question.settings),
                        }
                    )
                    for question in concrete_questions
                ),
            ),
        )


def _resolved_operator_versions(
    definition: MetricsDefinition,
) -> tuple[tuple[str, str], ...]:
    """Resolve each question to its live registered operator and version."""

    from dr_code.metrics.registry import REGISTRY

    resolutions: list[tuple[str, str]] = []
    for question in definition.questions:
        operator_class = REGISTRY.get(question.metric.value)
        if operator_class is None:
            raise ValueError(
                f"unknown metric operator {question.metric.value!r}"
            )
        resolutions.append(
            (question.metric.value, str(operator_class.VERSION))
        )
    return tuple(resolutions)


class MetricExtractionConfig(FrozenModel):
    """One ``MetricExtractionTemplate`` materialized against an assignment.

    Nests the concrete ``MetricsDefinition`` the engine executes, so a config
    and the questions it names are the same object rather than two parallel
    descriptions kept in step.
    """

    definition_ref: DefinitionRef
    assignment: tuple[tuple[str, NormalizedJson], ...]
    definition: MetricsDefinition

    @field_validator("assignment", mode="before")
    @classmethod
    def normalize_assignment(cls, value: object) -> object:
        return _normalized_assignment(value)

    @field_serializer("assignment")
    def serialize_assignment(
        self, value: tuple[tuple[str, NormalizedJson], ...]
    ) -> list[list[JsonValue]]:
        return [[name, denormalize_json(item)] for name, item in value]

    def coordinate(self) -> ConfigCoordinate:
        return ConfigCoordinate(
            definition_ref=self.definition_ref,
            assignment=self.assignment,
        )

    @model_validator(mode="after")
    def validate_composition(self) -> Self:
        _validate_ref_schema(
            self.definition_ref, SCHEMA_METRIC_EXTRACTION_DEFINITION
        )
        if (
            self.definition.definition_id != self.definition_ref.definition_id
            or self.definition.version != self.definition_ref.version
        ):
            raise ValueError(
                "metric extraction config definition must carry the "
                "coordinate its definition reference names"
            )
        _resolved_operator_versions(self.definition)
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

    def ref(self) -> DefinitionRef:
        return DefinitionRef(
            definition_id=self.definition_id,
            version=self.version,
            schema_name=SCHEMA_EVALUATION_PROCEDURE_DEFINITION,
        )

    def materialize(
        self,
        *,
        preprocessing: PreprocessingConfig,
        metric_extraction: MetricExtractionConfig,
    ) -> EvaluationProcedureConfig:
        return EvaluationProcedureConfig(
            definition_ref=self.ref(),
            trace_source=EvaluationTraceSource.PREPROCESSING,
            preprocessing_config=preprocessing.coordinate(),
            preprocessing_definition=preprocessing.definition_coordinate(),
            metric_extraction_config=metric_extraction.coordinate(),
        )

    def materialize_external(
        self,
        *,
        metric_extraction: MetricExtractionConfig,
    ) -> EvaluationProcedureConfig:
        """Bind metric extraction to traces ingested as truly external."""

        return EvaluationProcedureConfig(
            definition_ref=self.ref(),
            trace_source=EvaluationTraceSource.EXTERNAL,
            preprocessing_config=None,
            preprocessing_definition=None,
            metric_extraction_config=metric_extraction.coordinate(),
        )


class EvaluationTraceSource(StrEnum):
    """The producer contract accepted by an evaluation procedure."""

    PREPROCESSING = "preprocessing"
    EXTERNAL = "external"


class EvaluationProcedureConfig(FrozenModel):
    definition_ref: DefinitionRef
    trace_source: EvaluationTraceSource
    preprocessing_config: ConfigCoordinate | None
    preprocessing_definition: PreprocessingDefinitionCoordinate | None
    metric_extraction_config: ConfigCoordinate

    def coordinate(self) -> ConfigCoordinate:
        return ConfigCoordinate(definition_ref=self.definition_ref)

    @model_validator(mode="after")
    def validate_composition(self) -> Self:
        _validate_ref_schema(
            self.definition_ref,
            SCHEMA_EVALUATION_PROCEDURE_DEFINITION,
        )
        if self.trace_source is EvaluationTraceSource.PREPROCESSING:
            if (
                self.preprocessing_config is None
                or self.preprocessing_definition is None
            ):
                raise ValueError(
                    "preprocessing evaluation procedures require a "
                    "preprocessing config and definition coordinate"
                )
            return self
        if self.preprocessing_config is not None:
            raise ValueError(
                "external evaluation procedures cannot reference a "
                "preprocessing config"
            )
        if self.preprocessing_definition is not None:
            raise ValueError(
                "external evaluation procedures cannot reference a "
                "preprocessing definition"
            )
        return self

    def validate_trace_producer(self, producer: object) -> None:
        """Validate one trace producer against this explicit source contract."""

        from dr_code.trace import WiringError
        from dr_code.trace.provenance import (
            ExternalPreprocessingTraceProducer,
            ExternalTraceProducer,
            PreprocessingTraceProducer,
        )

        if not isinstance(
            producer,
            ExternalTraceProducer
            | PreprocessingTraceProducer
            | ExternalPreprocessingTraceProducer,
        ):
            raise WiringError(
                "evaluation trace producer must be a TraceProducer"
            )
        if self.trace_source is EvaluationTraceSource.EXTERNAL:
            if not isinstance(producer, ExternalTraceProducer):
                raise WiringError(
                    "external evaluation procedure requires an external trace"
                )
            return
        if isinstance(producer, ExternalTraceProducer):
            raise WiringError(
                "preprocessing evaluation procedure does not accept "
                "external traces"
            )
        if producer.definition != self.preprocessing_definition:
            raise WiringError(
                "evaluation procedure does not reference this trace's "
                "preprocessing definition"
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

    def ref(self) -> DefinitionRef:
        return DefinitionRef(
            definition_id=self.definition_id,
            version=self.version,
            schema_name=SCHEMA_AGGREGATION_DEFINITION,
        )

    def materialize(
        self, assignment: dict[str, NormalizedJson]
    ) -> AggregationConfig:
        resolved = resolve_assignment(self.variables, assignment)
        return AggregationConfig(
            definition_ref=self.ref(),
            assignment=tuple(resolved.items()),
        )


class AggregationConfig(FrozenModel):
    definition_ref: DefinitionRef
    assignment: tuple[tuple[str, NormalizedJson], ...]

    @field_validator("assignment", mode="before")
    @classmethod
    def normalize_assignment(cls, value: object) -> object:
        return _normalized_assignment(value)

    @field_serializer("assignment")
    def serialize_assignment(
        self, value: tuple[tuple[str, NormalizedJson], ...]
    ) -> list[list[JsonValue]]:
        return [[name, denormalize_json(item)] for name, item in value]

    def assignment_dict(self) -> dict[str, NormalizedJson]:
        return dict(self.assignment)

    def coordinate(self) -> ConfigCoordinate:
        return ConfigCoordinate(
            definition_ref=self.definition_ref,
            assignment=self.assignment,
        )

    @model_validator(mode="after")
    def validate_composition(self) -> Self:
        _validate_ref_schema(
            self.definition_ref, SCHEMA_AGGREGATION_DEFINITION
        )
        _validated_assignment(
            _CANONICAL_AGGREGATION_VARIABLES, self.assignment
        )
        return self


class EvalDefinition(FrozenModel):
    """Declare the composite evaluation configuration."""

    definition_id: str
    version: str
    variables: tuple[VariableSpec, ...] = (
        VariableSpec(name="sampling_config"),
        VariableSpec(name="evaluation_procedure_config"),
        VariableSpec(name="aggregation_config"),
    )

    @model_validator(mode="after")
    def validate_definition(self) -> Self:
        _validate_definition_coordinates(
            definition_id=self.definition_id,
            version=self.version,
            variables=self.variables,
        )
        if tuple(variable.name for variable in self.variables) != (
            "sampling_config",
            "evaluation_procedure_config",
            "aggregation_config",
        ):
            raise VariableError(
                "eval variables must name its three component configs"
            )
        return self

    def ref(self) -> DefinitionRef:
        return DefinitionRef(
            definition_id=self.definition_id,
            version=self.version,
            schema_name=SCHEMA_EVAL_DEFINITION,
        )

    def materialize(
        self,
        *,
        sampling: SamplingConfig,
        evaluation_procedure: EvaluationProcedureConfig,
        aggregation: AggregationConfig,
    ) -> EvalConfig:
        return EvalConfig(
            definition_ref=self.ref(),
            sampling_config=sampling.coordinate(),
            evaluation_procedure_config=evaluation_procedure.coordinate(),
            aggregation_config=aggregation.coordinate(),
        )


class EvalConfig(FrozenModel):
    definition_ref: DefinitionRef
    sampling_config: ConfigCoordinate
    evaluation_procedure_config: ConfigCoordinate
    aggregation_config: ConfigCoordinate

    def coordinate(self) -> ConfigCoordinate:
        return ConfigCoordinate(definition_ref=self.definition_ref)

    @model_validator(mode="after")
    def validate_composition(self) -> Self:
        _validate_ref_schema(self.definition_ref, SCHEMA_EVAL_DEFINITION)
        _validate_ref_schema(
            self.sampling_config.definition_ref,
            SCHEMA_SAMPLING_DEFINITION,
        )
        _validate_ref_schema(
            self.evaluation_procedure_config.definition_ref,
            SCHEMA_EVALUATION_PROCEDURE_DEFINITION,
        )
        _validate_ref_schema(
            self.aggregation_config.definition_ref,
            SCHEMA_AGGREGATION_DEFINITION,
        )
        return self


__all__ = [
    "SCHEMA_AGGREGATION_DEFINITION",
    "SCHEMA_EVALUATION_PROCEDURE_DEFINITION",
    "SCHEMA_EVAL_DEFINITION",
    "SCHEMA_METRIC_EXTRACTION_DEFINITION",
    "SCHEMA_PREPROCESSING_DEFINITION",
    "SCHEMA_SAMPLING_DEFINITION",
    "AggregationConfig",
    "AggregationDefinition",
    "ConfigCoordinate",
    "DefinitionRef",
    "EvalConfig",
    "EvalDefinition",
    "EvaluationProcedureConfig",
    "EvaluationProcedureDefinition",
    "EvaluationTraceSource",
    "MetricExtractionConfig",
    "MetricExtractionTemplate",
    "MetricQuestionTemplate",
    "PreprocessingConfig",
    "PreprocessingStepTemplate",
    "PreprocessingTemplate",
    "SamplingConfig",
    "SamplingDefinition",
]
