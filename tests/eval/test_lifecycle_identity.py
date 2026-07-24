"""Coordinate and Definition-to-Config contracts for the evaluation kernel."""

from __future__ import annotations

import math

import pytest
from dr_serialize import StrictJsonError
from pydantic import JsonValue, ValidationError

from dr_code.eval import (
    AggregationDefinition,
    DatasetCoordinate,
    EvalDefinition,
    EvaluationProcedureDefinition,
    MetricExtractionTemplate,
    MetricQuestionTemplate,
    PreprocessingStepTemplate,
    PreprocessingTemplate,
    RepeatPlan,
    SamplingDefinition,
    TaskSet,
    VariableError,
    VariableReference,
    VariableSpec,
    resolve_assignment,
)
from dr_code.eval.variables import (
    JsonArray,
    JsonObject,
    denormalize_json,
    normalize_json,
)

_DATASET = DatasetCoordinate(
    dataset_id="fixture",
    dataset_split="test",
    dataset_revision="r1",
)


def _components():
    task_set = TaskSet(
        manifest_id="tasks",
        version="1",
        dataset=_DATASET,
        source_task_identities=("task",),
        task_identities=("task",),
    )
    repeat_plan = RepeatPlan(
        plan_id="repeats",
        version="1",
        task_identities=("task",),
        repeat_count=1,
        seeds=(("task#0", 7),),
    )
    sampling = SamplingDefinition(
        definition_id="samp", version="1"
    ).materialize(task_set=task_set, repeat_plan=repeat_plan)
    preprocessing = PreprocessingTemplate(
        definition_id="pre",
        version="1",
        steps=(
            PreprocessingStepTemplate(
                instance_name="sf",
                step="return_all",
            ),
        ),
    ).materialize()
    metric = MetricExtractionTemplate(
        definition_id="met",
        version="1",
        questions=(
            MetricQuestionTemplate(
                metric="code_leakage",
                on="output",
            ),
        ),
    ).materialize()
    procedure = EvaluationProcedureDefinition(
        definition_id="proc", version="1"
    ).materialize(
        preprocessing=preprocessing,
        metric_extraction=metric,
    )
    aggregation = AggregationDefinition(
        definition_id="agg", version="1"
    ).materialize({"reduction": "mean"})
    composite = EvalDefinition(definition_id="ev", version="1").materialize(
        sampling=sampling,
        evaluation_procedure=procedure,
        aggregation=aggregation,
    )
    return sampling, preprocessing, metric, procedure, aggregation, composite


def test_configs_name_themselves_by_definition_coordinates() -> None:
    actual = tuple(
        (
            item.definition_ref.definition_id,
            item.definition_ref.version,
            item.definition_ref.schema_name,
        )
        for item in _components()
    )
    assert actual == (
        ("samp", "1", "dr_code.sampling.definition"),
        ("pre", "1", "dr_code.preprocessing.definition"),
        ("met", "1", "dr_code.metric_extraction.definition"),
        ("proc", "1", "dr_code.evaluation_procedure.definition"),
        ("agg", "1", "dr_code.aggregation.definition"),
        ("ev", "1", "dr_code.eval.definition"),
    )


def test_resolved_step_and_operator_versions_are_materialized() -> None:
    _, preprocessing, metric, procedure, _, _ = _components()
    assert preprocessing.definition_coordinate().steps[0].instance_name == (
        "sf"
    )
    assert (
        preprocessing.definition_coordinate()
        .steps[0]
        .component.registered_name
        == "return_all"
    )
    assert (
        preprocessing.definition_coordinate().steps[0].component.version == "0"
    )
    assert tuple(
        question.metric.value for question in metric.definition.questions
    ) == ("code_leakage",)
    assert procedure.preprocessing_config == preprocessing.coordinate()
    assert procedure.metric_extraction_config == metric.coordinate()


def test_config_assignments_are_complete_and_constrained() -> None:
    definition = AggregationDefinition(definition_id="a", version="1")
    with pytest.raises(VariableError, match="unassigned"):
        definition.materialize({})
    with pytest.raises(VariableError, match="unknown"):
        definition.materialize(
            {
                "reduction": "mean",
                "extra": True,
            }
        )


@pytest.mark.parametrize(
    ("allowed", "value"),
    [
        (1, True),
        (True, 1),
        (1, 1.0),
        ({"nested": [1]}, {"nested": [True]}),
    ],
)
def test_variable_allowed_values_use_exact_recursive_json_equality(
    allowed: JsonValue,
    value: JsonValue,
) -> None:
    spec = VariableSpec(name="value", allowed=(allowed,))
    with pytest.raises(VariableError, match="not an allowed value"):
        resolve_assignment((spec,), {"value": value})
    with pytest.raises(ValidationError, match="default"):
        VariableSpec(
            name="value",
            allowed=(allowed,),
            default=value,
            has_default=True,
        )


def test_eval_definition_enforces_its_three_component_variables() -> None:
    with pytest.raises(ValidationError, match="three component configs"):
        EvalDefinition(
            definition_id="ev",
            version="1",
            variables=(
                VariableSpec(name="sampling_config"),
                VariableSpec(name="evaluation_procedure_config"),
                VariableSpec(name="unexpected_component"),
            ),
        )


def test_eval_config_names_each_component_by_coordinate() -> None:
    sampling, _, _, procedure, aggregation, composite = _components()
    assert composite.sampling_config == sampling.coordinate()
    assert composite.evaluation_procedure_config == procedure.coordinate()
    assert composite.aggregation_config == aggregation.coordinate()


def test_normalized_json_rejects_nonfinite_values() -> None:
    with pytest.raises(StrictJsonError):
        normalize_json({"value": math.inf})


def test_normalized_json_objects_are_key_order_independent() -> None:
    forward = normalize_json({"a": 1, "b": {"y": [1, 2], "x": None}})
    reverse = normalize_json({"b": {"x": None, "y": [1, 2]}, "a": 1})

    assert isinstance(forward, JsonObject)
    assert isinstance(reverse, JsonObject)
    assert forward == reverse
    assert hash(forward) == hash(reverse)
    assert forward.model_dump(mode="json") == reverse.model_dump(mode="json")
    assert forward.model_dump_json() == reverse.model_dump_json()
    assert denormalize_json(forward) == denormalize_json(reverse)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ([1], [True]),
        ([1], [1.0]),
        ([1.0], [True]),
        ([{"x": 1}], [{"x": True}]),
        ([[1]], [[True]]),
        ([1, 2], [2, 1]),
        ([1], [1, 1]),
    ],
)
def test_normalized_json_arrays_are_type_exact(
    left: JsonValue,
    right: JsonValue,
) -> None:
    assert normalize_json(left) != normalize_json(right)


@pytest.mark.parametrize(
    "value",
    [
        [1, 2, 3],
        [True, False],
        [1.0, 2.0],
        [{"a": 1}, {"b": [None]}],
        [],
    ],
)
def test_normalized_json_arrays_of_equal_content_and_order_are_equal(
    value: JsonValue,
) -> None:
    left = normalize_json(value)
    right = normalize_json(value)

    assert left == right
    assert hash(left) == hash(right)


def test_normalized_json_arrays_keep_their_declared_order() -> None:
    array = normalize_json([{"b": 1, "a": 2}, "second"])

    assert array == JsonArray(
        items=(JsonObject(entries=(("a", 2), ("b", 1))), "second")
    )
    assert denormalize_json(array) == [{"a": 2, "b": 1}, "second"]


def test_materialized_metric_settings_ignore_mapping_order() -> None:
    forward = MetricQuestionTemplate(
        metric="compressed_length",
        on="output",
        settings={"compression": {"method": "gzip", "level": 9}},
    )
    reverse = MetricQuestionTemplate(
        metric="compressed_length",
        on="output",
        settings={"compression": {"level": 9, "method": "gzip"}},
    )

    assert tuple(name for name, _value in forward.settings) == ("compression",)
    assert forward.settings_dict() == {
        "compression": {"method": "gzip", "level": 9},
    }

    forward_definition = MetricExtractionTemplate(
        definition_id="metrics",
        version="1",
        questions=(forward,),
    )
    reverse_definition = MetricExtractionTemplate(
        definition_id="metrics",
        version="1",
        questions=(reverse,),
    )
    assert forward_definition.materialize() == reverse_definition.materialize()


def test_metric_question_binding_is_named_by_its_triple() -> None:
    binding = MetricQuestionTemplate(metric="code_leakage", on="output")

    assert (binding.metric, binding.on) == ("code_leakage", "output")
    assert binding == MetricQuestionTemplate(
        metric="code_leakage",
        on="output",
        settings={"task_names": []},
    ).model_copy(update={"settings": binding.settings})


def test_preprocessing_settings_normalize_recursively() -> None:
    # ``PreprocessingStepTemplate`` normalizes its settings without consulting
    # the registry; ``PreprocessingTemplate`` validates every step against the
    # registry in a model validator, so registry validation is
    # construction-time, not materialize-time. This test covers the
    # normalization boundary on its own, sequence values included.
    def _step(value: object) -> PreprocessingStepTemplate:
        return PreprocessingStepTemplate(
            instance_name="tabs",
            step="expand_tabs",
            settings={"tab_width": value},
        )

    forward = _step(2)
    assert forward == _step(2)

    # A sequence-valued setting normalizes recursively, nested arrays and
    # mixed scalar types included, and is order-sensitive.
    nested = _step([1, [2, "three"], 4])
    assert nested.settings == (
        (
            "tab_width",
            JsonArray(
                items=(1, JsonArray(items=(2, "three")), 4),
            ),
        ),
    )
    assert nested == _step([1, [2, "three"], 4])
    assert nested != _step([1, [2, "three"], 4.0])
    assert nested != _step([[2, "three"], 1, 4])

    # A sequence reaching the template through a variable default normalizes
    # the same way, so the variables path shares one normalization rule.
    templated = PreprocessingTemplate(
        definition_id="pre",
        version="1",
        steps=(
            PreprocessingStepTemplate(
                instance_name="tabs",
                step="expand_tabs",
                settings={"tab_width": VariableReference(variable="widths")},
            ),
        ),
        variables=(
            VariableSpec(
                name="widths",
                default=[1, [2, "three"], 4],
                has_default=True,
            ),
        ),
    )
    assert templated.variables[0].default == JsonArray(
        items=(1, JsonArray(items=(2, "three")), 4),
    )

    # Template identity over a step whose settings the registry validates at
    # construction time.
    def _tabs() -> PreprocessingStepTemplate:
        return PreprocessingStepTemplate(
            instance_name="tabs",
            step="expand_tabs",
            settings={"tab_width": 2},
        )

    assert PreprocessingTemplate(
        definition_id="pre",
        version="1",
        steps=(_tabs(),),
    ) == PreprocessingTemplate(
        definition_id="pre",
        version="1",
        steps=(_tabs(),),
    )

    # Registry validation really is construction-time: an unknown step is
    # rejected before anything is materialized.
    with pytest.raises(ValidationError):
        PreprocessingTemplate(
            definition_id="pre",
            version="1",
            steps=(
                PreprocessingStepTemplate(
                    instance_name="nope",
                    step="no_such_step",
                ),
            ),
        )


@pytest.mark.parametrize("invalid", [math.inf, math.nan, {"values": {1, 2}}])
def test_preprocessing_settings_reject_non_json_values(
    invalid: object,
) -> None:
    with pytest.raises(StrictJsonError):
        PreprocessingStepTemplate(
            instance_name="extract",
            step="return_all",
            settings={"invalid": invalid},
        )


def test_metric_settings_ignore_nested_mapping_order() -> None:
    forward = MetricQuestionTemplate(
        metric="compressed_length",
        on="output",
        settings={
            "compression": {
                "method": "gzip",
                "level": 9,
            }
        },
    )
    reverse = MetricQuestionTemplate(
        metric="compressed_length",
        on="output",
        settings={
            "compression": {
                "level": 9,
                "method": "gzip",
            }
        },
    )

    forward_config = MetricExtractionTemplate(
        definition_id="metrics",
        version="1",
        questions=(forward,),
    ).materialize()
    reverse_config = MetricExtractionTemplate(
        definition_id="metrics",
        version="1",
        questions=(reverse,),
    ).materialize()

    assert forward_config == reverse_config
    compression = forward_config.definition.questions[0].settings.model_dump(
        mode="json"
    )["compression"]
    assert compression == {"method": "gzip", "level": 9}


def test_metric_questions_reject_reordered_setting_duplicates() -> None:
    forward = MetricQuestionTemplate(
        metric="compressed_length",
        on="output",
        settings={
            "compression": {
                "method": "gzip",
                "level": 9,
            },
            "reference_key": None,
        },
    )
    reverse = MetricQuestionTemplate(
        metric="compressed_length",
        on="output",
        settings={
            "reference_key": None,
            "compression": {
                "level": 9,
                "method": "gzip",
            },
        },
    )

    with pytest.raises(
        ValueError,
        match="metric questions must have unique",
    ):
        MetricExtractionTemplate(
            definition_id="metrics",
            version="1",
            questions=(forward, reverse),
        )


def test_metric_setting_sequence_order_remains_significant() -> None:
    forward = MetricQuestionTemplate(
        metric="code_leakage",
        on="output",
        settings={"task_names": ["add", "subtract"]},
    )
    reverse = MetricQuestionTemplate(
        metric="code_leakage",
        on="output",
        settings={"task_names": ["subtract", "add"]},
    )

    assert forward != reverse
    assert forward.settings_dict()["task_names"] == ["add", "subtract"]
    assert reverse.settings_dict()["task_names"] == ["subtract", "add"]
    assert MetricExtractionTemplate(
        definition_id="metrics",
        version="1",
        questions=(forward,),
    ) != MetricExtractionTemplate(
        definition_id="metrics",
        version="1",
        questions=(reverse,),
    )


@pytest.mark.parametrize("invalid", [math.inf, math.nan, {"values": {1, 2}}])
def test_metric_settings_reject_non_json_values(invalid: object) -> None:
    with pytest.raises(StrictJsonError):
        MetricQuestionTemplate(
            metric="text_stats",
            on="output",
            settings={"invalid": invalid},
        )


def test_composite_coordinate_changes_with_each_component() -> None:
    sampling, _, _, procedure, aggregation, base = _components()
    changed_task_set = TaskSet(
        manifest_id="tasks",
        version="2",
        dataset=_DATASET,
        source_task_identities=("task",),
        task_identities=("task",),
    )
    changed_repeat_plan = RepeatPlan(
        plan_id="repeats",
        version="1",
        task_identities=("task",),
        repeat_count=1,
        seeds=(("task#0", 7),),
    )
    changed_sampling = SamplingDefinition(
        definition_id="samp", version="1"
    ).materialize(
        task_set=changed_task_set,
        repeat_plan=changed_repeat_plan,
    )
    changed = EvalDefinition(definition_id="ev", version="1").materialize(
        sampling=changed_sampling,
        evaluation_procedure=procedure,
        aggregation=aggregation,
    )
    assert changed.sampling_config != sampling.coordinate()
    assert changed != base


@pytest.mark.parametrize("component_index", range(6))
def test_every_config_round_trips_through_serialization(
    component_index: int,
) -> None:
    config = _components()[component_index]
    restored = type(config).model_validate_json(config.model_dump_json())
    assert restored == config
    assert restored.coordinate() == config.coordinate()
    assert hash(restored.definition_ref) == hash(config.definition_ref)


@pytest.mark.parametrize("component_index", range(6))
def test_configs_reject_a_mismatched_definition_schema(
    component_index: int,
) -> None:
    config = _components()[component_index]
    forged = config.model_dump(mode="json")
    forged["definition_ref"]["schema_name"] = "dr_code.not.this.schema"
    with pytest.raises(
        ValidationError, match="definition reference schema must be"
    ):
        type(config).model_validate(forged)


def test_canonical_variable_contracts_reject_custom_policies() -> None:
    with pytest.raises(ValidationError, match="canonical"):
        AggregationDefinition(
            definition_id="agg",
            version="1",
            variables=(
                VariableSpec(
                    name="reduction",
                    allowed=("mean", "median"),
                ),
            ),
        )
    with pytest.raises(VariableError, match="not an allowed value"):
        AggregationDefinition(definition_id="agg", version="1").materialize(
            {"reduction": "median"}
        )
    with pytest.raises(ValidationError, match="does not define"):
        EvaluationProcedureDefinition(
            definition_id="procedure",
            version="1",
            variables=(VariableSpec(name="zero_denominator"),),
        )


def test_normalized_settings_are_hashable_and_round_trip() -> None:
    spec = VariableSpec(name="names", allowed=(["a"],))
    definition = MetricExtractionTemplate(
        definition_id="metrics",
        version="1",
        variables=(spec,),
        questions=(
            MetricQuestionTemplate(
                metric="code_leakage",
                on="output",
                settings={"task_names": VariableReference(variable="names")},
            ),
        ),
    )
    assert hash(definition) == hash(
        MetricExtractionTemplate.model_validate(
            definition.model_dump(mode="python")
        )
    )

    config = definition.materialize({"names": ["a"]})
    assert hash(config.definition_ref) == hash(definition.ref())
    task_names = config.definition.questions[0].settings.task_names
    assert tuple(task_names) == ("a",)
    assert type(config).model_validate_json(config.model_dump_json()) == config

    preprocessing = PreprocessingTemplate(
        definition_id="pre",
        version="1",
        steps=(
            PreprocessingStepTemplate(
                instance_name="tabs",
                step="expand_tabs",
                settings={"tab_width": 2},
            ),
        ),
    )
    config = preprocessing.materialize()
    assert config.definition.steps[0].settings.tab_width == 2
    assert type(config).model_validate_json(config.model_dump_json()) == config


def test_substitution_rejects_concrete_metric_question_collisions() -> None:
    definition = MetricExtractionTemplate(
        definition_id="metrics",
        version="1",
        variables=(VariableSpec(name="a"), VariableSpec(name="b")),
        questions=(
            MetricQuestionTemplate(
                metric="code_leakage",
                on="output",
                settings={"task_names": VariableReference(variable="a")},
            ),
            MetricQuestionTemplate(
                metric="code_leakage",
                on="output",
                settings={"task_names": VariableReference(variable="b")},
            ),
        ),
    )
    with pytest.raises(ValueError, match="unique \\(metric, on, settings\\)"):
        definition.materialize({"a": [], "b": []})


def test_definition_ref_carries_only_coordinates() -> None:
    config = _components()[2]
    ref = config.definition_ref
    assert type(ref).model_validate_json(ref.model_dump_json()) == ref
    assert set(ref.model_dump()) == {
        "definition_id",
        "version",
        "schema_name",
    }

    empty = ref.model_dump(mode="json")
    empty["definition_id"] = ""
    with pytest.raises(ValidationError, match="must be non-empty"):
        type(ref).model_validate(empty)


def test_config_rejects_duplicate_serialized_assignment_names() -> None:
    config = _components()[4]
    forged = config.model_dump(mode="json")
    forged["assignment"].append(forged["assignment"][0])
    with pytest.raises(
        ValidationError, match="assignment names must be unique"
    ):
        type(config).model_validate(forged)


def test_typed_variable_substitution_controls_preprocessing_config() -> None:
    definition = PreprocessingTemplate(
        definition_id="variable-preprocessing",
        version="1",
        variables=(VariableSpec(name="tab_width", allowed=(2, 4)),),
        steps=(
            PreprocessingStepTemplate(
                instance_name="tabs",
                step="expand_tabs",
                settings={
                    "tab_width": VariableReference(variable="tab_width")
                },
            ),
        ),
    )
    width_two = definition.materialize({"tab_width": 2})
    width_four = definition.materialize({"tab_width": 4})
    assert width_two.definition.steps[0].settings.tab_width == 2
    assert width_four.definition.steps[0].settings.tab_width == 4
    assert width_two.coordinate() != width_four.coordinate()
    assert (
        width_two.definition_coordinate().definition_id
        == width_four.definition_coordinate().definition_id
    )
    with pytest.raises(VariableError, match="unassigned"):
        definition.materialize()


def test_definition_rejects_unused_and_undefined_variable_references() -> None:
    with pytest.raises(ValidationError, match="unused variable"):
        PreprocessingTemplate(
            definition_id="unused",
            version="1",
            variables=(VariableSpec(name="unused"),),
            steps=(),
        )
    with pytest.raises(ValidationError, match="undefined variable"):
        PreprocessingTemplate(
            definition_id="undefined",
            version="1",
            steps=(
                PreprocessingStepTemplate(
                    instance_name="tabs",
                    step="expand_tabs",
                    settings={
                        "tab_width": VariableReference(variable="missing")
                    },
                ),
            ),
        )


@pytest.mark.parametrize("level", [True, 1.0])
def test_operator_settings_reject_numeric_aliases(
    level: object,
) -> None:
    with pytest.raises(ValidationError):
        MetricExtractionTemplate(
            definition_id="strict-settings",
            version="1",
            questions=(
                MetricQuestionTemplate(
                    metric="compressed_length",
                    on="output",
                    settings={
                        "compression": {"method": "gzip", "level": level}
                    },
                ),
            ),
        )


def test_operator_defaults_match_explicit_settings() -> None:
    omitted = MetricQuestionTemplate(metric="code_leakage", on="output")
    explicit = MetricQuestionTemplate(
        metric="code_leakage",
        on="output",
        settings={"task_names": []},
    )
    omitted_config = MetricExtractionTemplate(
        definition_id="metrics",
        version="1",
        questions=(omitted,),
    ).materialize()
    explicit_config = MetricExtractionTemplate(
        definition_id="metrics",
        version="1",
        questions=(explicit,),
    ).materialize()
    assert omitted_config.definition == explicit_config.definition
