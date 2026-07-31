"""Identity and Definition-to-Config contracts for the evaluation kernel."""

from __future__ import annotations

import math

import pytest
from dr_serialize import StrictJsonError
from pydantic import JsonValue, ValidationError

from dr_code.eval import (
    AggregationDefinition,
    EvalDefinition,
    EvaluationProcedureDefinition,
    MetricExtractionDefinition,
    MetricQuestionBinding,
    PreprocessingDefinition,
    PreprocessingStepBinding,
    RepeatPlan,
    SamplingDefinition,
    TaskSet,
    VariableError,
    VariableReference,
    VariableSpec,
    identity_hash_for,
    resolved_operator_identity,
    resolved_step_identity,
    resolve_assignment,
)
from dr_code.preprocessing import run_preprocessing
from dr_code.trace import TextArtifact
from dr_code.eval.identity import (
    SCHEMA_AGGREGATION_CONFIG,
    SCHEMA_EVALUATION_PROCEDURE_CONFIG,
    SCHEMA_EVAL_CONFIG,
    SCHEMA_METRIC_EXTRACTION_CONFIG,
    SCHEMA_PREPROCESSING_CONFIG,
    SCHEMA_SAMPLING_CONFIG,
)


def _components():
    task_set = TaskSet(
        manifest_id="tasks",
        version="1",
        dataset_id="fixture",
        dataset_split="test",
        dataset_revision="r1",
        source_content_hash="a" * 64,
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
    preprocessing = PreprocessingDefinition(
        definition_id="pre",
        version="1",
        steps=(
            PreprocessingStepBinding(
                instance_name="return_all",
                step="return_all",
            ),
        ),
    ).materialize()
    metric = MetricExtractionDefinition(
        definition_id="met",
        version="1",
        questions=(
            MetricQuestionBinding(
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


def test_configs_have_stable_full_sha256_identities() -> None:
    expected = (
        "df81e5ad266c408bfc17ca576ce0e5febdb91b2e02e6ee4c298b1168936b641f",
        "5b2e11c696428497d9d735cb2a2b8f98a7e535e765f7934b1a28fa8bc0e786c2",
        "1e5d83675e61e483e5b54455eb33bb32bf548e3059268364e4879549ed3d6a67",
        "e4aeb74e4568130e2a1ba023c6bcf494114b31cbbf8be3179f812a728aea34f0",
        "373b234c512b09f897fc42107cb7f6d7b848c5f6a48e4dc70d0ba39727e3522b",
        "32f202ec7a45e2c4cbe36937d544adbbbc492bac8127ebf8feba6c8d43c2bec2",
    )
    actual = tuple(item.config_identity_hash for item in _components())
    assert actual == expected
    assert all(len(identity) == 64 for identity in actual)


def test_resolved_step_and_operator_versions_are_materialized() -> None:
    _, preprocessing, metric, procedure, _, _ = _components()
    assert preprocessing.resolved_step_versions == (
        (
            "return_all",
            "return_all",
            *resolved_step_identity("return_all"),
        ),
    )
    assert metric.resolved_operator_versions == (
        (
            metric.questions[0].identity_hash(),
            "code_leakage",
            *resolved_operator_identity("code_leakage"),
        ),
    )
    assert procedure.preprocessing_config_hash == (
        preprocessing.config_identity_hash
    )
    assert procedure.metric_extraction_config_hash == (
        metric.config_identity_hash
    )


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


def test_eval_materialize_enforces_composite_variables() -> None:
    sampling, _, _, procedure, aggregation, _ = _components()
    definition = EvalDefinition(
        definition_id="ev",
        version="1",
        variables=(
            VariableSpec(
                name="sampling_config_hash",
                allowed=("not-the-sampling-config",),
            ),
            VariableSpec(name="evaluation_procedure_config_hash"),
            VariableSpec(name="aggregation_config_hash"),
        ),
    )

    with pytest.raises(VariableError, match="sampling_config_hash"):
        definition.materialize(
            sampling=sampling,
            evaluation_procedure=procedure,
            aggregation=aggregation,
        )


def test_identity_rejects_nonfinite_json() -> None:
    with pytest.raises(StrictJsonError):
        identity_hash_for(
            schema="dr_code.test",
            payload={"value": math.inf},
        )


def test_metric_settings_ignore_top_level_mapping_order() -> None:
    forward = MetricQuestionBinding(
        metric="compressed_length",
        on="output",
        settings={"compression": {"method": "gzip", "level": 9}},
    )
    reverse = MetricQuestionBinding(
        metric="compressed_length",
        on="output",
        settings={"compression": {"level": 9, "method": "gzip"}},
    )

    assert forward == reverse
    assert tuple(name for name, _value in forward.settings) == ("compression",)
    assert forward.settings_dict() == {
        "compression": {"level": 9, "method": "gzip"},
    }

    forward_definition = MetricExtractionDefinition(
        definition_id="metrics",
        version="1",
        questions=(forward,),
    )
    reverse_definition = MetricExtractionDefinition(
        definition_id="metrics",
        version="1",
        questions=(reverse,),
    )
    assert forward_definition.identity_hash() == (
        reverse_definition.identity_hash()
    )
    assert (
        forward_definition.materialize().config_identity_hash
        == reverse_definition.materialize().config_identity_hash
    )


def test_metric_question_binding_has_schema_tagged_golden_identity() -> None:
    binding = MetricQuestionBinding(metric="code_leakage", on="output")

    assert (
        binding.identity_hash()
        == "0de9c1d49cf0e80c13e19a784b74a721c435b032a9797807104c301d7d5415ca"
    )


def test_preprocessing_settings_use_recursive_canonical_json() -> None:
    forward = PreprocessingStepBinding(
        instance_name="extract",
        step="extract_candidates",
        settings={"alternatives": ["fenced_blocks", "escaped_python"]},
    )
    reverse = PreprocessingStepBinding(
        instance_name="extract",
        step="extract_candidates",
        settings={"alternatives": ["fenced_blocks", "escaped_python"]},
    )

    assert forward == reverse
    assert forward.settings == (
        (
            "alternatives",
            ("fenced_blocks", "escaped_python"),
        ),
    )
    assert forward.model_dump_json() == reverse.model_dump_json()


@pytest.mark.parametrize("invalid", [math.inf, math.nan, {"values": {1, 2}}])
def test_preprocessing_settings_reject_non_json_values(
    invalid: object,
) -> None:
    with pytest.raises(StrictJsonError):
        PreprocessingStepBinding(
            instance_name="extract",
            step="return_all",
            settings={"invalid": invalid},
        )


def test_metric_settings_ignore_nested_mapping_order() -> None:
    forward = MetricQuestionBinding(
        metric="compressed_length",
        on="output",
        settings={
            "compression": {
                "method": "gzip",
                "level": 9,
            }
        },
    )
    reverse = MetricQuestionBinding(
        metric="compressed_length",
        on="output",
        settings={
            "compression": {
                "level": 9,
                "method": "gzip",
            }
        },
    )

    assert forward == reverse
    assert forward.settings_dict() == reverse.settings_dict()
    settings = forward.settings_dict()
    assert settings == {
        "compression": {
            "level": 9,
            "method": "gzip",
        }
    }
    compression = settings["compression"]
    assert isinstance(compression, dict)
    assert tuple(compression) == ("level", "method")


def test_metric_questions_reject_reordered_setting_duplicates() -> None:
    forward = MetricQuestionBinding(
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
    reverse = MetricQuestionBinding(
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
        MetricExtractionDefinition(
            definition_id="metrics",
            version="1",
            questions=(forward, reverse),
        )


def test_metric_setting_sequence_order_remains_significant() -> None:
    forward = MetricQuestionBinding(
        metric="code_leakage",
        on="output",
        settings={"task_names": ["add", "subtract"]},
    )
    reverse = MetricQuestionBinding(
        metric="code_leakage",
        on="output",
        settings={"task_names": ["subtract", "add"]},
    )

    assert forward != reverse
    assert forward.settings_dict()["task_names"] == ["add", "subtract"]
    assert reverse.settings_dict()["task_names"] == ["subtract", "add"]
    assert (
        MetricExtractionDefinition(
            definition_id="metrics",
            version="1",
            questions=(forward,),
        ).identity_hash()
        != MetricExtractionDefinition(
            definition_id="metrics",
            version="1",
            questions=(reverse,),
        ).identity_hash()
    )


@pytest.mark.parametrize("invalid", [math.inf, math.nan, {"values": {1, 2}}])
def test_metric_settings_reject_non_json_values(invalid: object) -> None:
    with pytest.raises(StrictJsonError):
        MetricQuestionBinding(
            metric="text_stats",
            on="output",
            settings={"invalid": invalid},
        )


def test_composite_identity_changes_with_each_component() -> None:
    sampling, _, _, procedure, aggregation, base = _components()
    changed_task_set = TaskSet(
        manifest_id="tasks",
        version="2",
        dataset_id="fixture",
        dataset_split="test",
        dataset_revision="r1",
        source_content_hash="a" * 64,
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
    assert changed.sampling_config_hash != sampling.config_identity_hash
    assert changed.config_identity_hash != base.config_identity_hash


@pytest.mark.parametrize("component_index", range(6))
def test_every_config_round_trips_and_rejects_forged_hash(
    component_index: int,
) -> None:
    config = _components()[component_index]
    restored = type(config).model_validate_json(config.model_dump_json())
    assert restored == config
    assert hash(restored) == hash(config)
    assert hash(restored.definition_ref) == hash(config.definition_ref)

    forged = config.model_dump(mode="json")
    forged["config_identity_hash"] = "0" * 64
    with pytest.raises(ValidationError, match="config identity hash mismatch"):
        type(config).model_validate(forged)

    malformed = config.model_dump(mode="json")
    malformed["config_identity_hash"] = "A" * 64
    with pytest.raises(ValidationError, match="lowercase 64-character"):
        type(config).model_validate(malformed)


@pytest.mark.parametrize("component_index", range(6))
def test_configs_reject_rehashed_noncanonical_definition_payloads(
    component_index: int,
) -> None:
    config = _components()[component_index]
    forged = config.model_dump(mode="json")
    ref = forged["definition_ref"]
    payload = ref["identity_payload"]
    payload["unexpected"] = True
    ref["identity_hash"] = identity_hash_for(
        schema=ref["schema_name"],
        payload=payload,
    )
    if component_index == 0:
        schema = SCHEMA_SAMPLING_CONFIG
        config_payload = {
            "definition_identity": ref["identity_hash"],
            "assignment": forged["assignment"],
            "task_set": forged["task_set"],
            "repeat_plan": forged["repeat_plan"],
        }
    elif component_index == 1:
        schema = SCHEMA_PREPROCESSING_CONFIG
        config_payload = {
            "definition_identity": ref["identity_hash"],
            "assignment": forged["assignment"],
            "steps": forged["steps"],
            "resolved_step_versions": forged["resolved_step_versions"],
        }
    elif component_index == 2:
        schema = SCHEMA_METRIC_EXTRACTION_CONFIG
        config_payload = {
            "definition_identity": ref["identity_hash"],
            "assignment": forged["assignment"],
            "questions": forged["questions"],
            "resolved_operator_versions": forged["resolved_operator_versions"],
        }
    elif component_index == 3:
        schema = SCHEMA_EVALUATION_PROCEDURE_CONFIG
        config_payload = {
            "definition_identity": ref["identity_hash"],
            "trace_source": forged["trace_source"],
            "preprocessing_config": forged["preprocessing_config_hash"],
            "metric_extraction_config": forged[
                "metric_extraction_config_hash"
            ],
        }
    elif component_index == 4:
        schema = SCHEMA_AGGREGATION_CONFIG
        config_payload = {
            "definition_identity": ref["identity_hash"],
            "assignment": forged["assignment"],
        }
    else:
        schema = SCHEMA_EVAL_CONFIG
        config_payload = {
            "definition_identity": ref["identity_hash"],
            "sampling_config": forged["sampling_config_hash"],
            "evaluation_procedure_config": forged[
                "evaluation_procedure_config_hash"
            ],
            "aggregation_config": forged["aggregation_config_hash"],
        }
    forged["config_identity_hash"] = identity_hash_for(
        schema=schema,
        payload=config_payload,
    )

    with pytest.raises(ValidationError):
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


def test_identity_json_is_deeply_immutable_and_round_trips() -> None:
    spec = VariableSpec(name="names", allowed=(["a"],))
    definition = MetricExtractionDefinition(
        definition_id="metrics",
        version="1",
        variables=(spec,),
        questions=(
            MetricQuestionBinding(
                metric="code_leakage",
                on="output",
                settings={"task_names": VariableReference(variable="names")},
            ),
        ),
    )
    identity_before = definition.identity_hash()
    with pytest.raises(AttributeError):
        spec.allowed[0].append("b")
    assert definition.identity_hash() == identity_before

    config = definition.materialize({"names": ["a"]})
    with pytest.raises(TypeError, match="do not support mutation"):
        config.definition_ref.identity_payload["version"] = "forged"
    with pytest.raises(TypeError):
        dict.__setitem__(
            config.definition_ref.identity_payload,
            "version",
            "forged",
        )
    with pytest.raises(TypeError, match="do not support mutation"):
        config.definition_ref.identity_payload._items = ()  # type: ignore[attr-defined]
    assert hash(config.definition_ref.identity_payload) == hash(
        definition.ref().identity_payload
    )
    task_names = dict(config.questions[0].settings)["task_names"]
    with pytest.raises(AttributeError):
        task_names.append("b")
    assert type(config).model_validate_json(config.model_dump_json()) == config

    preprocessing = PreprocessingStepBinding(
        instance_name="extract",
        step="extract_candidates",
        settings={"alternatives": ["fenced_blocks", "escaped_python"]},
    )
    alternatives = dict(preprocessing.settings)["alternatives"]
    with pytest.raises(AttributeError):
        alternatives.append("raw_text")


def test_substitution_rejects_concrete_metric_question_collisions() -> None:
    definition = MetricExtractionDefinition(
        definition_id="metrics",
        version="1",
        variables=(VariableSpec(name="a"), VariableSpec(name="b")),
        questions=(
            MetricQuestionBinding(
                metric="code_leakage",
                on="output",
                settings={"task_names": VariableReference(variable="a")},
            ),
            MetricQuestionBinding(
                metric="code_leakage",
                on="output",
                settings={"task_names": VariableReference(variable="b")},
            ),
        ),
    )
    with pytest.raises(ValueError, match="substituted metric questions"):
        definition.materialize({"a": [], "b": []})


def test_definition_ref_authenticates_payload_and_coordinates() -> None:
    config = _components()[2]
    ref = config.definition_ref
    assert type(ref).model_validate_json(ref.model_dump_json()) == ref

    forged_hash = ref.model_dump(mode="json")
    forged_hash["identity_hash"] = "0" * 64
    with pytest.raises(
        ValidationError, match="definition reference identity hash mismatch"
    ):
        type(ref).model_validate(forged_hash)

    forged_id = ref.model_dump(mode="json")
    forged_id["definition_id"] = "forged"
    with pytest.raises(ValidationError, match="id does not match"):
        type(ref).model_validate(forged_id)


def test_config_rejects_duplicate_serialized_assignment_names() -> None:
    config = _components()[4]
    forged = config.model_dump(mode="json")
    forged["assignment"].append(forged["assignment"][0])
    with pytest.raises(
        ValidationError, match="assignment names must be unique"
    ):
        type(config).model_validate(forged)


def test_typed_variable_substitution_controls_preprocessing_config() -> None:
    definition = PreprocessingDefinition(
        definition_id="variable-preprocessing",
        version="1",
        variables=(VariableSpec(name="tab_width", allowed=(2, 4)),),
        steps=(
            PreprocessingStepBinding(
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
    assert dict(width_two.steps[0].settings)["tab_width"] == 2
    assert dict(width_four.steps[0].settings)["tab_width"] == 4
    assert width_two.config_identity_hash != width_four.config_identity_hash
    width_two_trace = run_preprocessing(width_two, TextArtifact(text="\t"))
    width_four_trace = run_preprocessing(width_four, TextArtifact(text="\t"))
    assert width_two_trace.value("output") == TextArtifact(text="  ")
    assert width_four_trace.value("output") == TextArtifact(text="    ")
    assert (
        width_two_trace.producer.definition_hash
        == width_four_trace.producer.definition_hash
    )
    assert (
        width_two_trace.producer.preprocessing_config_hash
        != width_four_trace.producer.preprocessing_config_hash
    )
    with pytest.raises(VariableError, match="unassigned"):
        definition.materialize()


def test_definition_rejects_unused_and_undefined_variable_references() -> None:
    with pytest.raises(ValidationError, match="unused variable"):
        PreprocessingDefinition(
            definition_id="unused",
            version="1",
            variables=(VariableSpec(name="unused"),),
            steps=(),
        )
    with pytest.raises(ValidationError, match="undefined variable"):
        PreprocessingDefinition(
            definition_id="undefined",
            version="1",
            steps=(
                PreprocessingStepBinding(
                    instance_name="tabs",
                    step="expand_tabs",
                    settings={
                        "tab_width": VariableReference(variable="missing")
                    },
                ),
            ),
        )


@pytest.mark.parametrize("level", [True, 1.0])
def test_operator_settings_reject_numeric_aliases_before_hashing(
    level: object,
) -> None:
    with pytest.raises(ValidationError):
        MetricExtractionDefinition(
            definition_id="strict-settings",
            version="1",
            questions=(
                MetricQuestionBinding(
                    metric="compressed_length",
                    on="output",
                    settings={
                        "compression": {"method": "gzip", "level": level}
                    },
                ),
            ),
        )


def test_operator_defaults_are_canonical_identity_settings() -> None:
    omitted = MetricQuestionBinding(metric="code_leakage", on="output")
    explicit = MetricQuestionBinding(
        metric="code_leakage",
        on="output",
        settings={"task_names": []},
    )
    assert omitted.identity_hash() == explicit.identity_hash()
