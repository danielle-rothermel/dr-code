"""Definition -> Config lifecycle and identity tests for all six pairs."""

from __future__ import annotations

import pytest

from dr_code.eval.lifecycle import (
    AggregationDefinition,
    EvalDefinition,
    EvaluationProcedureDefinition,
    MetricExtractionDefinition,
    MetricQuestionBinding,
    PreprocessingDefinition,
    PreprocessingStepBinding,
    SamplingConfig,
    SamplingDefinition,
)
from dr_code.eval.variables import VariableError


def _sampling_config() -> SamplingConfig:
    definition = SamplingDefinition(definition_id="samp", version="1")
    return definition.materialize(
        {"task_set_hash": "ts1", "repeat_plan_hash": "rp1"}
    )


def _preprocessing_config():
    definition = PreprocessingDefinition(
        definition_id="pre",
        version="1",
        steps=(
            PreprocessingStepBinding(
                instance_name="sf", step="select_first"
            ),
        ),
    )
    return definition, definition.materialize()


def _metric_config():
    definition = MetricExtractionDefinition(
        definition_id="met",
        version="1",
        questions=(
            MetricQuestionBinding(metric="code_leakage", on="output"),
        ),
    )
    return definition, definition.materialize()


def test_config_carries_typed_definition_reference() -> None:
    definition = SamplingDefinition(definition_id="samp", version="1")
    config = _sampling_config()
    ref = config.definition_ref
    assert ref.definition_id == "samp"
    assert ref.version == "1"
    assert ref.identity_hash == definition.identity_hash()
    assert len(config.config_identity_hash) == 64
    assert config.config_identity_hash.islower()


def test_incomplete_assignment_rejected() -> None:
    definition = SamplingDefinition(definition_id="samp", version="1")
    with pytest.raises(VariableError, match="unassigned"):
        definition.materialize({"task_set_hash": "ts1"})


def test_unknown_variable_rejected() -> None:
    definition = SamplingDefinition(definition_id="samp", version="1")
    with pytest.raises(VariableError, match="unknown"):
        definition.materialize(
            {
                "task_set_hash": "ts1",
                "repeat_plan_hash": "rp1",
                "bogus": "x",
            }
        )


def test_out_of_range_value_rejected() -> None:
    definition = AggregationDefinition(definition_id="agg", version="1")
    with pytest.raises(VariableError, match="allowed"):
        definition.materialize({"reduction": "median"})


def test_definition_is_sole_owner_via_materialize() -> None:
    # Config is only constructed through the Definition's materialize;
    # the private _create signals the ownership boundary.
    definition, config = _preprocessing_config()
    assert config.definition_ref.identity_hash == definition.identity_hash()


def test_preprocessing_identity_folds_resolved_step_version() -> None:
    _definition, config = _preprocessing_config()
    assert config.resolved_step_versions == (("sf", "select_first", "1"),)


def test_metric_identity_folds_resolved_operator_version() -> None:
    _definition, config = _metric_config()
    # code_leakage resolves to VERSION "2" in the operator registry.
    assert config.resolved_operator_versions == (("code_leakage", "2"),)


def test_procedure_hash_covers_component_configs() -> None:
    _pd, preprocessing = _preprocessing_config()
    _md, metric = _metric_config()
    procedure_def = EvaluationProcedureDefinition(
        definition_id="proc", version="1"
    )
    procedure = procedure_def.materialize(
        preprocessing=preprocessing,
        metric_extraction=metric,
        assignment={"zero_denominator": "not_applicable"},
    )
    assert (
        procedure.preprocessing_config_hash
        == preprocessing.config_identity_hash
    )
    assert (
        procedure.metric_extraction_config_hash
        == metric.config_identity_hash
    )


def _full_eval_config(*, zero_denominator: str = "not_applicable"):
    sampling = _sampling_config()
    _pd, preprocessing = _preprocessing_config()
    _md, metric = _metric_config()
    procedure = EvaluationProcedureDefinition(
        definition_id="proc", version="1"
    ).materialize(
        preprocessing=preprocessing,
        metric_extraction=metric,
        assignment={"zero_denominator": zero_denominator},
    )
    aggregation = AggregationDefinition(
        definition_id="agg", version="1"
    ).materialize({"reduction": "mean"})
    eval_config = EvalDefinition(
        definition_id="ev", version="1"
    ).materialize(
        sampling=sampling,
        evaluation_procedure=procedure,
        aggregation=aggregation,
    )
    return sampling, procedure, aggregation, eval_config


def test_eval_config_hash_covers_all_three_components() -> None:
    sampling, procedure, aggregation, eval_config = _full_eval_config()
    assert eval_config.sampling_config_hash == sampling.config_identity_hash
    assert (
        eval_config.evaluation_procedure_config_hash
        == procedure.config_identity_hash
    )
    assert (
        eval_config.aggregation_config_hash
        == aggregation.config_identity_hash
    )
    assert len(eval_config.config_identity_hash) == 64


def test_procedure_change_alters_eval_config_hash() -> None:
    _s1, _p1, _a1, base = _full_eval_config(
        zero_denominator="not_applicable"
    )
    _s2, _p2, _a2, changed = _full_eval_config(zero_denominator="error")
    # A Procedure Config change alters both the procedure identity and the
    # composite eval config identity.
    assert (
        base.evaluation_procedure_config_hash
        != changed.evaluation_procedure_config_hash
    )
    assert base.config_identity_hash != changed.config_identity_hash


def test_sampling_only_change_alters_only_eval_config_hash() -> None:
    _s1, procedure1, _a1, base = _full_eval_config()

    # Change sampling only; procedure identity is unchanged.
    sampling2 = SamplingDefinition(
        definition_id="samp", version="1"
    ).materialize({"task_set_hash": "ts2", "repeat_plan_hash": "rp1"})
    _pd, preprocessing = _preprocessing_config()
    _md, metric = _metric_config()
    procedure2 = EvaluationProcedureDefinition(
        definition_id="proc", version="1"
    ).materialize(
        preprocessing=preprocessing,
        metric_extraction=metric,
        assignment={"zero_denominator": "not_applicable"},
    )
    aggregation = AggregationDefinition(
        definition_id="agg", version="1"
    ).materialize({"reduction": "mean"})
    changed = EvalDefinition(
        definition_id="ev", version="1"
    ).materialize(
        sampling=sampling2,
        evaluation_procedure=procedure2,
        aggregation=aggregation,
    )
    assert (
        procedure1.config_identity_hash == procedure2.config_identity_hash
    )
    assert base.config_identity_hash != changed.config_identity_hash


def test_config_hash_is_deterministic() -> None:
    first = _sampling_config()
    second = _sampling_config()
    assert first.config_identity_hash == second.config_identity_hash
