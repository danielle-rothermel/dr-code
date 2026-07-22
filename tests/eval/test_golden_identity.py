"""Golden Identity Hashes, including resolved-operator-version changes.

These pin the exact 64-char SHA-256 Identity Hashes so an accidental
payload change (field order, wrapping, schema name) is caught. A change
here is a deliberate breaking identity change and must be reviewed.
"""

from __future__ import annotations

import pytest

from dr_code.eval import resolved_versions
from dr_code.eval.lifecycle import (
    AggregationDefinition,
    EvalDefinition,
    EvaluationProcedureDefinition,
    MetricExtractionDefinition,
    MetricQuestionBinding,
    PreprocessingDefinition,
    PreprocessingStepBinding,
    SamplingDefinition,
)

GOLDEN_SAMPLING = (
    "bfcbb4eb54f32308ab5ffa49e864d03d94aeab9a63adeaef0fb18acebd93ff03"
)
GOLDEN_PREPROCESSING = (
    "a6ee0cbe127ecc284ec325cf3e4440d4ba12b08ff6af3113cf0e329bb0102437"
)
GOLDEN_METRIC = (
    "f9546c02e8a2bd91e387044cde3e88f6066eebb6cb2e40d8cd75b98015ead16b"
)
GOLDEN_PROCEDURE = (
    "0f9b8e117f3a2bdd26bf67cc61903c458d09a9ccee29fee6fcb39601e2850340"
)
GOLDEN_AGGREGATION = (
    "373b234c512b09f897fc42107cb7f6d7b848c5f6a48e4dc70d0ba39727e3522b"
)
GOLDEN_EVAL = (
    "e6a51cac4411be34428682c5e6652ae66a5f3eb8ce774a36e7f3cd0f07f49e9d"
)


def _sampling():
    return SamplingDefinition(definition_id="samp", version="1").materialize(
        {"task_set_hash": "ts1", "repeat_plan_hash": "rp1"}
    )


def _preprocessing():
    return PreprocessingDefinition(
        definition_id="pre",
        version="1",
        steps=(
            PreprocessingStepBinding(instance_name="sf", step="select_first"),
        ),
    ).materialize()


def _metric():
    return MetricExtractionDefinition(
        definition_id="met",
        version="1",
        questions=(MetricQuestionBinding(metric="code_leakage", on="output"),),
    ).materialize()


def _procedure(preprocessing, metric):
    return EvaluationProcedureDefinition(
        definition_id="proc", version="1"
    ).materialize(
        preprocessing=preprocessing,
        metric_extraction=metric,
        assignment={"zero_denominator": "not_applicable"},
    )


def _aggregation():
    return AggregationDefinition(definition_id="agg", version="1").materialize(
        {"reduction": "mean"}
    )


def test_golden_config_hashes() -> None:
    preprocessing = _preprocessing()
    metric = _metric()
    procedure = _procedure(preprocessing, metric)
    assert _sampling().config_identity_hash == GOLDEN_SAMPLING
    assert preprocessing.config_identity_hash == GOLDEN_PREPROCESSING
    assert metric.config_identity_hash == GOLDEN_METRIC
    assert procedure.config_identity_hash == GOLDEN_PROCEDURE
    assert _aggregation().config_identity_hash == GOLDEN_AGGREGATION
    eval_config = EvalDefinition(
        definition_id="ev", version="1"
    ).materialize(
        sampling=_sampling(),
        evaluation_procedure=procedure,
        aggregation=_aggregation(),
    )
    assert eval_config.config_identity_hash == GOLDEN_EVAL


def test_resolved_operator_version_change_changes_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bumping an operator's resolved VERSION changes the Config identity
    even though the declared question set is unchanged."""

    baseline = _metric().config_identity_hash
    assert baseline == GOLDEN_METRIC

    original = resolved_versions.resolved_operator_version

    def bumped(metric_name: str) -> str:
        if metric_name == "code_leakage":
            return "999"
        return original(metric_name)

    monkeypatch.setattr(
        "dr_code.eval.lifecycle.resolved_operator_version", bumped
    )
    changed = _metric().config_identity_hash
    assert changed != baseline


def test_resolved_step_version_change_changes_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _preprocessing().config_identity_hash
    assert baseline == GOLDEN_PREPROCESSING

    original = resolved_versions.resolved_step_version

    def bumped(step_name: str) -> str:
        if step_name == "select_first":
            return "999"
        return original(step_name)

    monkeypatch.setattr(
        "dr_code.eval.lifecycle.resolved_step_version", bumped
    )
    changed = _preprocessing().config_identity_hash
    assert changed != baseline
