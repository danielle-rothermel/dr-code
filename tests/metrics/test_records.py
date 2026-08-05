"""Metrics vocabulary and record contracts.

Covers ``MetricName`` / ``RecordStatus`` / ``MetricFactUnit`` (StrEnum
members), the three record variants of the ``MetricRecord`` discriminated
union, their shared identity, the golden serialized literals pinning the
persisted wire format, registry-free deserialization, and ``record_rows``
flattening with ``"{metric}.{name}"`` fact columns and ``.unit`` siblings
that prevent collisions across metrics.
"""

from __future__ import annotations

import json

import pytest

EXPECTED_METRIC_NAMES = {
    "text_stats",
    "code_leakage",
    "parse_outcome",
    "ast_stats",
    "compressed_length",
    "code_test",
}
EXPECTED_RECORD_STATUSES = {
    "measured",
    "not_applicable",
    "operator_failure",
}
EXPECTED_FACT_UNITS = {
    "count",
    "ratio",
    "percent",
    "characters",
    "bytes",
    "lines",
    "depth",
    "boolean",
    "identifier",
    "text",
}


# ===========================================================================
# MetricName / RecordStatus / MetricFactUnit enums.
# ===========================================================================


def test_metric_name_is_a_strenum_of_the_six_families() -> None:
    from dr_code.metrics import MetricName

    assert {name.value for name in MetricName} == EXPECTED_METRIC_NAMES


def test_metric_name_members_round_trip_through_their_string_values() -> None:
    from dr_code.metrics import MetricName

    for value in EXPECTED_METRIC_NAMES:
        name = MetricName(value)
        assert name.value == value
        assert name == str(name)  # StrEnum serializes to plain JSON


def test_record_status_is_the_three_way_answer_taxonomy() -> None:
    from dr_code.metrics import RecordStatus

    assert {
        status.value for status in RecordStatus
    } == EXPECTED_RECORD_STATUSES


def test_record_status_members_round_trip_through_their_string_values() -> (
    None
):
    from dr_code.metrics import RecordStatus

    for value in EXPECTED_RECORD_STATUSES:
        status = RecordStatus(value)
        assert status.value == value


def test_metric_fact_unit_is_the_closed_unit_vocabulary() -> None:
    from dr_code.metrics import MetricFactUnit

    assert {unit.value for unit in MetricFactUnit} == EXPECTED_FACT_UNITS


def test_every_operator_declares_a_unit_for_every_fact_it_emits() -> None:
    """Units are declared at the operator, not guessed at assembly time."""
    from dr_code.metrics import MetricFactUnit
    from dr_code.metrics.operators.ast_stats import AstStatsResult
    from dr_code.metrics.operators.code_leakage import CodeLeakageResult
    from dr_code.metrics.operators.code_test import CodeTestResult
    from dr_code.metrics.operators.compressed_length import (
        CompressedLengthResult,
        CompressedLengthWithReferenceResult,
    )
    from dr_code.metrics.operators.parse_outcome import ParseOutcomeResult
    from dr_code.metrics.operators.text_stats import TextStatsResult

    for result_class in (
        AstStatsResult,
        CodeLeakageResult,
        CodeTestResult,
        CompressedLengthResult,
        CompressedLengthWithReferenceResult,
        ParseOutcomeResult,
        TextStatsResult,
    ):
        assert set(result_class.model_fields) == set(result_class.UNITS), (
            result_class.__name__
        )
        for unit in result_class.UNITS.values():
            assert isinstance(unit, MetricFactUnit), result_class.__name__


def test_a_result_field_without_a_declared_unit_fails_loudly() -> None:
    """A new fact cannot reach a record carrying an unlabelled value."""
    from dr_code.metrics.operators.base import OperatorResult

    class UndeclaredResult(OperatorResult):
        UNITS = {}

        widget_count: int

    with pytest.raises(ValueError, match="declares no unit"):
        UndeclaredResult(widget_count=1).to_facts()


# ===========================================================================
# Record builders.
# ===========================================================================


def _producer():
    from dr_code.trace import (
        ComponentCoordinate,
        PreprocessingDefinitionCoordinate,
        PreprocessingTraceProducer,
        StepCoordinate,
    )

    return PreprocessingTraceProducer(
        definition=PreprocessingDefinitionCoordinate(
            definition_id="pre",
            version="v1",
            steps=(
                StepCoordinate(
                    instance_name="step",
                    component=ComponentCoordinate(
                        registered_name="normalize_unicode",
                        version="0",
                    ),
                ),
            ),
        )
    )


def _question_coordinate(metric=None, on_key="input", settings=()):
    from dr_code.metrics import MetricName, MetricQuestionCoordinate

    return MetricQuestionCoordinate(
        metric=MetricName.TEXT_STATS if metric is None else metric,
        on_key=on_key,
        settings=settings,
    )


def _identity(question=None, **overrides):
    from dr_code.metrics import (
        MetricRecordIdentity,
        MetricsDefinitionCoordinate,
    )

    question = question if question is not None else _question_coordinate()
    base: dict[str, object] = {
        "question": question,
        "metric_version": "1",
        "producer": _producer(),
        "metrics_definition": MetricsDefinitionCoordinate(
            definition_id="def",
            version="1",
            questions=(question,),
        ),
    }
    base.update(overrides)
    return MetricRecordIdentity(**base)


def _fact(name="character_count", value=4, unit=None):
    from dr_code.metrics import MetricFact, MetricFactUnit

    return MetricFact(
        name=name,
        value=value,
        unit=MetricFactUnit.COUNT if unit is None else unit,
    )


def _measured(**overrides):
    from dr_code.metrics import MeasuredRecord

    base: dict[str, object] = {
        "identity": _identity(),
        "facts": (_fact(),),
    }
    base.update(overrides)
    return MeasuredRecord(**base)


def _absent():
    from dr_code.trace import Absent

    return Absent(
        failed_step="extract",
        failure_code="no_candidates_extracted",
        cause="no code extracted",
        propagated_through=("clean",),
    )


def _not_applicable(**overrides):
    from dr_code.metrics import NotApplicableRecord

    base: dict[str, object] = {
        "identity": _identity(),
        "absence": _absent(),
    }
    base.update(overrides)
    return NotApplicableRecord(**base)


def _operator_failure(**overrides):
    from dr_code.metrics import OperatorFailure, OperatorFailureRecord

    base: dict[str, object] = {
        "identity": _identity(),
        "failure": OperatorFailure(
            failure_type="ValueError", failure_message="boom"
        ),
    }
    base.update(overrides)
    return OperatorFailureRecord(**base)


# ===========================================================================
# Schema version and the closed discriminated union.
# ===========================================================================


def test_every_record_variant_carries_the_initial_schema_version() -> None:
    from dr_code.metrics import METRIC_RECORD_SCHEMA_VERSION

    assert METRIC_RECORD_SCHEMA_VERSION == 1
    for record in (_measured(), _not_applicable(), _operator_failure()):
        assert record.schema_version == METRIC_RECORD_SCHEMA_VERSION


def test_records_reject_any_other_schema_version() -> None:
    from pydantic import ValidationError

    from dr_code.metrics import MeasuredRecord

    with pytest.raises(ValidationError):
        MeasuredRecord(
            schema_version=2,  # type: ignore[arg-type]
            identity=_identity(),
            facts=(_fact(),),
        )


def test_record_union_is_discriminated_by_status() -> None:
    from dr_code.metrics import (
        METRIC_RECORD_ADAPTER,
        MeasuredRecord,
        NotApplicableRecord,
        OperatorFailureRecord,
    )

    for record, expected in (
        (_measured(), MeasuredRecord),
        (_not_applicable(), NotApplicableRecord),
        (_operator_failure(), OperatorFailureRecord),
    ):
        restored = METRIC_RECORD_ADAPTER.validate_json(
            record.model_dump_json()
        )
        assert type(restored) is expected
        assert restored == record


def test_record_variant_fields_are_the_documented_schema() -> None:
    from dr_code.metrics import (
        MeasuredRecord,
        NotApplicableRecord,
        OperatorFailureRecord,
    )

    shared = {"schema_version", "status", "identity"}
    assert set(MeasuredRecord.model_fields) == shared | {"facts"}
    assert set(NotApplicableRecord.model_fields) == shared | {"absence"}
    assert set(OperatorFailureRecord.model_fields) == shared | {"failure"}


def test_records_are_frozen() -> None:
    record = _measured()
    with pytest.raises(Exception):  # noqa: PT011 — FrozenModel raises
        record.facts = record.facts  # type: ignore[misc]


def test_equal_records_compare_equal() -> None:
    """Records participate in structured comparison across runs."""
    assert _measured() == _measured()
    assert _not_applicable() == _not_applicable()
    assert _operator_failure() == _operator_failure()


def test_measured_and_not_applicable_records_are_never_equal() -> None:
    assert _measured() != _not_applicable()
    assert _not_applicable() != _operator_failure()


# ===========================================================================
# Shared identity: composition and internal consistency.
# ===========================================================================


def test_identity_carries_the_question_and_both_coordinates() -> None:
    identity = _identity()
    assert identity.question.metric.value == "text_stats"
    assert identity.question.on_key == "input"
    assert identity.metric_version == "1"
    assert identity.producer.kind == "preprocessing"
    assert identity.producer.definition.definition_id == "pre"
    assert identity.producer.definition.version == "v1"
    assert identity.metrics_definition.definition_id == "def"
    assert identity.metrics_definition.version == "1"
    assert identity.metrics_definition.questions == (identity.question,)


def test_identity_must_name_a_question_its_definition_declares() -> None:
    from pydantic import ValidationError

    from dr_code.metrics import MetricName, MetricsDefinitionCoordinate

    elsewhere = MetricsDefinitionCoordinate(
        definition_id="def",
        version="1",
        questions=(
            _question_coordinate(
                metric=MetricName.TEXT_STATS, on_key="output"
            ),
        ),
    )
    with pytest.raises(ValidationError):
        _identity(
            question=_question_coordinate(metric=MetricName.AST_STATS),
            metrics_definition=elsewhere,
        )


def test_identity_on_key_must_match_the_declared_question() -> None:
    from pydantic import ValidationError

    from dr_code.metrics import MetricsDefinitionCoordinate

    definition = MetricsDefinitionCoordinate(
        definition_id="def",
        version="1",
        questions=(_question_coordinate(on_key="output"),),
    )
    with pytest.raises(ValidationError):
        _identity(
            question=_question_coordinate(on_key="input"),
            metrics_definition=definition,
        )


def test_identity_settings_must_match_the_declared_question() -> None:
    from pydantic import ValidationError

    from dr_code.metrics import MetricName, MetricsDefinitionCoordinate
    from dr_code.trace import ComponentSetting

    declared = _question_coordinate(
        metric=MetricName.CODE_LEAKAGE,
        settings=(ComponentSetting(name="task_names", value=("declared",)),),
    )
    definition = MetricsDefinitionCoordinate(
        definition_id="def",
        version="1",
        questions=(declared,),
    )
    with pytest.raises(ValidationError):
        _identity(
            question=_question_coordinate(
                metric=MetricName.CODE_LEAKAGE,
                settings=(
                    ComponentSetting(name="task_names", value=("other",)),
                ),
            ),
            metrics_definition=definition,
        )


def test_identity_matches_a_question_among_several() -> None:
    from dr_code.metrics import MetricName, MetricsDefinitionCoordinate

    question = _question_coordinate(metric=MetricName.TEXT_STATS)
    definition = MetricsDefinitionCoordinate(
        definition_id="def",
        version="1",
        questions=(
            _question_coordinate(metric=MetricName.AST_STATS, on_key="output"),
            question,
        ),
    )
    identity = _identity(question=question, metrics_definition=definition)
    assert identity.question is question


def test_definition_coordinate_rejects_duplicate_questions() -> None:
    from pydantic import ValidationError

    from dr_code.metrics import MetricsDefinitionCoordinate

    question = _question_coordinate()
    with pytest.raises(ValidationError):
        MetricsDefinitionCoordinate(
            definition_id="def",
            version="1",
            questions=(question, question),
        )


# ===========================================================================
# MetricFact: strict finite scalars with an explicit unit.
# ===========================================================================


def test_metric_fact_accepts_every_scalar_type() -> None:
    from dr_code.metrics import MetricFactUnit

    facts = (
        _fact(name="int_val", value=42),
        _fact(name="float_val", value=3.14, unit=MetricFactUnit.RATIO),
        _fact(name="str_val", value="hello", unit=MetricFactUnit.IDENTIFIER),
        _fact(name="bool_val", value=True, unit=MetricFactUnit.BOOLEAN),
        _fact(name="none_val", value=None, unit=MetricFactUnit.TEXT),
    )
    record = _measured(facts=facts)
    assert [fact.value for fact in record.facts] == [
        42,
        3.14,
        "hello",
        True,
        None,
    ]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_metric_fact_rejects_non_finite_values(value: float) -> None:
    from pydantic import ValidationError

    from dr_code.metrics import MetricFactUnit

    with pytest.raises(ValidationError):
        _fact(name="ratio", value=value, unit=MetricFactUnit.RATIO)


def test_metric_fact_requires_a_unit_from_the_closed_vocabulary() -> None:
    from pydantic import ValidationError

    from dr_code.metrics import MetricFact

    with pytest.raises(ValidationError):
        MetricFact(name="count", value=1, unit="furlongs")


@pytest.mark.parametrize("name", ["character_count.unit", "a.b", ".", "x."])
def test_metric_fact_rejects_a_dotted_name(name: str) -> None:
    # A fact named ``x.unit`` would occupy fact ``x``'s unit column in
    # ``record_rows``; banning the separator is what makes that scheme
    # collision-free.
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="must not contain"):
        _fact(name=name, value=1)


def test_facts_preserve_operator_declaration_order() -> None:
    facts = (
        _fact(name="second", value=2),
        _fact(name="first", value=1),
    )
    assert [fact.name for fact in _measured(facts=facts).facts] == [
        "second",
        "first",
    ]


def test_measured_records_require_at_least_one_fact() -> None:
    """An empty fact tuple is indistinguishable from the no-answer shape."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _measured(facts=())


def test_measured_records_reject_duplicate_fact_names() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _measured(facts=(_fact(name="count"), _fact(name="count")))


# ===========================================================================
# Not-applicable and operator-failure payloads.
# ===========================================================================


def test_not_applicable_record_nests_the_complete_absent() -> None:
    record = _not_applicable()
    assert record.absence == _absent()
    assert record.absence.failed_step == "extract"
    assert record.absence.failure_code == "no_candidates_extracted"
    assert record.absence.cause == "no code extracted"
    assert record.absence.propagated_through == ("clean",)


def test_operator_failure_record_nests_a_structured_failure() -> None:
    record = _operator_failure()
    assert record.failure.failure_type == "ValueError"
    assert record.failure.failure_message == "boom"


def test_operator_failure_requires_type_and_message() -> None:
    from pydantic import ValidationError

    from dr_code.metrics import OperatorFailure

    with pytest.raises(ValidationError):
        OperatorFailure(failure_type="ValueError")
    with pytest.raises(ValidationError):
        OperatorFailure(failure_message="boom")


# ===========================================================================
# Golden serialized literals: the persisted wire format.
# ===========================================================================

# The exact persisted shape of a representative measured record. These keys
# and values are the storage contract: changing any of them changes what
# archived records mean, so this literal is pinned rather than derived from
# the model's field names.
_GOLDEN_MEASURED_RECORD = {
    "schema_version": 1,
    "status": "measured",
    "identity": {
        "question": {
            "metric": "code_leakage",
            "on_key": "output",
            "settings": [
                {"name": "task_names", "value": ["add_one"]},
            ],
        },
        "metric_version": "0",
        "producer": {
            "kind": "preprocessing",
            "definition": {
                "definition_id": "pre",
                "version": "v1",
                "steps": [
                    {
                        "instance_name": "step",
                        "component": {
                            "registered_name": "normalize_unicode",
                            "version": "0",
                            "settings": [],
                        },
                    }
                ],
            },
        },
        "metrics_definition": {
            "definition_id": "def",
            "version": "1",
            "questions": [
                {
                    "metric": "code_leakage",
                    "on_key": "output",
                    "settings": [
                        {"name": "task_names", "value": ["add_one"]},
                    ],
                }
            ],
        },
    },
    "facts": [
        {"name": "keyword_count", "value": 2, "unit": "count"},
        {
            "name": "punctuation_density",
            "value": 0.25,
            "unit": "ratio",
        },
    ],
}


def _golden_record():
    from dr_code.metrics import MetricFactUnit, MetricName
    from dr_code.trace import ComponentSetting

    question = _question_coordinate(
        metric=MetricName.CODE_LEAKAGE,
        on_key="output",
        settings=(ComponentSetting(name="task_names", value=("add_one",)),),
    )
    return _measured(
        identity=_identity(question=question, metric_version="0"),
        facts=(
            _fact(name="keyword_count", value=2),
            _fact(
                name="punctuation_density",
                value=0.25,
                unit=MetricFactUnit.RATIO,
            ),
        ),
    )


def test_measured_record_serializes_to_the_golden_literals() -> None:
    assert (
        json.loads(_golden_record().model_dump_json())
        == _GOLDEN_MEASURED_RECORD
    )


def test_golden_literals_load_back_to_an_equal_record() -> None:
    from dr_code.metrics import METRIC_RECORD_ADAPTER

    restored = METRIC_RECORD_ADAPTER.validate_python(_GOLDEN_MEASURED_RECORD)
    assert restored == _golden_record()


# The two non-measured variants share the measured golden's identity block,
# so these pin what distinguishes them: the status discriminator and the
# variant-specific payload keys.
_GOLDEN_NOT_APPLICABLE_RECORD = {
    "schema_version": 1,
    "status": "not_applicable",
    "identity": _GOLDEN_MEASURED_RECORD["identity"],
    "absence": {
        "kind": "absent",
        "failed_step": "extract",
        "failure_code": "no_candidates_extracted",
        "cause": "no code extracted",
        "propagated_through": ["clean"],
    },
}

_GOLDEN_OPERATOR_FAILURE_RECORD = {
    "schema_version": 1,
    "status": "operator_failure",
    "identity": _GOLDEN_MEASURED_RECORD["identity"],
    "failure": {
        "failure_type": "ValueError",
        "failure_message": "boom",
    },
}


def _golden_not_applicable():
    return _not_applicable(identity=_golden_record().identity)


def _golden_operator_failure():
    return _operator_failure(identity=_golden_record().identity)


def test_not_applicable_record_serializes_to_the_golden_literals() -> None:
    assert (
        json.loads(_golden_not_applicable().model_dump_json())
        == _GOLDEN_NOT_APPLICABLE_RECORD
    )


def test_golden_not_applicable_literals_load_back_equal() -> None:
    from dr_code.metrics import METRIC_RECORD_ADAPTER

    restored = METRIC_RECORD_ADAPTER.validate_python(
        _GOLDEN_NOT_APPLICABLE_RECORD
    )
    assert restored == _golden_not_applicable()


def test_operator_failure_record_serializes_to_the_golden_literals() -> None:
    assert (
        json.loads(_golden_operator_failure().model_dump_json())
        == _GOLDEN_OPERATOR_FAILURE_RECORD
    )


def test_golden_operator_failure_literals_load_back_equal() -> None:
    from dr_code.metrics import METRIC_RECORD_ADAPTER

    restored = METRIC_RECORD_ADAPTER.validate_python(
        _GOLDEN_OPERATOR_FAILURE_RECORD
    )
    assert restored == _golden_operator_failure()


# ===========================================================================
# Registry-free deserialization: archived records outlive the registry.
# ===========================================================================


class _PoisonedRegistry:
    """A registry that fails any read, however it is reached.

    An empty registry only proves a lookup found nothing; this proves no
    lookup happens at all, which is the actual guarantee.
    """

    def __getitem__(self, key: object) -> object:
        raise AssertionError(
            f"record deserialization consulted the registry: [{key!r}]"
        )

    def get(self, key: object, default: object = None) -> object:
        raise AssertionError(
            f"record deserialization consulted the registry: get({key!r})"
        )

    def __contains__(self, key: object) -> bool:
        raise AssertionError(
            f"record deserialization consulted the registry: {key!r} in ..."
        )

    def __iter__(self) -> object:
        raise AssertionError("record deserialization iterated the registry")


def test_record_loads_when_its_metric_is_absent_from_the_live_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Archived records stay loadable after the registry moves on."""
    from dr_code.metrics import METRIC_RECORD_ADAPTER
    import dr_code.metrics.registry as registry_module

    payload = json.dumps(_GOLDEN_MEASURED_RECORD)
    monkeypatch.setattr(
        registry_module, "REGISTRY", _PoisonedRegistry(), raising=True
    )

    restored = METRIC_RECORD_ADAPTER.validate_json(payload)
    assert restored.identity.question.metric.value == "code_leakage"
    assert restored.identity.question.settings[0].name == "task_names"
    assert restored.identity.question.settings[0].value == ("add_one",)


def test_record_loads_settings_the_live_operator_no_longer_accepts() -> None:
    """Settings persist as bounded entries, not as a live settings model."""
    from dr_code.metrics import METRIC_RECORD_ADAPTER

    payload = json.loads(json.dumps(_GOLDEN_MEASURED_RECORD))
    retired = [{"name": "retired_setting", "value": 7}]
    payload["identity"]["question"]["settings"] = retired
    payload["identity"]["metrics_definition"]["questions"][0]["settings"] = (
        retired
    )

    restored = METRIC_RECORD_ADAPTER.validate_python(payload)
    assert restored.identity.question.settings[0].name == "retired_setting"
    assert restored.identity.question.settings[0].value == 7


def test_engine_produced_records_round_trip(text_trace) -> None:
    """Whatever the engine emits is loadable; the rule mirrors its
    derivation."""
    from dr_code.metrics import METRIC_RECORD_ADAPTER

    for record in _engine_records(text_trace):
        assert (
            METRIC_RECORD_ADAPTER.validate_json(record.model_dump_json())
            == record
        )


def _engine_records(text_trace) -> tuple[object, ...]:
    from dr_code.metrics import MetricName, MetricQuestion, MetricsDefinition
    from dr_code.metrics.engine.engine import extract_metrics

    definition = MetricsDefinition(
        definition_id="round-trip",
        version="1",
        questions=(
            MetricQuestion(
                metric=MetricName.COMPRESSED_LENGTH,
                on="output",
                settings={"compression": {"method": "gzip", "level": 9}},
            ),
            MetricQuestion(
                metric=MetricName.CODE_LEAKAGE,
                on="output",
                settings={"task_names": ["add_one"]},
            ),
        ),
    )
    return extract_metrics(definition, text_trace("hello world"))


@pytest.mark.parametrize(
    "metric_value,settings",
    [
        pytest.param(
            "compressed_length",
            {"compression": {"method": "gzip", "level": 9}},
            id="compressed_length-gzip",
        ),
        pytest.param(
            "compressed_length",
            {"compression": {"method": "zstd", "level": 3}},
            id="compressed_length-zstd",
        ),
        pytest.param(
            "code_leakage",
            {"task_names": ["alpha", "beta"]},
            id="code_leakage-task-names",
        ),
    ],
)
def test_every_status_shape_round_trips_with_non_trivial_settings(
    metric_value: str, settings: dict[str, object]
) -> None:
    from dr_code.metrics import (
        METRIC_RECORD_ADAPTER,
        MetricName,
        MetricQuestion,
        MetricQuestionCoordinate,
    )

    question = MetricQuestionCoordinate.of(
        MetricQuestion(
            metric=MetricName(metric_value), on="input", settings=settings
        )
    )
    identity = _identity(question=question)
    for record in (
        _measured(identity=identity),
        _not_applicable(identity=identity),
        _operator_failure(identity=identity),
    ):
        restored = METRIC_RECORD_ADAPTER.validate_json(
            record.model_dump_json()
        )
        assert restored == record
        assert restored.identity.question.settings == question.settings


# ===========================================================================
# record_rows: flat rows with metric-prefixed fact columns.
# ===========================================================================


def test_record_rows_returns_one_row_per_record() -> None:
    from dr_code.metrics import MetricName, record_rows

    other = _question_coordinate(metric=MetricName.AST_STATS)
    rows = record_rows(
        [_measured(), _measured(identity=_identity(question=other))]
    )
    assert len(rows) == 2
    assert all(isinstance(row, dict) for row in rows)


def test_record_rows_empty_input_returns_empty_list() -> None:
    from dr_code.metrics import record_rows

    assert record_rows([]) == []


def test_record_rows_prefix_fact_columns_with_metric_and_name() -> None:
    from dr_code.metrics import record_rows

    row = record_rows(
        [
            _measured(
                facts=(
                    _fact(name="character_count", value=4),
                    _fact(name="word_count", value=1),
                )
            )
        ]
    )[0]
    assert row["text_stats.character_count"] == 4
    assert row["text_stats.word_count"] == 1
    # Raw fact names never appear unprefixed, avoiding cross-metric
    # collisions.
    assert "character_count" not in row
    assert "word_count" not in row


def test_record_rows_carry_each_facts_unit_in_a_sibling_column() -> None:
    from dr_code.metrics import MetricFactUnit, record_rows

    row = record_rows(
        [
            _measured(
                facts=(_fact(name="byte_count", unit=MetricFactUnit.BYTES),)
            )
        ]
    )[0]
    assert row["text_stats.byte_count.unit"] == MetricFactUnit.BYTES
    # The unit stays out of the value column name, so the same fact lines up
    # across rows regardless of the unit it was measured in.
    assert "text_stats.byte_count.bytes" not in row


def test_record_rows_include_identity_and_lineage_columns() -> None:
    from dr_code.metrics import MetricName, RecordStatus, record_rows

    row = record_rows([_measured()])[0]
    assert row["schema_version"] == 1
    assert row["metric"] == MetricName.TEXT_STATS
    assert row["metric_version"] == "1"
    assert row["on_key"] == "input"
    assert row["question_settings"] == ()
    assert row["producer"]["definition"]["definition_id"] == "pre"
    assert row["metrics_definition"]["definition_id"] == "def"
    assert row["metrics_definition"]["version"] == "1"
    assert row["status"] == RecordStatus.MEASURED


def test_record_rows_fact_columns_are_collision_free_across_metrics() -> None:
    from dr_code.metrics import MetricName, record_rows

    ast = _question_coordinate(metric=MetricName.AST_STATS)
    rows = record_rows(
        [
            _measured(facts=(_fact(name="count", value=1),)),
            _measured(
                identity=_identity(question=ast),
                facts=(_fact(name="count", value=2),),
            ),
        ]
    )
    assert rows[0]["text_stats.count"] == 1
    assert rows[1]["ast_stats.count"] == 2


def test_record_rows_status_column_distinguishes_absence_from_zero() -> None:
    """Not-applicable ≠ measured zero: a status column, not a magic value."""
    from dr_code.metrics import RecordStatus, record_rows

    rows = record_rows(
        [
            _measured(facts=(_fact(name="count", value=0),)),
            _not_applicable(),
        ]
    )
    assert rows[0]["status"] == RecordStatus.MEASURED
    assert rows[0]["text_stats.count"] == 0
    assert rows[1]["status"] == RecordStatus.NOT_APPLICABLE
    assert "text_stats.count" not in rows[1]
    assert rows[1]["absence"]["failed_step"] == "extract"
    assert rows[1]["absence"]["failure_code"] == "no_candidates_extracted"
    assert rows[1]["absence"]["cause"] == "no code extracted"


def test_record_rows_carry_the_operator_failure_payload() -> None:
    from dr_code.metrics import record_rows

    row = record_rows([_operator_failure()])[0]
    assert row["failure"]["failure_type"] == "ValueError"
    assert row["failure"]["failure_message"] == "boom"


def test_record_rows_preserve_declaration_order() -> None:
    from dr_code.metrics import MetricName, record_rows

    rows = record_rows(
        [
            _measured(),
            _measured(
                identity=_identity(
                    question=_question_coordinate(
                        metric=MetricName.CODE_LEAKAGE
                    )
                )
            ),
            _measured(
                identity=_identity(
                    question=_question_coordinate(metric=MetricName.AST_STATS)
                )
            ),
        ]
    )
    assert [row["metric"] for row in rows] == [
        MetricName.TEXT_STATS,
        MetricName.CODE_LEAKAGE,
        MetricName.AST_STATS,
    ]


# ===========================================================================
# Settings belong to the named metric; the discriminator is required.
# ===========================================================================


def test_metric_question_rejects_settings_from_another_operator() -> None:
    """Another operator's settings model is revalidated, not waved through."""
    from pydantic import ValidationError

    from dr_code.metrics import MetricName, MetricQuestion
    from dr_code.metrics.operators.code_leakage import CodeLeakageSettings

    with pytest.raises(ValidationError):
        MetricQuestion(
            metric=MetricName.TEXT_STATS,
            on="output",
            settings=CodeLeakageSettings(task_names=("x",)),
        )


def test_metric_question_accepts_its_own_settings_instance_and_dict() -> None:
    from dr_code.metrics import MetricName, MetricQuestion
    from dr_code.metrics.operators.code_leakage import CodeLeakageSettings

    expected = CodeLeakageSettings(task_names=("x",))
    from_instance = MetricQuestion(
        metric=MetricName.CODE_LEAKAGE, on="output", settings=expected
    )
    from_dict = MetricQuestion(
        metric=MetricName.CODE_LEAKAGE,
        on="output",
        settings={"task_names": ["x"]},
    )
    assert from_instance.settings == expected
    assert from_dict.settings == expected


def test_metric_question_missing_metric_raises_validation_error() -> None:
    from pydantic import ValidationError

    from dr_code.metrics import MetricQuestion

    with pytest.raises(ValidationError):
        MetricQuestion.model_validate({"on": "output", "settings": {}})


def test_question_coordinate_projects_a_declared_questions_settings() -> None:
    from dr_code.metrics import (
        MetricName,
        MetricQuestion,
        MetricQuestionCoordinate,
    )
    from dr_code.trace import ComponentSetting

    coordinate = MetricQuestionCoordinate.of(
        MetricQuestion(
            metric=MetricName.CODE_LEAKAGE,
            on="output",
            settings={"task_names": ["x", "y"]},
        )
    )
    assert coordinate.metric is MetricName.CODE_LEAKAGE
    assert coordinate.on_key == "output"
    assert coordinate.settings == (
        ComponentSetting(name="task_names", value=("x", "y")),
    )
