"""Definition contracts (plan section: ``definition.py``).

Covers ``MetricQuestion`` / ``MetricsDefinition`` — frozen, equality-based
comparability, the unique ``(metric, on, settings)`` validator, settings as
part of identity, and ``metrics_definition_hash`` determinism.

``dr_code.metrics`` is imported lazily inside each test so the suite collects
cleanly against the missing package and fails hard (never skips) when absent.
"""

from __future__ import annotations

import pytest


def _question(**overrides: object):
    from dr_code.metrics import MetricName, MetricQuestion

    base: dict[str, object] = {
        "metric": MetricName.TEXT_STATS,
        "on": "input",
        "settings": {},
    }
    base.update(overrides)
    return MetricQuestion(**base)


def _definition(
    questions=None,
    **overrides: object,
):
    from dr_code.metrics import MetricsDefinition

    base: dict[str, object] = {
        "definition_id": "def",
        "version": "1",
        "questions": questions or (_question(),),
    }
    base.update(overrides)
    return MetricsDefinition(**base)


# ===========================================================================
# MetricQuestion.
# ===========================================================================

def test_metric_question_carries_metric_on_settings() -> None:
    question = _question()
    assert question.settings == {}
    assert question.on == "input"


def test_metric_question_defaults_empty_settings() -> None:
    from dr_code.metrics import MetricName, MetricQuestion

    question = MetricQuestion(metric=MetricName.TEXT_STATS, on="input")
    assert question.settings == {}


def test_metric_question_carries_a_settings_dict() -> None:
    from dr_code.metrics import MetricName, MetricQuestion

    question = MetricQuestion(
        metric=MetricName.CODE_LEAKAGE,
        on="selected",
        settings={"task_names": ["add_one", "HumanEval/0"]},
    )
    assert question.settings == {"task_names": ["add_one", "HumanEval/0"]}


def test_metric_question_field_set_is_exactly_metric_on_settings() -> None:
    """Precise schema: questions carry only the three identity fields."""
    from dr_code.metrics.definition import MetricQuestion

    assert set(MetricQuestion.model_fields) == {"metric", "on", "settings"}


def test_metric_question_is_frozen() -> None:
    question = _question()
    with pytest.raises(Exception):  # noqa: PT011 — FrozenModel raises
        question.on = "output"  # type: ignore[misc]


def test_metric_questions_compare_equal_by_value() -> None:
    """Equality is the comparability contract; deterministic content
    identity is metrics_definition_hash (JSON-based), not Python __hash__."""
    assert _question() == _question()


def test_metric_questions_differ_on_settings() -> None:
    from dr_code.metrics import MetricName

    a = _question(
        metric=MetricName.COMPRESSED_LENGTH,
        settings={"method": "gzip", "level": 9},
    )
    b = _question(
        metric=MetricName.COMPRESSED_LENGTH,
        settings={"method": "zstd", "level": 3},
    )
    assert a != b


def test_metric_question_settings_are_order_independent() -> None:
    """Dict key ordering does not affect equality (settings are identity)."""
    from dr_code.metrics import MetricName, MetricQuestion

    a = MetricQuestion(
        metric=MetricName.COMPRESSED_LENGTH,
        on="input",
        settings={"method": "gzip", "level": 9},
    )
    b = MetricQuestion(
        metric=MetricName.COMPRESSED_LENGTH,
        on="input",
        settings={"level": 9, "method": "gzip"},
    )
    assert a == b


# ===========================================================================
# MetricsDefinition.
# ===========================================================================

def test_metrics_definition_carries_id_version_questions() -> None:
    from dr_code.metrics import MetricName

    definition = _definition(
        definition_id="humaneval-metrics",
        version="v1",
        questions=(_question(metric=MetricName.TEXT_STATS, on="input"),),
    )
    assert definition.definition_id == "humaneval-metrics"
    assert definition.version == "v1"
    assert len(definition.questions) == 1


def test_metrics_definition_field_set_is_exactly_id_version_questions() -> None:
    from dr_code.metrics.definition import MetricsDefinition

    assert set(MetricsDefinition.model_fields) == {
        "definition_id",
        "version",
        "questions",
    }


def test_metrics_definition_questions_is_a_tuple() -> None:
    definition = _definition(
        questions=(_question(on="input"), _question(on="output")),
    )
    assert isinstance(definition.questions, tuple)


def test_metrics_definition_is_frozen() -> None:
    definition = _definition()
    with pytest.raises(Exception):  # noqa: PT011
        definition.version = "2"  # type: ignore[misc]


def test_metrics_definition_questions_are_required() -> None:
    """The plan stub declares ``questions`` with no default — it is required."""
    from dr_code.metrics import MetricsDefinition

    with pytest.raises(Exception):  # noqa: PT011
        MetricsDefinition(definition_id="def", version="1")  # type: ignore[call-arg]


def test_metrics_definitions_compare_equal_by_value() -> None:
    assert _definition() == _definition()


def test_metrics_definition_json_round_trip_is_lossless() -> None:
    from dr_code.metrics import MetricName
    from dr_code.metrics.definition import MetricsDefinition

    definition = _definition(
        questions=(
            _question(
                metric=MetricName.CODE_LEAKAGE,
                on="selected",
                settings={"task_names": ["add_one"]},
            ),
            _question(on="input"),
        ),
    )
    restored = MetricsDefinition.model_validate_json(
        definition.model_dump_json()
    )
    assert restored == definition
    assert restored.questions[0].settings == {"task_names": ["add_one"]}


# ---------------------------------------------------------------------------
# Uniqueness of (metric, on, settings) triples.
# ---------------------------------------------------------------------------

def test_duplicate_metric_on_settings_triple_is_rejected() -> None:
    """Distinct questions need a distinct triple; a duplicate is a wiring bug.

    The import is resolved before the assertion so the test fails hard
    (ModuleNotFoundError) against the missing package rather than swallowing
    the import error inside ``pytest.raises``.
    """
    from dr_code.metrics import MetricName  # noqa: F401 — resolve before assert

    assert _question().metric == MetricName.TEXT_STATS
    with pytest.raises(Exception):  # noqa: PT011 — validator raises
        _definition(
            questions=(
                _question(),
                _question(),
            ),
        )


def test_same_metric_different_on_key_is_allowed() -> None:
    definition = _definition(
        questions=(_question(on="input"), _question(on="output")),
    )
    assert len(definition.questions) == 2


def test_same_metric_on_key_different_settings_is_allowed() -> None:
    """Settings participate in identity: two codec levels are two questions."""
    from dr_code.metrics import MetricName

    definition = _definition(
        questions=(
            _question(
                metric=MetricName.COMPRESSED_LENGTH,
                settings={"method": "gzip", "level": 6},
            ),
            _question(
                metric=MetricName.COMPRESSED_LENGTH,
                settings={"method": "gzip", "level": 9},
            ),
        ),
    )
    assert len(definition.questions) == 2


# ===========================================================================
# metrics_definition_hash (M2) — deterministic, content-addressed identity.
# ===========================================================================

def test_definition_hash_is_a_nonempty_string() -> None:
    from dr_code.metrics import metrics_definition_hash

    digest = metrics_definition_hash(_definition())
    assert isinstance(digest, str)
    assert len(digest) > 0


def test_definition_hash_is_deterministic() -> None:
    from dr_code.metrics import metrics_definition_hash

    assert metrics_definition_hash(_definition()) == metrics_definition_hash(
        _definition()
    )


def test_definition_hash_is_128_char_blake2b_hex() -> None:
    """trace.identity.stable_hash uses BLAKE2b (64-byte digest ⇒ 128 hex)."""
    from dr_code.metrics import metrics_definition_hash

    digest = metrics_definition_hash(_definition())
    assert len(digest) == 128
    assert all(c in "0123456789abcdef" for c in digest)


def test_equal_definitions_have_equal_hashes() -> None:
    from dr_code.metrics import metrics_definition_hash

    assert metrics_definition_hash(_definition()) == metrics_definition_hash(
        _definition()
    )


def test_definition_hash_is_stable_for_settings_key_reorder() -> None:
    """A JSON sort_keys hash is field/dict-key-order proof (persisted sweeps)."""
    from dr_code.metrics import MetricName
    from dr_code.metrics import metrics_definition_hash

    a = _definition(
        questions=(
            _question(
                metric=MetricName.COMPRESSED_LENGTH,
                settings={"method": "gzip", "level": 9},
            ),
        ),
    )
    b = _definition(
        questions=(
            _question(
                metric=MetricName.COMPRESSED_LENGTH,
                settings={"level": 9, "method": "gzip"},
            ),
        ),
    )
    assert metrics_definition_hash(a) == metrics_definition_hash(b)


def test_definition_hash_changes_with_metric() -> None:
    from dr_code.metrics import MetricName
    from dr_code.metrics import metrics_definition_hash

    a = _definition(questions=(_question(metric=MetricName.TEXT_STATS),))
    b = _definition(questions=(_question(metric=MetricName.CODE_LEAKAGE),))
    assert metrics_definition_hash(a) != metrics_definition_hash(b)


def test_definition_hash_changes_with_on_key() -> None:
    from dr_code.metrics import metrics_definition_hash

    a = _definition(questions=(_question(on="input"),))
    b = _definition(questions=(_question(on="output"),))
    assert metrics_definition_hash(a) != metrics_definition_hash(b)


def test_definition_hash_changes_with_settings() -> None:
    from dr_code.metrics import MetricName
    from dr_code.metrics import metrics_definition_hash

    a = _definition(
        questions=(
            _question(
                metric=MetricName.COMPRESSED_LENGTH,
                settings={"method": "gzip", "level": 6},
            ),
        ),
    )
    b = _definition(
        questions=(
            _question(
                metric=MetricName.COMPRESSED_LENGTH,
                settings={"method": "gzip", "level": 9},
            ),
        ),
    )
    assert metrics_definition_hash(a) != metrics_definition_hash(b)


def test_definition_hash_changes_with_version_or_id() -> None:
    from dr_code.metrics import metrics_definition_hash

    base = _definition()
    assert metrics_definition_hash(base) != metrics_definition_hash(
        _definition(version="2")
    )
    assert metrics_definition_hash(base) != metrics_definition_hash(
        _definition(definition_id="other")
    )
