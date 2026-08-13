from __future__ import annotations

from ._helpers import (
    SAMPLE_TEXT,
    _definition,
    _extract,
    _question,
    _text_trace,
    _value,
)


_TEXT_STATS_GOLDEN = {
    "character_count": 141,
    "byte_count": 141,
    "line_count": 7,
    "nonempty_line_count": 6,
    "word_count": 24,
    "average_word_length": 3.7916666666666665,
    "punctuation_count": 18,
    "symbol_count": 6,
}


def test_text_stats_match_golden_values_field_for_field() -> None:
    record = _extract(
        _definition([_question("text_stats")]), _text_trace(SAMPLE_TEXT)
    )[0]

    for field, expected in _TEXT_STATS_GOLDEN.items():
        assert _value(record, field) == expected, field


def test_text_stats_empty_text_has_zero_counts() -> None:
    record = _extract(_definition([_question("text_stats")]), _text_trace(""))[
        0
    ]
    assert _value(record, "character_count") == 0
    assert _value(record, "line_count") == 0


def test_text_stats_distinguishes_unicode_characters_from_utf8_bytes() -> None:
    record = _extract(
        _definition([_question("text_stats")]), _text_trace("é🙂")
    )[0]

    assert _value(record, "character_count") == 2
    assert _value(record, "byte_count") == 6
