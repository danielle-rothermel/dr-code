from __future__ import annotations

import gzip

import pytest
import zstandard
from pydantic import ValidationError

from dr_code.trace import CodeArtifact, TextArtifact, external_trace

from ._helpers import (
    SAMPLE_TEXT,
    _definition,
    _extract,
    _facts,
    _question,
    _text_trace,
    _value,
)


_ZSTD_GOLDEN_LEVEL = 3


def _reference_trace(text: str, reference: str):
    return external_trace(
        {
            "input": TextArtifact(text=text),
            "output": TextArtifact(text=text),
            "reference": CodeArtifact(source=reference),
        }
    )


def test_compressed_length_gzip_level_9_matches_golden(task) -> None:
    reference = task.ground_truth_code

    expected_compressed = len(
        gzip.compress(SAMPLE_TEXT.encode("utf-8"), compresslevel=9)
    )
    ground_truth_bytes = len(reference.encode("utf-8"))
    expected_ratio = expected_compressed / ground_truth_bytes
    expected_percent_reduction = (1.0 - expected_ratio) * 100.0

    record = _extract(
        _definition(
            [
                _question(
                    "compressed_length",
                    compression={"method": "gzip", "level": 9},
                    reference_key="reference",
                )
            ]
        ),
        _reference_trace(SAMPLE_TEXT, reference),
    )[0]
    assert _value(record, "compressed_bytes") == expected_compressed
    assert _value(record, "representation_bytes") == len(
        SAMPLE_TEXT.encode("utf-8")
    )
    assert _value(record, "ratio_to_reference") == expected_ratio
    assert _value(record, "percent_reduction") == expected_percent_reduction


def test_compressed_length_zstd_level_3_matches_golden(task) -> None:
    reference = task.ground_truth_code

    record = _extract(
        _definition(
            [
                _question(
                    "compressed_length",
                    compression={"method": "zstd", "level": 3},
                    reference_key="reference",
                )
            ]
        ),
        _reference_trace(SAMPLE_TEXT, reference),
    )[0]
    assert _value(record, "compressed_bytes") == len(
        zstandard.ZstdCompressor(level=_ZSTD_GOLDEN_LEVEL).compress(
            SAMPLE_TEXT.encode("utf-8")
        )
    )


def test_compressed_length_without_reference_has_no_ratio() -> None:
    record = _extract(
        _definition(
            [
                _question(
                    "compressed_length",
                    compression={"method": "gzip", "level": 9},
                )
            ]
        ),
        _text_trace(SAMPLE_TEXT),
    )[0]
    assert _value(record, "compressed_bytes") == len(
        gzip.compress(SAMPLE_TEXT.encode("utf-8"), compresslevel=9)
    )
    assert "representation_bytes" in _facts(record)
    assert "ratio_to_reference" not in _facts(record)


@pytest.mark.parametrize(
    ("method", "level"),
    [
        pytest.param("gzip", 0, id="gzip-min"),
        pytest.param("gzip", 9, id="gzip-max"),
        pytest.param("zstd", -1, id="zstd-negative"),
        pytest.param("zstd", 1, id="zstd-min-positive"),
        pytest.param("zstd", 22, id="zstd-max"),
    ],
)
def test_compressed_length_accepts_valid_level_edges(
    method: str,
    level: int,
) -> None:
    question = _question(
        "compressed_length",
        compression={"method": method, "level": level},
    )

    record = _extract(_definition([question]), _text_trace(SAMPLE_TEXT))[0]

    assert record.status.value == "measured"
    assert _value(record, "compressed_bytes") > 0


@pytest.mark.parametrize(
    ("settings", "error_type", "error_loc"),
    [
        pytest.param(
            {"compression": {"method": "gzip", "level": -1}},
            "value_error",
            ("compression", "gzip"),
            id="gzip-below-min",
        ),
        pytest.param(
            {"compression": {"method": "gzip", "level": 10}},
            "value_error",
            ("compression", "gzip"),
            id="gzip-above-max",
        ),
        pytest.param(
            {"compression": {"method": "zstd", "level": 0}},
            "value_error",
            ("compression", "zstd"),
            id="zstd-zero",
        ),
        pytest.param(
            {"compression": {"method": "zstd", "level": 23}},
            "value_error",
            ("compression", "zstd"),
            id="zstd-above-max",
        ),
        pytest.param(
            {"compression": {"method": "brotli", "level": 1}},
            "union_tag_invalid",
            ("compression",),
            id="unknown-method",
        ),
        pytest.param(
            {
                "compression": {"method": "gzip", "level": 1},
                "reference_key": "",
            },
            "value_error",
            (),
            id="empty-reference-key",
        ),
    ],
)
def test_compressed_length_rejects_invalid_settings(
    settings: dict[str, object],
    error_type: str,
    error_loc: tuple[str, ...],
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        _question("compressed_length", **settings)

    assert [
        (error["type"], error["loc"]) for error in exc_info.value.errors()
    ] == [(error_type, error_loc)]


def test_compressed_level_is_part_of_identity() -> None:
    from dr_code.metrics import METRIC_RECORD_ADAPTER

    trace = _text_trace(SAMPLE_TEXT)
    level_1 = _extract(
        _definition(
            [
                _question(
                    "compressed_length",
                    compression={"method": "gzip", "level": 1},
                )
            ]
        ),
        trace,
    )[0]
    level_9 = _extract(
        _definition(
            [
                _question(
                    "compressed_length",
                    compression={"method": "gzip", "level": 9},
                )
            ]
        ),
        trace,
    )[0]
    restored_1 = METRIC_RECORD_ADAPTER.validate_json(level_1.model_dump_json())
    restored_9 = METRIC_RECORD_ADAPTER.validate_json(level_9.model_dump_json())

    assert restored_1.identity != restored_9.identity
    assert {
        setting.name: setting.value
        for setting in restored_1.identity.question.settings
    } == {
        "compression.method": "gzip",
        "compression.level": 1,
        "reference_key": None,
    }
    assert {
        setting.name: setting.value
        for setting in restored_9.identity.question.settings
    } == {
        "compression.method": "gzip",
        "compression.level": 9,
        "reference_key": None,
    }
    assert (
        restored_1.identity.question.settings
        == restored_1.identity.metrics_definition.questions[0].settings
    )
    assert (
        restored_9.identity.question.settings
        == restored_9.identity.metrics_definition.questions[0].settings
    )
