from __future__ import annotations

from dr_code.classifier.extraction import (
    extract_parse_failures,
    extract_test_failures,
)
from dr_code.classifier.taxonomy import FailureKind
from dr_code.viewer.analytics import ViewerAnalytics
from dr_code.viewer.database import ViewerDatabase
from viewer.helpers import write_bundle


def _analytics(tmp_path, **kwargs):
    descriptor = write_bundle(tmp_path / "bundle", **kwargs)
    database = ViewerDatabase(":memory:")
    analytics = ViewerAnalytics(database, [descriptor])
    return analytics, descriptor


def test_parse_failures_cover_every_nonblank_no_candidate_row(
    tmp_path,
) -> None:
    analytics, descriptor = _analytics(tmp_path)
    items, total = extract_parse_failures(
        analytics._database.connection, descriptor
    )
    # blank is filtered (is_nonblank False); missing has no output. The five
    # remaining nonblank/no-candidate rows are parse failures.
    assert total == 5
    assert {item.sample_id for item in items} == {
        "no-code",
        "no-code-alt",
        "no-code-null",
        "compile-fail",
        "top-fail",
    }
    for item in items:
        assert item.kind is FailureKind.PARSE
        assert item.text  # raw decoder output carried
        assert item.dataset_id == "Task"


def test_parse_failures_carry_the_raw_decoder_output(tmp_path) -> None:
    analytics, descriptor = _analytics(tmp_path)
    items, _ = extract_parse_failures(
        analytics._database.connection, descriptor
    )
    by_sample = {item.sample_id: item for item in items}
    assert by_sample["compile-fail"].text == "def broken("
    assert by_sample["top-fail"].text == "answer = 42"


def test_parse_limit_caps_by_stable_sample_order(tmp_path) -> None:
    analytics, descriptor = _analytics(tmp_path)
    items, total = extract_parse_failures(
        analytics._database.connection, descriptor, limit=2
    )
    assert total == 5
    assert [item.sample_id for item in items] == ["compile-fail", "no-code"]


def test_test_failures_extract_compiled_but_failed(tmp_path) -> None:
    analytics, descriptor = _analytics(tmp_path)
    items, total = extract_test_failures(
        analytics._database.connection, descriptor
    )
    assert total == 1
    assert items[0].sample_id == "fail"
    assert items[0].kind is FailureKind.TEST
    assert "failed" in items[0].text


def test_test_failures_empty_without_evaluation(tmp_path) -> None:
    analytics, descriptor = _analytics(tmp_path, with_evaluation=False)
    items, total = extract_test_failures(
        analytics._database.connection, descriptor
    )
    assert items == []
    assert total == 0
