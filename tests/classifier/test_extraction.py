from __future__ import annotations

import json
from dataclasses import replace
from typing import cast

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import dr_code.classifier.extraction as extraction_module
from dr_code.classifier.extraction import (
    extract_parse_failures,
    extract_test_failures,
    stream_failures,
)
from dr_code.viewer.analytics import ViewerAnalytics
from dr_code.viewer.database import ViewerDatabase
from viewer.helpers import write_bundle
from viewer.helpers import CORPUS_ROWS, CORPUS_SCHEMA


def test_public_extraction_is_stable_capped_and_reports_population(
    tmp_path,
) -> None:
    descriptor = write_bundle(
        tmp_path / "bundle",
        dataset_id="organization/benchmark",
        task_namespace="IndependentTask",
    )
    analytics = ViewerAnalytics(ViewerDatabase(":memory:"), [descriptor])

    parse_items, parse_total = extract_parse_failures(
        analytics, descriptor.run_id, limit=2
    )
    assert parse_total == 5
    assert [item.sample_id for item in parse_items] == [
        "compile-fail",
        "no-code",
    ]
    assert {item.dataset_id for item in parse_items} == {
        "organization/benchmark"
    }
    assert parse_items[0].task_id is None
    assert parse_items[0].task_identity is None
    assert "__corpus_context" not in parse_items[0].rendered_input
    assert "def broken(" in parse_items[0].rendered_input

    test_items, test_total = extract_test_failures(
        analytics, descriptor.run_id, limit=1
    )
    assert test_total == 1
    assert test_items[0].sample_id == "fail"
    assert test_items[0].candidate_id
    assert test_items[0].evaluation_key
    assert test_items[0].task_id == "IndependentTask/6"
    assert len(test_items[0].task_identity or "") == 64
    assert "def fail_me()" in test_items[0].rendered_input
    assert '"outcome":"tests_failed"' in test_items[0].rendered_input
    assert '"failed_count":1' in test_items[0].rendered_input
    assert '"error_count":0' in test_items[0].rendered_input
    assert '"timeout_count":0' in test_items[0].rendered_input
    assert '"coverage_complete":true' in test_items[0].rendered_input
    assert '"function_count":1' in test_items[0].rendered_input
    assert '"best_function_name":"f"' in test_items[0].rendered_input
    assert "failure_type" not in test_items[0].rendered_input
    assert "failure_message" not in test_items[0].rendered_input
    assert '"task_id":"IndependentTask/6"' in test_items[0].rendered_input


def test_unlimited_extraction_streams_in_bounded_stable_pages(
    tmp_path,
    monkeypatch,
) -> None:
    descriptor = write_bundle(tmp_path / "bundle")
    analytics = ViewerAnalytics(ViewerDatabase(":memory:"), [descriptor])
    parse_calls: list[tuple[int | None, int]] = []
    original_parse = analytics.parse_failures_for_classification

    def parse_page(
        run_id: str,
        *,
        limit: int | None = None,
        offset: int = 0,
    ):
        parse_calls.append((limit, offset))
        return original_parse(run_id, limit=limit, offset=offset)

    monkeypatch.setattr(
        analytics,
        "parse_failures_for_classification",
        parse_page,
    )
    monkeypatch.setattr(extraction_module, "EXTRACTION_BATCH_SIZE", 2)

    extracted = stream_failures(
        analytics,
        descriptor.run_id,
        parse_limit=None,
        test_limit=None,
    )
    items = tuple(extracted.items)

    assert len(items) == 6
    assert parse_calls == [(2, 0), (2, 2), (1, 4)]


def test_test_extraction_is_empty_without_evaluation(tmp_path) -> None:
    descriptor = write_bundle(tmp_path / "bundle", with_evaluation=False)
    analytics = ViewerAnalytics(ViewerDatabase(":memory:"), [descriptor])
    items, total = extract_test_failures(analytics, descriptor.run_id)
    assert items == ()
    assert total == 0


def test_extraction_rejects_null_dataset_identity(
    tmp_path,
    monkeypatch,
) -> None:
    descriptor = write_bundle(tmp_path / "bundle")
    analytics = ViewerAnalytics(ViewerDatabase(":memory:"), [descriptor])
    page = analytics.parse_failures_for_classification(
        descriptor.run_id,
        limit=1,
    )
    invalid = replace(page.items[0], dataset_id=cast(str, None))
    monkeypatch.setattr(
        analytics,
        "parse_failures_for_classification",
        lambda *args, **kwargs: replace(page, items=(invalid,)),
    )

    with pytest.raises(ValueError, match="nonnull dataset identity"):
        extract_parse_failures(analytics, descriptor.run_id, limit=1)


def test_classification_caps_are_bound_in_sql_while_counts_stay_uncapped(
    tmp_path,
) -> None:
    descriptor = write_bundle(tmp_path / "bundle")
    analytics = ViewerAnalytics(ViewerDatabase(":memory:"), [descriptor])
    connection = analytics._connection  # noqa: SLF001

    class QuerySpy:
        queries: list[tuple[str, object]]

        def __init__(self) -> None:
            self.queries = []

        def execute(
            self,
            query: str,
            parameters: object = None,
        ) -> duckdb.DuckDBPyConnection | QuerySpy:
            self.queries.append((query, parameters))
            if parameters is None:
                connection.execute(query)
            else:
                connection.execute(query, parameters)
            return self

        @property
        def description(self) -> object:
            return connection.description

        def fetchone(self) -> object:
            return connection.fetchone()

        def fetchall(self) -> object:
            return connection.fetchall()

    spy = QuerySpy()
    analytics._database._connection = cast(  # noqa: SLF001
        duckdb.DuckDBPyConnection,
        spy,
    )
    try:
        parse = analytics.parse_failures_for_classification(
            descriptor.run_id,
            limit=2,
        )
        tests = analytics.candidate_test_failures_for_classification(
            descriptor.run_id,
            limit=1,
        )
    finally:
        analytics._database._connection = connection  # noqa: SLF001

    assert parse.total == 5
    assert len(parse.items) == 2
    assert tests.total == 1
    assert len(tests.items) == 1
    bounded = [
        (query, parameters)
        for query, parameters in spy.queries
        if "__classifier_" in query
    ]
    assert len(bounded) == 2
    assert all("LIMIT ?" in query for query, _ in bounded)
    assert [parameters[-2] for _, parameters in bounded] == [2, 1]
    assert [parameters[-1] for _, parameters in bounded] == [0, 0]


def test_zero_parse_population_reports_zero_without_unbounded_materialization(
    tmp_path,
) -> None:
    descriptor = write_bundle(
        tmp_path / "bundle",
        with_evaluation=False,
        parse_failures_are_nonblank=False,
    )
    analytics = ViewerAnalytics(ViewerDatabase(":memory:"), [descriptor])

    page = analytics.parse_failures_for_classification(
        descriptor.run_id,
        limit=2,
    )

    assert page.total == 0
    assert page.items == ()


def test_nested_nonfinite_context_uses_shared_json_normalization(
    tmp_path,
) -> None:
    corpus = tmp_path / "corpus.parquet"
    rows = [
        {
            "sample_id": sample_id,
            "task_id": task_id,
            "source_kind": source_kind,
            "decoder_output": decoder_output,
            "nested": {"values": [float("nan"), float("inf"), float("-inf")]},
        }
        for sample_id, task_id, source_kind, decoder_output in CORPUS_ROWS
    ]
    corpus_schema = CORPUS_SCHEMA.append(
        pa.field(
            "nested",
            pa.struct(
                [
                    pa.field(
                        "values",
                        pa.list_(pa.float64()),
                    )
                ]
            ),
            nullable=False,
        )
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=corpus_schema), corpus)
    descriptor = write_bundle(
        tmp_path / "bundle",
        corpus_path=corpus,
        dataset_id="organization/benchmark",
    )
    analytics = ViewerAnalytics(ViewerDatabase(":memory:"), [descriptor])

    items, _ = extract_parse_failures(
        analytics,
        descriptor.run_id,
        limit=1,
    )
    marker = "Evidence follows as canonical JSON:\n"
    evidence_line = items[0].rendered_input.split(marker, 1)[1].splitlines()[0]
    evidence = json.loads(evidence_line)

    assert evidence["task_context"]["task_id"] == "Task/3"
    assert evidence["task_context"]["nested"]["values"] == [
        "NaN",
        "Infinity",
        "-Infinity",
    ]


def test_real_empty_named_context_field_survives_synthetic_empty_sentinel(
    tmp_path,
) -> None:
    base_fields = (
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("decoder_output", pa.string()),
    )
    base_rows = [
        {
            "sample_id": sample_id,
            "decoder_output": decoder_output,
        }
        for sample_id, _, _, decoder_output in CORPUS_ROWS
    ]
    empty_corpus = tmp_path / "empty-context.parquet"
    pq.write_table(
        pa.Table.from_pylist(base_rows, schema=pa.schema(base_fields)),
        empty_corpus,
    )
    empty_descriptor = write_bundle(
        tmp_path / "empty-bundle",
        corpus_path=empty_corpus,
        with_evaluation=False,
    )
    empty_analytics = ViewerAnalytics(
        ViewerDatabase(":memory:"),
        [empty_descriptor],
    )

    real_corpus = tmp_path / "real-empty-field.parquet"
    real_rows = [{**row, "__empty": "real"} for row in base_rows]
    pq.write_table(
        pa.Table.from_pylist(
            real_rows,
            schema=pa.schema(
                (
                    *base_fields,
                    pa.field("__empty", pa.string(), nullable=False),
                )
            ),
        ),
        real_corpus,
    )
    real_descriptor = write_bundle(
        tmp_path / "real-bundle",
        corpus_path=real_corpus,
        with_evaluation=False,
    )
    real_analytics = ViewerAnalytics(
        ViewerDatabase(":memory:"),
        [real_descriptor],
    )

    empty_page = empty_analytics.parse_failures_for_classification(
        empty_descriptor.run_id,
        limit=1,
    )
    real_page = real_analytics.parse_failures_for_classification(
        real_descriptor.run_id,
        limit=1,
    )

    assert empty_page.items[0].task_context == {}
    assert real_page.items[0].task_context == {"__empty": "real"}
