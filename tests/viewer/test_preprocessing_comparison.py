from __future__ import annotations

import json
import hashlib
import sqlite3
import threading
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import dr_code.corpus.preprocessing_comparison as comparison_module
from dr_code.corpus.preprocessing_comparison import (
    PreprocessingComparisonArtifacts,
    PreprocessingComparisonError,
    compare_preprocessing_runs,
)
from dr_code.corpus.run_descriptor import RunDescriptor
from viewer.helpers import write_bundle


def test_comparison_is_deterministic_and_schema_pinned(tmp_path: Path) -> None:
    before = write_bundle(
        tmp_path / "before", run_id="before", with_evaluation=False
    )
    after = write_bundle(
        tmp_path / "after",
        run_id="after",
        corpus_path=before.corpus_path,
        with_evaluation=False,
        no_code_causes=("changed", "alternate", None),
    )
    first = compare_preprocessing_runs(
        corpus_path=before.corpus_path,
        before_run=before.preprocessing_manifest_path.parent,
        after_run=after.preprocessing_manifest_path.parent,
        output_dir=tmp_path / "comparison-one",
    )
    second = compare_preprocessing_runs(
        corpus_path=before.corpus_path,
        before_run=before.preprocessing_manifest_path.parent,
        after_run=after.preprocessing_manifest_path.parent,
        output_dir=tmp_path / "comparison-two",
    )

    assert first.summary_path.read_bytes() == second.summary_path.read_bytes()
    assert (
        first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    )
    for name, path in first.relation_paths.items():
        assert path.read_bytes() == second.relation_paths[name].read_bytes()
    transitions = pq.read_table(
        first.relation_paths["sample_outcome_transitions"]
    ).to_pylist()
    changed = next(row for row in transitions if row["sample_id"] == "no-code")
    assert changed["semantic_result_changed"] is True
    assert changed["changed_fields"] == ["cause"]
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert all(
        {"filename", "row_count", "sha256", "schema"} == set(coordinates)
        for coordinates in manifest["relations"].values()
    )


def test_comparison_includes_optional_evaluation_relations(
    tmp_path: Path,
) -> None:
    before = write_bundle(tmp_path / "before", run_id="before")
    after = write_bundle(
        tmp_path / "after",
        run_id="after",
        corpus_path=before.corpus_path,
    )

    artifacts = compare_preprocessing_runs(
        corpus_path=before.corpus_path,
        before_run=before.preprocessing_manifest_path.parent,
        after_run=after.preprocessing_manifest_path.parent,
        before_evaluation=before.evaluation_root_path,
        after_evaluation=after.evaluation_root_path,
        output_dir=tmp_path / "comparison",
    )

    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert summary["evaluation"]["included"] is True
    assert (
        pq.read_table(
            artifacts.relation_paths["evaluation_membership_changes"]
        ).num_rows
        == 2
    )


def test_comparison_holds_admitted_relations_through_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = write_bundle(
        tmp_path / "before", run_id="before", with_evaluation=False
    )
    after = write_bundle(
        tmp_path / "after",
        run_id="after",
        corpus_path=before.corpus_path,
        with_evaluation=False,
        no_code_causes=("original", "alternate", None),
    )
    replacement = write_bundle(
        tmp_path / "replacement",
        run_id="replacement",
        corpus_path=before.corpus_path,
        with_evaluation=False,
        no_code_causes=("replacement", "alternate", None),
    )
    original_store = comparison_module._comparison_store

    @contextmanager
    def replace_then_open(
        root: Path,
    ) -> Iterator[sqlite3.Connection]:
        replacement.results_path.replace(after.results_path)
        with original_store(root) as connection:
            yield connection

    monkeypatch.setattr(
        comparison_module, "_comparison_store", replace_then_open
    )
    artifacts = compare_preprocessing_runs(
        corpus_path=before.corpus_path,
        before_run=before.preprocessing_manifest_path.parent,
        after_run=after.preprocessing_manifest_path.parent,
        output_dir=tmp_path / "comparison",
    )

    transitions = pq.read_table(
        artifacts.relation_paths["sample_outcome_transitions"]
    ).to_pylist()
    no_code = next(row for row in transitions if row["sample_id"] == "no-code")
    assert no_code["after_cause"] == "original"


def test_comparison_spills_and_streams_many_small_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copies = 16
    before = write_bundle(
        tmp_path / "before", run_id="before", with_evaluation=False
    )
    after = write_bundle(
        tmp_path / "after",
        run_id="after",
        corpus_path=before.corpus_path,
        with_evaluation=False,
        no_code_causes=("changed", "alternate", None),
    )
    _repeat_corpus(before.corpus_path, copies)
    _repeat_preprocessing_run(before, copies)
    _repeat_preprocessing_run(after, copies)

    def fail_full_materialization(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("comparison used an obsolete full materializer")

    monkeypatch.setattr(comparison_module, "_INPUT_BATCH_SIZE", 1)
    monkeypatch.setattr(comparison_module, "_OUTPUT_BATCH_SIZE", 1)
    monkeypatch.setattr(
        comparison_module.pq, "read_table", fail_full_materialization
    )
    for obsolete in (
        "_read_run",
        "_comparison_relations",
        "_comparison_summary",
    ):
        monkeypatch.setattr(
            comparison_module,
            obsolete,
            fail_full_materialization,
            raising=False,
        )

    artifacts = compare_preprocessing_runs(
        corpus_path=before.corpus_path,
        before_run=before.preprocessing_manifest_path.parent,
        after_run=after.preprocessing_manifest_path.parent,
        output_dir=tmp_path / "comparison",
    )

    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert summary["sample_outcome_transitions"] == {
        "identity_rows": 9 * copies,
        "changed_identity_rows": copies,
        "by_change": {
            "semantic_result_changed": copies,
            "unchanged": 8 * copies,
        },
        "output_identity_changed_count": 0,
        "outcome_changed_count": 0,
        "semantic_result_changed_count": copies,
        "transitions": [
            {
                "before_outcome": "decoder_output_blank",
                "after_outcome": "decoder_output_blank",
                "count": copies,
            },
            {
                "before_outcome": "decoder_output_missing",
                "after_outcome": "decoder_output_missing",
                "count": copies,
            },
            {
                "before_outcome": "function_candidates_extracted",
                "after_outcome": "function_candidates_extracted",
                "count": 2 * copies,
            },
            {
                "before_outcome": "no_code_candidates",
                "after_outcome": "no_code_candidates",
                "count": 3 * copies,
            },
            {
                "before_outcome": "no_compilable_candidate",
                "after_outcome": "no_compilable_candidate",
                "count": copies,
            },
            {
                "before_outcome": "no_top_level_function_candidate",
                "after_outcome": "no_top_level_function_candidate",
                "count": copies,
            },
        ],
    }
    assert summary["candidate_changes"] == {
        "identity_rows": 2 * copies,
        "changed_identity_rows": 0,
        "by_change": {"unchanged": 2 * copies},
        "before_count": 2 * copies,
        "after_count": 2 * copies,
        "count_delta": 0,
    }
    assert summary["reconciliation"] == {
        "sample_identity_rows": 9 * copies,
        "sample_rows_match_corpus": True,
        "candidate_before_count": 2 * copies,
        "candidate_after_count": 2 * copies,
        "provenance_before_count": 2 * copies,
        "provenance_after_count": 2 * copies,
        "evaluation_membership_before_count": 0,
        "evaluation_membership_after_count": 0,
        "evaluation_result_identity_rows": 0,
    }
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert {
        name: coordinates["row_count"]
        for name, coordinates in manifest["relations"].items()
    } == {
        "sample_outcome_transitions": 9 * copies,
        "candidate_changes": 2 * copies,
        "provenance_path_deltas": 2 * copies,
        "evaluation_membership_changes": 0,
        "evaluation_result_changes": 0,
    }
    for path in artifacts.relation_paths.values():
        parquet = pq.ParquetFile(path)
        assert parquet.num_row_groups == parquet.metadata.num_rows


def _repeat_corpus(path: Path, copies: int) -> None:
    table = pq.read_table(path)
    repeated = [
        {**row, "sample_id": f"{row['sample_id']}/{copy:04d}"}
        for copy in range(copies)
        for row in table.to_pylist()
    ]
    pq.write_table(pa.Table.from_pylist(repeated, schema=table.schema), path)


def _repeat_preprocessing_run(descriptor: RunDescriptor, copies: int) -> None:
    paths = {
        "results": descriptor.results_path,
        "candidates": descriptor.candidates_path,
        "step_facts": descriptor.step_facts_path,
        "rejections": descriptor.rejections_path,
    }
    repeated_relations: dict[str, list[dict[str, object]]] = {}
    for name, path in paths.items():
        table = pq.read_table(path)
        repeated = [
            {**row, "sample_id": f"{row['sample_id']}/{copy:04d}"}
            for copy in range(copies)
            for row in table.to_pylist()
        ]
        pq.write_table(
            pa.Table.from_pylist(repeated, schema=table.schema), path
        )
        repeated_relations[name] = repeated

    corpus = pq.ParquetFile(descriptor.corpus_path)
    manifest = json.loads(
        descriptor.preprocessing_manifest_path.read_text(encoding="utf-8")
    )
    manifest["input"].update(
        {
            "sha256": hashlib.sha256(
                descriptor.corpus_path.read_bytes()
            ).hexdigest(),
            "size": descriptor.corpus_path.stat().st_size,
            "expected_rows": corpus.metadata.num_rows,
            "expected_row_groups": corpus.num_row_groups,
            "row_groups": [
                {
                    "index": index,
                    "rows": corpus.metadata.row_group(index).num_rows,
                    "total_byte_size": (
                        corpus.metadata.row_group(index).total_byte_size
                    ),
                }
                for index in range(corpus.num_row_groups)
            ],
        }
    )
    manifest["completed_row_groups"] = list(range(corpus.num_row_groups))
    manifest["relation_totals"] = {
        name: len(rows) for name, rows in repeated_relations.items()
    }
    manifest["outcome_totals"] = dict(
        Counter(row["outcome"] for row in repeated_relations["results"])
    )
    manifest["relation_sha256"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }
    descriptor.preprocessing_manifest_path.write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_comparison_refuses_existing_output_and_one_sided_evaluation(
    tmp_path: Path,
) -> None:
    before = write_bundle(tmp_path / "before", run_id="before")
    after = write_bundle(
        tmp_path / "after",
        run_id="after",
        corpus_path=before.corpus_path,
    )
    existing = tmp_path / "existing"
    existing.mkdir()

    with pytest.raises(FileExistsError):
        compare_preprocessing_runs(
            corpus_path=before.corpus_path,
            before_run=before.preprocessing_manifest_path.parent,
            after_run=after.preprocessing_manifest_path.parent,
            output_dir=existing,
        )
    with pytest.raises(
        PreprocessingComparisonError, match="supplied together"
    ):
        compare_preprocessing_runs(
            corpus_path=before.corpus_path,
            before_run=before.preprocessing_manifest_path.parent,
            after_run=after.preprocessing_manifest_path.parent,
            before_evaluation=before.evaluation_root_path,
            output_dir=tmp_path / "one-sided",
        )


def test_comparison_rejects_different_corpora(tmp_path: Path) -> None:
    before = write_bundle(
        tmp_path / "before", run_id="before", with_evaluation=False
    )
    after = write_bundle(
        tmp_path / "after", run_id="after", with_evaluation=False
    )
    corpus_path = after.corpus_path
    corpus_path.write_bytes(corpus_path.read_bytes() + b"different")
    manifest_path = after.preprocessing_manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["input"]["sha256"] = hashlib.sha256(
        corpus_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        PreprocessingComparisonError, match="fingerprint mismatch"
    ):
        compare_preprocessing_runs(
            corpus_path=before.corpus_path,
            before_run=before.preprocessing_manifest_path.parent,
            after_run=after.preprocessing_manifest_path.parent,
            output_dir=tmp_path / "comparison",
        )


def test_comparison_rejects_source_changed_under_unchanged_candidate_id(
    tmp_path: Path,
) -> None:
    before = write_bundle(
        tmp_path / "before", run_id="before", with_evaluation=False
    )
    after = write_bundle(
        tmp_path / "after",
        run_id="after",
        corpus_path=before.corpus_path,
        with_evaluation=False,
    )
    candidates_path = after.candidates_path
    table = pq.read_table(candidates_path)
    rows = table.to_pylist()
    changed_source = "def identity_was_reused():\n    return 2"
    rows[0]["cleaned_source"] = changed_source
    rows[0]["source_sha256"] = hashlib.sha256(
        changed_source.encode("utf-8")
    ).hexdigest()
    pq.write_table(
        pa.Table.from_pylist(rows, schema=table.schema), candidates_path
    )
    manifest_path = after.preprocessing_manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["relation_sha256"]["candidates"] = hashlib.sha256(
        candidates_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        PreprocessingComparisonError,
        match="candidate_id is not content-derived",
    ):
        compare_preprocessing_runs(
            corpus_path=before.corpus_path,
            before_run=before.preprocessing_manifest_path.parent,
            after_run=after.preprocessing_manifest_path.parent,
            output_dir=tmp_path / "comparison",
        )


def test_concurrent_comparison_publication_preserves_one_complete_output(
    tmp_path: Path,
) -> None:
    before = write_bundle(
        tmp_path / "before", run_id="before", with_evaluation=False
    )
    first_after = write_bundle(
        tmp_path / "first-after",
        run_id="first-after",
        corpus_path=before.corpus_path,
        with_evaluation=False,
        no_code_causes=("first", "alternate", None),
    )
    second_after = write_bundle(
        tmp_path / "second-after",
        run_id="second-after",
        corpus_path=before.corpus_path,
        with_evaluation=False,
        no_code_causes=("second", "alternate", None),
    )
    destination = tmp_path / "comparison"
    barrier = threading.Barrier(2)

    def publish(
        after: RunDescriptor,
    ) -> PreprocessingComparisonArtifacts:
        barrier.wait()
        return compare_preprocessing_runs(
            corpus_path=before.corpus_path,
            before_run=before.preprocessing_manifest_path.parent,
            after_run=after.preprocessing_manifest_path.parent,
            output_dir=destination,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(publish, value)
            for value in (first_after, second_after)
        ]
        successes = []
        failures = []
        for future in futures:
            try:
                successes.append(future.result())
            except FileExistsError as exc:
                failures.append(exc)

    assert len(successes) == len(failures) == 1
    manifest = json.loads(
        (destination / "comparison_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["complete"] is True
    assert manifest["after"]["preprocessing_manifest_sha256"] in {
        first_after.preprocessing_manifest_sha256,
        second_after.preprocessing_manifest_sha256,
    }
    assert all(path.is_file() for path in successes[0].relation_paths.values())
    assert not list(tmp_path.glob(".comparison.*.tmp"))
