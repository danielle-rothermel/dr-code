from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import dr_code.corpus.preprocessing_analysis as analysis_module
from dr_code.corpus.preprocessing_artifacts import PROJECTED_ARTIFACT_SCHEMAS
from dr_code.corpus.preprocessing_analysis import (
    PreprocessingAnalysisArtifacts,
    TABLE_SCHEMAS,
    analyze_preprocessing_corpus,
)
from dr_code.corpus.run_descriptor import RunDescriptor, file_sha256
from viewer.helpers import CORPUS_SCHEMA, write_bundle


def test_analysis_writes_compact_authenticated_artifacts(
    tmp_path: Path,
) -> None:
    descriptor = write_bundle(tmp_path / "bundle")
    output = tmp_path / "analysis"

    artifacts = analyze_preprocessing_corpus(
        dataset_id=descriptor.dataset_id,
        corpus_path=descriptor.corpus_path,
        run_dir=descriptor.preprocessing_manifest_path.parent,
        candidate_evaluation=descriptor.evaluation_root_path,
        output_dir=output,
    )

    expected_files = {
        artifacts.manifest_path,
        artifacts.summary_path,
        *artifacts.table_paths.values(),
    }
    assert {path for path in output.rglob("*") if path.is_file()} == (
        expected_files
    )
    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert summary["counts"] == {
        "samples": 9,
        "final_candidates": 2,
        "failures": 6,
        "evaluated_candidates": 2,
        "evaluated_executions": 2,
    }
    assert "tables" not in summary
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert set(manifest["tables"]) == set(TABLE_SCHEMAS)
    for name, schema in TABLE_SCHEMAS.items():
        parquet = pq.ParquetFile(artifacts.table_paths[name])
        assert parquet.schema_arrow.equals(schema)
        assert len(manifest["tables"][name]["sha256"]) == 64


def test_analysis_is_deterministic_and_append_only(tmp_path: Path) -> None:
    descriptor = write_bundle(tmp_path / "bundle", with_evaluation=False)
    first = analyze_preprocessing_corpus(
        dataset_id=descriptor.dataset_id,
        corpus_path=descriptor.corpus_path,
        run_dir=descriptor.preprocessing_manifest_path.parent,
        output_dir=tmp_path / "first",
    )
    second = analyze_preprocessing_corpus(
        dataset_id=descriptor.dataset_id,
        corpus_path=descriptor.corpus_path,
        run_dir=descriptor.preprocessing_manifest_path.parent,
        output_dir=tmp_path / "second",
    )

    assert first.summary_path.read_bytes() == second.summary_path.read_bytes()
    assert (
        first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    )
    for name in TABLE_SCHEMAS:
        assert (
            first.table_paths[name].read_bytes()
            == second.table_paths[name].read_bytes()
        )
    with pytest.raises(FileExistsError):
        analyze_preprocessing_corpus(
            dataset_id=descriptor.dataset_id,
            corpus_path=descriptor.corpus_path,
            run_dir=descriptor.preprocessing_manifest_path.parent,
            output_dir=first.output_dir,
        )


def test_analysis_identity_includes_preprocessing_only_dataset(
    tmp_path: Path,
) -> None:
    descriptor = write_bundle(
        tmp_path / "bundle",
        with_evaluation=False,
    )
    first = analyze_preprocessing_corpus(
        dataset_id="dataset/one",
        corpus_path=descriptor.corpus_path,
        run_dir=descriptor.preprocessing_manifest_path.parent,
        output_dir=tmp_path / "dataset-one",
    )
    second = analyze_preprocessing_corpus(
        dataset_id="dataset/two",
        corpus_path=descriptor.corpus_path,
        run_dir=descriptor.preprocessing_manifest_path.parent,
        output_dir=tmp_path / "dataset-two",
    )
    first_manifest = json.loads(
        first.manifest_path.read_text(encoding="utf-8")
    )
    second_manifest = json.loads(
        second.manifest_path.read_text(encoding="utf-8")
    )
    first_summary = json.loads(first.summary_path.read_text(encoding="utf-8"))
    second_summary = json.loads(
        second.summary_path.read_text(encoding="utf-8")
    )

    assert first_manifest["inputs"]["dataset"] == {"dataset_id": "dataset/one"}
    assert second_manifest["inputs"]["dataset"] == {
        "dataset_id": "dataset/two"
    }
    assert first_summary["run"]["dataset_id"] == "dataset/one"
    assert second_summary["run"]["dataset_id"] == "dataset/two"
    assert (
        first.manifest_path.read_bytes() != second.manifest_path.read_bytes()
    )
    assert first.summary_path.read_bytes() != second.summary_path.read_bytes()
    assert all(
        first.table_paths[name].read_bytes()
        == second.table_paths[name].read_bytes()
        for name in TABLE_SCHEMAS
    )


def test_concurrent_analysis_publication_preserves_one_complete_output(
    tmp_path: Path,
) -> None:
    first = write_bundle(
        tmp_path / "first-input",
        run_id="first",
        with_evaluation=False,
    )
    second = write_bundle(
        tmp_path / "second-input",
        run_id="second",
        with_evaluation=False,
    )
    destination = tmp_path / "analysis"
    barrier = threading.Barrier(2)

    def publish(descriptor: RunDescriptor) -> PreprocessingAnalysisArtifacts:
        barrier.wait()
        return analyze_preprocessing_corpus(
            dataset_id=descriptor.dataset_id,
            corpus_path=descriptor.corpus_path,
            run_dir=descriptor.preprocessing_manifest_path.parent,
            output_dir=destination,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(publish, value) for value in (first, second)
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
        (destination / "analysis_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["complete"] is True
    assert manifest["inputs"]["preprocessing"]["manifest_sha256"] in {
        first.preprocessing_manifest_sha256,
        second.preprocessing_manifest_sha256,
    }
    assert all(path.is_file() for path in successes[0].table_paths.values())
    assert not list(tmp_path.glob(".analysis.*.tmp"))


def test_analysis_streams_batches_and_accepts_complete_zero_row_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor = write_bundle(tmp_path / "bundle", with_evaluation=False)
    corpus_path = descriptor.corpus_path
    pq.write_table(pa.Table.from_pylist([], schema=CORPUS_SCHEMA), corpus_path)
    run = descriptor.preprocessing_manifest_path.parent
    for name, schema in PROJECTED_ARTIFACT_SCHEMAS.items():
        pq.write_table(
            pa.Table.from_pylist([], schema=schema), run / f"{name}.parquet"
        )

    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus = pq.ParquetFile(corpus_path)
    manifest["input"].update(
        {
            "sha256": file_sha256(corpus_path),
            "size": corpus_path.stat().st_size,
            "schema_hex": corpus.schema_arrow.serialize().to_pybytes().hex(),
            "expected_rows": 0,
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
        name: 0 for name in PROJECTED_ARTIFACT_SCHEMAS
    }
    manifest["outcome_totals"] = {}
    manifest["relation_sha256"] = {
        name: file_sha256(run / f"{name}.parquet")
        for name in PROJECTED_ARTIFACT_SCHEMAS
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(
        pq,
        "read_table",
        lambda *_args, **_kwargs: pytest.fail(
            "analysis must not materialize a whole Parquet table"
        ),
    )
    artifacts = analyze_preprocessing_corpus(
        dataset_id=descriptor.dataset_id,
        corpus_path=corpus_path,
        run_dir=run,
        output_dir=tmp_path / "analysis",
    )

    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert summary["counts"] == {
        "samples": 0,
        "final_candidates": 0,
        "failures": 0,
        "evaluated_candidates": 0,
        "evaluated_executions": 0,
    }
    for name, schema in TABLE_SCHEMAS.items():
        relation = pq.ParquetFile(artifacts.table_paths[name])
        assert relation.metadata.num_rows == 0
        assert relation.schema_arrow.equals(schema)


def test_analysis_holds_admitted_relations_through_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = write_bundle(
        tmp_path / "original",
        with_evaluation=False,
        no_code_causes=("original", "alternate", None),
    )
    replacement = write_bundle(
        tmp_path / "replacement",
        corpus_path=original.corpus_path,
        with_evaluation=False,
        no_code_causes=("replacement", "alternate", None),
    )
    original_summarize = analysis_module._summarize

    def replace_then_summarize(
        descriptor: RunDescriptor,
        *,
        table_paths: dict[str, Path],
    ) -> tuple[dict[str, object], dict[str, int]]:
        replacement.results_path.replace(original.results_path)
        return original_summarize(descriptor, table_paths=table_paths)

    monkeypatch.setattr(analysis_module, "_summarize", replace_then_summarize)
    artifacts = analyze_preprocessing_corpus(
        dataset_id=original.dataset_id,
        corpus_path=original.corpus_path,
        run_dir=original.preprocessing_manifest_path.parent,
        output_dir=tmp_path / "analysis",
    )

    failures = pq.read_table(artifacts.table_paths["failures"]).to_pylist()
    no_code = next(
        row
        for row in failures
        if row["failure_code"] == "no_code_candidates"
        and row["cause"] not in {"alternate", None}
    )
    assert no_code["cause"] == "original"


def test_analysis_spills_evaluation_join_to_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor = write_bundle(tmp_path / "bundle")
    result_rows = [
        {
            "evaluation_key": f"key-{index}",
            "outcome": "passed" if index % 2 == 0 else "tests_failed",
        }
        for index in range(2_048)
    ]
    membership_rows = [
        {"evaluation_key": row["evaluation_key"]}
        for row in result_rows
        for _ in range(2)
    ]
    results_path = tmp_path / "evaluation-results.parquet"
    memberships_path = tmp_path / "evaluation-memberships.parquet"
    pq.write_table(pa.Table.from_pylist(result_rows), results_path)
    pq.write_table(pa.Table.from_pylist(membership_rows), memberships_path)
    admitted = replace(
        descriptor,
        candidate_results_path=results_path,
        candidate_membership_path=memberships_path,
    )
    database_paths: list[Path] = []
    original_connect = analysis_module.sqlite3.connect

    def connect(path: Path) -> analysis_module.sqlite3.Connection:
        database_paths.append(Path(path))
        return original_connect(path)

    monkeypatch.setattr(analysis_module.sqlite3, "connect", connect)
    table_root = tmp_path / "tables"
    table_root.mkdir()
    table_paths = {
        name: table_root / f"{name}.parquet" for name in TABLE_SCHEMAS
    }
    summary, _table_rows = analysis_module._summarize(
        admitted,
        table_paths=table_paths,
    )

    assert database_paths and database_paths[0].name == "analysis.sqlite3"
    assert summary["counts"]["evaluated_candidates"] == 4_096
    assert summary["counts"]["evaluated_executions"] == 2_048
    evaluation_counts = {
        row["outcome"]: (
            row["candidate_count"],
            row["execution_count"],
        )
        for row in pq.read_table(
            table_paths["evaluation_outcomes"]
        ).to_pylist()
    }
    assert evaluation_counts == {
        "passed": (2_048, 1_024),
        "tests_failed": (2_048, 1_024),
    }
