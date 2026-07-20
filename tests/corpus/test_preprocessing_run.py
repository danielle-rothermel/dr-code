"""Integration tests for resumable generation-corpus preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dr_code.corpus.preprocessing_run import (
    CorpusRunError,
    run_preprocessing_corpus,
)


def _write_input(path: Path, *, row_group_size: int = 1) -> Path:
    schema = pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("decoder_output", pa.string(), nullable=True),
        ]
    )
    table = pa.Table.from_arrays(
        [
            pa.array(["missing", "blank", "literal", "success"]),
            pa.array([None, "\n\t", "[1, 2]", "def f():\n    return 1\n"]),
        ],
        schema=schema,
    )
    pq.write_table(table, path, row_group_size=row_group_size)
    return path


def _manifest(directory: Path) -> dict[str, object]:
    return json.loads((directory / "manifest.json").read_text())


def _write_success_then_missing_input(path: Path) -> Path:
    schema = pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("decoder_output", pa.string(), nullable=True),
        ]
    )
    pq.write_table(
        pa.Table.from_arrays(
            [
                pa.array(["success", "missing"]),
                pa.array(["def f():\n    return 1\n", None]),
            ],
            schema=schema,
        ),
        path,
        row_group_size=1,
    )
    return path


def test_run_is_resumable_and_publishes_complete_relations(
    tmp_path: Path,
) -> None:
    input_path = _write_input(tmp_path / "input.parquet")
    output_root = tmp_path / "runs"

    partial_dir = run_preprocessing_corpus(
        input_path=input_path,
        output_root=output_root,
        run_id="tiny",
        batch_size=1,
        max_row_groups=2,
    )
    partial_manifest = _manifest(partial_dir)
    assert partial_dir == output_root / "tiny.partial"
    assert partial_manifest["complete"] is False
    assert partial_manifest["completed_row_groups"] == [0, 1]

    completed_dir = run_preprocessing_corpus(
        input_path=input_path,
        output_root=output_root,
        run_id="tiny",
        batch_size=1,
    )
    manifest = _manifest(completed_dir)
    assert completed_dir == output_root / "tiny"
    assert not (output_root / "tiny.partial").exists()
    assert manifest["complete"] is True
    assert manifest["completed_row_groups"] == [0, 1, 2, 3]
    assert manifest["relation_totals"]["results"] == 4

    results = pq.read_table(completed_dir / "results.parquet")
    assert results.num_rows == 4
    assert len(set(results.column("sample_id").to_pylist())) == 4
    assert results.column("decoder_output_presence").to_pylist() == [
        "missing",
        "present",
        "present",
        "present",
    ]
    assert results.column("outcome").to_pylist() == [
        "decoder_output_missing",
        "decoder_output_blank",
        "no_code_candidates",
        "function_candidates_extracted",
    ]
    candidates = pq.read_table(completed_dir / "candidates.parquet")
    assert candidates.column("candidate_id").null_count == 0
    assert candidates.num_rows == 1


def test_resume_refuses_manifest_fingerprint_mismatch(tmp_path: Path) -> None:
    input_path = _write_input(tmp_path / "input.parquet")
    output_root = tmp_path / "runs"
    run_preprocessing_corpus(
        input_path=input_path,
        output_root=output_root,
        run_id="tiny",
        batch_size=1,
        max_row_groups=1,
    )

    with pytest.raises(CorpusRunError, match="batch_size"):
        run_preprocessing_corpus(
            input_path=input_path,
            output_root=output_root,
            run_id="tiny",
            batch_size=2,
        )


def test_complete_run_is_immutable(tmp_path: Path) -> None:
    input_path = _write_input(tmp_path / "input.parquet")
    output_root = tmp_path / "runs"
    run_preprocessing_corpus(
        input_path=input_path,
        output_root=output_root,
        run_id="tiny",
        batch_size=2,
    )

    with pytest.raises(FileExistsError, match="completed run already exists"):
        run_preprocessing_corpus(
            input_path=input_path,
            output_root=output_root,
            run_id="tiny",
            batch_size=2,
        )


def test_resume_publishes_after_manifest_was_written_before_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dr_code.corpus import preprocessing_run

    input_path = _write_input(tmp_path / "input.parquet")
    output_root = tmp_path / "runs"
    destination = output_root / "crash"
    original_replace = preprocessing_run.os.replace

    def fail_final_publish(source: Path | str, target: Path | str) -> None:
        if Path(target) == destination:
            raise OSError("simulated publish interruption")
        original_replace(source, target)

    monkeypatch.setattr(preprocessing_run.os, "replace", fail_final_publish)
    with pytest.raises(OSError, match="simulated publish interruption"):
        run_preprocessing_corpus(
            input_path=input_path,
            output_root=output_root,
            run_id="crash",
        )

    partial_dir = output_root / "crash.partial"
    assert _manifest(partial_dir)["complete"] is True
    monkeypatch.setattr(preprocessing_run.os, "replace", original_replace)
    assert (
        run_preprocessing_corpus(
            input_path=input_path,
            output_root=output_root,
            run_id="crash",
        )
        == destination
    )


def test_initial_partial_creation_is_atomic_and_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dr_code.corpus import preprocessing_run

    input_path = _write_input(tmp_path / "input.parquet")
    output_root = tmp_path / "runs"
    partial_dir = output_root / "initial.partial"
    original_write_manifest = preprocessing_run._write_manifest

    def fail_manifest_write(directory: Path, manifest: object) -> None:
        raise OSError("simulated initial manifest interruption")

    monkeypatch.setattr(
        preprocessing_run, "_write_manifest", fail_manifest_write
    )
    with pytest.raises(
        OSError, match="simulated initial manifest interruption"
    ):
        run_preprocessing_corpus(
            input_path=input_path,
            output_root=output_root,
            run_id="initial",
        )
    assert not partial_dir.exists()

    monkeypatch.setattr(
        preprocessing_run, "_write_manifest", original_write_manifest
    )
    completed_dir = run_preprocessing_corpus(
        input_path=input_path,
        output_root=output_root,
        run_id="initial",
    )
    assert _manifest(completed_dir)["complete"] is True


def test_row_group_processing_uses_streaming_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = _write_input(tmp_path / "input.parquet", row_group_size=4)

    def read_row_group_is_forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("runner must use ParquetFile.iter_batches")

    monkeypatch.setattr(
        pq.ParquetFile, "read_row_group", read_row_group_is_forbidden
    )
    partial_dir = run_preprocessing_corpus(
        input_path=input_path,
        output_root=tmp_path / "runs",
        run_id="streaming",
        batch_size=1,
        max_row_groups=1,
    )

    results_part = pq.ParquetFile(
        partial_dir / "parts" / "row_group_00000000" / "results.parquet"
    )
    assert results_part.metadata.num_rows == 4
    assert results_part.num_row_groups == 4


def test_completion_rejects_noncontiguous_candidate_indexes(
    tmp_path: Path,
) -> None:
    input_path = _write_success_then_missing_input(tmp_path / "input.parquet")
    output_root = tmp_path / "runs"
    partial_dir = run_preprocessing_corpus(
        input_path=input_path,
        output_root=output_root,
        run_id="invalid-candidates",
        max_row_groups=1,
    )
    candidate_path = (
        partial_dir / "parts" / "row_group_00000000" / "candidates.parquet"
    )
    candidates = pq.read_table(candidate_path)
    rows = candidates.to_pylist()
    rows[0]["candidate_index"] = 1
    pq.write_table(
        pa.Table.from_pylist(rows, schema=candidates.schema), candidate_path
    )

    with pytest.raises(CorpusRunError, match="not contiguous"):
        run_preprocessing_corpus(
            input_path=input_path,
            output_root=output_root,
            run_id="invalid-candidates",
        )


def test_completion_rejects_same_size_results_sample_id_corruption(
    tmp_path: Path,
) -> None:
    input_path = _write_success_then_missing_input(tmp_path / "input.parquet")
    output_root = tmp_path / "runs"
    partial_dir = run_preprocessing_corpus(
        input_path=input_path,
        output_root=output_root,
        run_id="invalid-results",
        max_row_groups=1,
    )
    results_path = (
        partial_dir / "parts" / "row_group_00000000" / "results.parquet"
    )
    results = pq.read_table(results_path)
    rows = results.to_pylist()
    rows[0]["sample_id"] = "same-size-bogus"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=results.schema), results_path
    )

    with pytest.raises(CorpusRunError, match="sample_id does not match"):
        run_preprocessing_corpus(
            input_path=input_path,
            output_root=output_root,
            run_id="invalid-results",
        )


def test_completion_rejects_child_sample_id_foreign_key(
    tmp_path: Path,
) -> None:
    input_path = _write_success_then_missing_input(tmp_path / "input.parquet")
    output_root = tmp_path / "runs"
    partial_dir = run_preprocessing_corpus(
        input_path=input_path,
        output_root=output_root,
        run_id="invalid-child",
        max_row_groups=1,
    )
    step_facts_path = (
        partial_dir / "parts" / "row_group_00000000" / "step_facts.parquet"
    )
    step_facts = pq.read_table(step_facts_path)
    rows = step_facts.to_pylist()
    assert rows
    rows[0]["sample_id"] = "same-size-bogus"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=step_facts.schema), step_facts_path
    )

    with pytest.raises(CorpusRunError, match="step_facts contains sample_id"):
        run_preprocessing_corpus(
            input_path=input_path,
            output_root=output_root,
            run_id="invalid-child",
        )


def test_completion_rejects_duplicate_input_sample_ids(tmp_path: Path) -> None:
    input_path = tmp_path / "duplicate-input.parquet"
    schema = pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("decoder_output", pa.string(), nullable=True),
        ]
    )
    pq.write_table(
        pa.Table.from_arrays(
            [pa.array(["duplicate", "duplicate"]), pa.array([None, None])],
            schema=schema,
        ),
        input_path,
    )

    with pytest.raises(CorpusRunError, match="duplicate sample_id"):
        run_preprocessing_corpus(
            input_path=input_path,
            output_root=tmp_path / "runs",
            run_id="duplicate-input",
        )


def test_omitted_run_id_is_generated(tmp_path: Path) -> None:
    completed_dir = run_preprocessing_corpus(
        input_path=_write_input(tmp_path / "input.parquet"),
        output_root=tmp_path / "runs",
    )

    assert completed_dir.name.startswith("preprocessing-")
    assert _manifest(completed_dir)["run_id"] == completed_dir.name


def test_input_schema_requires_sample_id_and_decoder_output(
    tmp_path: Path,
) -> None:
    invalid_input = tmp_path / "invalid.parquet"
    pq.write_table(
        pa.table({"sample_id": ["one"]}),
        invalid_input,
    )

    with pytest.raises(CorpusRunError, match="decoder_output"):
        run_preprocessing_corpus(
            input_path=invalid_input,
            output_root=tmp_path / "runs",
            run_id="tiny",
        )


@pytest.mark.parametrize("run_id", (".", "..", "nested/run"))
def test_run_id_must_be_safe_path_segment(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(CorpusRunError, match="path segment"):
        run_preprocessing_corpus(
            input_path=_write_input(tmp_path / "input.parquet"),
            output_root=tmp_path / "runs",
            run_id=run_id,
        )
