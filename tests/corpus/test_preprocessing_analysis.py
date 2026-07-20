"""Tests for compact preprocessing corpus analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dr_code.corpus.preprocessing_analysis import (
    PreprocessingAnalysisError,
    analyze_preprocessing_corpus,
)
from dr_code.corpus.preprocessing_run import run_preprocessing_corpus


def _input(path: Path) -> Path:
    schema = pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("source_kind", pa.string(), nullable=False),
            pa.field("source_database", pa.string(), nullable=False),
            pa.field("source_table", pa.string(), nullable=False),
            pa.field("model", pa.string()),
            pa.field("encoder_model", pa.string()),
            pa.field("decoder_model", pa.string()),
            pa.field("prompt_fidelity", pa.string(), nullable=False),
            pa.field("is_retry", pa.bool_(), nullable=False),
            pa.field("is_partial", pa.bool_(), nullable=False),
            pa.field("task_id", pa.string()),
            pa.field("date", pa.timestamp("us", tz="UTC")),
            pa.field("decoder_output", pa.string()),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "sample_id": "missing",
                    "source_kind": "alpha",
                    "source_database": "db",
                    "source_table": "table",
                    "model": None,
                    "encoder_model": None,
                    "decoder_model": None,
                    "prompt_fidelity": "unavailable",
                    "is_retry": False,
                    "is_partial": False,
                    "task_id": "HumanEval/0",
                    "date": None,
                    "decoder_output": None,
                },
                {
                    "sample_id": "blank",
                    "source_kind": "",
                    "source_database": "db",
                    "source_table": "table",
                    "model": "model-a",
                    "encoder_model": None,
                    "decoder_model": None,
                    "prompt_fidelity": "full",
                    "is_retry": True,
                    "is_partial": False,
                    "task_id": "HumanEval/1",
                    "date": None,
                    "decoder_output": " \n\t",
                },
                {
                    "sample_id": "success",
                    "source_kind": "alpha",
                    "source_database": "db",
                    "source_table": "table",
                    "model": "model-a",
                    "encoder_model": "encoder",
                    "decoder_model": "decoder",
                    "prompt_fidelity": "full",
                    "is_retry": False,
                    "is_partial": True,
                    "task_id": "HumanEval/2",
                    "date": None,
                    "decoder_output": "```python\ndef f():\n    return 1\n```",
                },
            ],
            schema=schema,
        ),
        path,
    )
    return path


def _completed_run(tmp_path: Path) -> tuple[Path, Path]:
    corpus = _input(tmp_path / "corpus.parquet")
    run = run_preprocessing_corpus(
        input_path=corpus, output_root=tmp_path / "runs", run_id="tiny"
    )
    return corpus, run


def test_analysis_writes_reconciled_compact_deliverables(
    tmp_path: Path,
) -> None:
    corpus, run = _completed_run(tmp_path)
    output = analyze_preprocessing_corpus(
        corpus_path=corpus, run_dir=run, output_dir=tmp_path / "analysis"
    )

    summary = json.loads((output / "summary.json").read_text())
    assert str(tmp_path) not in json.dumps(summary)
    assert summary["provenance"]["corpus"]["label"] == "corpus.parquet"
    assert summary["denominators"] == {"all": 3, "nonblank": 1, "present": 2}
    source_kinds = summary["source_kind_reconciliation"]
    assert sum(row["sample_count"] for row in source_kinds) == 3
    assert (
        next(row for row in source_kinds if row["source_kind"] == "<blank>")[
            "source_kind_value_state"
        ]
        == "blank"
    )
    assert (
        sum(row["decoder_output_missing_count"] for row in source_kinds) == 1
    )
    assert sum(row["decoder_output_blank_count"] for row in source_kinds) == 1
    assert (
        sum(row["decoder_output_nonblank_count"] for row in source_kinds) == 1
    )
    assert summary["candidate_invariants"]["final_candidate_rows"] == 1
    assert {row["outcome"] for row in summary["outcomes"]} == {
        "decoder_output_blank",
        "decoder_output_missing",
        "function_candidates_extracted",
    }
    missing = next(
        row
        for row in summary["outcomes"]
        if row["outcome"] == "decoder_output_missing"
    )
    assert missing["count_present"] == 0
    assert missing["rate_of_present"] == 0.0
    assert (output / "report.md").is_file()
    viewer = json.loads((output / "viewer-data.json").read_text())
    assert {
        "headline",
        "failure_modes",
        "origin_contribution",
        "outcome_by_dimension",
        "examples",
    }.issubset(viewer)
    assert (output / "tables" / "outcome_by_dimension.parquet").is_file()
    assert len(viewer["examples"]) == 3


def test_analysis_rejects_candidate_count_mismatch(tmp_path: Path) -> None:
    corpus, run = _completed_run(tmp_path)
    candidates_path = run / "candidates.parquet"
    candidates = pq.read_table(candidates_path)
    pq.write_table(candidates.slice(0, 0), candidates_path)

    with pytest.raises(
        PreprocessingAnalysisError, match="candidate count mismatch"
    ):
        analyze_preprocessing_corpus(
            corpus_path=corpus, run_dir=run, output_dir=tmp_path / "analysis"
        )
