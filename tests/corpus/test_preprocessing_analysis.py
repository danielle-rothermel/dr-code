"""Tests for compact preprocessing corpus analysis."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dr_code.corpus.preprocessing_analysis import (
    PreprocessingAnalysisArtifacts,
    PreprocessingAnalysisError,
    analyze_preprocessing_corpus,
)
from dr_code.corpus import analyze_preprocessing_corpus as public_analyze
from dr_code.corpus.preprocessing_run import run_preprocessing_corpus
from dr_code.corpus.candidate_evaluation import (
    MEMBERSHIP_SCHEMA,
    RESULTS_SCHEMA as CANDIDATE_RESULTS_SCHEMA,
    humaneval_metrics_definition,
)
from dr_code.metrics.definition import metrics_definition_hash


def _corpus_schema() -> pa.Schema:
    return pa.schema(
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


def _input(path: Path) -> Path:
    schema = _corpus_schema()
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


def _evaluated_corpus(path: Path) -> Path:
    rows = []
    for index, category in enumerate(
        ("passed", "failed", "timed_out", "infrastructure_failure")
    ):
        rows.append(
            {
                "sample_id": category,
                "source_kind": "alpha" if index < 2 else "beta",
                "source_database": "db",
                "source_table": "table",
                "model": "model-a" if index % 2 == 0 else "model-b",
                "encoder_model": "encoder",
                "decoder_model": "decoder",
                "prompt_fidelity": "full",
                "is_retry": False,
                "is_partial": False,
                "task_id": f"HumanEval/{index}",
                "date": None,
                "decoder_output": (
                    f"def solution_{index}(x):\n    return x + {index}\n"
                ),
            }
        )
    pq.write_table(
        pa.Table.from_pylist(rows, schema=_corpus_schema()), path
    )
    return path


def _failure_browser_corpus(path: Path) -> Path:
    rows = []
    for index in range(101):
        raw = f"just prose {index:03d}"
        if index == 0:
            raw += "x" * 1_500
        rows.append(
            {
                "sample_id": f"prose-{index:03d}",
                "source_kind": "synthetic",
                "source_database": "db",
                "source_table": "failures",
                "model": "model-a",
                "encoder_model": None,
                "decoder_model": "decoder-a",
                "prompt_fidelity": "full",
                "is_retry": False,
                "is_partial": False,
                "task_id": f"HumanEval/{index}",
                "date": None,
                "decoder_output": raw,
            }
        )
    for sample_id, raw in (
        ("assignment", "x = 1"),
        ("blank", " \n\t"),
        ("missing", None),
        ("success", "```python\ndef f():\n    return 1\n```"),
    ):
        rows.append(
            {
                "sample_id": sample_id,
                "source_kind": "synthetic",
                "source_database": "db",
                "source_table": "failures",
                "model": "model-a",
                "encoder_model": None,
                "decoder_model": "decoder-a",
                "prompt_fidelity": "full",
                "is_retry": False,
                "is_partial": False,
                "task_id": "HumanEval/999",
                "date": None,
                "decoder_output": raw,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows, schema=_corpus_schema()), path)
    return path


def _evaluation_relations(
    directory: Path,
    run: Path,
    corpus: Path,
    *,
    all_pass: bool = False,
) -> tuple[Path, Path, Path]:
    directory.mkdir()
    definition = humaneval_metrics_definition()
    metrics_profile = "humaneval-metrics@v1"
    operator = "code_test@1"
    definition_hash = metrics_definition_hash(definition)
    operator_settings = definition.questions[0].settings
    execution_fingerprint = "e" * 64
    corpus_by_id = {
        row["sample_id"]: row for row in pq.read_table(corpus).to_pylist()
    }
    memberships = []
    candidate_results = []
    for candidate in pq.read_table(run / "candidates.parquet").to_pylist():
        sample_id = candidate["sample_id"]
        assert isinstance(sample_id, str)
        source = candidate["cleaned_source"]
        source_sha256 = candidate["source_sha256"]
        source_row = corpus_by_id[sample_id]
        task_id = source_row["task_id"]
        task_fingerprint = hashlib.sha256(task_id.encode()).hexdigest()
        evaluation_key = hashlib.sha256(
            json.dumps(
                {
                    "task_id": task_id,
                    "task_fingerprint": task_fingerprint,
                    "candidate_source_sha256": source_sha256,
                    "metrics_definition_hash": definition_hash,
                    "metrics_profile": metrics_profile,
                    "operator": operator,
                    "settings": operator_settings,
                    "execution_fingerprint": execution_fingerprint,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        category = "passed" if all_pass else sample_id
        infrastructure = category == "infrastructure_failure"
        official_outcome = {
            "passed": "passed",
            "failed": "tests_failed",
            "timed_out": "timed_out",
            "infrastructure_failure": None,
        }[category]
        memberships.append(
            {
                "sample_id": sample_id,
                "candidate_id": candidate["candidate_id"],
                "candidate_index": candidate["candidate_index"],
                "task_id": task_id,
                "source_kind": source_row["source_kind"],
                "source_sha256": source_sha256,
                "task_fingerprint": task_fingerprint,
                "evaluation_key": evaluation_key,
                "metrics_profile": metrics_profile,
                "operator": operator,
            }
        )
        candidate_results.append(
            {
                "evaluation_key": evaluation_key,
                "task_id": task_id,
                "cleaned_source": source,
                "source_sha256": source_sha256,
                "task_fingerprint": task_fingerprint,
                "metrics_profile": metrics_profile,
                "operator": operator,
                "record_status": (
                    "infrastructure_failure" if infrastructure else "measured"
                ),
                "failure_type": "SandboxError" if infrastructure else None,
                "failure_message": (
                    "runtime unavailable" if infrastructure else None
                ),
                "outcome": official_outcome,
                "function_count": None if infrastructure else 1,
                "best_function_name": None if infrastructure else f"solution_{sample_id}",
                "total_cases": None if infrastructure else 2,
                "passed_count": (
                    None if infrastructure else 2 if category == "passed" else 1
                ),
                "failed_count": (
                    None if infrastructure else 1 if category == "failed" else 0
                ),
                "error_count": None if infrastructure else 0,
                "timeout_count": (
                    None if infrastructure else 1 if category == "timed_out" else 0
                ),
                "coverage_complete": None if infrastructure else True,
            }
        )
    membership_path = directory / "candidate_membership.parquet"
    results_path = directory / "candidate_results.parquet"
    pq.write_table(
        pa.Table.from_pylist(memberships, schema=MEMBERSHIP_SCHEMA),
        membership_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            candidate_results, schema=CANDIDATE_RESULTS_SCHEMA
        ),
        results_path,
    )
    manifest_path = directory / "candidate_evaluation_manifest.json"
    manifest = {
        "schema_version": 1,
        "preprocessing_manifest_sha256": _file_sha256(
            run / "manifest.json"
        ),
        "preprocessing_candidates_sha256": _file_sha256(
            run / "candidates.parquet"
        ),
        "preprocessing_results_sha256": _file_sha256(
            run / "results.parquet"
        ),
        "corpus_sha256": _file_sha256(corpus),
        "snapshot_sha256": "s" * 64,
        "metrics_definition": definition.model_dump(mode="json"),
        "metrics_definition_hash": definition_hash,
        "operator": operator,
        "operator_settings": operator_settings,
        "metrics_profile": metrics_profile,
        "python": "3.fixture",
        "python_implementation": "CPython",
        "trusted_source_sha256": {"runner": "r" * 64},
        "sandbox_image": None,
        "runner_identity": "fixture-runner@1",
        "execution_fingerprint": execution_fingerprint,
        "membership_rows": len(memberships),
        "result_rows": len(candidate_results),
        "complete": True,
        "completed_at": "2026-07-19T00:00:00+00:00",
    }
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )
    return membership_path, results_path, manifest_path


def _evaluated_run(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    corpus = _evaluated_corpus(tmp_path / "evaluated-corpus.parquet")
    run = run_preprocessing_corpus(
        input_path=corpus,
        output_root=tmp_path / "evaluated-runs",
        run_id="evaluated",
    )
    membership, candidate_results, manifest = _evaluation_relations(
        tmp_path / "evaluation", run, corpus
    )
    return corpus, run, membership, candidate_results, manifest


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    artifacts = analyze_preprocessing_corpus(
        corpus_path=corpus, run_dir=run, output_dir=tmp_path / "analysis"
    )
    output = artifacts.output_dir
    assert isinstance(artifacts, PreprocessingAnalysisArtifacts)
    assert public_analyze is analyze_preprocessing_corpus
    assert artifacts.summary_path == output / "summary.json"

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
        "operation_contribution",
        "outcome_by_dimension",
        "examples",
    }.issubset(viewer)
    assert "failure_browser" not in viewer
    empty_failure_manifest = json.loads(
        (artifacts.failure_examples_path / "manifest.json").read_text()
    )
    assert empty_failure_manifest == {
        "artifact_id": empty_failure_manifest["artifact_id"],
        "groups": [],
        "schema_version": 1,
        "total_count": 0,
    }
    assert len(empty_failure_manifest["artifact_id"]) == 20
    assert (output / "tables" / "outcome_by_dimension.parquet").is_file()
    assert len(viewer["examples"]) == 3


def test_analysis_rejects_candidate_count_mismatch(tmp_path: Path) -> None:
    corpus, run = _completed_run(tmp_path)
    candidates_path = run / "candidates.parquet"
    candidates = pq.read_table(candidates_path)
    pq.write_table(candidates.slice(0, 0), candidates_path)

    with pytest.raises(
        PreprocessingAnalysisError,
        match="manifest row count does not match candidates",
    ):
        analyze_preprocessing_corpus(
            corpus_path=corpus, run_dir=run, output_dir=tmp_path / "analysis"
        )


def test_analysis_writes_complete_lazy_failure_browser(
    tmp_path: Path,
) -> None:
    corpus = _failure_browser_corpus(tmp_path / "failure-corpus.parquet")
    run = run_preprocessing_corpus(
        input_path=corpus,
        output_root=tmp_path / "failure-runs",
        run_id="failures",
    )
    results_path = run / "results.parquet"
    results_table = pq.read_table(results_path)
    result_rows = results_table.to_pylist()
    next(
        row for row in result_rows if row["sample_id"] == "prose-100"
    )["failed_step"] = "alternate_terminal"
    pq.write_table(
        pa.Table.from_pylist(result_rows, schema=results_table.schema),
        results_path,
    )
    rejections_path = run / "rejections.parquet"
    rejections_table = pq.read_table(rejections_path)
    rejection_rows = rejections_table.to_pylist()
    for row in rejection_rows:
        if row["sample_id"] == "assignment":
            row["reason_code"] = None
    pq.write_table(
        pa.Table.from_pylist(rejection_rows, schema=rejections_table.schema),
        rejections_path,
    )
    first = analyze_preprocessing_corpus(
        corpus_path=corpus,
        run_dir=run,
        output_dir=tmp_path / "analysis-a",
    )
    second = analyze_preprocessing_corpus(
        corpus_path=corpus,
        run_dir=run,
        output_dir=tmp_path / "analysis-b",
    )

    assert first.failure_examples_path == (
        first.output_dir / "failure-examples"
    )
    manifest = json.loads(
        (first.failure_examples_path / "manifest.json").read_text()
    )
    viewer = json.loads(first.viewer_data_path.read_text())
    assert viewer["failure_browser"] == manifest
    assert manifest["schema_version"] == 1
    artifact_id = manifest["artifact_id"]
    assert len(artifact_id) == 20
    int(artifact_id, 16)
    assert manifest["total_count"] == 102
    assert [
        (group["failure_code"], group["failed_step"])
        for group in manifest["groups"]
    ] == [
        ("no_code_candidates", "alternate_terminal"),
        ("no_code_candidates", "extract_candidates"),
        (
            "no_top_level_function_candidate",
            "filter_has_top_level_function",
        ),
    ]

    sample_ids = []
    entries_by_id = {}
    detail_paths = set()
    for group in manifest["groups"]:
        index = json.loads(
            (first.failure_examples_path / group["index_path"]).read_text()
        )
        assert index["schema_version"] == 1
        assert index["failure_code"] == group["failure_code"]
        assert index["failed_step"] == group["failed_step"]
        assert index["count"] == group["count"] == len(index["entries"])
        path_parts = group["index_path"].split("/")
        assert path_parts[0] == artifact_id
        assert path_parts[1].startswith("group-")
        assert group["failure_code"] not in group["index_path"]
        assert [entry["sample_id"] for entry in index["entries"]] == sorted(
            entry["sample_id"] for entry in index["entries"]
        )
        sample_ids.extend(entry["sample_id"] for entry in index["entries"])
        entries_by_id.update(
            (entry["sample_id"], entry) for entry in index["entries"]
        )
        detail_paths.update(
            entry["detail_shard"] for entry in index["entries"]
        )

    assert len(sample_ids) == len(set(sample_ids)) == 102
    assert {"blank", "missing", "success"}.isdisjoint(sample_ids)
    assert len(detail_paths) == 3
    assert all(path.split("/")[0] == artifact_id for path in detail_paths)
    examples_by_id = {}
    for detail_path in sorted(detail_paths):
        shard = json.loads(
            (first.failure_examples_path / detail_path).read_text()
        )
        assert shard["schema_version"] == 1
        assert shard["failure_code"] in {
            "no_code_candidates",
            "no_top_level_function_candidate",
        }
        assert len(shard["examples"]) <= 100
        for example in shard["examples"]:
            assert all(
                fact["step_name"]
                == entries_by_id[example["sample_id"]]["failed_step"]
                for fact in example["facts"]
            )
            examples_by_id[example["sample_id"]] = example
    assert len(examples_by_id) == 102
    long_raw = examples_by_id["prose-000"]["raw_decoder_output"]
    assert long_raw.endswith("… [truncated]")
    assert entries_by_id["prose-000"]["raw_character_count"] > len(long_raw)
    assert entries_by_id["prose-000"]["detail_shard"] in detail_paths
    assert entries_by_id["assignment"]["rejection_reasons"] == ["<null>"]
    assert {
        rejection["reason_code"]
        for rejection in examples_by_id["assignment"]["rejections"]
    } == {"<null>"}
    assert {
        fact["step_name"] for fact in examples_by_id["prose-000"]["facts"]
    } == {"extract_candidates"}

    for relative_path in sorted(
        path.relative_to(first.failure_examples_path)
        for path in first.failure_examples_path.rglob("*.json")
    ):
        assert (first.failure_examples_path / relative_path).read_bytes() == (
            second.failure_examples_path / relative_path
        ).read_bytes()

    changed_corpus = pq.read_table(corpus)
    changed_rows = changed_corpus.to_pylist()
    next(row for row in changed_rows if row["sample_id"] == "prose-000")[
        "decoder_output"
    ] += " changed"
    pq.write_table(
        pa.Table.from_pylist(changed_rows, schema=changed_corpus.schema), corpus
    )
    changed = analyze_preprocessing_corpus(
        corpus_path=corpus,
        run_dir=run,
        output_dir=tmp_path / "analysis-changed",
    )
    changed_manifest = json.loads(
        (changed.failure_examples_path / "manifest.json").read_text()
    )
    assert changed_manifest["artifact_id"] != artifact_id


def test_analysis_joins_candidate_evaluation_and_is_deterministic(
    tmp_path: Path,
) -> None:
    corpus, run, membership, candidate_results, manifest = _evaluated_run(
        tmp_path
    )
    first = analyze_preprocessing_corpus(
        corpus_path=corpus,
        run_dir=run,
        output_dir=tmp_path / "analysis-a",
        candidate_membership_path=membership,
        candidate_results_path=candidate_results,
        candidate_evaluation_manifest_path=manifest,
    )
    second = analyze_preprocessing_corpus(
        corpus_path=corpus,
        run_dir=run,
        output_dir=tmp_path / "analysis-b",
        candidate_membership_path=membership,
        candidate_results_path=candidate_results,
        candidate_evaluation_manifest_path=manifest,
    )

    summary = json.loads(first.summary_path.read_text())
    evaluation = summary["candidate_evaluation"]
    assert evaluation["available"] is True
    assert evaluation["candidate_membership_count"] == 4
    assert evaluation["provenance"]["semantic_coordinates"][
        "runner_identity"
    ] == "fixture-runner@1"
    assert {
        row["test_outcome"]: row["candidate_count"]
        for row in evaluation["candidate_outcomes"]
    } == {
        "failed": 1,
        "infrastructure_failure": 1,
        "passed": 1,
        "timed_out": 1,
    }
    assert {
        row["best_test_outcome"]: row["sample_count"]
        for row in evaluation["sample_best_outcomes"]
    } == {
        "failed": 1,
        "infrastructure_failure": 1,
        "passed": 1,
        "timed_out": 1,
    }
    by_dimension = pq.read_table(
        first.table_paths["test_success_by_dimension"]
    ).to_pylist()
    alpha = next(
        row
        for row in by_dimension
        if row["dimension"] == "source_kind" and row["value"] == "alpha"
    )
    assert alpha["sample_count"] == 2
    assert alpha["passed_count"] == 1
    assert alpha["pass_rate_of_evaluated_samples"] == 0.5
    viewer = json.loads(first.viewer_data_path.read_text())
    evaluation_viewer = viewer["candidate_evaluation"]
    assert evaluation_viewer["test_success_by_operation"]
    assert all(
        set(row["operation"]) == {"kind", "details"}
        for row in evaluation_viewer["test_success_by_operation"]
    )
    assert {
        example["test_outcome"] for example in evaluation_viewer["examples"]
    } == {"passed", "failed", "timed_out", "infrastructure_failure"}
    assert all(
        len(example["cleaned_source"]) <= 1_214
        for example in evaluation_viewer["examples"]
    )
    assert "Candidate evaluation funnel" in first.report_path.read_text()
    assert "fixture-runner@1" in first.report_path.read_text()

    for relative_path in (
        "summary.json",
        "viewer-data.json",
        "report.md",
    ):
        assert (first.output_dir / relative_path).read_bytes() == (
            second.output_dir / relative_path
        ).read_bytes()
    for name in first.table_paths:
        assert first.table_paths[name].read_bytes() == second.table_paths[
            name
        ].read_bytes()


def test_analysis_reports_completed_evaluation_with_zero_candidates(
    tmp_path: Path,
) -> None:
    corpus = _input(tmp_path / "source-corpus.parquet")
    zero_candidate_corpus = tmp_path / "zero-candidate-corpus.parquet"
    pq.write_table(pq.read_table(corpus).slice(0, 1), zero_candidate_corpus)
    run = run_preprocessing_corpus(
        input_path=zero_candidate_corpus,
        output_root=tmp_path / "zero-candidate-runs",
        run_id="zero-candidate",
    )
    membership, candidate_results, manifest = _evaluation_relations(
        tmp_path / "zero-candidate-evaluation",
        run,
        zero_candidate_corpus,
    )

    artifacts = analyze_preprocessing_corpus(
        corpus_path=zero_candidate_corpus,
        run_dir=run,
        output_dir=tmp_path / "zero-candidate-analysis",
        candidate_membership_path=membership,
        candidate_results_path=candidate_results,
        candidate_evaluation_manifest_path=manifest,
    )

    summary = json.loads(artifacts.summary_path.read_text())
    evaluation = summary["candidate_evaluation"]
    assert evaluation["available"] is True
    assert evaluation["candidate_membership_count"] == 0
    assert all(row["rate"] is None for row in evaluation["funnel"])
    report = artifacts.report_path.read_text()
    assert "| extracted final candidates | 0 | n/a |" in report
    assert "No extracted samples were available for candidate testing." in report


def test_analysis_rejects_incomplete_or_invalid_evaluation_join(
    tmp_path: Path,
) -> None:
    corpus, run, membership, candidate_results, manifest = _evaluated_run(
        tmp_path
    )

    with pytest.raises(
        PreprocessingAnalysisError, match="must be supplied together"
    ):
        analyze_preprocessing_corpus(
            corpus_path=corpus,
            run_dir=run,
            output_dir=tmp_path / "missing-pair",
            candidate_membership_path=membership,
            candidate_results_path=candidate_results,
        )

    rows = pq.read_table(membership).to_pylist()
    rows[0]["source_sha256"] = "0" * 64
    corrupt_membership = tmp_path / "corrupt-membership.parquet"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=MEMBERSHIP_SCHEMA),
        corrupt_membership,
    )
    with pytest.raises(
        PreprocessingAnalysisError,
        match="membership/result coordinates differ",
    ):
        analyze_preprocessing_corpus(
            corpus_path=corpus,
            run_dir=run,
            output_dir=tmp_path / "invalid-join",
            candidate_membership_path=corrupt_membership,
            candidate_results_path=candidate_results,
            candidate_evaluation_manifest_path=manifest,
        )

    with pytest.raises(
        PreprocessingAnalysisError, match="file does not exist"
    ):
        analyze_preprocessing_corpus(
            corpus_path=corpus,
            run_dir=run,
            output_dir=tmp_path / "missing-file",
            candidate_membership_path=tmp_path / "does-not-exist.parquet",
            candidate_results_path=candidate_results,
            candidate_evaluation_manifest_path=manifest,
        )


@pytest.mark.parametrize(
    ("sample_id", "changes", "error"),
    [
        (
            "passed",
            {"passed_count": 0, "failed_count": 2},
            "outcome contradicts measured facts",
        ),
        (
            "passed",
            {"outcome": "tests_failed"},
            "outcome contradicts measured facts",
        ),
        (
            "timed_out",
            {"outcome": "passed"},
            "outcome contradicts measured facts",
        ),
        (
            "passed",
            {"passed_count": 1},
            "coverage contradicts case counts",
        ),
        (
            "passed",
            {"coverage_complete": False},
            "coverage contradicts case counts",
        ),
        (
            "passed",
            {"failed_count": -1},
            "invalid non-negative integer",
        ),
        (
            "passed",
            {"failure_type": "UnexpectedError"},
            "must not have failure diagnostics",
        ),
        (
            "infrastructure_failure",
            {"failure_message": None},
            "requires failure diagnostics",
        ),
        (
            "passed",
            {
                "function_count": 0,
                "best_function_name": None,
                "coverage_complete": False,
                "outcome": "no_top_level_functions",
            },
            "case statuses without a selected function",
        ),
    ],
)
def test_analysis_rejects_contradictory_candidate_test_facts(
    tmp_path: Path,
    sample_id: str,
    changes: dict[str, object],
    error: str,
) -> None:
    corpus, run, membership, candidate_results, manifest = _evaluated_run(
        tmp_path
    )
    membership_rows = pq.read_table(membership).to_pylist()
    evaluation_key = next(
        row["evaluation_key"]
        for row in membership_rows
        if row["sample_id"] == sample_id
    )
    rows = pq.read_table(candidate_results).to_pylist()
    target = next(
        row for row in rows if row["evaluation_key"] == evaluation_key
    )
    target.update(changes)
    corrupt_results = tmp_path / "contradictory-results.parquet"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=CANDIDATE_RESULTS_SCHEMA),
        corrupt_results,
    )

    with pytest.raises(PreprocessingAnalysisError, match=error):
        analyze_preprocessing_corpus(
            corpus_path=corpus,
            run_dir=run,
            output_dir=tmp_path / "contradictory-analysis",
            candidate_membership_path=membership,
            candidate_results_path=corrupt_results,
            candidate_evaluation_manifest_path=manifest,
        )


def test_analysis_accepts_duplicate_name_stacked_status_counts(
    tmp_path: Path,
) -> None:
    corpus, run, membership, candidate_results, manifest = _evaluated_run(
        tmp_path
    )
    membership_rows = pq.read_table(membership).to_pylist()
    evaluation_key = next(
        row["evaluation_key"]
        for row in membership_rows
        if row["sample_id"] == "passed"
    )
    rows = pq.read_table(candidate_results).to_pylist()
    target = next(
        row for row in rows if row["evaluation_key"] == evaluation_key
    )
    target.update(
        {
            "function_count": 2,
            "total_cases": 2,
            "passed_count": 4,
            "failed_count": 0,
            "error_count": 0,
            "timeout_count": 0,
            "coverage_complete": False,
            "outcome": "evaluation_incomplete",
        }
    )
    stacked_results = tmp_path / "stacked-results.parquet"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=CANDIDATE_RESULTS_SCHEMA),
        stacked_results,
    )

    artifacts = analyze_preprocessing_corpus(
        corpus_path=corpus,
        run_dir=run,
        output_dir=tmp_path / "stacked-analysis",
        candidate_membership_path=membership,
        candidate_results_path=stacked_results,
        candidate_evaluation_manifest_path=manifest,
    )

    summary = json.loads(artifacts.summary_path.read_text())
    outcomes = summary["candidate_evaluation"]["candidate_outcomes"]
    assert sum(
        row["candidate_count"]
        for row in outcomes
        if row["official_outcome"] == "evaluation_incomplete"
    ) == 1


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("missing_execution_fingerprint", "schema v1 manifest is missing"),
        ("membership_row_count", "membership_rows mismatch"),
        ("result_row_count", "result_rows mismatch"),
        ("corpus_hash", "corpus_sha256 mismatch"),
        (
            "preprocessing_manifest_hash",
            "preprocessing_manifest_sha256 mismatch",
        ),
        (
            "preprocessing_candidates_hash",
            "preprocessing_candidates_sha256 mismatch",
        ),
        (
            "preprocessing_results_hash",
            "preprocessing_results_sha256 mismatch",
        ),
        ("incomplete", "manifest is incomplete"),
        ("schema", "unsupported.*schema_version"),
        ("execution_fingerprint", "evaluation_key does not match"),
    ],
)
def test_analysis_rejects_invalid_evaluation_manifest(
    tmp_path: Path, mutation: str, error: str
) -> None:
    corpus, run, membership, candidate_results, manifest = _evaluated_run(
        tmp_path
    )
    value = json.loads(manifest.read_text())
    if mutation == "missing_execution_fingerprint":
        value.pop("execution_fingerprint")
    elif mutation == "membership_row_count":
        value["membership_rows"] += 1
    elif mutation == "result_row_count":
        value["result_rows"] += 1
    elif mutation == "corpus_hash":
        value["corpus_sha256"] = "0" * 64
    elif mutation.endswith("_hash"):
        value[f"{mutation.removesuffix('_hash')}_sha256"] = "0" * 64
    elif mutation == "incomplete":
        value["complete"] = False
    elif mutation == "execution_fingerprint":
        value["execution_fingerprint"] = "0" * 64
    else:
        value["schema_version"] = 2
    invalid_manifest = tmp_path / f"{mutation}.json"
    invalid_manifest.write_text(json.dumps(value))

    with pytest.raises(PreprocessingAnalysisError, match=error):
        analyze_preprocessing_corpus(
            corpus_path=corpus,
            run_dir=run,
            output_dir=tmp_path / f"{mutation}-analysis",
            candidate_membership_path=membership,
            candidate_results_path=candidate_results,
            candidate_evaluation_manifest_path=invalid_manifest,
        )


@pytest.mark.parametrize(
    ("field", "mixed_value"),
    [("metrics_profile", "mixed@1"), ("operator", "other_operator@1")],
)
def test_analysis_rejects_mixed_evaluation_coordinates(
    tmp_path: Path, field: str, mixed_value: str
) -> None:
    corpus, run, membership, candidate_results, manifest = _evaluated_run(
        tmp_path
    )
    result_rows = pq.read_table(candidate_results).to_pylist()
    membership_rows = pq.read_table(membership).to_pylist()
    changed_key = result_rows[0]["evaluation_key"]
    result_rows[0][field] = mixed_value
    changed_membership = next(
        row
        for row in membership_rows
        if row["evaluation_key"] == changed_key
    )
    changed_membership[field] = mixed_value
    mixed_results = tmp_path / "mixed-results.parquet"
    mixed_membership = tmp_path / "mixed-membership.parquet"
    pq.write_table(
        pa.Table.from_pylist(result_rows, schema=CANDIDATE_RESULTS_SCHEMA),
        mixed_results,
    )
    pq.write_table(
        pa.Table.from_pylist(membership_rows, schema=MEMBERSHIP_SCHEMA),
        mixed_membership,
    )

    with pytest.raises(PreprocessingAnalysisError, match=field):
        analyze_preprocessing_corpus(
            corpus_path=corpus,
            run_dir=run,
            output_dir=tmp_path / "mixed-analysis",
            candidate_membership_path=mixed_membership,
            candidate_results_path=mixed_results,
            candidate_evaluation_manifest_path=manifest,
        )


def test_analysis_rejects_invalid_evaluation_manifest_json(
    tmp_path: Path,
) -> None:
    corpus, run, membership, candidate_results, _manifest = _evaluated_run(
        tmp_path
    )
    invalid_manifest = tmp_path / "invalid-manifest.json"
    invalid_manifest.write_text("{")

    with pytest.raises(
        PreprocessingAnalysisError, match="manifest is invalid JSON"
    ):
        analyze_preprocessing_corpus(
            corpus_path=corpus,
            run_dir=run,
            output_dir=tmp_path / "invalid-manifest-analysis",
            candidate_membership_path=membership,
            candidate_results_path=candidate_results,
            candidate_evaluation_manifest_path=invalid_manifest,
        )


def test_report_handles_all_success_without_rejections(tmp_path: Path) -> None:
    corpus = _evaluated_corpus(tmp_path / "all-success-corpus.parquet")
    run = run_preprocessing_corpus(
        input_path=corpus,
        output_root=tmp_path / "all-success-runs",
        run_id="all-success",
    )
    membership, candidate_results, manifest = _evaluation_relations(
        tmp_path / "all-success-evaluation",
        run,
        corpus,
        all_pass=True,
    )

    artifacts = analyze_preprocessing_corpus(
        corpus_path=corpus,
        run_dir=run,
        output_dir=tmp_path / "all-success-analysis",
        candidate_membership_path=membership,
        candidate_results_path=candidate_results,
        candidate_evaluation_manifest_path=manifest,
    )

    assert "100.00%" in artifacts.report_path.read_text()


def test_report_handles_corpus_without_candidate_origins(tmp_path: Path) -> None:
    template = _input(tmp_path / "template.parquet")
    corpus = tmp_path / "missing-only.parquet"
    pq.write_table(pq.read_table(template).slice(0, 1), corpus)
    run = run_preprocessing_corpus(
        input_path=corpus,
        output_root=tmp_path / "missing-only-runs",
        run_id="missing-only",
    )

    artifacts = analyze_preprocessing_corpus(
        corpus_path=corpus,
        run_dir=run,
        output_dir=tmp_path / "missing-only-analysis",
    )

    assert artifacts.report_path.is_file()
