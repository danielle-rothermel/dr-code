from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dr_code.corpus.candidate_evaluation import MEMBERSHIP_SCHEMA
from dr_code.corpus.candidate_evaluation import (
    RESULTS_SCHEMA as EVALUATION_RESULTS_SCHEMA,
)
from dr_code.corpus.preprocessing_artifacts import (
    CANDIDATES_SCHEMA,
    RESULTS_SCHEMA,
)
from dr_code.corpus.preprocessing_comparison import (
    PreprocessingComparisonError,
    _sample_transition_rows,
    compare_preprocessing_runs,
)
from viewer.helpers import write_bundle


def test_comparison_is_deterministic_and_normalizes_legacy_provenance(
    tmp_path: Path,
) -> None:
    before = write_bundle(
        tmp_path / "before",
        run_id="before",
        preprocessing_schema_version=1,
    )
    after = write_bundle(
        tmp_path / "after",
        run_id="after",
        corpus_path=before.corpus_path,
        preprocessing_schema_version=2,
    )

    first = compare_preprocessing_runs(
        corpus_path=before.corpus_path,
        before_run=before.preprocessing_manifest_path.parent,
        after_run=after.preprocessing_manifest_path.parent,
        before_evaluation=before.evaluation_manifest_path.parent,
        after_evaluation=after.evaluation_manifest_path.parent,
        output_dir=tmp_path / "comparison-1",
    )
    second = compare_preprocessing_runs(
        corpus_path=before.corpus_path,
        before_run=before.preprocessing_manifest_path.parent,
        after_run=after.preprocessing_manifest_path.parent,
        before_evaluation=before.evaluation_manifest_path.parent,
        after_evaluation=after.evaluation_manifest_path.parent,
        output_dir=tmp_path / "comparison-2",
    )

    summary = json.loads(first.summary_path.read_text(encoding="utf-8"))
    provenance = pq.read_table(
        first.relation_paths["provenance_path_deltas"]
    ).to_pylist()
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))

    assert summary["sample_outcome_transitions"]["identity_rows"] == 9
    assert summary["candidate_changes"]["before_count"] == 2
    assert summary["candidate_changes"]["after_count"] == 2
    assert summary["provenance_path_deltas"]["before_count"] == 2
    assert summary["provenance_path_deltas"]["after_count"] == 2
    assert summary["evaluation"]["membership_changes"]["identity_rows"] == 2
    assert summary["evaluation"]["result_changes"]["identity_rows"] == 2
    assert summary["reconciliation"]["sample_rows_match_corpus"] is True
    assert {row["change"] for row in provenance} == {"added", "removed"}
    assert all(
        set(json.loads(row["path_json"])) == {"path"} for row in provenance
    )
    assert manifest["before"]["preprocessing_schema_version"] == 1
    assert manifest["after"]["preprocessing_schema_version"] == 2
    assert manifest["complete"] is True
    filenames = [path.name for path in first.relation_paths.values()]
    filenames.extend(["comparison_summary.json", "comparison_manifest.json"])
    for name in filenames:
        assert (first.output_dir / name).read_bytes() == (
            second.output_dir / name
        ).read_bytes()


def test_comparison_exports_outcome_candidate_source_and_evaluation_changes(
    tmp_path: Path,
) -> None:
    before = write_bundle(tmp_path / "before", run_id="before")
    after = write_bundle(
        tmp_path / "after",
        run_id="after",
        corpus_path=before.corpus_path,
    )
    _change_after_preprocessing(after.preprocessing_manifest_path.parent)
    assert after.evaluation_manifest_path is not None
    _change_after_evaluation(after.evaluation_manifest_path.parent)

    artifacts = compare_preprocessing_runs(
        corpus_path=before.corpus_path,
        before_run=before.preprocessing_manifest_path.parent,
        after_run=after.preprocessing_manifest_path.parent,
        before_evaluation=before.evaluation_manifest_path.parent,
        after_evaluation=after.evaluation_manifest_path.parent,
        output_dir=tmp_path / "comparison",
    )

    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    samples = pq.read_table(
        artifacts.relation_paths["sample_outcome_transitions"]
    ).to_pylist()
    candidates = pq.read_table(
        artifacts.relation_paths["candidate_changes"]
    ).to_pylist()
    evaluation_results = pq.read_table(
        artifacts.relation_paths["evaluation_result_changes"]
    ).to_pylist()

    assert summary["sample_outcome_transitions"]["outcome_changed_count"] == 1
    assert summary["candidate_changes"]["changed_identity_rows"] == 2
    assert summary["candidate_changes"]["count_delta"] == -1
    assert (
        summary["evaluation"]["result_changes"]["changed_identity_rows"] == 2
    )
    assert summary["evaluation"]["coordinates"]["changed_fields"] == [
        "runner_identity"
    ]
    assert sum(row["outcome_changed"] for row in samples) == 1
    assert {row["change"] for row in candidates} == {"modified", "removed"}
    assert sum(row["source_changed"] for row in candidates) == 1
    assert {
        row["candidate_id"]: row["change"] for row in evaluation_results
    } == {"candidate-fail": "removed", "candidate-pass": "modified"}


def test_comparison_refuses_existing_output_and_one_sided_evaluation(
    tmp_path: Path,
) -> None:
    before = write_bundle(tmp_path / "before", run_id="before")
    after = write_bundle(
        tmp_path / "after",
        run_id="after",
        corpus_path=before.corpus_path,
    )
    destination = tmp_path / "existing"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("untouched", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        compare_preprocessing_runs(
            corpus_path=before.corpus_path,
            before_run=before.preprocessing_manifest_path.parent,
            after_run=after.preprocessing_manifest_path.parent,
            output_dir=destination,
        )
    assert marker.read_text(encoding="utf-8") == "untouched"

    with pytest.raises(
        PreprocessingComparisonError, match="supplied together"
    ):
        compare_preprocessing_runs(
            corpus_path=before.corpus_path,
            before_run=before.preprocessing_manifest_path.parent,
            after_run=after.preprocessing_manifest_path.parent,
            before_evaluation=before.evaluation_manifest_path.parent,
            output_dir=tmp_path / "one-sided",
        )
    assert not (tmp_path / "one-sided").exists()


def test_sample_transition_marks_attribution_only_change(
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
    results_path = after.results_path
    rows = pq.read_table(results_path).to_pylist()
    for row in rows:
        if row["sample_id"] == "no-code":
            row["cause"] = "changed attribution"
            row["propagated_through"] = ["extract_candidates"]
    pq.write_table(
        pa.Table.from_pylist(rows, schema=RESULTS_SCHEMA), results_path
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
    changed = [row for row in transitions if row["semantic_result_changed"]]
    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert [(row["sample_id"], row["changed_fields"]) for row in changed] == [
        ("no-code", ["cause", "propagated_through"])
    ]
    assert changed[0]["before_propagated_through"] == []
    assert changed[0]["after_propagated_through"] == ["extract_candidates"]
    assert summary["sample_outcome_transitions"]["changed_identity_rows"] == 1
    assert (
        summary["sample_outcome_transitions"]["semantic_result_changed_count"]
        == 1
    )


def test_sample_transition_marks_final_candidate_count_only_change(
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
    results_path = after.results_path
    results = pq.read_table(results_path).to_pylist()
    for row in results:
        if row["sample_id"] == "pass":
            row["final_candidate_count"] = 2
    pq.write_table(
        pa.Table.from_pylist(results, schema=RESULTS_SCHEMA), results_path
    )
    candidates_path = after.candidates_path
    candidates = pq.read_table(candidates_path).to_pylist()
    extra = dict(next(row for row in candidates if row["sample_id"] == "pass"))
    source = "def pass_me_too():\n    return 2"
    extra.update(
        candidate_id="candidate-pass-extra",
        candidate_index=1,
        cleaned_source=source,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    )
    candidates.append(extra)
    pq.write_table(
        pa.Table.from_pylist(candidates, schema=CANDIDATES_SCHEMA),
        candidates_path,
    )
    manifest_path = after.preprocessing_manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["relation_totals"]["candidates"] = len(candidates)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
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
    changed = [row for row in transitions if row["semantic_result_changed"]]
    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert [(row["sample_id"], row["changed_fields"]) for row in changed] == [
        ("pass", ["final_candidate_count"])
    ]
    assert summary["sample_outcome_transitions"]["changed_identity_rows"] == 1
    assert (
        summary["sample_outcome_transitions"]["semantic_result_changed_count"]
        == 1
    )


@pytest.mark.parametrize(
    ("field", "after_value"),
    [
        ("raw_output_sha256", "b" * 64),
        ("decoder_output_presence", "changed-presence"),
        ("outcome", "changed-outcome"),
        ("outcome_code", "changed-code"),
        ("failure_code", "changed-failure"),
        ("failed_step", "changed-step"),
        ("cause", "changed-cause"),
        ("propagated_through", ["changed-step"]),
        ("final_candidate_count", 2),
    ],
)
def test_each_semantic_result_field_marks_sample_transition_changed(
    field: str, after_value: object
) -> None:
    before = {
        "raw_output_sha256": "a" * 64,
        "decoder_output_presence": "present",
        "outcome": "function_candidates_extracted",
        "outcome_code": "function_candidates_extracted",
        "failure_code": None,
        "failed_step": None,
        "cause": None,
        "propagated_through": None,
        "final_candidate_count": 1,
    }
    after = {**before, field: after_value}

    row = _sample_transition_rows(
        {"sample": before}, {"sample": after}, ("sample",)
    )[0]

    assert row["semantic_result_changed"] is True
    assert row["changed_fields"] == [field]
    assert row["change"] == "semantic_result_changed"


def test_comparison_rejects_run_from_a_different_corpus(
    tmp_path: Path,
) -> None:
    before = write_bundle(
        tmp_path / "before", run_id="before", with_evaluation=False
    )
    after = write_bundle(
        tmp_path / "after", run_id="after", with_evaluation=False
    )
    after_corpus = after.corpus_path
    table = pq.read_table(after_corpus)
    rows = table.to_pylist()
    rows[0]["decoder_output"] = "different corpus identity"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=table.schema), after_corpus
    )
    manifest_path = after.preprocessing_manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["input"]["sha256"] = _sha256_file(after_corpus)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(
        PreprocessingComparisonError, match="after immutable bundle is invalid"
    ):
        compare_preprocessing_runs(
            corpus_path=before.corpus_path,
            before_run=before.preprocessing_manifest_path.parent,
            after_run=after.preprocessing_manifest_path.parent,
            output_dir=tmp_path / "comparison",
        )


def test_comparison_cli_writes_the_requested_new_directory(
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
    destination = tmp_path / "comparison"

    result = subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).parents[2]
                / "scripts"
                / "compare_preprocessing_runs.py"
            ),
            "--corpus",
            str(before.corpus_path),
            "--before-run",
            str(before.preprocessing_manifest_path.parent),
            "--after-run",
            str(after.preprocessing_manifest_path.parent),
            "--output-dir",
            str(destination),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(destination.resolve())
    assert (destination / "comparison_manifest.json").is_file()


def _change_after_preprocessing(run: Path) -> None:
    results_path = run / "results.parquet"
    results = pq.read_table(results_path).to_pylist()
    for row in results:
        if row["sample_id"] == "fail":
            row["outcome"] = "changed_outcome"
            row["outcome_code"] = "changed_outcome"
            row["final_candidate_count"] = 0
    pq.write_table(
        pa.Table.from_pylist(results, schema=RESULTS_SCHEMA), results_path
    )

    candidates_path = run / "candidates.parquet"
    candidates = pq.read_table(candidates_path).to_pylist()
    changed = []
    for row in candidates:
        if row["candidate_id"] == "candidate-fail":
            continue
        source = "def pass_me():\n    return 2"
        row["cleaned_source"] = source
        row["source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
        changed.append(row)
    pq.write_table(
        pa.Table.from_pylist(changed, schema=CANDIDATES_SCHEMA),
        candidates_path,
    )
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["relation_totals"]["candidates"] = len(changed)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )


def _change_after_evaluation(evaluation: Path) -> None:
    membership_path = evaluation / "candidate_membership.parquet"
    memberships = pq.read_table(membership_path).to_pylist()
    source = "def pass_me():\n    return 2"
    source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    changed_memberships = []
    for row in memberships:
        if row["evaluation_key"] == "key-fail":
            continue
        row["source_sha256"] = source_sha256
        changed_memberships.append(row)
    pq.write_table(
        pa.Table.from_pylist(changed_memberships, schema=MEMBERSHIP_SCHEMA),
        membership_path,
    )

    results_path = evaluation / "candidate_results.parquet"
    rows = pq.read_table(results_path).to_pylist()
    changed_results = []
    for row in rows:
        if row["evaluation_key"] == "key-fail":
            continue
        row["cleaned_source"] = source
        row["source_sha256"] = source_sha256
        row["outcome"] = "failed"
        row["passed_count"] = 0
        row["failed_count"] = 1
        changed_results.append(row)
    pq.write_table(
        pa.Table.from_pylist(
            changed_results, schema=EVALUATION_RESULTS_SCHEMA
        ),
        results_path,
    )
    manifest_path = evaluation / "candidate_evaluation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runner_identity"] = "changed-runner"
    manifest["preprocessing_manifest_sha256"] = _sha256_file(
        evaluation.parent / "run" / "manifest.json"
    )
    manifest["preprocessing_candidates_sha256"] = _sha256_file(
        evaluation.parent / "run" / "candidates.parquet"
    )
    manifest["preprocessing_results_sha256"] = _sha256_file(
        evaluation.parent / "run" / "results.parquet"
    )
    manifest["membership_rows"] = len(changed_memberships)
    manifest["result_rows"] = len(changed_results)
    manifest["candidate_membership_sha256"] = _sha256_file(membership_path)
    manifest["candidate_results_sha256"] = _sha256_file(results_path)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
