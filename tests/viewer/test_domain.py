from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dr_code.corpus.preprocessing_artifacts import (
    CANDIDATES_SCHEMA,
    STEP_FACTS_SCHEMA,
)
from dr_code.viewer.domain import RunDescriptor, RunValidationError
from viewer.helpers import write_bundle


def test_descriptor_validates_complete_manifest_backed_bundle(
    tmp_path,
) -> None:
    descriptor = write_bundle(tmp_path / "bundle")

    assert descriptor.run_id == "fixture-run"
    assert descriptor.preprocessing_schema_version == 2
    assert descriptor.has_evaluation
    assert descriptor.corpus_path.is_absolute()
    assert set(descriptor.artifact_sha256) == {
        "results",
        "candidates",
        "step_facts",
        "rejections",
        "candidate_membership",
        "candidate_results",
    }
    assert descriptor.evaluation_manifest_path is not None
    evaluation_manifest = json.loads(
        descriptor.evaluation_manifest_path.read_text(encoding="utf-8")
    )
    assert (
        evaluation_manifest["candidate_membership_sha256"]
        == descriptor.artifact_sha256["candidate_membership"]
    )
    assert (
        evaluation_manifest["candidate_results_sha256"]
        == descriptor.artifact_sha256["candidate_results"]
    )
    # New unified runs expose the canonical eval-kernel identity.
    assert len(descriptor.definition_identity) == 64


def test_descriptor_reads_legacy_definition_hash_manifest(tmp_path) -> None:
    # Decision 3: the viewer keeps reading OLD artifacts read-only. A legacy
    # manifest has only the BLAKE2 definition_hash and no canonical identity.
    bundle = write_bundle(tmp_path / "legacy")
    manifest_path = bundle.preprocessing_manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    legacy_hash = manifest.pop("preprocessing_definition_identity")
    manifest.pop("identity_scheme", None)
    manifest["definition_hash"] = legacy_hash
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    reloaded = RunDescriptor.from_paths(
        label=bundle.label,
        corpus_path=bundle.corpus_path,
        preprocessing=manifest_path.parent,
        candidate_evaluation=None,
    )
    assert reloaded.definition_identity == legacy_hash


def test_descriptor_accepts_legacy_evaluation_without_artifact_hashes(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    write_bundle(bundle)
    manifest_path = (
        bundle / "evaluation" / "candidate_evaluation_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("candidate_membership_sha256")
    manifest.pop("candidate_results_sha256")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    descriptor = RunDescriptor.from_paths(
        label="legacy-evaluation",
        corpus_path=bundle / "corpus.parquet",
        preprocessing=bundle / "run",
        candidate_evaluation=bundle / "evaluation",
    )

    assert descriptor.has_evaluation


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("membership_mismatch", "candidate_membership_sha256 mismatch"),
        ("results_mismatch", "candidate_results_sha256 mismatch"),
        ("partial_hashes", "missing artifact hash field"),
        ("malformed_hash", "invalid 'candidate_results_sha256'"),
    ],
)
def test_descriptor_rejects_invalid_evaluation_artifact_hashes(
    tmp_path: Path, mutation: str, error: str
) -> None:
    bundle = tmp_path / "bundle"
    write_bundle(bundle)
    manifest_path = (
        bundle / "evaluation" / "candidate_evaluation_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "membership_mismatch":
        manifest["candidate_membership_sha256"] = "0" * 64
    elif mutation == "results_mismatch":
        manifest["candidate_results_sha256"] = "0" * 64
    elif mutation == "partial_hashes":
        manifest.pop("candidate_results_sha256")
    else:
        manifest["candidate_results_sha256"] = "not-a-sha256"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RunValidationError, match=error):
        RunDescriptor.from_paths(
            label="bad",
            corpus_path=bundle / "corpus.parquet",
            preprocessing=bundle / "run",
            candidate_evaluation=bundle / "evaluation",
        )


def test_descriptor_accepts_legacy_preprocessing_artifacts(tmp_path) -> None:
    descriptor = write_bundle(
        tmp_path / "legacy",
        preprocessing_schema_version=1,
    )

    assert descriptor.preprocessing_schema_version == 1


def test_descriptor_file_resolves_relative_paths(tmp_path) -> None:
    bundle = tmp_path / "bundle"
    expected = write_bundle(bundle)
    descriptor_path = bundle / "descriptor.json"
    descriptor_path.write_text(
        json.dumps(
            {
                "label": "relative",
                "corpus": "corpus.parquet",
                "preprocessing": "run",
                "candidate_evaluation": "evaluation",
            }
        ),
        encoding="utf-8",
    )

    actual = RunDescriptor.from_file(descriptor_path)

    assert actual.run_id == expected.run_id
    assert actual.label == "relative"
    assert actual.corpus_sha256 == expected.corpus_sha256


def test_descriptor_rejects_artifact_schema_drift(tmp_path) -> None:
    bundle = tmp_path / "bundle"
    write_bundle(bundle)
    results_path = bundle / "run" / "results.parquet"
    table = pq.read_table(results_path).drop(["cause"])
    pq.write_table(table, results_path)

    with pytest.raises(RunValidationError, match="unexpected schema"):
        RunDescriptor.from_paths(
            label="bad",
            corpus_path=bundle / "corpus.parquet",
            preprocessing=bundle / "run",
        )


def test_descriptor_rejects_invalid_origin_details_json(tmp_path) -> None:
    bundle = tmp_path / "bundle"
    write_bundle(bundle, with_evaluation=False)
    candidates_path = bundle / "run" / "candidates.parquet"
    rows = pq.read_table(candidates_path).to_pylist()
    rows[0]["origins"][0]["path"][0]["details_json"] = "not-json"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=CANDIDATES_SCHEMA), candidates_path
    )

    with pytest.raises(RunValidationError, match="details_json is invalid"):
        RunDescriptor.from_paths(
            label="bad",
            corpus_path=bundle / "corpus.parquet",
            preprocessing=bundle / "run",
        )


def test_descriptor_rejects_corpus_fingerprint_mismatch(tmp_path) -> None:
    bundle = tmp_path / "bundle"
    write_bundle(bundle)
    manifest_path = bundle / "run" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["input"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RunValidationError, match="corpus fingerprint"):
        RunDescriptor.from_paths(
            label="bad",
            corpus_path=bundle / "corpus.parquet",
            preprocessing=bundle / "run",
        )


def test_descriptor_rejects_cross_run_evaluation_bundle(tmp_path) -> None:
    bundle = tmp_path / "bundle"
    write_bundle(bundle)
    manifest_path = (
        bundle / "evaluation" / "candidate_evaluation_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["preprocessing_results_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RunValidationError, match="results_sha256 mismatch"):
        RunDescriptor.from_paths(
            label="bad",
            corpus_path=bundle / "corpus.parquet",
            preprocessing=bundle / "run",
            candidate_evaluation=bundle / "evaluation",
        )


def test_descriptor_rejects_reordered_viewer_stage_contract(tmp_path) -> None:
    bundle = tmp_path / "bundle"
    write_bundle(bundle, with_evaluation=False)
    manifest_path = bundle / "run" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["definition"]["steps"].reverse()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RunValidationError, match="out of order"):
        RunDescriptor.from_paths(
            label="bad",
            corpus_path=bundle / "corpus.parquet",
            preprocessing=bundle / "run",
        )


def test_descriptor_rejects_duplicate_persisted_stage_fact(tmp_path) -> None:
    bundle = tmp_path / "bundle"
    write_bundle(bundle, with_evaluation=False)
    facts_path = bundle / "run" / "step_facts.parquet"
    rows = pq.read_table(facts_path).to_pylist()
    rows.append(dict(rows[0]))
    pq.write_table(
        pa.Table.from_pylist(rows, schema=STEP_FACTS_SCHEMA),
        facts_path,
    )
    manifest_path = bundle / "run" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["relation_totals"]["step_facts"] = len(rows)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        RunValidationError, match="duplicate viewer stage fact"
    ):
        RunDescriptor.from_paths(
            label="bad",
            corpus_path=bundle / "corpus.parquet",
            preprocessing=bundle / "run",
        )


@pytest.mark.parametrize("value", [-1, 2**63, True, "1", None])
def test_descriptor_rejects_invalid_persisted_stage_fact_value(
    tmp_path, value: object
) -> None:
    bundle = tmp_path / "bundle"
    write_bundle(bundle, with_evaluation=False)
    facts_path = bundle / "run" / "step_facts.parquet"
    rows = pq.read_table(facts_path).to_pylist()
    extract = next(
        row for row in rows if row["step_name"] == "extract_candidates"
    )
    extract["facts_json"] = json.dumps({"candidate_count": value})
    pq.write_table(
        pa.Table.from_pylist(rows, schema=STEP_FACTS_SCHEMA),
        facts_path,
    )

    with pytest.raises(RunValidationError, match="candidate_count"):
        RunDescriptor.from_paths(
            label="bad",
            corpus_path=bundle / "corpus.parquet",
            preprocessing=bundle / "run",
        )


def test_descriptor_accepts_signed_int64_max_stage_fact(tmp_path) -> None:
    bundle = tmp_path / "bundle"
    write_bundle(bundle, with_evaluation=False)
    facts_path = bundle / "run" / "step_facts.parquet"
    rows = pq.read_table(facts_path).to_pylist()
    extract = next(
        row for row in rows if row["step_name"] == "extract_candidates"
    )
    extract["facts_json"] = json.dumps({"candidate_count": 2**63 - 1})
    pq.write_table(
        pa.Table.from_pylist(rows, schema=STEP_FACTS_SCHEMA), facts_path
    )

    descriptor = RunDescriptor.from_paths(
        label="max",
        corpus_path=bundle / "corpus.parquet",
        preprocessing=bundle / "run",
    )

    assert descriptor.run_id == "fixture-run"


@pytest.mark.parametrize("remove_all", [False, True])
def test_descriptor_rejects_missing_stage_fact_coverage(
    tmp_path, *, remove_all: bool
) -> None:
    bundle = tmp_path / "bundle"
    write_bundle(bundle, with_evaluation=False)
    facts_path = bundle / "run" / "step_facts.parquet"
    rows = pq.read_table(facts_path).to_pylist()
    matching = [
        index
        for index, row in enumerate(rows)
        if row["step_name"] == "extract_candidates"
    ]
    removed = set(matching if remove_all else matching[:1])
    rows = [row for index, row in enumerate(rows) if index not in removed]
    _rewrite_stage_facts(bundle, rows)

    with pytest.raises(RunValidationError, match="coverage mismatch"):
        RunDescriptor.from_paths(
            label="bad",
            corpus_path=bundle / "corpus.parquet",
            preprocessing=bundle / "run",
        )


def test_descriptor_rejects_stage_fact_for_ineligible_sample(tmp_path) -> None:
    bundle = tmp_path / "bundle"
    write_bundle(bundle, with_evaluation=False)
    facts_path = bundle / "run" / "step_facts.parquet"
    rows = pq.read_table(facts_path).to_pylist()
    rows.append(
        {
            "sample_id": "no-code",
            "step_name": "filter_compilable",
            "facts_json": '{"survivor_candidate_count":0}',
        }
    )
    _rewrite_stage_facts(bundle, rows)

    with pytest.raises(RunValidationError, match="extra=1"):
        RunDescriptor.from_paths(
            label="bad",
            corpus_path=bundle / "corpus.parquet",
            preprocessing=bundle / "run",
        )


def _rewrite_stage_facts(bundle: Path, rows: list[dict[str, object]]) -> None:
    facts_path = bundle / "run" / "step_facts.parquet"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=STEP_FACTS_SCHEMA), facts_path
    )
    manifest_path = bundle / "run" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["relation_totals"]["step_facts"] = len(rows)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
