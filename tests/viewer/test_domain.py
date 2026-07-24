from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path
from collections.abc import Callable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import dr_code.corpus.stable_files as stable_files_module
from dr_code.corpus.candidate_evaluation import (
    evaluate_preprocessing_candidates,
)
from dr_code.corpus.candidate_evaluation_contract import (
    CANDIDATE_EVALUATION_COORDINATE_FIELDS,
    candidate_evaluation_identity,
)
from dr_code.corpus.preprocessing_analysis import (
    analyze_preprocessing_corpus,
)
from dr_code.corpus.preprocessing_run import run_preprocessing_corpus
from dr_code.corpus.evaluation_generation import (
    publish_generation_directory,
    resolve_current_generation,
    staged_generation_directory,
    switch_current,
)
from dr_code.corpus.run_descriptor import (
    RunDescriptor,
    RunValidationError,
)
from dr_code.execution.subprocess import run_python_subprocess
from dr_code.eval import identity_hash_for
from dr_code.synthetic.humaneval_loader import packaged_snapshot_bytes
from dr_code.viewer.analytics import ViewerAnalytics
from dr_code.viewer.database import ViewerDatabase
from dr_code.viewer.domain import (
    ANNOTATION_NOTE_MAX_LENGTH,
    ANNOTATION_TAG_IDS_MAX_COUNT,
    TAG_NAME_MAX_LENGTH,
    TAG_NAME_WHITESPACE_CODE_POINTS,
    InvalidQueryError,
    normalize_annotation_tag_ids,
    normalize_tag_name,
    validate_annotation_note,
)
from viewer.helpers import write_bundle


def test_annotation_contract_boundaries_and_normalization() -> None:
    assert validate_annotation_note("n" * ANNOTATION_NOTE_MAX_LENGTH)
    assert validate_annotation_note("😀" * ANNOTATION_NOTE_MAX_LENGTH)
    with pytest.raises(InvalidQueryError, match="10000 characters"):
        validate_annotation_note("n" * (ANNOTATION_NOTE_MAX_LENGTH + 1))

    tag_ids = [
        f"tag-{index:03}" for index in range(ANNOTATION_TAG_IDS_MAX_COUNT)
    ]
    assert normalize_annotation_tag_ids([*tag_ids, tag_ids[0]]) == tuple(
        tag_ids
    )
    with pytest.raises(InvalidQueryError, match="100 distinct tag IDs"):
        normalize_annotation_tag_ids(
            [*tag_ids, f"tag-{ANNOTATION_TAG_IDS_MAX_COUNT:03}"]
        )

    assert normalize_tag_name("  left   right  ") == (
        "left right",
        "left right",
    )
    assert normalize_tag_name("x" * TAG_NAME_MAX_LENGTH) == (
        "x" * TAG_NAME_MAX_LENGTH,
        "x" * TAG_NAME_MAX_LENGTH,
    )
    with pytest.raises(InvalidQueryError, match="after normalization"):
        normalize_tag_name("x" * (TAG_NAME_MAX_LENGTH + 1))
    assert normalize_tag_name("😀" * TAG_NAME_MAX_LENGTH) == (
        "😀" * TAG_NAME_MAX_LENGTH,
        "😀" * TAG_NAME_MAX_LENGTH,
    )


def test_annotation_text_contract_pins_scalars_and_whitespace() -> None:
    expected_whitespace = (
        0x0009,
        0x000A,
        0x000B,
        0x000C,
        0x000D,
        0x0020,
        0x0085,
        0x00A0,
        0x1680,
        0x2000,
        0x2001,
        0x2002,
        0x2003,
        0x2004,
        0x2005,
        0x2006,
        0x2007,
        0x2008,
        0x2009,
        0x200A,
        0x2028,
        0x2029,
        0x202F,
        0x205F,
        0x3000,
    )
    assert TAG_NAME_WHITESPACE_CODE_POINTS == expected_whitespace
    separators = "".join(chr(code_point) for code_point in expected_whitespace)
    assert normalize_tag_name(f"left{separators}right") == (
        "left right",
        "left right",
    )
    for code_point in (0x001C, 0x001D, 0x001E, 0x001F, 0xFEFF):
        separator = chr(code_point)
        assert normalize_tag_name(f"left{separator}right") == (
            f"left{separator}right",
            f"left{separator}right",
        )

    assert validate_annotation_note("😀") == "😀"
    for surrogate in ("\ud800", "\udfff"):
        with pytest.raises(InvalidQueryError, match="Unicode scalar values"):
            normalize_tag_name(surrogate)
        with pytest.raises(InvalidQueryError, match="Unicode scalar values"):
            validate_annotation_note(surrogate)
        with pytest.raises(InvalidQueryError, match="Unicode scalar values"):
            normalize_annotation_tag_ids([surrogate])


def test_typescript_annotation_contract_matches_python_constants() -> None:
    contract_path = (
        Path(__file__).parents[2]
        / "viewer/packages/preprocessing-analysis/src/annotation-contract.ts"
    )
    source = contract_path.read_text(encoding="utf-8")

    assert (
        "export const ANNOTATION_NOTE_MAX_LENGTH = "
        f"{ANNOTATION_NOTE_MAX_LENGTH:_};"
    ) in source
    assert (
        "export const ANNOTATION_TAG_IDS_MAX_COUNT = "
        f"{ANNOTATION_TAG_IDS_MAX_COUNT:_};"
    ) in source
    assert (
        f"export const TAG_NAME_MAX_LENGTH = {TAG_NAME_MAX_LENGTH:_};"
        in source
    )
    for code_point in TAG_NAME_WHITESPACE_CODE_POINTS:
        assert f"  0x{code_point:04x}," in source


def test_helper_bundle_facts_are_admitted_as_canonical_waterfall(
    tmp_path: Path,
) -> None:
    descriptor = write_bundle(tmp_path / "bundle")
    step_facts = {
        (row["sample_id"], row["step_name"]): json.loads(row["facts_json"])
        for row in pq.read_table(descriptor.step_facts_path).to_pylist()
    }

    assert descriptor.run_id == "fixture-run"
    assert descriptor.preprocessing_schema_version == 4
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
    assert len(descriptor.definition_identity) == 64
    assert len(descriptor.preprocessing_identity) == 64
    assert len(descriptor.evaluation_identity or "") == 64
    assert step_facts[("blank", "require_nonblank_text")] == {
        "is_nonblank": False,
        "text_character_count": 0,
    }
    assert step_facts[("compile-fail", "require_nonblank_text")] == {
        "is_nonblank": True,
        "text_character_count": len("def broken("),
    }
    assert step_facts[("no-code", "extract_candidates")] == {
        "candidate_count": 0,
        "operation_counts": [],
        "paths": [],
    }


def test_descriptor_uses_one_captured_preprocessing_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = write_bundle(
        tmp_path / "original",
        run_id="original",
        with_evaluation=False,
    )
    replacement = write_bundle(
        tmp_path / "replacement",
        run_id="replacement",
        corpus_path=original.corpus_path,
        with_evaluation=False,
    )
    original_hash = original.preprocessing_manifest_sha256
    source_manifest = original.preprocessing_manifest_path
    replacement_manifest = replacement.preprocessing_manifest_path
    copy_and_hash = stable_files_module._copy_and_hash
    replaced = False

    def replace_after_capture(
        source: Path,
        destination: Path,
        *,
        label: str,
        max_bytes: int | None = None,
    ) -> stable_files_module.StableFile:
        nonlocal replaced
        captured = copy_and_hash(
            source,
            destination,
            label=label,
            max_bytes=max_bytes,
        )
        if source == source_manifest and not replaced:
            replaced = True
            replacement_manifest.replace(source_manifest)
        return captured

    monkeypatch.setattr(
        stable_files_module, "_copy_and_hash", replace_after_capture
    )
    admitted = RunDescriptor.from_paths(
        label="race",
        corpus_path=original.corpus_path,
        preprocessing=source_manifest.parent,
    )

    assert replaced
    assert (
        json.loads(source_manifest.read_text(encoding="utf-8"))["run_id"]
        == "replacement"
    )
    assert admitted.run_id == "original"
    assert admitted.preprocessing_manifest_sha256 == original_hash


def test_descriptor_uses_one_captured_evaluation_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_root = tmp_path / "original"
    original = write_bundle(original_root)
    assert original.evaluation_root_path is not None
    original_evaluation = original.evaluation_root_path
    original_generation = resolve_current_generation(original_evaluation)
    replacement_root = tmp_path / "replacement"
    replacement = write_bundle(
        replacement_root,
        corpus_path=original.corpus_path,
    )
    assert replacement.evaluation_root_path is not None
    replacement_evaluation = replacement.evaluation_root_path

    def mutate(staged: Path) -> None:
        manifest_path = staged / "candidate_evaluation_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["reuse_result_sources"] = ["replacement"]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _rewrite_current_generation(replacement_root, mutate)
    replacement_generation = resolve_current_generation(replacement_evaluation)
    target_generation = (
        original_evaluation
        / "generations"
        / replacement_generation.generation_id
    )
    shutil.copytree(replacement_generation.generation_dir, target_generation)
    pointer_path = original_evaluation / "CURRENT.json"
    original_pointer_sha256 = hashlib.sha256(
        pointer_path.read_bytes()
    ).hexdigest()
    replacement_pointer = tmp_path / "replacement-CURRENT.json"
    shutil.copyfile(
        replacement_evaluation / "CURRENT.json",
        replacement_pointer,
    )
    copy_and_hash = stable_files_module._copy_and_hash
    replaced = False

    def replace_after_capture(
        source: Path,
        destination: Path,
        *,
        label: str,
        max_bytes: int | None = None,
    ) -> stable_files_module.StableFile:
        nonlocal replaced
        captured = copy_and_hash(
            source,
            destination,
            label=label,
            max_bytes=max_bytes,
        )
        if source == pointer_path and not replaced:
            replaced = True
            replacement_pointer.replace(pointer_path)
        return captured

    monkeypatch.setattr(
        stable_files_module, "_copy_and_hash", replace_after_capture
    )
    admitted = RunDescriptor.from_paths(
        label="race",
        corpus_path=original.corpus_path,
        preprocessing=original.preprocessing_manifest_path.parent,
        candidate_evaluation=original_evaluation,
    )

    assert replaced
    assert (
        json.loads(pointer_path.read_text(encoding="utf-8"))["generation_id"]
        == replacement_generation.generation_id
    )
    assert admitted.evaluation_generation_id == (
        original_generation.generation_id
    )
    assert admitted.evaluation_pointer_sha256 == original_pointer_sha256


def test_descriptor_coordinates_are_deeply_immutable_and_json_stable(
    tmp_path: Path,
) -> None:
    descriptor = write_bundle(tmp_path / "bundle")
    coordinates = descriptor.evaluation_coordinates
    assert coordinates is not None
    dataset = coordinates["dataset"]
    assert isinstance(dataset, Mapping)
    with pytest.raises(TypeError):
        dataset["split"] = "forged"  # type: ignore[index]
    definition_ref = coordinates["metric_extraction_definition_ref"]
    assert isinstance(definition_ref, Mapping)
    definition_payload = definition_ref["identity_payload"]
    assert isinstance(definition_payload, Mapping)
    assert isinstance(definition_payload["questions"], tuple)
    assert descriptor.to_json() == descriptor.to_json()
    serialized = json.loads(descriptor.to_json())
    assert serialized["evaluation_coordinates"]["dataset"]["split"] == "test"
    assert isinstance(
        serialized["evaluation_coordinates"][
            "metric_extraction_definition_ref"
        ]["identity_payload"]["questions"],
        list,
    )


def test_descriptor_rejects_stale_schema_and_forged_question_identity(
    tmp_path: Path,
) -> None:
    for mutation in ("schema", "question"):
        bundle = tmp_path / mutation
        descriptor = write_bundle(bundle)

        def mutate(staged: Path) -> None:
            manifest_path = staged / "candidate_evaluation_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if mutation == "schema":
                manifest["schema_version"] = 4
            else:
                manifest["question_identity_hash"] = "0" * 64
            if mutation == "question":
                manifest["evaluation_identity"] = (
                    candidate_evaluation_identity(
                        {
                            field: manifest[field]
                            for field in CANDIDATE_EVALUATION_COORDINATE_FIELDS
                        }
                    )
                )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        _rewrite_current_generation(bundle, mutate)
        expected = (
            "schema_version 6"
            if mutation == "schema"
            else "identities are not canonical"
        )
        with pytest.raises(RunValidationError, match=expected):
            RunDescriptor.from_paths(
                label="bad",
                corpus_path=descriptor.corpus_path,
                preprocessing=descriptor.preprocessing_manifest_path.parent,
                candidate_evaluation=bundle / "evaluation",
            )


def test_descriptor_translates_unknown_metric_to_validation_error(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "unknown-metric"
    descriptor = write_bundle(bundle)

    def mutate(staged: Path) -> None:
        manifest_path = staged / "candidate_evaluation_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        definition_ref = manifest["metric_extraction_definition_ref"]
        assert isinstance(definition_ref, dict)
        definition_payload = definition_ref["identity_payload"]
        assert isinstance(definition_payload, dict)
        questions = definition_payload["questions"]
        assert isinstance(questions, list)
        question = questions[0]
        assert isinstance(question, dict)
        question["metric"] = "unregistered_metric"
        definition_ref["identity_hash"] = identity_hash_for(
            schema=definition_ref["schema_name"],
            payload=definition_payload,
        )
        manifest["metric_extraction_definition_identity"] = definition_ref[
            "identity_hash"
        ]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _rewrite_current_generation(bundle, mutate)

    with pytest.raises(RunValidationError, match="coordinates are invalid"):
        RunDescriptor.from_paths(
            label="bad",
            corpus_path=descriptor.corpus_path,
            preprocessing=descriptor.preprocessing_manifest_path.parent,
            candidate_evaluation=bundle / "evaluation",
        )


def test_descriptor_generation_survives_bundle_relocation(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original"
    before = write_bundle(original)
    relocated = tmp_path / "relocated"
    shutil.move(original, relocated)

    after = RunDescriptor.from_paths(
        label="relocated",
        corpus_path=relocated / "corpus.parquet",
        preprocessing=relocated / "run",
        candidate_evaluation=relocated / "evaluation",
    )

    assert after.evaluation_generation_id == before.evaluation_generation_id
    assert after.evaluation_pointer_sha256 == before.evaluation_pointer_sha256
    assert after.evaluation_identity == before.evaluation_identity


def test_descriptor_rejects_candidate_duplicate_across_row_groups(
    tmp_path: Path,
) -> None:
    descriptor = write_bundle(tmp_path / "bundle", with_evaluation=False)
    candidates_path = descriptor.candidates_path
    table = pq.read_table(candidates_path)
    rows = table.to_pylist()
    pq.write_table(
        pa.Table.from_pylist([*rows, rows[0]], schema=table.schema),
        candidates_path,
        row_group_size=1,
    )
    assert pq.ParquetFile(candidates_path).num_row_groups == 3
    manifest_path = descriptor.preprocessing_manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["relation_totals"]["candidates"] = 3
    manifest["relation_sha256"]["candidates"] = hashlib.sha256(
        candidates_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        RunValidationError, match="candidates contains sample_id absent"
    ):
        RunDescriptor.from_paths(
            label="duplicate",
            corpus_path=descriptor.corpus_path,
            preprocessing=manifest_path.parent,
        )


def test_real_schema_six_producer_flows_through_analysis_and_viewer(
    tmp_path: Path,
) -> None:
    snapshot_bytes = packaged_snapshot_bytes()
    canonical_task = json.loads(snapshot_bytes)["rows"][0]
    corpus = tmp_path / "corpus.parquet"
    pq.write_table(
        pa.table(
            {
                "sample_id": ["one"],
                "decoder_output": [
                    canonical_task["prompt"]
                    + canonical_task["canonical_solution"]
                ],
                "task_id": [canonical_task["task_id"]],
                "source_kind": ["fixture"],
            }
        ).cast(
            pa.schema(
                [
                    pa.field("sample_id", pa.string(), nullable=False),
                    pa.field("decoder_output", pa.string()),
                    pa.field("task_id", pa.string(), nullable=False),
                    pa.field("source_kind", pa.string(), nullable=False),
                ]
            )
        ),
        corpus,
    )
    preprocessing = run_preprocessing_corpus(
        input_path=corpus,
        output_root=tmp_path / "runs",
        run_id="producer",
    )
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_bytes(snapshot_bytes)
    evaluation = tmp_path / "evaluation"
    evaluate_preprocessing_candidates(
        preprocessing_run=preprocessing,
        corpus_path=corpus,
        output_dir=evaluation,
        snapshot_path=snapshot,
        max_workers=1,
        run_in_subprocess=run_python_subprocess,
        runner_identity="test:python-isolated@v1",
    )

    descriptor = RunDescriptor.from_paths(
        label="producer",
        corpus_path=corpus,
        preprocessing=preprocessing,
        candidate_evaluation=evaluation,
    )
    produced_manifest = json.loads(
        descriptor.evaluation_manifest_path.read_text(encoding="utf-8")
    )
    assert (
        descriptor.preprocessing_identity
        == (produced_manifest["preprocessing_run"]["identity"])
    )
    assert (
        descriptor.evaluation_identity
        == produced_manifest["evaluation_identity"]
    )
    analysis = analyze_preprocessing_corpus(
        corpus_path=corpus,
        run_dir=preprocessing,
        candidate_evaluation=evaluation,
        output_dir=tmp_path / "analysis",
    )
    analysis_manifest = json.loads(
        analysis.manifest_path.read_text(encoding="utf-8")
    )
    assert (
        analysis_manifest["inputs"]["candidate_evaluation"]["coordinates"][
            "evaluation_identity"
        ]
        == descriptor.evaluation_identity
    )
    with ViewerDatabase(":memory:") as database:
        waterfall = ViewerAnalytics(database, [descriptor]).waterfall(
            descriptor.run_id
        )
    assert waterfall.stages[-1].stage_id == "has_passing_candidate"
    assert waterfall.stages[-1].count == 1


def test_descriptor_file_has_one_exact_relative_path_contract(
    tmp_path: Path,
) -> None:
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


@pytest.mark.parametrize(
    "extra_field",
    ["corpus_path", "preprocessing_manifest", "unrecognized"],
)
def test_descriptor_rejects_noncanonical_fields(
    tmp_path: Path, extra_field: str
) -> None:
    bundle = tmp_path / "bundle"
    write_bundle(bundle, with_evaluation=False)
    descriptor_path = bundle / "descriptor.json"
    descriptor_path.write_text(
        json.dumps(
            {
                "label": "run",
                "corpus": "corpus.parquet",
                "preprocessing": "run",
                extra_field: "value",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RunValidationError, match="unknown field"):
        RunDescriptor.from_file(descriptor_path)


def test_descriptor_rejects_incomplete_relation_hashes(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    write_bundle(bundle, with_evaluation=False)
    manifest_path = bundle / "run" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["relation_sha256"]["results"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RunValidationError, match="hashes are incomplete"):
        RunDescriptor.from_paths(
            label="bad",
            corpus_path=bundle / "corpus.parquet",
            preprocessing=bundle / "run",
        )


def test_descriptor_rejects_current_artifact_hash_mismatch(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    descriptor = write_bundle(bundle)
    assert descriptor.candidate_results_path is not None
    descriptor.candidate_results_path.write_bytes(b"drift")

    with pytest.raises(RunValidationError, match="hash"):
        RunDescriptor.from_paths(
            label="bad",
            corpus_path=bundle / "corpus.parquet",
            preprocessing=bundle / "run",
            candidate_evaluation=bundle / "evaluation",
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "obsolete_preprocessing_coordinate",
        "preprocessing_relation_drift",
        "forged_identity",
        "unknown_field",
    ],
)
def test_descriptor_rejects_noncanonical_schema_six_manifest(
    tmp_path: Path, mutation: str
) -> None:
    bundle = tmp_path / mutation
    descriptor = write_bundle(bundle)

    def mutate(staged: Path) -> None:
        manifest_path = staged / "candidate_evaluation_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if mutation == "obsolete_preprocessing_coordinate":
            manifest["preprocessing_run"] = {
                "manifest_sha256": descriptor.preprocessing_manifest_sha256,
                "relations": manifest["preprocessing_run"]["relations"],
            }
            manifest["evaluation_identity"] = candidate_evaluation_identity(
                {
                    field: manifest[field]
                    for field in CANDIDATE_EVALUATION_COORDINATE_FIELDS
                }
            )
        elif mutation == "preprocessing_relation_drift":
            manifest["preprocessing_run"]["relations"]["results"]["sha256"] = (
                "0" * 64
            )
            manifest["evaluation_identity"] = candidate_evaluation_identity(
                {
                    field: manifest[field]
                    for field in CANDIDATE_EVALUATION_COORDINATE_FIELDS
                }
            )
        elif mutation == "forged_identity":
            manifest["evaluation_identity"] = "0" * 64
        else:
            manifest["unknown"] = True
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _rewrite_current_generation(bundle, mutate)

    with pytest.raises(RunValidationError):
        RunDescriptor.from_paths(
            label="bad",
            corpus_path=descriptor.corpus_path,
            preprocessing=descriptor.preprocessing_manifest_path.parent,
            candidate_evaluation=bundle / "evaluation",
        )


def test_descriptor_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    bundle = tmp_path / "duplicate"
    descriptor = write_bundle(bundle)
    manifest_path = bundle / "run" / "manifest.json"
    text = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(
        text.replace(
            '"complete": true',
            '"complete": true, "complete": true',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(RunValidationError, match="not valid JSON"):
        RunDescriptor.from_paths(
            label="bad",
            corpus_path=descriptor.corpus_path,
            preprocessing=descriptor.preprocessing_manifest_path.parent,
        )


@pytest.mark.parametrize(
    ("field", "mutate", "message"),
    [
        (
            "size",
            lambda manifest: manifest["input"].__setitem__(
                "size", manifest["input"]["size"] + 1
            ),
            "corpus coordinate mismatch: size",
        ),
        (
            "row_groups",
            lambda manifest: manifest["input"]["row_groups"][0].__setitem__(
                "rows", manifest["input"]["row_groups"][0]["rows"] + 1
            ),
            "corpus coordinate mismatch: row_groups",
        ),
        (
            "completed_row_groups",
            lambda manifest: manifest.__setitem__("completed_row_groups", []),
            "do not cover the captured corpus",
        ),
        (
            "outcome_totals",
            lambda manifest: manifest.__setitem__(
                "outcome_totals", {"forged": 9}
            ),
            "outcome_totals mismatch",
        ),
        (
            "strict_integer",
            lambda manifest: manifest.__setitem__("batch_size", True),
            "must be a non-negative integer",
        ),
    ],
)
def test_descriptor_reconciles_all_preprocessing_manifest_claims(
    tmp_path: Path,
    field: str,
    mutate: Callable[[dict[str, object]], None],
    message: str,
) -> None:
    bundle = tmp_path / field
    descriptor = write_bundle(bundle, with_evaluation=False)
    manifest_path = descriptor.preprocessing_manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RunValidationError, match=message):
        RunDescriptor.from_paths(
            label="bad",
            corpus_path=descriptor.corpus_path,
            preprocessing=manifest_path.parent,
        )


def test_descriptor_rejects_non_finite_manifest_json(tmp_path: Path) -> None:
    bundle = tmp_path / "non-finite"
    descriptor = write_bundle(bundle, with_evaluation=False)
    manifest_path = descriptor.preprocessing_manifest_path
    text = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(
        text.replace('"batch_size": 1000', '"batch_size": NaN'),
        encoding="utf-8",
    )

    with pytest.raises(RunValidationError, match="not valid JSON"):
        RunDescriptor.from_paths(
            label="bad",
            corpus_path=descriptor.corpus_path,
            preprocessing=manifest_path.parent,
        )


def test_descriptor_rejects_noncurrent_schema(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    write_bundle(bundle, with_evaluation=False)
    manifest_path = bundle / "run" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RunValidationError, match="requires schema_version 4"):
        RunDescriptor.from_paths(
            label="bad",
            corpus_path=bundle / "corpus.parquet",
            preprocessing=bundle / "run",
        )


def test_descriptor_rejects_stale_preprocessing_source_coordinates(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    descriptor = write_bundle(bundle, with_evaluation=False)
    manifest_path = descriptor.preprocessing_manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"] = {
        "git_commit": "legacy",
        "source_tree_sha256": "9" * 64,
        "python_implementation": "CPython",
        "python_version": "fixture",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        RunValidationError, match="source coordinates are invalid"
    ):
        RunDescriptor.from_paths(
            label="bad",
            corpus_path=descriptor.corpus_path,
            preprocessing=manifest_path.parent,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "record_status_totals",
            {"measured": 1},
            "record_status_totals mismatch",
        ),
        (
            "reused_result_rows",
            1,
            "reused_result_rows mismatch",
        ),
        (
            "reuse_result_sources",
            [{}],
            "reuse source 0 is invalid",
        ),
    ],
)
def test_descriptor_reconciles_evaluation_manifest_summaries(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    bundle = tmp_path / message.replace(" ", "-")
    descriptor = write_bundle(bundle)

    def mutate(staged: Path) -> None:
        manifest_path = staged / "candidate_evaluation_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest[field] = value
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _rewrite_current_generation(bundle, mutate)

    with pytest.raises(RunValidationError, match=message):
        RunDescriptor.from_paths(
            label="bad",
            corpus_path=descriptor.corpus_path,
            preprocessing=descriptor.preprocessing_manifest_path.parent,
            candidate_evaluation=bundle / "evaluation",
        )


@pytest.mark.parametrize(
    ("relation", "field", "value", "message"),
    [
        (
            "candidate_membership",
            "task_id",
            "Task/999",
            "membership/corpus semantic mismatches",
        ),
        (
            "candidate_results",
            "metrics_profile",
            "contradictory@v1",
            "result/manifest coordinate mismatches",
        ),
        (
            "candidate_results",
            "cleaned_source",
            "def changed():\n    return 0",
            "content fingerprint mismatches",
        ),
    ],
)
def test_descriptor_rejects_hash_consistent_evaluation_contradictions(
    tmp_path: Path,
    relation: str,
    field: str,
    value: str,
    message: str,
) -> None:
    bundle = tmp_path / relation
    descriptor = write_bundle(bundle)

    def mutate(staged: Path) -> None:
        relation_path = staged / f"{relation}.parquet"
        table = pq.read_table(relation_path)
        rows = table.to_pylist()
        rows[0][field] = value
        pq.write_table(
            pa.Table.from_pylist(rows, schema=table.schema), relation_path
        )
        _record_evaluation_relation_hash(staged, relation)

    _rewrite_current_generation(bundle, mutate)

    with pytest.raises(RunValidationError, match=message):
        RunDescriptor.from_paths(
            label="bad",
            corpus_path=descriptor.corpus_path,
            preprocessing=descriptor.preprocessing_manifest_path.parent,
            candidate_evaluation=bundle / "evaluation",
        )


def test_descriptor_rejects_hash_consistent_arbitrary_evaluation_key(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "arbitrary-key"
    descriptor = write_bundle(bundle)
    arbitrary_key = "0" * 64

    def mutate(staged: Path) -> None:
        for relation in ("candidate_membership", "candidate_results"):
            relation_path = staged / f"{relation}.parquet"
            table = pq.read_table(relation_path)
            rows = table.to_pylist()
            rows[0]["evaluation_key"] = arbitrary_key
            pq.write_table(
                pa.Table.from_pylist(rows, schema=table.schema), relation_path
            )
            _record_evaluation_relation_hash(staged, relation)

    _rewrite_current_generation(bundle, mutate)

    with pytest.raises(
        RunValidationError,
        match="evaluation_key does not match canonical value",
    ):
        RunDescriptor.from_paths(
            label="bad",
            corpus_path=descriptor.corpus_path,
            preprocessing=descriptor.preprocessing_manifest_path.parent,
            candidate_evaluation=bundle / "evaluation",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("record_status", "invented", "invalid record_status"),
        ("outcome", "invented", "outcome does not match canonical value"),
        (
            "failed_count",
            1,
            "observed case count must not exceed total_cases",
        ),
    ],
)
def test_descriptor_rejects_hash_consistent_result_contract_violations(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    bundle = tmp_path / field
    descriptor = write_bundle(bundle)

    def mutate(staged: Path) -> None:
        results_path = staged / "candidate_results.parquet"
        table = pq.read_table(results_path)
        rows = table.to_pylist()
        passed = next(row for row in rows if row["outcome"] == "passed")
        passed[field] = value
        pq.write_table(
            pa.Table.from_pylist(rows, schema=table.schema), results_path
        )
        _record_evaluation_relation_hash(staged, "candidate_results")

    _rewrite_current_generation(bundle, mutate)

    with pytest.raises(RunValidationError, match=message):
        RunDescriptor.from_paths(
            label="bad",
            corpus_path=descriptor.corpus_path,
            preprocessing=descriptor.preprocessing_manifest_path.parent,
            candidate_evaluation=bundle / "evaluation",
        )


def _record_evaluation_relation_hash(staged: Path, relation: str) -> None:
    relation_path = staged / f"{relation}.parquet"
    manifest_path = staged / "candidate_evaluation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[f"{relation}_sha256"] = hashlib.sha256(
        relation_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _rewrite_current_generation(
    bundle: Path, mutate: Callable[[Path], None]
) -> None:
    root = bundle / "evaluation"
    current = resolve_current_generation(root)
    with staged_generation_directory(root) as staged:
        for source in (
            current.manifest_path,
            current.membership_path,
            current.results_path,
        ):
            shutil.copyfile(source, staged / source.name)
        mutate(staged)
        generation = publish_generation_directory(root, staged)
    switch_current(root, generation)
