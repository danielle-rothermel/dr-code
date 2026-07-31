from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from dr_code.corpus.preprocessing_artifacts import (
    AtomicProjectedPartWriter,
    PROJECTED_ARTIFACT_SCHEMAS,
    combine_projected_parts,
    project_preprocessing_result,
    validate_origin_paths,
    write_projected_part,
)
from dr_code.corpus.preprocessing_contract import PROJECTED_PART_SCHEMA_VERSION
from dr_code.preprocessing.definitions import (
    HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
)
from dr_code.preprocessing.runner import bind_preprocessing
from dr_code.trace import TextArtifact


def test_projection_preserves_missing_empty_and_trace_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = project_preprocessing_result("missing", None, None)
    assert missing.results[0]["decoder_output_presence"] == "missing"
    assert missing.results[0]["raw_output_sha256"] is None

    source = "def f():\n    return 1\n"
    trace = bind_preprocessing(
        HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION.materialize()
    ).run(TextArtifact(text=source))

    from dr_code.preprocessing import identification

    def forbidden_inspection(*args: object, **kwargs: object) -> None:
        raise AssertionError("corpus projection must not reparse source")

    monkeypatch.setattr(identification, "_inspect", forbidden_inspection)
    projected = project_preprocessing_result("present", source, trace)

    assert projected.results[0]["decoder_output_presence"] == "present"
    assert projected.results[0]["final_candidate_count"] == 1
    assert projected.candidates[0]["cleaned_source"] == source.rstrip()
    origins = validate_origin_paths(projected.candidates[0]["origins"])
    assert origins[0]["path"]
    assert projected.candidates[0]["parse_ok"] is True
    assert projected.candidates[0]["compile_ok"] is True


def test_schema_two_origin_paths_require_ordered_path() -> None:
    with pytest.raises(ValueError, match="path"):
        validate_origin_paths(
            [{"variant": "whole_response", "strategy": "identity"}]
        )


@pytest.mark.parametrize(
    "details_json",
    ['{"value":NaN}', '{"value":Infinity}', '{"value":1,"value":2}'],
)
def test_origin_details_reject_noncanonical_json_constants_and_duplicates(
    details_json: str,
) -> None:
    with pytest.raises(ValueError, match="invalid JSON"):
        validate_origin_paths(
            [{"path": [{"kind": "raw", "details_json": details_json}]}]
        )


def test_projected_part_has_exact_hashed_relation_manifest(
    tmp_path: Path,
) -> None:
    part = write_projected_part(
        tmp_path,
        "row_group_00000000",
        project_preprocessing_result("missing", None, None),
    )

    manifest = json.loads(
        (
            tmp_path / "parts" / "row_group_00000000" / "manifest.json"
        ).read_text()
    )
    assert manifest["schema_version"] == PROJECTED_PART_SCHEMA_VERSION
    assert set(manifest["relations"]) == set(PROJECTED_ARTIFACT_SCHEMAS)
    assert manifest["relations"]["results"]["rows"] == 1
    for relation, schema in PROJECTED_ARTIFACT_SCHEMAS.items():
        path = part.relation_paths[relation]
        assert pq.read_schema(path).equals(schema)
        assert (
            manifest["relations"][relation]["sha256"]
            == (part.relation_sha256[relation])
        )


def test_reserved_temporary_part_id_is_rejected_before_mutation(
    tmp_path: Path,
) -> None:
    sentinel = tmp_path / "parts" / ".sentinel.tmp"
    sentinel.mkdir(parents=True)
    marker = sentinel / "marker"
    marker.write_bytes(b"do not touch")

    with pytest.raises(ValueError, match="part_id"):
        write_projected_part(
            tmp_path,
            ".sentinel.tmp",
            project_preprocessing_result("missing", None, None),
        )

    assert marker.read_bytes() == b"do not touch"


def test_concurrent_part_writers_have_one_exclusive_valid_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dr_code.corpus import preprocessing_artifacts

    barrier = threading.Barrier(2)
    original_publish = preprocessing_artifacts.publish_staged_output_directory

    def synchronized_publish(source: Path, destination: Path) -> None:
        barrier.wait(timeout=10)
        original_publish(source, destination)

    monkeypatch.setattr(
        preprocessing_artifacts,
        "publish_staged_output_directory",
        synchronized_publish,
    )
    projected = project_preprocessing_result("missing", None, None)

    def write() -> object:
        with AtomicProjectedPartWriter(tmp_path, "shared") as writer:
            writer.append(projected)
        return writer.part

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(write) for _ in range(2)]
        outcomes: list[object] = []
        for future in futures:
            try:
                outcomes.append(future.result(timeout=10))
            except FileExistsError as exc:
                outcomes.append(exc)

    assert (
        sum(isinstance(outcome, FileExistsError) for outcome in outcomes) == 1
    )
    winner = tmp_path / "parts" / "shared"
    manifest = json.loads((winner / "manifest.json").read_text())
    assert manifest["relations"]["results"]["rows"] == 1
    for relation, schema in PROJECTED_ARTIFACT_SCHEMAS.items():
        path = winner / f"{relation}.parquet"
        parquet = pq.ParquetFile(path)
        assert parquet.schema_arrow.equals(schema)
        assert (
            preprocessing_artifacts.file_sha256(path)
            == manifest["relations"][relation]["sha256"]
        )
    assert not list((tmp_path / "parts").glob(".shared.*.tmp"))


def test_part_publication_flushes_files_and_directories_in_crash_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dr_code.corpus import preprocessing_artifacts

    events: list[str] = []
    original_file = preprocessing_artifacts.fsync_file
    original_directory = preprocessing_artifacts.fsync_directory
    original_publish = preprocessing_artifacts.publish_staged_output_directory

    def fsync_file(path: Path) -> None:
        events.append(f"file:{path.name}")
        original_file(path)

    def fsync_directory(path: Path) -> None:
        events.append(f"directory:{path.name}")
        original_directory(path)

    def publish(source: Path, destination: Path) -> None:
        events.append("rename")
        original_publish(source, destination)

    monkeypatch.setattr(preprocessing_artifacts, "fsync_file", fsync_file)
    monkeypatch.setattr(
        preprocessing_artifacts, "fsync_directory", fsync_directory
    )
    monkeypatch.setattr(
        preprocessing_artifacts,
        "publish_staged_output_directory",
        publish,
    )

    write_projected_part(
        tmp_path,
        "durable",
        project_preprocessing_result("missing", None, None),
    )

    rename_index = events.index("rename")
    assert {
        f"file:{relation}.parquet" for relation in PROJECTED_ARTIFACT_SCHEMAS
    } <= set(events[:rename_index])
    assert any(
        event.startswith("directory:.durable.")
        for event in events[:rename_index]
    )
    assert events[rename_index + 1] == "directory:parts"


def test_final_relation_publication_flushes_file_before_each_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dr_code.corpus import preprocessing_artifacts

    write_projected_part(
        tmp_path,
        "part",
        project_preprocessing_result("missing", None, None),
    )
    events: list[str] = []
    original_file = preprocessing_artifacts.fsync_file
    original_directory = preprocessing_artifacts.fsync_directory
    original_replace = preprocessing_artifacts.os.replace

    def fsync_file(path: Path) -> None:
        events.append(f"file:{path.name}")
        original_file(path)

    def fsync_directory(path: Path) -> None:
        events.append(f"directory:{path.name}")
        original_directory(path)

    def replace(source: Path | str, destination: Path | str) -> None:
        events.append(f"rename:{Path(destination).name}")
        original_replace(source, destination)

    monkeypatch.setattr(preprocessing_artifacts, "fsync_file", fsync_file)
    monkeypatch.setattr(
        preprocessing_artifacts, "fsync_directory", fsync_directory
    )
    monkeypatch.setattr(preprocessing_artifacts.os, "replace", replace)

    combine_projected_parts(tmp_path, ["part"])

    for relation in PROJECTED_ARTIFACT_SCHEMAS:
        rename = events.index(f"rename:{relation}.parquet")
        assert events[rename - 1] == f"file:.{relation}.parquet.tmp"
        assert events[rename + 1] == f"directory:{tmp_path.name}"
