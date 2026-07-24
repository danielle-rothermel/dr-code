from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from dr_code.corpus.evaluation_generation import (
    CURRENT_FILENAME,
    EvaluationGenerationError,
    publish_generation_directory,
    resolve_current_generation,
    staged_generation_directory,
    switch_current,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _staged(root: Path, label: str) -> Path:
    directory = root / f"staged-{label}"
    directory.mkdir()
    _write_staged(directory, label)
    return directory


def _write_staged(directory: Path, label: str) -> None:
    membership = directory / "candidate_membership.parquet"
    results = directory / "candidate_results.parquet"
    membership.write_bytes(f"membership-{label}".encode())
    results.write_bytes(f"results-{label}".encode())
    (directory / "candidate_evaluation_manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "candidate_membership_sha256": _sha256(membership),
                "candidate_results_sha256": _sha256(results),
            },
            sort_keys=True,
        )
    )


def test_crash_before_pointer_switch_preserves_prior_generation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluation"
    first = publish_generation_directory(root, _staged(tmp_path, "first"))
    switch_current(root, first)

    second = publish_generation_directory(root, _staged(tmp_path, "second"))
    assert (
        resolve_current_generation(root).generation_id == first.generation_id
    )
    assert first.generation_dir.is_dir()
    assert second.generation_dir.is_dir()

    switch_current(root, second)
    assert (
        resolve_current_generation(root).generation_id == second.generation_id
    )
    assert first.generation_dir.is_dir()


def test_pointer_rejects_traversal_symlink_incomplete_and_hash_drift(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluation"
    generation = publish_generation_directory(root, _staged(tmp_path, "valid"))
    switch_current(root, generation)
    pointer_path = root / CURRENT_FILENAME
    pointer = json.loads(pointer_path.read_text())

    pointer["generation_id"] = "../escape"
    pointer_path.write_text(json.dumps(pointer))
    with pytest.raises(EvaluationGenerationError, match="generation_id"):
        resolve_current_generation(root)

    switch_current(root, generation)
    generation.results_path.write_bytes(b"drift")
    with pytest.raises(EvaluationGenerationError, match="hash"):
        resolve_current_generation(root)

    incomplete_root = tmp_path / "incomplete"
    incomplete = _staged(tmp_path, "incomplete")
    manifest_path = incomplete / "candidate_evaluation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["complete"] = False
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(EvaluationGenerationError, match="incomplete"):
        publish_generation_directory(incomplete_root, incomplete)

    symlink_root = tmp_path / "symlink"
    symlink_root.mkdir()
    (symlink_root / "CURRENT.json").symlink_to(pointer_path)
    with pytest.raises(EvaluationGenerationError, match="symlink"):
        resolve_current_generation(symlink_root)


def test_pointer_rejects_duplicate_keys_and_flat_only_directory(
    tmp_path: Path,
) -> None:
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "candidate_results.parquet").write_bytes(b"legacy")
    with pytest.raises(EvaluationGenerationError, match="flat"):
        resolve_current_generation(flat)

    root = tmp_path / "duplicates"
    generation = publish_generation_directory(
        root, _staged(tmp_path, "duplicates")
    )
    switch_current(root, generation)
    pointer = (root / CURRENT_FILENAME).read_text()
    (root / CURRENT_FILENAME).write_text(
        pointer.replace(
            '"schema_version": 1',
            '"schema_version": 1, "schema_version": 1',
        )
    )
    with pytest.raises(EvaluationGenerationError, match="duplicate key"):
        resolve_current_generation(root)


def test_concurrent_pointer_switches_never_cross_delete_temporary_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluation"
    generations = [
        publish_generation_directory(root, _staged(tmp_path, label))
        for label in ("one", "two")
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        resolved = list(
            executor.map(lambda item: switch_current(root, item), generations)
        )
    current = resolve_current_generation(root)
    assert current.generation_id in {
        generation.generation_id for generation in resolved
    }
    assert not list(root.glob(".CURRENT.json.*.tmp"))


def test_same_parent_publication_is_durable_before_pointer_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dr_code.corpus import evaluation_generation

    root = tmp_path / "evaluation"
    events: list[str] = []
    original_fsync_file = evaluation_generation.fsync_file
    original_fsync_directory = evaluation_generation.fsync_directory
    original_publish = evaluation_generation.publish_staged_output_directory
    original_replace = evaluation_generation.os.replace

    def fsync_file(path: Path) -> None:
        events.append(f"file:{path.name}")
        original_fsync_file(path)

    def fsync_directory(path: Path) -> None:
        events.append(f"directory:{path.name}")
        original_fsync_directory(path)

    def publish(source: Path, destination: Path) -> None:
        events.append("generation-rename")
        original_publish(source, destination)

    def replace(source: Path | str, destination: Path | str) -> None:
        if Path(destination).name == CURRENT_FILENAME:
            events.append("pointer-switch")
        original_replace(source, destination)

    monkeypatch.setattr(evaluation_generation, "fsync_file", fsync_file)
    monkeypatch.setattr(
        evaluation_generation, "fsync_directory", fsync_directory
    )
    monkeypatch.setattr(
        evaluation_generation, "publish_staged_output_directory", publish
    )
    monkeypatch.setattr(evaluation_generation.os, "replace", replace)

    with staged_generation_directory(root) as staging:
        _write_staged(staging, "durable")
        generation = publish_generation_directory(root, staging)
    switch_current(root, generation)

    rename_index = events.index("generation-rename")
    pointer_index = events.index("pointer-switch")
    assert set(events[:rename_index]) == {
        "file:candidate_evaluation_manifest.json",
        "file:candidate_membership.parquet",
        "file:candidate_results.parquet",
        f"directory:{staging.name}",
    }
    assert events[rename_index + 1] == "directory:generations"
    assert rename_index < pointer_index
