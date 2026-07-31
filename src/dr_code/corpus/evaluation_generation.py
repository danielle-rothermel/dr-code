"""Immutable candidate-evaluation generations selected by one atomic pointer."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from dr_code.corpus.atomic_directory import (
    publish_staged_output_directory,
    staged_output_directory,
)
from dr_code.corpus.durability import fsync_directory, fsync_file

CURRENT_FILENAME: Final = "CURRENT.json"
GENERATIONS_DIRECTORY: Final = "generations"
POINTER_SCHEMA_VERSION: Final = 1
MANIFEST_FILENAME: Final = "candidate_evaluation_manifest.json"
MEMBERSHIP_FILENAME: Final = "candidate_membership.parquet"
RESULTS_FILENAME: Final = "candidate_results.parquet"
_ARTIFACT_FILENAMES: Final = (
    MANIFEST_FILENAME,
    MEMBERSHIP_FILENAME,
    RESULTS_FILENAME,
)
_POINTER_FIELDS: Final = {
    "schema_version",
    "generation_id",
    "manifest_sha256",
    "candidate_membership_sha256",
    "candidate_results_sha256",
}


class EvaluationGenerationError(ValueError):
    """An evaluation pointer or immutable generation violates its contract."""


@dataclass(frozen=True, slots=True)
class EvaluationGeneration:
    root: Path
    generation_id: str
    generation_dir: Path
    manifest_path: Path
    membership_path: Path
    results_path: Path
    pointer: dict[str, object]


@dataclass(frozen=True, slots=True)
class StagedCurrentSwitch:
    """A fully authenticated pointer file ready for a fast terminal switch."""

    root: Path
    generation: EvaluationGeneration
    temporary_path: Path


def publish_generation_directory(
    root: Path,
    staged_directory: Path,
) -> EvaluationGeneration:
    """Publish complete staged artifacts without replacing prior generations."""

    root = root.resolve()
    staged = staged_directory.resolve(strict=True)
    hashes = {
        filename: _file_sha256(staged / filename)
        for filename in _ARTIFACT_FILENAMES
    }
    manifest = _read_json_object(
        staged / MANIFEST_FILENAME, "candidate evaluation manifest"
    )
    _validate_complete_manifest(manifest, hashes)
    generation_id = _generation_id(hashes)
    generations = root / GENERATIONS_DIRECTORY
    destination = generations / generation_id
    pointer = _pointer(generation_id, hashes)
    if destination.exists():
        existing = _generation_from_pointer(root, pointer)
        return existing
    if staged.parent == generations.resolve():
        for filename in _ARTIFACT_FILENAMES:
            fsync_file(staged / filename)
        fsync_directory(staged)
        try:
            publish_staged_output_directory(staged, destination)
        except FileExistsError:
            return _generation_from_pointer(root, pointer)
        fsync_directory(generations)
        fsync_directory(root)
        return _generation_from_pointer(root, pointer)
    try:
        with staged_output_directory(destination) as temporary:
            for filename in _ARTIFACT_FILENAMES:
                shutil.copyfile(staged / filename, temporary / filename)
                fsync_file(temporary / filename)
            fsync_directory(temporary)
    except FileExistsError:
        return _generation_from_pointer(root, pointer)
    fsync_directory(generations)
    fsync_directory(root)
    return _generation_from_pointer(root, pointer)


@contextmanager
def staged_generation_directory(root: Path) -> Iterator[Path]:
    """Yield a unique private directory beside immutable generations."""

    generations = root.resolve() / GENERATIONS_DIRECTORY
    generations.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=".generation.",
            suffix=".tmp",
            dir=generations,
        )
    )
    try:
        yield temporary
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def switch_current(
    root: Path,
    generation: EvaluationGeneration,
) -> EvaluationGeneration:
    """Atomically select an already-published immutable generation."""

    with staged_current_switch(root, generation) as staged:
        return publish_staged_current_switch(staged)


@contextmanager
def staged_current_switch(
    root: Path,
    generation: EvaluationGeneration,
) -> Iterator[StagedCurrentSwitch]:
    """Authenticate and durably stage CURRENT before a terminal lease fence."""

    resolved_root = root.resolve()
    if generation.root != resolved_root:
        raise EvaluationGenerationError(
            "evaluation generation belongs to a different root"
        )
    _generation_from_pointer(resolved_root, generation.pointer)
    resolved_root.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{CURRENT_FILENAME}.",
        suffix=".tmp",
        dir=resolved_root,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    generation.pointer,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        yield StagedCurrentSwitch(
            root=resolved_root,
            generation=generation,
            temporary_path=temporary,
        )
    finally:
        temporary.unlink(missing_ok=True)


def publish_staged_current_switch(
    staged: StagedCurrentSwitch,
) -> EvaluationGeneration:
    """Switch CURRENT without hashing or validation after the durable replace."""

    os.replace(staged.temporary_path, staged.root / CURRENT_FILENAME)
    fsync_directory(staged.root)
    return staged.generation


def resolve_current_generation(root: Path | str) -> EvaluationGeneration:
    """Resolve and authenticate exactly the generation selected by CURRENT."""

    requested = Path(root).expanduser()
    if requested.is_symlink():
        raise EvaluationGenerationError(
            "evaluation root must not be a symlink"
        )
    directory = requested.resolve(strict=True)
    pointer_path = directory / CURRENT_FILENAME
    if pointer_path.is_symlink():
        raise EvaluationGenerationError(
            "evaluation CURRENT.json must not be a symlink"
        )
    if not pointer_path.is_file():
        if any((directory / name).exists() for name in _ARTIFACT_FILENAMES):
            raise EvaluationGenerationError(
                "flat evaluation artifacts are unsupported; CURRENT.json "
                "must select an immutable generation"
            )
        raise EvaluationGenerationError(
            f"evaluation directory has no {CURRENT_FILENAME}: {directory}"
        )
    pointer = _read_json_object(pointer_path, "evaluation pointer")
    return _generation_from_pointer(directory, pointer)


def validate_evaluation_root(root: Path | str) -> None:
    """Reject legacy or corrupt roots before resumable state is mutated."""

    requested = Path(root).expanduser()
    if requested.is_symlink():
        raise EvaluationGenerationError(
            "evaluation root must be a non-symlink directory"
        )
    if not requested.exists():
        return
    if not requested.is_dir():
        raise EvaluationGenerationError(
            "evaluation root must be a non-symlink directory"
        )
    pointer_path = requested / CURRENT_FILENAME
    if pointer_path.exists() or pointer_path.is_symlink():
        resolve_current_generation(requested)
        return
    if any((requested / name).exists() for name in _ARTIFACT_FILENAMES):
        raise EvaluationGenerationError(
            "flat evaluation artifacts are unsupported; CURRENT.json "
            "must select an immutable generation"
        )


def validate_captured_generation(
    generation: EvaluationGeneration,
    *,
    manifest_sha256: str,
    membership_sha256: str,
    results_sha256: str,
) -> None:
    """Bind separately captured artifact bytes to one resolved pointer."""

    hashes = {
        MANIFEST_FILENAME: manifest_sha256,
        MEMBERSHIP_FILENAME: membership_sha256,
        RESULTS_FILENAME: results_sha256,
    }
    expected = {
        MANIFEST_FILENAME: generation.pointer.get("manifest_sha256"),
        MEMBERSHIP_FILENAME: generation.pointer.get(
            "candidate_membership_sha256"
        ),
        RESULTS_FILENAME: generation.pointer.get("candidate_results_sha256"),
    }
    if any(
        not _is_sha256(actual) or actual != expected[filename]
        for filename, actual in hashes.items()
    ):
        raise EvaluationGenerationError(
            "captured evaluation generation hash does not match CURRENT.json"
        )
    if generation.generation_id != _generation_id(hashes):
        raise EvaluationGenerationError(
            "captured evaluation generation identity is not content-derived"
        )


def _generation_from_pointer(
    root: Path,
    pointer: dict[str, object],
) -> EvaluationGeneration:
    if set(pointer) != _POINTER_FIELDS:
        raise EvaluationGenerationError(
            "evaluation pointer schema does not match schema_version 1"
        )
    if pointer.get("schema_version") != POINTER_SCHEMA_VERSION:
        raise EvaluationGenerationError(
            "evaluation pointer requires schema_version 1"
        )
    generation_id = _validated_generation_id(pointer.get("generation_id"))
    generations = root / GENERATIONS_DIRECTORY
    if generations.is_symlink():
        raise EvaluationGenerationError(
            "evaluation generations directory must not be a symlink"
        )
    generation_dir = generations / generation_id
    if generation_dir.is_symlink() or not generation_dir.is_dir():
        raise EvaluationGenerationError(
            f"evaluation generation is missing or invalid: {generation_id}"
        )
    resolved_generation = generation_dir.resolve(strict=True)
    expected_parent = generations.resolve(strict=True)
    if resolved_generation.parent != expected_parent:
        raise EvaluationGenerationError(
            "evaluation generation escapes the generations directory"
        )
    paths = {
        MANIFEST_FILENAME: resolved_generation / MANIFEST_FILENAME,
        MEMBERSHIP_FILENAME: resolved_generation / MEMBERSHIP_FILENAME,
        RESULTS_FILENAME: resolved_generation / RESULTS_FILENAME,
    }
    if {path.name for path in resolved_generation.iterdir()} != set(paths):
        raise EvaluationGenerationError(
            "evaluation generation contains unexpected artifacts"
        )
    for filename, path in paths.items():
        if path.is_symlink() or not path.is_file():
            raise EvaluationGenerationError(
                f"evaluation generation artifact is missing: {filename}"
            )
    hashes = {name: _file_sha256(path) for name, path in paths.items()}
    expected_hashes = {
        MANIFEST_FILENAME: pointer.get("manifest_sha256"),
        MEMBERSHIP_FILENAME: pointer.get("candidate_membership_sha256"),
        RESULTS_FILENAME: pointer.get("candidate_results_sha256"),
    }
    if any(
        not _is_sha256(expected_hashes[name])
        or expected_hashes[name] != actual
        for name, actual in hashes.items()
    ):
        raise EvaluationGenerationError(
            "evaluation generation hash does not match CURRENT.json"
        )
    expected_generation_id = _generation_id(hashes)
    if generation_id != expected_generation_id:
        raise EvaluationGenerationError(
            "evaluation generation identity is not content-derived"
        )
    manifest = _read_json_object(
        paths[MANIFEST_FILENAME], "candidate evaluation manifest"
    )
    _validate_complete_manifest(manifest, hashes)
    return EvaluationGeneration(
        root=root,
        generation_id=generation_id,
        generation_dir=resolved_generation,
        manifest_path=paths[MANIFEST_FILENAME],
        membership_path=paths[MEMBERSHIP_FILENAME],
        results_path=paths[RESULTS_FILENAME],
        pointer=dict(pointer),
    )


def _validate_complete_manifest(
    manifest: dict[str, object],
    hashes: dict[str, str],
) -> None:
    if manifest.get("complete") is not True:
        raise EvaluationGenerationError(
            "evaluation generation manifest is incomplete"
        )
    if (
        manifest.get("candidate_membership_sha256")
        != hashes[MEMBERSHIP_FILENAME]
        or manifest.get("candidate_results_sha256") != hashes[RESULTS_FILENAME]
    ):
        raise EvaluationGenerationError(
            "evaluation generation manifest artifact hashes are invalid"
        )


def _pointer(
    generation_id: str,
    hashes: dict[str, str],
) -> dict[str, object]:
    return {
        "schema_version": POINTER_SCHEMA_VERSION,
        "generation_id": generation_id,
        "manifest_sha256": hashes[MANIFEST_FILENAME],
        "candidate_membership_sha256": hashes[MEMBERSHIP_FILENAME],
        "candidate_results_sha256": hashes[RESULTS_FILENAME],
    }


def _generation_id(hashes: Mapping[str, str]) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "schema": "dr_code.corpus.evaluation_generation",
                "schema_version": POINTER_SCHEMA_VERSION,
                "artifacts": hashes,
            }
        ).encode("utf-8")
    ).hexdigest()


def _validated_generation_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not _is_sha256(value)
        or Path(value).name != value
        or value.startswith(".")
    ):
        raise EvaluationGenerationError(
            "evaluation pointer generation_id is invalid"
        )
    return value


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_number,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationGenerationError(f"{label} is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise EvaluationGenerationError(f"{label} must be a JSON object")
    return value


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EvaluationGenerationError(
                f"JSON object contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _reject_nonfinite_number(value: str) -> object:
    raise EvaluationGenerationError(
        f"JSON contains non-finite number {value!r}"
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


__all__ = [
    "CURRENT_FILENAME",
    "EvaluationGeneration",
    "EvaluationGenerationError",
    "GENERATIONS_DIRECTORY",
    "MANIFEST_FILENAME",
    "MEMBERSHIP_FILENAME",
    "POINTER_SCHEMA_VERSION",
    "RESULTS_FILENAME",
    "StagedCurrentSwitch",
    "publish_generation_directory",
    "publish_staged_current_switch",
    "resolve_current_generation",
    "staged_current_switch",
    "staged_generation_directory",
    "switch_current",
    "validate_captured_generation",
    "validate_evaluation_root",
]
