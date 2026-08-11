from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

from pydantic import JsonValue, ValidationError

from dr_code.generation_corpus.models import (
    DumpedPoolRow,
    PoolManifestEntry,
    SourceManifest,
)


def canonical_json(value: JsonValue) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def content_sha256(*values: JsonValue) -> str:
    payload: JsonValue = list(values)
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def source_record_id(row: DumpedPoolRow) -> str:
    return f"{row.project_name}:{row.pool_name}:{row.sample_id}"


def generation_id(row: DumpedPoolRow) -> str:
    # Persisted identity contract from the original HumanEval corpus builder.
    # The NUL separators and prefix are pinned by a golden test.
    identity = "\0".join(
        (
            "legacy_dr_llm_pool_attempt",
            row.project_name,
            row.pool_name,
            row.sample_id,
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def task_record_id(task_material: JsonValue) -> str:
    return content_sha256(task_material)


def read_manifest(path: Path) -> SourceManifest:
    try:
        manifest = SourceManifest.model_validate_json(
            path.read_text(encoding="utf-8"), strict=True
        )
    except (OSError, ValidationError) as exc:
        raise ValueError(f"invalid source manifest {path}: {exc}") from exc

    incomplete = [
        f"{entry.project_name}/{entry.pool_name}"
        for entry in manifest.pools
        if entry.row_count != entry.dumped_row_count
    ]
    if incomplete:
        raise ValueError(
            "source manifest contains incomplete pool dumps: "
            + ", ".join(incomplete)
        )
    pool_coordinates = [
        (entry.project_name, entry.pool_name) for entry in manifest.pools
    ]
    if len(pool_coordinates) != len(set(pool_coordinates)):
        raise ValueError("source manifest contains duplicate pool coordinates")
    file_names = [entry.file_name for entry in manifest.pools]
    if len(file_names) != len(set(file_names)):
        raise ValueError("source manifest contains duplicate pool files")
    for file_name in file_names:
        if Path(file_name).name != file_name:
            raise ValueError(
                f"source manifest file is not a basename: {file_name}"
            )
    return manifest


def iter_pool_rows(
    path: Path, entry: PoolManifestEntry
) -> Iterator[DumpedPoolRow]:
    if path.name != entry.file_name:
        raise ValueError(
            f"pool path {path.name!r} does not match manifest file "
            f"{entry.file_name!r}"
        )
    expected_keys = {
        column.name for column in entry.pool_schema_json.key_columns
    }
    count = 0
    try:
        with gzip.open(path, "rt", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                count += 1
                if not line.strip():
                    raise ValueError(f"blank pool row at {path}:{line_number}")
                try:
                    row = DumpedPoolRow.model_validate_json(line, strict=True)
                except ValidationError as exc:
                    raise ValueError(
                        f"invalid pool row at {path}:{line_number}: {exc}"
                    ) from exc
                actual_coordinate = (
                    row.project_name,
                    row.pool_name,
                    row.table_name,
                )
                expected_coordinate = (
                    entry.project_name,
                    entry.pool_name,
                    entry.table_name,
                )
                if actual_coordinate != expected_coordinate:
                    raise ValueError(
                        f"pool row coordinate mismatch at {path}:{line_number}: "
                        f"expected={expected_coordinate!r}, "
                        f"actual={actual_coordinate!r}"
                    )
                if set(row.key_values) != expected_keys:
                    raise ValueError(
                        f"pool row key coordinate mismatch at "
                        f"{path}:{line_number}: expected={sorted(expected_keys)!r}, "
                        f"actual={sorted(row.key_values)!r}"
                    )
                yield row
    except OSError as exc:
        raise ValueError(f"cannot read pool dump {path}: {exc}") from exc
    if count != entry.dumped_row_count:
        raise ValueError(
            f"pool dump row count mismatch for "
            f"{entry.project_name}/{entry.pool_name}: "
            f"manifest={entry.dumped_row_count}, file={count}"
        )


def iter_dump_rows(
    dump_directory: Path, manifest: SourceManifest
) -> Iterator[DumpedPoolRow]:
    for entry in manifest.pools:
        yield from iter_pool_rows(dump_directory / entry.file_name, entry)


__all__ = [
    "canonical_json",
    "content_sha256",
    "generation_id",
    "iter_dump_rows",
    "iter_pool_rows",
    "read_manifest",
    "source_record_id",
    "task_record_id",
]
