from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Final, cast

from pydantic import JsonValue

from dr_code.generation_corpus.models import (
    DatasetName,
    TaskMaterialFidelity,
    TaskRecord,
)
from dr_code.generation_corpus.pool_dump import canonical_json, task_record_id

_TASK_ID_RE: Final = re.compile(r"^HumanEval/(?P<index>\d+)$")
_DATA_SAMPLE_ID_RE: Final = re.compile(
    r"^human_eval/(?P<task_id>HumanEval/\d+)"
    r"(?:/gt_solution(?:@(?P<source_digest>[0-9a-f]{16}))?)?$"
)
_EXPECTED_TASK_IDS: Final = tuple(f"HumanEval/{index}" for index in range(164))
_EXPECTED_DATASET_ID: Final = "evalplus/humanevalplus"
_EXPECTED_REVISION: Final = "d32357cf319e50e9c8d8dab5ea876c72b0fd321b"
_EXPECTED_SNAPSHOT_SHA256: Final = (
    "b2daa45795b56b5e73dfc70e9993ef07c7c3bdf4b01ade42beae88387a961377"
)


def parse_human_eval_data_sample_id(
    data_sample_id: str,
) -> tuple[str, str | None]:
    """Parse the persisted HumanEval source-identity forms."""

    match = _DATA_SAMPLE_ID_RE.fullmatch(data_sample_id)
    if match is None:
        raise ValueError(
            f"invalid HumanEval data_sample_id {data_sample_id!r}"
        )
    task_id = match.group("task_id")
    if task_id not in _EXPECTED_TASK_IDS:
        raise ValueError(f"unknown HumanEval task in {data_sample_id!r}")
    return task_id, match.group("source_digest")


class HumanEvalTaskAdapter:
    """Resolve legacy HumanEval identities against one explicit snapshot."""

    dataset: DatasetName = DatasetName.HUMAN_EVAL

    def __init__(self, snapshot_path: Path) -> None:
        self._snapshot_path = snapshot_path
        (
            self._records_by_task_id,
            self._source_digests_by_task_id,
        ) = _load_snapshot(snapshot_path)

    def records(self) -> Iterable[TaskRecord]:
        return tuple(
            self._records_by_task_id[task_id] for task_id in _EXPECTED_TASK_IDS
        )

    def resolve(self, data_sample_id: str) -> TaskRecord | None:
        try:
            task_id, source_digest = parse_human_eval_data_sample_id(
                data_sample_id
            )
        except ValueError:
            return None
        if (
            source_digest is not None
            and source_digest != self._source_digests_by_task_id[task_id][:16]
        ):
            return None
        return self._records_by_task_id[task_id]


def _load_snapshot(
    path: Path,
) -> tuple[dict[str, TaskRecord], dict[str, str]]:
    try:
        snapshot_bytes = path.read_bytes()
        payload: object = json.loads(snapshot_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid HumanEval snapshot {path}: {exc}") from exc
    actual_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
    if actual_sha256 != _EXPECTED_SNAPSHOT_SHA256:
        raise ValueError(
            "HumanEval snapshot content hash mismatch: "
            f"expected={_EXPECTED_SNAPSHOT_SHA256}, actual={actual_sha256}"
        )
    if not isinstance(payload, dict):
        raise ValueError(f"invalid HumanEval snapshot {path}: expected object")

    header = payload.get("header")
    rows = payload.get("rows")
    if not isinstance(header, dict) or not isinstance(rows, list):
        raise ValueError(
            f"invalid HumanEval snapshot {path}: missing header or rows"
        )
    if header.get("schema_version") != 2:
        raise ValueError(
            f"unsupported HumanEval snapshot schema in {path}: "
            f"{header.get('schema_version')!r}"
        )
    dataset_id = header.get("dataset_id")
    revision = header.get("hf_revision")
    if dataset_id != _EXPECTED_DATASET_ID or revision != _EXPECTED_REVISION:
        raise ValueError(f"invalid HumanEval snapshot identity in {path}")
    if not isinstance(dataset_id, str):
        raise AssertionError("validated HumanEval dataset ID is not a string")
    if not isinstance(revision, str):
        raise AssertionError("validated HumanEval revision is not a string")

    records: dict[str, TaskRecord] = {}
    source_digests: dict[str, str] = {}
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, dict):
            raise ValueError(
                f"invalid HumanEval snapshot row {index}: expected object"
            )
        row = _validate_task_row(cast(dict[str, object], raw_row), index=index)
        task_id_value = row["task_id"]
        if not isinstance(task_id_value, str):
            raise AssertionError("validated HumanEval task ID is not a string")
        task_id = task_id_value
        if task_id in records:
            raise ValueError(f"duplicate HumanEval snapshot task {task_id}")
        task_json = canonical_json(row)
        digest = task_record_id(row)
        source_digest = _cleaned_source_digest(row)
        records[task_id] = TaskRecord(
            task_record_id=digest,
            dataset=DatasetName.HUMAN_EVAL,
            source_variant="humanevalplus_snapshot",
            task_id=task_id,
            language="python",
            dataset_id=dataset_id,
            split="test",
            data_sample_id=None,
            source_digest=source_digest,
            dataset_revision=revision,
            evaluator_kind="humaneval_plus",
            material_fidelity=TaskMaterialFidelity.PINNED_SNAPSHOT,
            task_json=task_json,
            content_sha256=digest,
        )
        source_digests[task_id] = source_digest

    actual_task_ids = tuple(sorted(records, key=_task_index))
    if actual_task_ids != _EXPECTED_TASK_IDS:
        missing = sorted(set(_EXPECTED_TASK_IDS).difference(records))
        unexpected = sorted(set(records).difference(_EXPECTED_TASK_IDS))
        raise ValueError(
            "HumanEval snapshot task set mismatch: "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    return records, source_digests


def _cleaned_source_digest(row: dict[str, JsonValue]) -> str:
    prompt = row["prompt"]
    canonical_solution = row["canonical_solution"]
    if not isinstance(prompt, str) or not isinstance(canonical_solution, str):
        raise AssertionError("validated HumanEval source fields are strings")
    try:
        tree = ast.parse(prompt + canonical_solution)
    except SyntaxError as exc:
        raise ValueError(
            f"invalid HumanEval source for {row['task_id']!r}: {exc}"
        ) from exc
    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            continue
        if not node.body:
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            node.body = node.body[1:] or [ast.Pass()]
    ast.fix_missing_locations(tree)
    cleaned_source = ast.unparse(tree) + "\n"
    return hashlib.sha256(
        f"gt_solution\0{cleaned_source}".encode("utf-8")
    ).hexdigest()


def _validate_task_row(
    row: dict[str, object], *, index: int
) -> dict[str, JsonValue]:
    required_strings = (
        "task_id",
        "prompt",
        "canonical_solution",
        "entry_point",
        "test",
    )
    for field in required_strings:
        value = row.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"invalid HumanEval snapshot row {index}: "
                f"{field} must be a nonempty string"
            )
    task_id = row["task_id"]
    if not isinstance(task_id, str) or _TASK_ID_RE.fullmatch(task_id) is None:
        raise ValueError(
            f"invalid HumanEval snapshot row {index}: task_id={task_id!r}"
        )
    try:
        # Round-trip through JSON to reject values outside the persisted JSON
        # boundary and detach task material from the mutable input object.
        normalized: object = json.loads(canonical_json(cast(JsonValue, row)))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid HumanEval snapshot row {index}: {exc}"
        ) from exc
    if not isinstance(normalized, dict):
        raise AssertionError("canonicalized HumanEval row is not an object")
    return cast(dict[str, JsonValue], normalized)


def _task_index(task_id: str) -> int:
    match = _TASK_ID_RE.fullmatch(task_id)
    if match is None:
        raise ValueError(f"invalid HumanEval task ID {task_id!r}")
    return int(match.group("index"))


__all__ = [
    "HumanEvalTaskAdapter",
    "parse_human_eval_data_sample_id",
]
