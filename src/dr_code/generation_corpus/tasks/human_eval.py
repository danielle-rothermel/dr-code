from __future__ import annotations

import ast
import hashlib
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Final, cast

from pydantic import JsonValue, ValidationError

from dr_code.generation_corpus.models import (
    DatasetName,
    TaskMaterialFidelity,
    TaskRecord,
)
from dr_code.generation_corpus.pool_dump import canonical_json, task_record_id
from dr_code.humaneval.sampling import (
    DEFAULT_HUMANEVAL_DATASET_NAME,
    DEFAULT_HUMANEVAL_DATASET_SPLIT,
    DEFAULT_HUMANEVAL_HF_REVISION,
    load_humaneval_raw_snapshot,
)

_TASK_ID_RE: Final = re.compile(r"^HumanEval/(?P<index>\d+)$")
_DATA_SAMPLE_ID_RE: Final = re.compile(
    r"^human_eval/(?P<task_id>HumanEval/\d+)"
    r"(?:/gt_solution(?:@(?P<source_digest>[0-9a-f]{16}))?)?$"
)
_EXPECTED_TASK_IDS: Final = tuple(f"HumanEval/{index}" for index in range(164))
_REQUIRED_ROW_FIELDS: Final = (
    "task_id",
    "prompt",
    "canonical_solution",
    "entry_point",
    "test",
)
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
    _verify_snapshot_bytes(path)
    try:
        snapshot = load_humaneval_raw_snapshot(path)
    except (OSError, ValidationError, ValueError) as exc:
        raise ValueError(f"invalid HumanEval snapshot {path}: {exc}") from exc
    # The adapter records the snapshot's rows as pinned material and applies no
    # overrides, so it checks the header's dataset identity itself rather than
    # the override set the sampling loader requires.
    if (
        snapshot.header.dataset_id != DEFAULT_HUMANEVAL_DATASET_NAME
        or snapshot.header.hf_revision != DEFAULT_HUMANEVAL_HF_REVISION
    ):
        raise ValueError(f"invalid HumanEval snapshot identity in {path}")

    records: dict[str, TaskRecord] = {}
    source_digests: dict[str, str] = {}
    for index, raw_row in enumerate(snapshot.rows):
        row = cast(dict[str, JsonValue], raw_row.model_dump(mode="json"))
        _require_nonempty_strings(row, index=index)
        task_id = row["task_id"]
        if not isinstance(task_id, str):
            raise AssertionError("validated HumanEval task ID is not a string")
        if _TASK_ID_RE.fullmatch(task_id) is None:
            raise ValueError(
                f"invalid HumanEval snapshot task_id {task_id!r} in {path}"
            )
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
            dataset_id=DEFAULT_HUMANEVAL_DATASET_NAME,
            split=DEFAULT_HUMANEVAL_DATASET_SPLIT,
            data_sample_id=None,
            source_digest=source_digest,
            dataset_revision=DEFAULT_HUMANEVAL_HF_REVISION,
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


def _require_nonempty_strings(
    row: dict[str, JsonValue], *, index: int
) -> None:
    """Reject a row whose recorded material is an empty string.

    The snapshot model types every field `StrictStr`, which admits `""`; the
    corpus record built from the row is only meaningful when each field carries
    content.
    """

    for field in _REQUIRED_ROW_FIELDS:
        value = row.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"invalid HumanEval snapshot row {index}: "
                f"{field} must be a nonempty string"
            )


def _verify_snapshot_bytes(path: Path) -> None:
    try:
        snapshot_bytes = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"invalid HumanEval snapshot {path}: {exc}") from exc
    actual_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
    if actual_sha256 != _EXPECTED_SNAPSHOT_SHA256:
        raise ValueError(
            "HumanEval snapshot content hash mismatch: "
            f"expected={_EXPECTED_SNAPSHOT_SHA256}, actual={actual_sha256}"
        )


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


def _task_index(task_id: str) -> int:
    match = _TASK_ID_RE.fullmatch(task_id)
    if match is None:
        raise ValueError(f"invalid HumanEval task ID {task_id!r}")
    return int(match.group("index"))


__all__ = [
    "HumanEvalTaskAdapter",
    "parse_human_eval_data_sample_id",
]
