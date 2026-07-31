"""Tests for HumanEvalPlus source selection."""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path

import pytest

from dr_code.humaneval.sampling import (
    HUMAN_EVAL_RAW_ROW_SNAPSHOT_SCHEMA_VERSION,
    HumanEvalRawRowsSnapshot,
    human_eval_overrides_digest,
)
from dr_code.synthetic import humaneval_loader
from dr_code.synthetic.humaneval_loader import (
    HumanEvalSource,
    load_humaneval_plus,
)


ROW = {
    "task_id": "HumanEval/0",
    "prompt": "def add(a, b):\n",
    "canonical_solution": "    return a + b\n",
    "entry_point": "add",
    "test": "",
}


def test_default_loader_uses_only_packaged_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[bytes, Mapping[str, object]]] = []

    def packaged_rows(
        content: bytes,
        **kwargs: object,
    ) -> list[dict[str, str]]:
        calls.append((content, kwargs))
        return [ROW]

    monkeypatch.setattr(
        humaneval_loader,
        "packaged_snapshot_bytes",
        lambda: b"packaged snapshot",
    )
    monkeypatch.setattr(
        humaneval_loader,
        "load_human_eval_snapshot_rows_bytes",
        packaged_rows,
    )
    monkeypatch.setattr(
        humaneval_loader,
        "_load_from_hf",
        lambda: (_ for _ in ()).throw(
            AssertionError("HF must not be consulted")
        ),
    )

    tasks = load_humaneval_plus()

    assert [task.task_id for task in tasks] == ["HumanEval/0"]
    assert calls == [
        (
            b"packaged snapshot",
            {
                "dataset_name": humaneval_loader.HF_DATASET_ID,
                "hf_revision": humaneval_loader.HF_REVISION,
            },
        )
    ]


def test_explicit_hf_loader_is_independent_and_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Mapping[str, object]] = []

    def available_source(**kwargs: object) -> list[dict[str, str]]:
        calls.append(kwargs)
        return [ROW]

    monkeypatch.setattr(
        humaneval_loader,
        "load_human_eval_rows",
        available_source,
    )
    monkeypatch.setattr(
        humaneval_loader,
        "packaged_snapshot_bytes",
        lambda: (_ for _ in ()).throw(
            AssertionError("snapshot must not be consulted")
        ),
    )

    tasks = load_humaneval_plus(source=HumanEvalSource.HF)

    assert [task.task_id for task in tasks] == ["HumanEval/0"]
    assert calls == [
        {
            "dataset_name": humaneval_loader.HF_DATASET_ID,
            "dataset_split": humaneval_loader.HF_SPLIT,
            "hf_revision": humaneval_loader.HF_REVISION,
        }
    ]


def test_packaged_snapshot_resource_matches_pinned_sha256() -> None:
    content = (
        files("dr_code.synthetic")
        .joinpath(humaneval_loader.SNAPSHOT_RESOURCE)
        .read_bytes()
    )
    snapshot = HumanEvalRawRowsSnapshot.model_validate_json(content)

    assert (
        hashlib.sha256(content).hexdigest() == humaneval_loader.SNAPSHOT_SHA256
    )
    assert snapshot.header.schema_version == (
        HUMAN_EVAL_RAW_ROW_SNAPSHOT_SCHEMA_VERSION
    )
    assert snapshot.header.dataset_id == humaneval_loader.HF_DATASET_ID
    assert snapshot.header.dataset_split == humaneval_loader.HF_SPLIT
    assert snapshot.header.hf_revision == humaneval_loader.HF_REVISION
    assert snapshot.header.overrides_digest == human_eval_overrides_digest()


def test_packaged_snapshot_is_byte_reproducible(tmp_path: Path) -> None:
    destination = tmp_path / "snapshot.json"

    humaneval_loader.save_snapshot(
        load_humaneval_plus(),
        destination,
    )

    assert (
        destination.read_bytes() == humaneval_loader.packaged_snapshot_bytes()
    )


def test_snapshot_regeneration_requires_explicit_destination() -> None:
    destination = inspect.signature(humaneval_loader.save_snapshot).parameters[
        "destination"
    ]

    assert destination.default is inspect.Parameter.empty
