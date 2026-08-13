from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from drc_generation_corpus import (
    canonical_json,
    content_sha256,
    generation_id,
    iter_pool_rows,
    read_manifest,
    source_record_id,
)


def _entry(*, count: int = 1) -> dict[str, Any]:
    return {
        "project_name": "code_comp_v0",
        "pool_name": "official_decoder_t0",
        "table_name": "pool_official_decoder_t0_samples",
        "file_name": "code_comp_v0__official_decoder_t0.jsonl.gz",
        "row_count": count,
        "dumped_row_count": count,
        "pool_schema_json": {
            "name": "official_decoder_t0",
            "key_columns": [
                {"name": "data_sample_id", "type": "text"},
                {"name": "dec_llm_config_id", "type": "text"},
            ],
        },
        "original_status": "stopped",
        "temporarily_started": True,
    }


def _manifest(*, count: int = 1) -> dict[str, Any]:
    return {
        "version": 1,
        "created_at": "2026-06-21T20:19:47.639734Z",
        "output_dir": "/source/dump",
        "pools": [_entry(count=count)],
    }


def _row() -> dict[str, Any]:
    return {
        "project_name": "code_comp_v0",
        "pool_name": "official_decoder_t0",
        "table_name": "pool_official_decoder_t0_samples",
        "sample_id": "sample-1",
        "key_values": {
            "data_sample_id": "human_eval/HumanEval/0/gt_solution@abc",
            "dec_llm_config_id": "openai/gpt-5-nano/minimal/v1",
        },
        "sample_idx": 0,
        "run_id": None,
        "request_json": {"prompt": "implement"},
        "response_json": {"text": "def f():\n    pass\n"},
        "finish_reason": "stop",
        "attempt_count": 2,
        "metadata_json": {"source_kind": "task_prompt"},
        "created_at": "2026-05-10T10:00:00Z",
        "hints": {
            "human_eval_task_id": "HumanEval/0",
            "human_eval_pro_task_id": None,
            "output_kind": "code_text",
            "output_json_path": "response_json.text",
            "decoder_input_description_source": "request.prompt",
        },
    }


def _write_dump(path: Path, rows: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def test_reads_exact_manifest_and_validates_pool_coordinates(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    manifest = read_manifest(manifest_path)
    dump_path = tmp_path / manifest.pools[0].file_name
    _write_dump(dump_path, [_row()])

    rows = list(iter_pool_rows(dump_path, manifest.pools[0]))

    assert len(rows) == 1
    assert rows[0].created_at == "2026-05-10T10:00:00Z"
    assert source_record_id(rows[0]) == (
        "code_comp_v0:official_decoder_t0:sample-1"
    )


def test_strict_models_reject_unknown_manifest_and_row_fields(
    tmp_path: Path,
) -> None:
    manifest_payload = _manifest()
    manifest_payload["unexpected"] = True
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        read_manifest(manifest_path)

    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    manifest = read_manifest(manifest_path)
    row = _row()
    row["unexpected"] = True
    dump_path = tmp_path / manifest.pools[0].file_name
    _write_dump(dump_path, [row])
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        list(iter_pool_rows(dump_path, manifest.pools[0]))


def test_rejects_manifest_and_physical_row_count_mismatches(
    tmp_path: Path,
) -> None:
    incomplete = _manifest()
    incomplete["pools"][0]["row_count"] = 2
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(incomplete), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete pool dumps"):
        read_manifest(manifest_path)

    manifest_path.write_text(json.dumps(_manifest(count=2)), encoding="utf-8")
    manifest = read_manifest(manifest_path)
    dump_path = tmp_path / manifest.pools[0].file_name
    _write_dump(dump_path, [_row()])
    with pytest.raises(ValueError, match="row count mismatch"):
        list(iter_pool_rows(dump_path, manifest.pools[0]))


def test_rejects_row_source_coordinate_drift(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    manifest = read_manifest(manifest_path)
    row = _row()
    row["pool_name"] = "another_pool"
    dump_path = tmp_path / manifest.pools[0].file_name
    _write_dump(dump_path, [row])

    with pytest.raises(ValueError, match="coordinate mismatch"):
        list(iter_pool_rows(dump_path, manifest.pools[0]))


def test_canonical_json_ids_and_hashes_are_stable() -> None:
    first = canonical_json({"z": ["é", None], "a": 1})
    second = canonical_json({"a": 1, "z": ["é", None]})
    row_payload = _row()
    from drc_generation_corpus import DumpedPoolRow

    row = DumpedPoolRow.model_validate(row_payload)

    assert first == second == '{"a":1,"z":["é",null]}'
    assert generation_id(row) == (
        "14417e040de531c9d00d397c31c9d499812eacb95e5df5761ba63ff29406af4e"
    )
    assert content_sha256("left", None, "right") == (
        "01a6d668d2a83f1b1e797d5ff389aed5026742324974e3f4c2141fbe8dc7cf0d"
    )
