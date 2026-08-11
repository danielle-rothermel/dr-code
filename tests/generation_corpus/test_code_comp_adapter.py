from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from dr_code.generation_corpus import CorpusWriter, read_manifest
from dr_code.generation_corpus.adapters.base import CorpusAdapter
from dr_code.generation_corpus.adapters.code_comp import MbppProCodeCompAdapter
from dr_code.generation_corpus.tasks import code_eval_pro as task_module
from dr_code.generation_corpus.tasks.base import TaskAdapter
from dr_code.generation_corpus.tasks.code_eval_pro import (
    BigCodeBenchLiteProTaskAdapter,
    ClassEvalTaskAdapter,
    HumanEvalProTaskAdapter,
    MbppProTaskAdapter,
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _write_task_cache(
    directory: Path,
    *,
    spec: task_module._TaskDatasetSpec,
    task_id: str,
    flawed_id: str,
    source_by_variant: dict[str, str],
) -> task_module._TaskDatasetSpec:
    directory.mkdir()
    raw_sample: dict[str, Any] = {"task_id": task_id, "validated": True}
    raw_sample.update(source_by_variant)
    canonical_source = next(iter(source_by_variant.values()))
    task = {
        "dataset": spec.dataset_id,
        "task_id": task_id,
        "source": {"code": canonical_source, "kind": "gt_solution"},
    }
    payload = {
        "raw_samples": {task_id: raw_sample},
        "flawed_raw_samples": {flawed_id: {"error": "synthetic flaw"}},
        "tasks": {task_id: task},
    }
    payload_path = directory / "payload.json.gz"
    with gzip.open(payload_path, "wt", encoding="utf-8") as file:
        json.dump(payload, file)

    source_entries = []
    for variant in spec.source_variants:
        source = source_by_variant[variant]
        if variant == "cleaned":
            source = task_module._clean_python_source(canonical_source)
        digest = hashlib.sha256(f"gt_solution\0{source}".encode()).hexdigest()
        source_entries.append([task_id, variant, digest])
    patched_spec = replace(
        spec,
        raw_sample_count=1,
        flawed_ids=(flawed_id,),
        task_count=1,
        payload_sha256=hashlib.sha256(payload_path.read_bytes()).hexdigest(),
        accepted_ids_sha256=_canonical_sha256([task_id]),
        source_digests_sha256=_canonical_sha256(source_entries),
    )
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": patched_spec.dataset_id,
                "split": patched_spec.split,
                "source_revision": patched_spec.revision,
                "cache_schema_version": patched_spec.cache_schema_version,
                "built_at": "2026-08-08T12:00:00+00:00",
                "raw_sample_count": 1,
                "flawed_count": 1,
                "task_count": 1,
            }
        ),
        encoding="utf-8",
    )
    return patched_spec


@pytest.mark.parametrize(
    ("adapter_type", "spec_name", "task_id", "flawed_id", "sources"),
    [
        (
            MbppProTaskAdapter,
            "_MBPP_PRO_TASK_SPEC",
            "MbppPro/0",
            "MbppPro/36",
            {"canonical": "def target():\n    return 1\n"},
        ),
        (
            HumanEvalProTaskAdapter,
            "_HUMANEVAL_PRO_TASK_SPEC",
            "HumanEvalPro/0",
            "HumanEvalPro/24",
            {"canonical": "def target():\n    return 1\n"},
        ),
        (
            ClassEvalTaskAdapter,
            "_CLASS_EVAL_TASK_SPEC",
            "ClassEval_0",
            "ClassEval_48",
            {
                "gt_code": "class Target:\n    pass\n",
                "gt_code_with_comments": "# retained\nclass Target:\n    pass\n",
            },
        ),
        (
            BigCodeBenchLiteProTaskAdapter,
            "_BIGCODEBENCH_LITE_PRO_TASK_SPEC",
            "BigCodeBenchLitePro/23",
            "BigCodeBenchLitePro/201",
            {
                "canonical": (
                    '"""module docs"""\n'
                    "def target():\n"
                    '    """function docs"""\n'
                    "    return 1\n"
                ),
                "cleaned": "unused: replaced by the pinned cleaner",
            },
        ),
    ],
)
def test_pinned_task_adapters_resolve_only_full_source_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    adapter_type: type[task_module._PinnedCacheTaskAdapter],
    spec_name: str,
    task_id: str,
    flawed_id: str,
    sources: dict[str, str],
) -> None:
    original_spec = getattr(task_module, spec_name)
    cache = tmp_path / spec_name
    patched_spec = _write_task_cache(
        cache,
        spec=original_spec,
        task_id=task_id,
        flawed_id=flawed_id,
        source_by_variant=sources,
    )
    monkeypatch.setattr(adapter_type, "_spec", patched_spec)

    adapter = adapter_type(cache)
    task_contract: TaskAdapter = adapter
    records = tuple(task_contract.records())

    assert len(records) == len(patched_spec.source_variants)
    assert {record.source_variant for record in records} == set(
        patched_spec.source_variants
    )
    assert all(len(record.source_digest or "") == 64 for record in records)
    assert all(
        task_contract.resolve(record.data_sample_id or "") == record
        for record in records
    )
    unknown_digest_id = (
        f"{patched_spec.definition.namespace}/{task_id}/"
        "gt_solution@0000000000000000"
    )
    assert task_contract.resolve(unknown_digest_id) is None
    with pytest.raises(ValueError, match="invalid .* data_sample_id"):
        task_contract.resolve(f"{patched_spec.definition.namespace}/{task_id}")


def test_task_adapter_rejects_manifest_payload_and_population_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_spec = task_module._MBPP_PRO_TASK_SPEC
    cache = tmp_path / "cache"
    patched_spec = _write_task_cache(
        cache,
        spec=original_spec,
        task_id="MbppPro/0",
        flawed_id="MbppPro/36",
        source_by_variant={"canonical": "def target():\n    return 1\n"},
    )
    monkeypatch.setattr(MbppProTaskAdapter, "_spec", patched_spec)

    manifest = json.loads((cache / "manifest.json").read_text())
    manifest["split"] = "validation"
    (cache / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest coordinate mismatch"):
        MbppProTaskAdapter(cache)

    manifest["split"] = patched_spec.split
    (cache / "manifest.json").write_text(json.dumps(manifest))
    with gzip.open(cache / "payload.json.gz", "rt", encoding="utf-8") as file:
        valid_payload = json.load(file)
    (cache / "payload.json.gz").write_bytes(
        (cache / "payload.json.gz").read_bytes() + b"drift"
    )
    with pytest.raises(ValueError, match="payload SHA-256 mismatch"):
        MbppProTaskAdapter(cache)

    with gzip.open(cache / "payload.json.gz", "wt", encoding="utf-8") as file:
        json.dump(valid_payload, file)
    corrected_payload_sha = hashlib.sha256(
        (cache / "payload.json.gz").read_bytes()
    ).hexdigest()
    monkeypatch.setattr(
        MbppProTaskAdapter,
        "_spec",
        replace(
            patched_spec,
            payload_sha256=corrected_payload_sha,
            accepted_ids_sha256="0" * 64,
        ),
    )
    with pytest.raises(ValueError, match="accepted ID set digest mismatch"):
        MbppProTaskAdapter(cache)

    monkeypatch.setattr(
        MbppProTaskAdapter,
        "_spec",
        replace(
            patched_spec,
            payload_sha256=corrected_payload_sha,
            source_digests_sha256="0" * 64,
        ),
    )
    with pytest.raises(ValueError, match="full source digest set mismatch"):
        MbppProTaskAdapter(cache)


_POOL_KEYS = {
    "direct_enc_t0": [
        "prompt_template_id",
        "data_sample_id",
        "llm_config_id",
        "status",
    ],
    "encoder_pool_t1": [
        "prompt_template_id",
        "data_sample_id",
        "llm_config_id",
        "status",
    ],
    "official_decoder_t0": [
        "enc_prompt_template_id",
        "data_sample_id",
        "enc_llm_config_id",
        "enc_sample_id",
        "dec_prompt_template_id",
        "dec_llm_config_id",
        "status",
    ],
    "reexport_shell_seed_decoder": [
        "enc_prompt_template_id",
        "data_sample_id",
        "enc_llm_config_id",
        "enc_sample_id",
        "dec_prompt_template_id",
        "dec_llm_config_id",
    ],
    "reexport_shell_seed_encoder": [
        "prompt_template_id",
        "data_sample_id",
        "llm_config_id",
    ],
    "reexport_smoke_20260510_155435": [
        "prompt_template_id",
        "data_sample_id",
        "llm_config_id",
    ],
    "tstfix_20260510_034336": [
        "prompt_template_id",
        "data_sample_id",
        "llm_config_id",
    ],
}


def _row(
    *,
    pool: str,
    sample_id: str,
    data_sample_id: str,
    key_values: dict[str, object],
    request: dict[str, object],
    response: dict[str, object] | None,
    metadata: dict[str, object],
    attempt_count: int,
    finish_reason: str | None,
    output_kind: str,
    output_path: str | None,
) -> dict[str, object]:
    return {
        "project_name": "code_comp_v0",
        "pool_name": pool,
        "table_name": f"pool_{pool}_samples",
        "sample_id": sample_id,
        "key_values": {"data_sample_id": data_sample_id, **key_values},
        "sample_idx": 0,
        "run_id": None,
        "request_json": request,
        "response_json": response,
        "finish_reason": finish_reason,
        "attempt_count": attempt_count,
        "metadata_json": metadata,
        "created_at": "2026-05-10T12:00:00Z",
        "hints": {
            "human_eval_task_id": None,
            "human_eval_pro_task_id": None,
            "output_kind": output_kind,
            "output_json_path": output_path,
            "decoder_input_description_source": (
                "request.prompt" if "prompt" in request else "missing"
            ),
        },
    }


def _write_pool_dump(
    dump_directory: Path,
    *,
    canonical_data_sample_id: str,
    invalid_decoder_sentinel: bool = False,
) -> Path:
    opaque_data_sample_id = "mbpp_pro/MbppPro/0/gt_solution@0000000000000000"
    exact_request: dict[str, object] = {
        "prompt": [{"role": "user", "content": "describe the source"}],
        "llm_config": {
            "provider": "synthetic",
            "model": "encoder",
            "reasoning": None,
        },
    }
    response: dict[str, object] = {
        "text": "persisted text",
        "provider": "synthetic",
        "model": "model",
        "finish_reason": "stop",
    }
    rows_by_pool: dict[str, list[dict[str, object]]] = {
        pool: [] for pool in _POOL_KEYS
    }
    rows_by_pool["direct_enc_t0"].append(
        _row(
            pool="direct_enc_t0",
            sample_id="failed-encoder",
            data_sample_id=canonical_data_sample_id,
            key_values={
                "prompt_template_id": "template",
                "llm_config_id": "config",
                "status": "failed",
            },
            request=exact_request,
            response={"error": "synthetic failure"},
            metadata={"fail_reason": "synthetic failure"},
            attempt_count=3,
            finish_reason="error",
            output_kind="not_code",
            output_path=None,
        )
    )
    rows_by_pool["encoder_pool_t1"].append(
        _row(
            pool="encoder_pool_t1",
            sample_id="opaque-encoder",
            data_sample_id=opaque_data_sample_id,
            key_values={
                "prompt_template_id": "template",
                "llm_config_id": "config",
                "status": "active",
            },
            request={"reason": "historical_migration", "unavailable": True},
            response=response,
            metadata={},
            attempt_count=0,
            finish_reason=None,
            output_kind="not_code",
            output_path=None,
        )
    )
    rows_by_pool["official_decoder_t0"].append(
        _row(
            pool="official_decoder_t0",
            sample_id="decoder",
            data_sample_id=canonical_data_sample_id,
            key_values={
                "enc_prompt_template_id": "sentinel/task_prompt",
                "enc_llm_config_id": "sentinel/non_llm",
                "enc_sample_id": f"sentinel/task_prompt/{canonical_data_sample_id}",
                "dec_prompt_template_id": "sentinel/official_prompt",
                "dec_llm_config_id": "synthetic/decoder",
                "status": "active",
            },
            request={"reason": "historical_migration", "unavailable": True},
            response={**response, "finish_reason": "length"},
            metadata={
                "source_kind": (
                    "encoder_sample"
                    if invalid_decoder_sentinel
                    else "task_prompt"
                )
            },
            attempt_count=0,
            finish_reason=None,
            output_kind="code_text",
            output_path="response_json.text",
        )
    )
    rows_by_pool["reexport_smoke_20260510_155435"].append(
        _row(
            pool="reexport_smoke_20260510_155435",
            sample_id="exact-encoder",
            data_sample_id=canonical_data_sample_id,
            key_values={
                "prompt_template_id": "template",
                "llm_config_id": "config",
            },
            request=exact_request,
            response=response,
            metadata={},
            attempt_count=1,
            finish_reason="stop",
            output_kind="not_code",
            output_path=None,
        )
    )

    entries = []
    for pool, key_names in _POOL_KEYS.items():
        rows = rows_by_pool[pool]
        file_name = f"code_comp_v0__{pool}.jsonl.gz"
        with gzip.open(
            dump_directory / file_name, "wt", encoding="utf-8"
        ) as file:
            for row in rows:
                file.write(json.dumps(row) + "\n")
        entries.append(
            {
                "project_name": "code_comp_v0",
                "pool_name": pool,
                "table_name": f"pool_{pool}_samples",
                "file_name": file_name,
                "row_count": len(rows),
                "dumped_row_count": len(rows),
                "pool_schema_json": {
                    "name": pool,
                    "key_columns": [
                        {"name": name, "type": "text"} for name in key_names
                    ],
                },
                "original_status": "stopped",
                "temporarily_started": False,
            }
        )
    manifest_path = dump_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "created_at": "2026-06-21T20:19:47Z",
                "output_dir": "/synthetic",
                "pools": entries,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _install_synthetic_mbpp_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, str]:
    cache = tmp_path / "task-cache"
    source = "def target():\n    return 1\n"
    spec = _write_task_cache(
        cache,
        spec=task_module._MBPP_PRO_TASK_SPEC,
        task_id="MbppPro/0",
        flawed_id="MbppPro/36",
        source_by_variant={"canonical": source},
    )
    monkeypatch.setattr(MbppProTaskAdapter, "_spec", spec)
    digest = hashlib.sha256(f"gt_solution\0{source}".encode()).hexdigest()
    return cache, f"mbpp_pro/MbppPro/0/gt_solution@{digest[:16]}"


def test_code_comp_adapter_keeps_raw_decoder_and_encoder_grains_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache, data_sample_id = _install_synthetic_mbpp_cache(
        tmp_path, monkeypatch
    )
    dump_directory = tmp_path / "dump"
    dump_directory.mkdir()
    manifest_path = _write_pool_dump(
        dump_directory, canonical_data_sample_id=data_sample_id
    )
    manifest = read_manifest(manifest_path)
    adapter = MbppProCodeCompAdapter(cache)
    adapter_contract: CorpusAdapter = adapter
    destination = tmp_path / "corpus"

    with CorpusWriter(
        destination,
        source_manifest_path=manifest_path,
        source_manifest=manifest,
        adapter_name=adapter.adapter_name,
        created_at="2026-08-08T12:00:00+00:00",
    ) as writer:
        adapter_contract.populate(
            dump_directory=dump_directory,
            source_manifest=manifest,
            writer=writer,
        )

    sources = pl.read_parquet(destination / "source_records.parquet")
    generations = pl.read_parquet(destination / "generations.parquet")
    encoders = pl.read_parquet(destination / "encoder_artifacts.parquet")
    requests = pl.read_parquet(destination / "requests.parquet")
    tasks = pl.read_parquet(destination / "tasks.parquet")

    assert sources.height == 4
    assert sorted(sources["attempt_count"].to_list()) == [0, 0, 1, 3]
    assert generations.height == requests.height == tasks.height == 1
    assert encoders.height == 2
    assert (
        generations.row(0, named=True)["prompt_fidelity"] == "recovered_task"
    )
    assert generations.row(0, named=True)["is_partial"] is True
    assert generations.row(0, named=True)["decoder_user_prompt"] is None
    assert requests.row(0, named=True)["encoder_source_record_id"] is None
    assert requests.row(0, named=True)["decoder_provider"] is None
    assert requests.row(0, named=True)["budget_mode"] == "no_budget"
    assert set(encoders["prompt_fidelity"].to_list()) == {
        "exact_request",
        "unavailable",
    }
    opaque = encoders.filter(
        pl.col("source_variant") == "opaque_encoder_pool_t1"
    )
    assert opaque.height == 1
    assert opaque.row(0, named=True)["task_record_id"] is None


def test_code_comp_adapter_rejects_invented_decoder_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache, data_sample_id = _install_synthetic_mbpp_cache(
        tmp_path, monkeypatch
    )
    dump_directory = tmp_path / "dump"
    dump_directory.mkdir()
    manifest_path = _write_pool_dump(
        dump_directory,
        canonical_data_sample_id=data_sample_id,
        invalid_decoder_sentinel=True,
    )
    manifest = read_manifest(manifest_path)
    adapter = MbppProCodeCompAdapter(cache)

    with pytest.raises(ValueError, match="direct decoder sentinel mismatch"):
        with CorpusWriter(
            tmp_path / "corpus",
            source_manifest_path=manifest_path,
            source_manifest=manifest,
            adapter_name=adapter.adapter_name,
            created_at="2026-08-08T12:00:00+00:00",
        ) as writer:
            adapter.populate(
                dump_directory=dump_directory,
                source_manifest=manifest,
                writer=writer,
            )
