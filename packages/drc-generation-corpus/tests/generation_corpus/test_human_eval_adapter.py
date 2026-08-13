from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, cast

import pytest

from drc_generation_corpus.adapters.base import CorpusAdapter
from drc_generation_corpus.adapters.human_eval import HumanEvalAdapter
from drc_generation_corpus.models import (
    EncoderArtifactRecord,
    GenerationMode,
    GenerationRecord,
    LifecycleState,
    PromptFidelity,
    RequestRecord,
    SourceManifest,
    SourceRecord,
    Stage,
    TaskRecord,
)
from drc_generation_corpus.pool_dump import read_manifest
from drc_generation_corpus.tasks.base import TaskAdapter
from drc_generation_corpus.tasks.human_eval import HumanEvalTaskAdapter
from drc_generation_corpus.writer import CorpusWriter

from _paths import HUMANEVALPLUS_SNAPSHOT as _HUMANEVAL_SNAPSHOT

_DESCRIPTION = "Returns True when a sufficiently close pair exists."
_CODE = "def has_close_elements(numbers, threshold):\n    return False\n"
_SOURCE_DIGEST = "4c72a0b83d08f3bc"


class _Sink:
    def __init__(self) -> None:
        self.generations: list[GenerationRecord] = []
        self.sources: list[SourceRecord] = []
        self.encoders: list[EncoderArtifactRecord] = []
        self.requests: list[RequestRecord] = []
        self.tasks: list[TaskRecord] = []

    def add_generation(self, record: GenerationRecord) -> None:
        self.generations.append(record)

    def add_source_record(self, record: SourceRecord) -> None:
        self.sources.append(record)

    def add_encoder_artifact(self, record: EncoderArtifactRecord) -> None:
        self.encoders.append(record)

    def add_request(self, record: RequestRecord) -> None:
        self.requests.append(record)

    def add_task(self, record: TaskRecord) -> None:
        self.tasks.append(record)


def _hints(task_id: str, *, encoder: bool = False) -> dict[str, Any]:
    return {
        "human_eval_task_id": task_id,
        "human_eval_pro_task_id": None,
        "output_kind": "not_code" if encoder else "code_text",
        "output_json_path": None if encoder else "response_json.text",
        "decoder_input_description_source": "metadata.source_text",
    }


def _llm_config() -> dict[str, Any]:
    return {
        "provider": "openai",
        "model": "gpt-5-nano",
        "reasoning": {"kind": "openai", "thinking_level": "minimal"},
        "sampling": {"temperature": 0.0, "top_p": 1.0},
        "max_tokens": 512,
    }


def _direct_row(
    sample_id: str,
    *,
    task_id: str = "HumanEval/0",
    request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "project_name": "code_comp_v0",
        "pool_name": "official_decoder_t0",
        "table_name": "pool_official_decoder_t0_samples",
        "sample_id": sample_id,
        "key_values": {
            "data_sample_id": (
                f"human_eval/{task_id}/gt_solution@{_SOURCE_DIGEST}"
            ),
            "dec_llm_config_id": "openai/gpt-5-nano/minimal/v1",
            "dec_prompt_template_id": "sentinel/official_prompt",
        },
        "sample_idx": 0,
        "run_id": "run-1",
        "request_json": request
        or {
            "llm_config": _llm_config(),
            "prompt": [{"role": "user", "content": "Implement it."}],
        },
        "response_json": {
            "text": _CODE,
            "provider": "openai",
            "model": "gpt-5-nano",
            "finish_reason": "length",
        },
        "finish_reason": None,
        "attempt_count": 3,
        "metadata_json": {"source_kind": "task_prompt"},
        "created_at": "2026-05-10T10:02:00Z",
        "hints": _hints(task_id),
    }


def _encoder_row(sample_id: str, *, linked: bool) -> dict[str, Any]:
    task_id = "HumanEval/0"
    return {
        "project_name": "code_comp_t1",
        "pool_name": "budget_enc_v0",
        "table_name": "pool_budget_enc_v0_samples",
        "sample_id": sample_id,
        "key_values": {
            "data_sample_id": (
                f"human_eval/{task_id}/gt_solution@{_SOURCE_DIGEST}"
            ),
            "llm_config_id": "openai/gpt-5-nano/minimal/v1",
            "prompt_template_id": (
                "encoder/template=code_then_description/var_budget=64"
            ),
        },
        "sample_idx": 0 if linked else 1,
        "run_id": None,
        "request_json": {
            "llm_config": _llm_config(),
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        "Describe the code using at most 64 characters"
                    ),
                }
            ],
        },
        "response_json": {
            "text": _DESCRIPTION,
            "provider": "openai",
            "model": "gpt-5-nano",
            "finish_reason": "stop",
        },
        "finish_reason": "stop",
        "attempt_count": 1,
        "metadata_json": {},
        "created_at": "2026-05-10T10:00:00Z",
        "hints": _hints(task_id, encoder=True),
    }


def _enc_dec_row(*, embedded_output: str = _DESCRIPTION) -> dict[str, Any]:
    task_id = "HumanEval/0"
    reference = "encoder_pool/budget_enc_v0/linked-encoder"
    row = _direct_row("enc-decoder")
    row["project_name"] = "code_comp_t1"
    row["pool_name"] = "budget_dec_v0"
    row["table_name"] = "pool_budget_dec_v0_samples"
    row["key_values"] = {
        "source_sample_id": reference,
        "dec_llm_config_id": "openai/gpt-5-nano/minimal/v1",
        "dec_prompt_template_id": "decoder/description_then_code",
    }
    row["metadata_json"] = {
        "data_sample_id": (
            f"human_eval/{task_id}/gt_solution@{_SOURCE_DIGEST}"
        ),
        "source_kind": "encoder_sample",
        "source_pool_name": "budget_enc_v0",
        "source_sample_id": reference,
        "source_text": embedded_output,
        "enc_llm_config_id": "openai/gpt-5-nano/minimal/v1",
        "enc_prompt_template_id": (
            "encoder/template=code_then_description/var_budget=64"
        ),
    }
    return row


def _entry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    project = first["project_name"]
    pool = first["pool_name"]
    return {
        "project_name": project,
        "pool_name": pool,
        "table_name": first["table_name"],
        "file_name": f"{project}__{pool}.jsonl.gz",
        "row_count": len(rows),
        "dumped_row_count": len(rows),
        "pool_schema_json": {
            "name": pool,
            "key_columns": [
                {"name": name, "type": "text"} for name in first["key_values"]
            ],
        },
        "original_status": "stopped",
        "temporarily_started": True,
    }


def _dump(
    tmp_path: Path, pools: list[list[dict[str, Any]]]
) -> tuple[Path, SourceManifest]:
    entries = [_entry(rows) for rows in pools]
    for entry, rows in zip(entries, pools, strict=True):
        with gzip.open(
            tmp_path / entry["file_name"], "wt", encoding="utf-8"
        ) as file:
            for row in rows:
                file.write(json.dumps(row) + "\n")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "created_at": "2026-06-21T20:19:47Z",
                "output_dir": str(tmp_path),
                "pools": entries,
            }
        ),
        encoding="utf-8",
    )
    return tmp_path, read_manifest(manifest_path)


def _extract(
    tmp_path: Path,
    pools: list[list[dict[str, Any]]],
    *,
    task_adapter: HumanEvalTaskAdapter,
) -> _Sink:
    dump_directory, manifest = _dump(tmp_path, pools)
    sink = _Sink()
    adapter = HumanEvalAdapter(task_adapter)
    adapter_contract: CorpusAdapter = adapter
    adapter_contract.populate(
        dump_directory=dump_directory,
        source_manifest=manifest,
        writer=cast(CorpusWriter, sink),
    )
    return sink


def test_loads_content_addressed_snapshot_tasks(
    humaneval_task_adapter: HumanEvalTaskAdapter,
) -> None:
    task_contract: TaskAdapter = humaneval_task_adapter

    records = tuple(task_contract.records())
    resolved = task_contract.resolve(
        f"human_eval/HumanEval/0/gt_solution@{_SOURCE_DIGEST}"
    )

    assert len(records) == 164
    assert resolved is not None
    assert resolved.task_record_id == resolved.content_sha256
    assert resolved.task_id == "HumanEval/0"
    assert task_contract.resolve("human_eval/HumanEval/0") == resolved
    assert (
        task_contract.resolve(f"mbpp/HumanEval/0/gt_solution@{_SOURCE_DIGEST}")
        is None
    )
    assert (
        task_contract.resolve(
            "human_eval/HumanEval/0/gt_solution@0000000000000000"
        )
        is None
    )


def test_rejects_modified_human_eval_snapshot(tmp_path: Path) -> None:
    payload = json.loads(_HUMANEVAL_SNAPSHOT.read_text(encoding="utf-8"))
    payload["rows"][0]["canonical_solution"] = "\n    return False\n"
    modified = tmp_path / "modified-snapshot.json"
    modified.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="content hash mismatch"):
        HumanEvalTaskAdapter(modified)


def test_extracts_direct_exact_and_recovered_prompts_without_deduplicating(
    tmp_path: Path,
    humaneval_task_adapter: HumanEvalTaskAdapter,
) -> None:
    recovered = _direct_row(
        "recovered",
        request={"reason": "historical_migration", "unavailable": True},
    )
    unknown_finish = _direct_row("unknown-finish")
    unknown_finish["response_json"].pop("finish_reason")
    sink = _extract(
        tmp_path,
        [
            [
                _direct_row("direct-1"),
                _direct_row("direct-2"),
                recovered,
                unknown_finish,
            ]
        ],
        task_adapter=humaneval_task_adapter,
    )

    assert (
        len(sink.sources) == len(sink.generations) == len(sink.requests) == 4
    )
    assert len(sink.tasks) == 164
    assert not sink.encoders
    assert len({row.generation_id for row in sink.generations}) == 4
    assert len({row.decoder_output for row in sink.generations}) == 1
    exact = sink.generations[0]
    assert exact.generation_mode is GenerationMode.DIRECT
    assert exact.lifecycle_state is LifecycleState.PENDING_VALIDATION
    assert exact.prompt_fidelity is PromptFidelity.EXACT_REQUEST
    assert exact.is_partial is True
    assert sink.requests[0].source_attempt_count == 3
    recovered_generation = sink.generations[2]
    assert (
        recovered_generation.prompt_fidelity is PromptFidelity.RECOVERED_TASK
    )
    assert recovered_generation.decoder_user_prompt is not None
    assert recovered_generation.decoder_user_prompt.startswith(
        "from typing import List"
    )
    assert sink.generations[3].is_partial is True


def test_resolves_explicit_encoder_lineage_and_keeps_grains_separate(
    tmp_path: Path,
    humaneval_task_adapter: HumanEvalTaskAdapter,
) -> None:
    sink = _extract(
        tmp_path,
        [
            [
                _encoder_row("linked-encoder", linked=True),
                _encoder_row("standalone-encoder", linked=False),
            ],
            [_enc_dec_row()],
        ],
        task_adapter=humaneval_task_adapter,
    )

    assert len(sink.sources) == 3
    assert len(sink.generations) == len(sink.requests) == 1
    assert len(sink.encoders) == 1
    assert sink.encoders[0].source_record_id.endswith("standalone-encoder")
    generation = sink.generations[0]
    request = sink.requests[0]
    assert generation.generation_mode is GenerationMode.ENCODER_DECODER
    assert generation.encoder_output == _DESCRIPTION
    assert request.encoder_source_record_id is not None
    assert request.max_characters == 64
    assert request.encoder_temperature == 0.0
    assert request.decoder_max_tokens == 512


def test_rejects_missing_encoder_lineage(
    tmp_path: Path,
    humaneval_task_adapter: HumanEvalTaskAdapter,
) -> None:
    row = _direct_row("missing-lineage")
    row["metadata_json"] = {"source_kind": "encoder_sample"}

    with pytest.raises(ValueError, match="without source_sample_id"):
        _extract(tmp_path, [[row]], task_adapter=humaneval_task_adapter)


def test_direct_prompt_source_reference_is_not_encoder_lineage(
    tmp_path: Path,
    humaneval_task_adapter: HumanEvalTaskAdapter,
) -> None:
    row = _direct_row("direct-docstring")
    source_reference = (
        "docstring_only/human_eval/HumanEval/0/"
        f"gt_solution@{_SOURCE_DIGEST}/source@test"
    )
    row["key_values"]["source_sample_id"] = source_reference
    row["metadata_json"] = {
        "source_kind": "docstring_only",
        "source_sample_id": source_reference,
    }

    sink = _extract(tmp_path, [[row]], task_adapter=humaneval_task_adapter)

    assert len(sink.generations) == len(sink.requests) == 1
    assert sink.generations[0].generation_mode is GenerationMode.DIRECT
    assert sink.requests[0].encoder_source_record_id is None


def test_preserves_unqualified_task_identity_for_encoder(
    tmp_path: Path,
    humaneval_task_adapter: HumanEvalTaskAdapter,
) -> None:
    row = _encoder_row("unqualified-encoder", linked=False)
    row["key_values"]["data_sample_id"] = "human_eval/HumanEval/0"

    sink = _extract(tmp_path, [[row]], task_adapter=humaneval_task_adapter)

    assert len(sink.encoders) == 1
    assert sink.encoders[0].source_variant == "unqualified_task"


def test_preserves_audited_unresolved_encoder_candidate(
    tmp_path: Path,
    humaneval_task_adapter: HumanEvalTaskAdapter,
) -> None:
    row = _direct_row("dca9c9034e234318b9e1b5a13703bf7b")
    row["project_name"] = "code_comp_v0"
    row["pool_name"] = "tde_20260510_0345"
    row["table_name"] = "pool_tde_20260510_0345_samples"
    row["metadata_json"] = {"source_kind": "encoder_sample"}
    row["key_values"].update(
        {
            "enc_llm_config_id": "codex/gpt-5.4-mini/low",
            "enc_prompt_template_id": "size6-task3-goal2",
            "enc_sample_id": "34fe4d0b93ef49e6a30b9ce1a0044860",
        }
    )

    sink = _extract(tmp_path, [[row]], task_adapter=humaneval_task_adapter)

    assert sink.generations[0].generation_mode is (
        GenerationMode.UNRESOLVED_ENCODER
    )
    assert sink.requests[0].encoder_source_record_id is None
    assert sink.generations[0].extraction_warning == (
        "encoder_reference_unavailable"
    )


def test_rejects_conflicting_encoder_lineage(
    tmp_path: Path,
    humaneval_task_adapter: HumanEvalTaskAdapter,
) -> None:
    with pytest.raises(ValueError, match="embeds encoder output"):
        _extract(
            tmp_path,
            [
                [_encoder_row("linked-encoder", linked=True)],
                [_enc_dec_row(embedded_output="conflicting description")],
            ],
            task_adapter=humaneval_task_adapter,
        )


def test_decoder_pool_owns_stage_for_blank_seed_row(
    tmp_path: Path,
    humaneval_task_adapter: HumanEvalTaskAdapter,
) -> None:
    row = _direct_row("blank-decoder")
    row["response_json"] = None
    row["finish_reason"] = None
    row["key_values"]["status"] = "active"
    row["hints"] = _hints("HumanEval/0", encoder=True)

    sink = _extract(tmp_path, [[row]], task_adapter=humaneval_task_adapter)

    assert not sink.generations
    assert sink.sources[0].stage is Stage.DECODER
    assert sink.sources[0].status == "active"
    assert sink.sources[0].lifecycle_state is LifecycleState.SEEDED


def test_rejects_conflicting_status_locations(
    tmp_path: Path,
    humaneval_task_adapter: HumanEvalTaskAdapter,
) -> None:
    row = _direct_row("conflicting-status")
    row["key_values"]["status"] = "active"
    row["metadata_json"]["status"] = "failed"

    with pytest.raises(ValueError, match="conflicting status values"):
        _extract(tmp_path, [[row]], task_adapter=humaneval_task_adapter)
