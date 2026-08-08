from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import polars as pl

_CORPUS_NAME = "legacy-humaneval-generation-corpus.parquet"
_REQUESTS_NAME = "legacy-humaneval-generation-requests.parquet"
_MANIFEST_NAME = "legacy-humaneval-generation-corpus.manifest.json"


def _hints(
    task_id: str,
    *,
    output_kind: str,
    output_path: str | None,
) -> dict[str, Any]:
    return {
        "human_eval_task_id": task_id,
        "human_eval_pro_task_id": None,
        "output_kind": output_kind,
        "output_json_path": output_path,
        "decoder_input_description_source": "metadata.source_text",
    }


def _llm_config(
    *,
    model: str = "gpt-5-nano",
    provider: str = "openai",
) -> dict[str, Any]:
    return {
        "effort": "na",
        "max_tokens": None,
        "mode": "api",
        "model": model,
        "provider": provider,
        "reasoning": {"kind": provider, "thinking_level": "minimal"},
        "sampling": {"temperature": 0.0, "top_p": 1.0},
    }


def _encoder_row(*, prompt_budget: int = 64) -> dict[str, Any]:
    description = "Returns True when any pair is sufficiently close."
    return {
        "project_name": "code_comp_t1",
        "pool_name": "budget_enc_v0",
        "table_name": "pool_budget_enc_v0_samples",
        "sample_id": "encoder-1",
        "key_values": {
            "data_sample_id": "human_eval/HumanEval/0/gt_solution@test",
            "llm_config_id": "openai/gpt-5-nano/minimal/v1",
            "prompt_template_id": (
                "encoder/template=code_then_description/var_budget=64"
            ),
        },
        "sample_idx": 0,
        "run_id": None,
        "request_json": {
            "llm_config": _llm_config(),
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        "def has_close_elements(numbers, threshold):\n"
                        "    return False\n\n"
                        "Provide a concise natural language description of "
                        f"the code using at most {prompt_budget} characters."
                    ),
                }
            ],
        },
        "response_json": {
            "text": description,
            "model": "gpt-5-nano",
            "provider": "openai",
        },
        "finish_reason": "stop",
        "attempt_count": 1,
        "metadata_json": {},
        "created_at": "2026-05-10T10:00:00Z",
        "hints": _hints(
            "HumanEval/0", output_kind="not_code", output_path=None
        ),
    }


def _decoder_row() -> dict[str, Any]:
    description = "Returns True when any pair is sufficiently close."
    return {
        "project_name": "code_comp_t1",
        "pool_name": "budget_dec_v0",
        "table_name": "pool_budget_dec_v0_samples",
        "sample_id": "decoder-1",
        "key_values": {
            "source_sample_id": ("encoder_pool/budget_enc_v0/encoder-1"),
            "dec_llm_config_id": "openai/gpt-5-nano/minimal/v1",
            "dec_prompt_template_id": "decoder/description_then_code",
        },
        "sample_idx": 0,
        "run_id": None,
        "request_json": {
            "llm_config": _llm_config(),
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        f"{description}\n\nWrite functional Python code."
                    ),
                }
            ],
        },
        "response_json": {
            "text": "def has_close_elements(numbers, threshold):\n    return False\n",
            "model": "gpt-5-nano",
            "provider": "openai",
        },
        "finish_reason": "stop",
        "attempt_count": 2,
        "metadata_json": {
            "data_sample_id": "human_eval/HumanEval/0/gt_solution@test",
            "enc_llm_config_id": "openai/gpt-5-nano/minimal/v1",
            "enc_prompt_template_id": (
                "encoder/template=code_then_description/var_budget=64"
            ),
            "source_kind": "encoder_sample",
            "source_pool_name": "budget_enc_v0",
            "source_sample_id": ("encoder_pool/budget_enc_v0/encoder-1"),
            "source_text": description,
        },
        "created_at": "2026-05-10T10:01:00Z",
        "hints": _hints(
            "HumanEval/0",
            output_kind="code_text",
            output_path="response_json.text",
        ),
    }


def _direct_row(
    *,
    sample_id: str,
    pool_name: str,
    source_kind: str,
    task_id: str,
    request_json: dict[str, Any],
    finish_reason: str | None = "stop",
) -> dict[str, Any]:
    return {
        "project_name": "code_comp_v0",
        "pool_name": pool_name,
        "table_name": f"pool_{pool_name}_samples",
        "sample_id": sample_id,
        "key_values": {
            "data_sample_id": f"human_eval/{task_id}/gt_solution@test",
            "dec_llm_config_id": "codex/gpt-5.4-mini/low",
            "dec_prompt_template_id": "sentinel/official_prompt",
        },
        "sample_idx": 0,
        "run_id": None,
        "request_json": request_json,
        "response_json": {
            "text": "def has_close_elements(numbers, threshold):\n    return False\n",
            "model": "gpt-5.4-mini",
            "provider": "codex",
        },
        "finish_reason": finish_reason,
        "attempt_count": 1,
        "metadata_json": {"source_kind": source_kind},
        "created_at": "2026-05-10T10:02:00Z",
        "hints": _hints(
            task_id,
            output_kind="code_text",
            output_path="response_json.text",
        ),
    }


def _unresolved_encoder_row() -> dict[str, Any]:
    row = _direct_row(
        sample_id="unresolved-encoder",
        pool_name="tde_20260510_0345",
        source_kind="encoder_sample",
        task_id="HumanEval/2",
        request_json={
            "llm_config": _llm_config(model="gpt-5.4-mini", provider="codex"),
            "prompt": [{"role": "user", "content": "Describe this code."}],
        },
    )
    row["key_values"]["dec_prompt_template_id"] = "encoder/anomalous"
    return row


def _pool_entry(
    project_name: str,
    pool_name: str,
    rows: list[dict[str, Any]],
    *,
    decoder: bool,
) -> dict[str, Any]:
    return {
        "project_name": project_name,
        "pool_name": pool_name,
        "table_name": f"pool_{pool_name}_samples",
        "file_name": f"{project_name}__{pool_name}.jsonl.gz",
        "row_count": len(rows),
        "dumped_row_count": len(rows),
        "pool_schema_json": {
            "name": pool_name,
            "key_columns": [
                {
                    "name": (
                        "dec_llm_config_id" if decoder else "llm_config_id"
                    ),
                    "type": "text",
                }
            ],
        },
    }


def _write_pool(
    dump_directory: Path,
    entry: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    with gzip.open(
        dump_directory / entry["file_name"], "wt", encoding="utf-8"
    ) as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def _dump_fixture(tmp_path: Path, *, prompt_budget: int = 64) -> Path:
    dump_directory = tmp_path / "dump"
    dump_directory.mkdir()
    exact_request = {
        "llm_config": _llm_config(model="gpt-5.4-mini", provider="codex"),
        "prompt": [
            {
                "role": "user",
                "content": "Implement the HumanEval function.",
            }
        ],
    }
    pools = [
        (
            "code_comp_t1",
            "budget_enc_v0",
            [_encoder_row(prompt_budget=prompt_budget)],
            False,
        ),
        ("code_comp_t1", "budget_dec_v0", [_decoder_row()], True),
        (
            "code_comp_v0",
            "dec_v0_orig",
            [
                _direct_row(
                    sample_id="direct-exact",
                    pool_name="dec_v0_orig",
                    source_kind="original_humaneval_prompt",
                    task_id="HumanEval/1",
                    request_json=exact_request,
                    finish_reason="length",
                )
            ],
            True,
        ),
        (
            "code_comp_v0",
            "official_decoder_t0",
            [
                _direct_row(
                    sample_id="direct-migrated",
                    pool_name="official_decoder_t0",
                    source_kind="task_prompt",
                    task_id="HumanEval/0",
                    request_json={
                        "reason": "historical_migration",
                        "unavailable": True,
                    },
                    finish_reason=None,
                )
            ],
            True,
        ),
        (
            "code_comp_v0",
            "tde_20260510_0345",
            [_unresolved_encoder_row()],
            True,
        ),
    ]
    entries: list[dict[str, Any]] = []
    for project_name, pool_name, rows, decoder in pools:
        entry = _pool_entry(project_name, pool_name, rows, decoder=decoder)
        entries.append(entry)
        _write_pool(dump_directory, entry, rows)
    (dump_directory / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "created_at": "2026-06-21T20:19:47Z",
                "output_dir": str(dump_directory),
                "pools": entries,
            }
        ),
        encoding="utf-8",
    )
    return dump_directory


def _run_builder(
    dump_directory: Path, output_directory: Path
) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).parents[2]
    return subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "build_legacy_humaneval_corpus.py"),
            str(dump_directory),
            "--snapshot",
            str(root / "tests" / "corpus" / "humanevalplus_snapshot.json"),
            "--output-dir",
            str(output_directory),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_reconstructs_canonical_corpus_and_request_sidecar(
    tmp_path: Path,
) -> None:
    dump_directory = _dump_fixture(tmp_path)
    output_directory = tmp_path / "output"

    completed = _run_builder(dump_directory, output_directory)

    assert completed.returncode == 0, completed.stderr
    assert "Validated 4 canonical rows" in completed.stdout
    corpus = pl.read_parquet(output_directory / _CORPUS_NAME)
    requests = pl.read_parquet(output_directory / _REQUESTS_NAME)
    manifest = json.loads(
        (output_directory / _MANIFEST_NAME).read_text(encoding="utf-8")
    )

    assert corpus.height == requests.height == 4
    assert corpus.get_column("sample_id").n_unique() == 4
    assert corpus.get_column("decoder_output").n_unique() == 1
    assert corpus.schema["date"] == pl.Datetime("us", "UTC")
    assert manifest["generation_mode_counts"] == {
        "direct": 2,
        "enc_dec": 1,
        "unresolved_encoder": 1,
    }
    assert manifest["budget_mode_counts"] == {
        "budget": 1,
        "no_budget": 2,
        "unresolved": 1,
    }

    enc_dec = requests.filter(pl.col("generation_mode") == "enc_dec").row(
        0, named=True
    )
    enc_dec_corpus = corpus.filter(
        pl.col("sample_id") == enc_dec["sample_id"]
    ).row(0, named=True)
    assert enc_dec["max_characters"] == 64
    assert enc_dec["encoder_temperature"] == 0.0
    assert enc_dec["decoder_temperature"] == 0.0
    assert enc_dec["encoder_source_sample_id"] == "encoder-1"
    assert enc_dec_corpus["encoder_model"] == "openai/gpt-5-nano"
    assert enc_dec_corpus["attempt_index"] == 1
    assert enc_dec_corpus["is_retry"] is True
    assert enc_dec_corpus["prompt_fidelity"] == "exact_request"

    direct = corpus.filter(pl.col("task_id") == "HumanEval/1").row(
        0, named=True
    )
    assert direct["encoder_model"] is None
    assert direct["is_partial"] is True

    migrated = corpus.join(
        requests.filter(pl.col("source_pool") == "official_decoder_t0").select(
            "sample_id"
        ),
        on="sample_id",
    ).row(0, named=True)
    assert migrated["prompt_fidelity"] == "semantic_only"
    assert migrated["decoder_user_prompt"].startswith(
        "from typing import List"
    )
    assert migrated["extraction_warning"] == "original_request_unavailable"

    unresolved = corpus.join(
        requests.filter(
            pl.col("generation_mode") == "unresolved_encoder"
        ).select("sample_id"),
        on="sample_id",
    ).row(0, named=True)
    assert unresolved["encoder_model"] is None
    assert unresolved["extraction_warning"] == "encoder_source_unresolved"


def test_rejects_budget_disagreement_without_publishing_outputs(
    tmp_path: Path,
) -> None:
    dump_directory = _dump_fixture(tmp_path, prompt_budget=32)
    output_directory = tmp_path / "output"

    completed = _run_builder(dump_directory, output_directory)

    assert completed.returncode == 1
    assert "encoder prompt budget disagrees with template" in completed.stderr
    assert not output_directory.exists()


def test_rejects_pool_file_row_count_mismatch_without_publishing_outputs(
    tmp_path: Path,
) -> None:
    dump_directory = _dump_fixture(tmp_path)
    manifest_path = dump_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pools"][0]["row_count"] += 1
    manifest["pools"][0]["dumped_row_count"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_directory = tmp_path / "output"

    completed = _run_builder(dump_directory, output_directory)

    assert completed.returncode == 1
    assert "pool dump row count mismatch" in completed.stderr
    assert not output_directory.exists()


def test_refuses_nonempty_output_directory(tmp_path: Path) -> None:
    dump_directory = _dump_fixture(tmp_path)
    output_directory = tmp_path / "output"
    output_directory.mkdir()
    (output_directory / "keep.txt").write_text("keep", encoding="utf-8")

    completed = _run_builder(dump_directory, output_directory)

    assert completed.returncode == 1
    assert (
        "refusing to write into non-empty output directory" in completed.stderr
    )
    assert (output_directory / "keep.txt").read_text(
        encoding="utf-8"
    ) == "keep"
