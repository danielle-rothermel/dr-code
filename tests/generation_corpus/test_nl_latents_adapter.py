from __future__ import annotations

import json
from pathlib import Path

import pytest

from dr_code.generation_corpus.adapters.nl_latents import (
    adapt_nl_latents_row,
    normalize_failure_category,
)
from dr_code.generation_corpus.models import (
    DumpedPoolRow,
    LifecycleState,
)
from dr_code.generation_corpus.pool_dump import canonical_json, content_sha256
from dr_code.generation_corpus.tasks.nl_latents import (
    NlLatentsFamily,
    NlLatentsLanguage,
    NlLatentsSplit,
    NlLatentsTaskAdapter,
    NlLatentsTaskCoordinate,
    NlLatentsTaskRoot,
)

_TASK_ID = "stateful_0123456789abcdef"
_CODE = "def f(xs: list[int]) -> int:\n    return sum(xs)"
_QUERIES: list[dict[str, object]] = [
    {"input": [1, 2], "output": 3, "tag": "coverage"}
]
_ENCODER_PROMPT = f"Describe this program under 100 characters.\n{_CODE}"
_DESCRIPTION = "Return the sum of the integers in xs."
_DECODER_SYSTEM = "Output only a single function and nothing else."
_DECODER_TASK = (
    "Implement exactly this function signature:\n\n"
    "def f(xs: list[int]) -> int:\n\n"
    "Use the standard interpretation of the task described below:\n\n"
    f"{_DESCRIPTION}"
)
_DECODED_CODE = "def f(xs: list[int]) -> int:\n    return sum(xs)"


def _task(
    *,
    task_id: str = _TASK_ID,
    queries: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "family": "stateful",
        "difficulty": 3,
        "code": _CODE,
        "queries": _QUERIES if queries is None else queries,
        "description": "Sum a list.",
        "spec": {"kind": "sum"},
        "axes": {},
        "trace": {"family": "stateful", "steps": []},
    }


def _write_task(
    archive: Path,
    root: str,
    task: dict[str, object],
    *,
    language: str = "python",
) -> None:
    path = archive / root / "stateful" / "d3" / language / "train.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(task) + "\n", encoding="utf-8")


def _archive(tmp_path: Path) -> Path:
    archive = tmp_path / "archive"
    archive.mkdir()
    _write_task(archive, "data/tasks", _task())
    return archive


def _validation(queries: list[dict[str, object]] | None = None) -> str:
    selected = _QUERIES if queries is None else queries
    return canonical_json(
        {
            "test_case_results": [
                {
                    "input_value": query["input"],
                    "expected_output": query["output"],
                    "actual_output": query["output"],
                    "passed": True,
                }
                for query in selected
            ],
            "test_pass_rate": 1.0,
        }
    )


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "actual_chars": len(_DESCRIPTION),
        "budget_ok": True,
        "dec_system": _DECODER_SYSTEM,
        "dec_task": _DECODER_TASK,
        "decoded_code": _DECODED_CODE,
        "description": _DESCRIPTION,
        "detail": "",
        "enc_prompt": _ENCODER_PROMPT,
        "failure_category": None,
        "metadata_json": "{}",
        "model_provenance_source": "run_pool_generation",
        "passed": True,
        "validation_json": _validation(),
    }
    payload.update(overrides)
    return payload


def _row(
    *,
    status: str = "active",
    payload: dict[str, object] | None = None,
) -> DumpedPoolRow:
    selected = _payload() if payload is None else payload
    pending = status == "pending"
    return DumpedPoolRow.model_validate(
        {
            "attempt_count": 0,
            "created_at": "2026-05-10T05:53:42Z",
            "finish_reason": None,
            "hints": {
                "decoder_input_description_source": "missing",
                "human_eval_pro_task_id": None,
                "human_eval_task_id": None,
                "output_json_path": (
                    None if pending else "response_json.decoded_code"
                ),
                "output_kind": "not_code" if pending else "decoded_code",
            },
            "key_values": {
                "budget": "100",
                "call_id": "__legacy_null_key__:call_id",
                "config_id": "config",
                "dec_model": "openrouter:openai/gpt-5-nano",
                "dec_reasoning_effort": (
                    "provider_default" if pending else "minimal"
                ),
                "difficulty": "3",
                "enc_model": "openrouter:openai/gpt-5-nano",
                "enc_reasoning_effort": (
                    "provider_default" if pending else "minimal"
                ),
                "family": "stateful",
                "language": "python",
                "split": "train",
                "status": status,
                "task_data_version": "tasks_v2_resampled_2026_02_11",
                "task_id": _TASK_ID,
            },
            "metadata_json": {},
            "pool_name": "nl_latents",
            "project_name": "nl_latents",
            "request_json": (
                selected
                if pending
                else {"reason": "historical_migration", "unavailable": True}
            ),
            "response_json": None if pending else selected,
            "run_id": None if pending else "historical_run",
            "sample_id": "sample-pending" if pending else "sample-active",
            "sample_idx": 0,
            "table_name": "pool_nl_latents_samples",
        }
    )


def _adapt(row: DumpedPoolRow, archive: Path):
    return adapt_nl_latents_row(
        row,
        source_file="nl_latents__nl_latents.jsonl.gz",
        line_number=1,
        task_adapter=NlLatentsTaskAdapter(archive),
    )


def test_active_and_pending_select_closed_payload_branches_and_exact_prompts(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path)
    active = _adapt(_row(), archive)
    pending_payload = _payload(
        detail="pending docker validation",
        passed=None,
        failure_category=None,
        validation_json=None,
    )
    pending = _adapt(_row(status="pending", payload=pending_payload), archive)

    assert active.generation is not None
    assert active.request is not None
    assert active.generation.content_sha256 == content_sha256(
        None,
        _ENCODER_PROMPT,
        _DESCRIPTION,
        _DECODER_SYSTEM,
        _DECODER_TASK,
        _DECODED_CODE,
    )
    assert active.generation.date_kind == "migration_import_created_at"
    assert active.request.source_kind == "response_json"
    assert active.old_evaluation_ready is True

    assert pending.generation is not None
    assert pending.request is not None
    assert (
        pending.generation.lifecycle_state is LifecycleState.PENDING_VALIDATION
    )
    assert pending.request.source_kind == "request_json"
    assert pending.request.response_json == "null"
    assert pending.old_evaluation_ready is False


def test_incomplete_rows_remain_at_raw_or_encoder_only_grains(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path)
    encoder_payload = _payload(
        decoded_code=None,
        validation_json=None,
        passed=False,
    )
    encoder_only = _adapt(_row(payload=encoder_payload), archive)
    pre_encoder_payload = _payload(
        actual_chars=None,
        budget_ok=None,
        dec_system=None,
        dec_task=None,
        decoded_code=None,
        description=None,
        validation_json=None,
        passed=False,
    )
    pre_encoder = _adapt(_row(payload=pre_encoder_payload), archive)

    assert encoder_only.generation is None
    assert encoder_only.request is None
    assert encoder_only.encoder_artifact is not None
    assert encoder_only.encoder_artifact.content_sha256 == content_sha256(
        None,
        _ENCODER_PROMPT,
        _DESCRIPTION,
    )
    assert (
        encoder_only.source_record.lifecycle_state
        is LifecycleState.ENCODER_ONLY
    )
    assert pre_encoder.generation is None
    assert pre_encoder.encoder_artifact is None
    assert pre_encoder.source_record.lifecycle_state is (
        LifecycleState.PRE_ENCODER_FAILURE
    )


def test_task_resolution_prefers_primary_and_seed_fallback_requires_workshop(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    seed_task = _task()
    _write_task(archive, "data/tasks_seed41_u5", seed_task)
    _write_task(
        archive,
        "data/tasks_workshop_core_f3_d2to5_seed41_u5",
        seed_task,
    )
    coordinate = NlLatentsTaskCoordinate(
        task_data_version="tasks_v2_resampled_2026_02_11",
        family=NlLatentsFamily.STATEFUL,
        difficulty=3,
        split=NlLatentsSplit.TRAIN,
        language=NlLatentsLanguage.PYTHON,
        task_id=_TASK_ID,
    )
    adapter = NlLatentsTaskAdapter(archive)

    mapping = adapter.resolve_coordinate(coordinate)

    assert mapping is not None
    assert mapping.task_root is NlLatentsTaskRoot.SEED41
    assert mapping.relative_jsonl_path == "stateful/d3/python/train.jsonl"
    assert str(archive) not in canonical_json(mapping.model_dump(mode="json"))
    assert len(mapping.task_sha256) == 64
    assert len(mapping.code_sha256) == 64
    assert len(mapping.query_sha256) == 64
    assert mapping.task_record_id == mapping.to_task_record().content_sha256
    assert adapter.resolve(coordinate.serialize()) is not None


def test_seed_fallback_fails_when_workshop_copy_differs(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    _write_task(archive, "data/tasks_seed41_u5", _task())
    _write_task(
        archive,
        "data/tasks_workshop_core_f3_d2to5_seed41_u5",
        _task(queries=[{"input": [9], "output": 9, "tag": "other"}]),
    )
    coordinate = NlLatentsTaskCoordinate(
        task_data_version="tasks_v2_resampled_2026_02_11",
        family=NlLatentsFamily.STATEFUL,
        difficulty=3,
        split=NlLatentsSplit.TRAIN,
        language=NlLatentsLanguage.PYTHON,
        task_id=_TASK_ID,
    )

    with pytest.raises(ValueError, match="differs from its workshop copy"):
        NlLatentsTaskAdapter(archive).resolve_coordinate(coordinate)


def test_known_ambiguity_explicitly_selects_primary_using_stored_cases(
    tmp_path: Path,
) -> None:
    task_id = "stateful_225d71b320455b55"
    primary_queries: list[dict[str, object]] = [
        {"input": [1], "output": 1, "tag": "primary"}
    ]
    seed_queries: list[dict[str, object]] = [
        {"input": [2], "output": 2, "tag": "seed"}
    ]
    archive = tmp_path / "archive"
    archive.mkdir()
    _write_task(
        archive,
        "data/tasks",
        _task(task_id=task_id, queries=primary_queries),
    )
    seed_task = _task(task_id=task_id, queries=seed_queries)
    _write_task(archive, "data/tasks_seed41_u5", seed_task)
    _write_task(
        archive,
        "data/tasks_workshop_core_f3_d2to5_seed41_u5",
        seed_task,
    )
    coordinate = NlLatentsTaskCoordinate(
        task_data_version="tasks_v2_resampled_2026_02_11",
        family=NlLatentsFamily.STATEFUL,
        difficulty=3,
        split=NlLatentsSplit.TRAIN,
        language=NlLatentsLanguage.PYTHON,
        task_id=task_id,
    )
    adapter = NlLatentsTaskAdapter(archive)

    mapping = adapter.resolve_coordinate(
        coordinate,
        validation_json=_validation(primary_queries),
    )
    adapter.assert_ambiguous_resolutions_validated()

    assert mapping is not None
    assert mapping.task_root is NlLatentsTaskRoot.PRIMARY
    with pytest.raises(ValueError, match="do not match the selected primary"):
        adapter.resolve_coordinate(
            coordinate,
            validation_json=_validation(seed_queries),
        )


def test_smoke_task_is_fixed_and_explicitly_not_execution_ready(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    coordinate = NlLatentsTaskCoordinate.model_validate(
        {
            "task_data_version": "tasks_v1_pre_resample_2026_02_10",
            "family": "smoke",
            "difficulty": 3,
            "split": "train",
            "language": "Python",
            "task_id": "check_02_add_one",
        }
    )

    mapping = NlLatentsTaskAdapter(archive).resolve_coordinate(coordinate)

    assert mapping is not None
    assert mapping.task_root is NlLatentsTaskRoot.EMBEDDED_SMOKE
    assert mapping.relative_jsonl_path is None
    assert mapping.code == "def f(x):\n    return x + 1"
    task_payload = json.loads(mapping.to_task_record().task_json)
    assert task_payload["execution_ready"] is False
    assert task_payload["entry_point"] == "f"


def test_encoder_prompt_must_end_with_exact_archived_code(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path)
    payload = _payload(enc_prompt=f"Describe this.\n{_CODE}\n")

    with pytest.raises(ValueError, match="does not end with exact archived"):
        _adapt(_row(payload=payload), archive)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"actual_chars": len(_DESCRIPTION) + 1}, "actual_chars"),
        (
            {"actual_chars": 101, "description": "x" * 101, "budget_ok": True},
            "budget_ok",
        ),
    ],
)
def test_character_budget_fails_closed(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    archive = _archive(tmp_path)

    with pytest.raises(ValueError, match=message):
        _adapt(_row(payload=_payload(**overrides)), archive)


def test_failure_category_aliases_are_closed_and_normalized() -> None:
    assert (
        normalize_failure_category("FailureCategory.TEST_FAIL") == "test_fail"
    )
    assert normalize_failure_category("runtime_error") == "runtime_error"
    with pytest.raises(
        ValueError, match="unknown NL Latents failure category"
    ):
        normalize_failure_category("timeout")
