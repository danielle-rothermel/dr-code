from __future__ import annotations

import pytest
from _builders import evaluation_slot
from dr_serialize import build_identity_document, identity_document_hash

from dr_code.evaluation import derive_work_key
from dr_code.evaluation.work_key import (
    WORK_KEY_SCHEMA,
    WORK_KEY_SCHEMA_VERSION,
)

_CONFIG_HASH = (
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
)

# Literal payload keys pin the persisted work-key derivation; deriving them
# from model field names would hide silent drift of stored identity.
_GOLDEN_WORK_KEY_PAYLOAD = {
    "experiment_config_hash": _CONFIG_HASH,
    "task_set_id": "task-set",
    "task_set_version": "1",
    "sampling_plan_id": "sampling-plan",
    "sampling_plan_version": "1",
    "task_id": "t0",
    "sample_index": 0,
}


def test_work_key_schema_literals_are_pinned() -> None:
    assert WORK_KEY_SCHEMA == "dr-code/generation-work-key-v1"
    assert WORK_KEY_SCHEMA_VERSION == 1


def test_work_key_hashes_the_golden_derivation_payload() -> None:
    expected = identity_document_hash(
        build_identity_document(
            schema="dr-code/generation-work-key-v1",
            schema_version=1,
            payload=_GOLDEN_WORK_KEY_PAYLOAD,
        )
    )
    assert (
        derive_work_key(evaluation_slot(), experiment_config_hash=_CONFIG_HASH)
        == expected
    )
    assert str(expected) == (
        "a125dce847b31ca610f049aac229197a24c95ff74206dab38ec89366e02a2c10"
    )


def test_work_key_is_deterministic_for_the_same_slot_and_config() -> None:
    first = derive_work_key(
        evaluation_slot(), experiment_config_hash=_CONFIG_HASH
    )
    second = derive_work_key(
        evaluation_slot(), experiment_config_hash=_CONFIG_HASH
    )
    assert first == second


def test_work_key_separates_sample_indices_within_one_task() -> None:
    first = derive_work_key(
        evaluation_slot(sample_index=0), experiment_config_hash=_CONFIG_HASH
    )
    second = derive_work_key(
        evaluation_slot(sample_index=1), experiment_config_hash=_CONFIG_HASH
    )
    assert first != second


def test_work_key_separates_tasks_and_experiment_configurations() -> None:
    base = derive_work_key(
        evaluation_slot(), experiment_config_hash=_CONFIG_HASH
    )
    other_task = derive_work_key(
        evaluation_slot(task_id="t2"), experiment_config_hash=_CONFIG_HASH
    )
    other_config = derive_work_key(
        evaluation_slot(), experiment_config_hash="f" * 64
    )
    assert len({base, other_task, other_config}) == 3


def test_work_key_rejects_an_empty_experiment_config_hash() -> None:
    with pytest.raises(ValueError, match="experiment_config_hash"):
        derive_work_key(evaluation_slot(), experiment_config_hash="")
