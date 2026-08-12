from __future__ import annotations

from typing import Final

from dr_serialize import (
    Sha256Digest,
    build_identity_document,
    identity_document_hash,
)

from dr_code.evaluation.identity import EvaluationSlotIdentity

# Persisted identity payload keys are an explicit wire contract. Never derive
# them from model field names, and never build this payload by iterating.
WORK_KEY_SCHEMA: Final = "dr-code/generation-work-key-v1"
WORK_KEY_SCHEMA_VERSION: Final = 1


def derive_work_key(
    slot: EvaluationSlotIdentity,
    /,
    *,
    experiment_config_hash: str,
) -> Sha256Digest:
    """Derive the content-addressed work key for one planned generation.

    The key binds an experiment configuration to the slot's addressing
    coordinates — its task set coordinate, its sampling plan coordinate, its
    task id, and its sample index — so the same configuration addresses the
    same work under the same task set and sampling plan coordinates.
    """

    if not experiment_config_hash:
        raise ValueError("experiment_config_hash must be a non-empty hash")
    return identity_document_hash(
        build_identity_document(
            schema=WORK_KEY_SCHEMA,
            schema_version=WORK_KEY_SCHEMA_VERSION,
            payload={
                "experiment_config_hash": experiment_config_hash,
                "task_set_id": slot.task_set.task_set_id,
                "task_set_version": slot.task_set.version,
                "sampling_plan_id": slot.sampling_plan.sampling_plan_id,
                "sampling_plan_version": slot.sampling_plan.version,
                "task_id": slot.task_id,
                "sample_index": slot.sample_index,
            },
        )
    )


__all__ = ["derive_work_key"]
