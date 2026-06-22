"""JobEnvelope helpers for AttemptRecord seeding."""

from __future__ import annotations

from dr_queues import JobEnvelope

from dr_code.models.attempts import AttemptRecord
from dr_code.pipeline.definition import PIPELINE_ID

_ATTEMPT_KEY = "attempt"


def stamp_run_id(
    records: list[AttemptRecord], run_id: str
) -> list[AttemptRecord]:
    """Return copies of records with run_id set."""
    return [record.model_copy(update={"run_id": run_id}) for record in records]


def attempt_to_payload(record: AttemptRecord) -> dict[str, object]:
    """Serialize an AttemptRecord into a job payload fragment."""
    return {"attempt": record.model_dump(mode="json")}


def attempt_from_job(job: JobEnvelope) -> AttemptRecord:
    """Deserialize AttemptRecord from a job payload."""
    raw = job.payload.get(_ATTEMPT_KEY)
    if raw is None:
        msg = "Job payload missing 'attempt' key"
        raise ValueError(msg)
    return AttemptRecord.model_validate(raw)


def build_seed_jobs(
    records: list[AttemptRecord],
    *,
    run_id: str,
    pipeline_id: str = PIPELINE_ID,
) -> list[JobEnvelope]:
    """Build one JobEnvelope per AttemptRecord for parse-stage seeding."""
    jobs: list[JobEnvelope] = []
    for index, record in enumerate(records):
        jobs.append(
            JobEnvelope(
                run_id=run_id,
                lane="default",
                repeat=index,
                step_index=0,
                pipeline_id=pipeline_id,
                payload=attempt_to_payload(record),
            ),
        )
    return jobs
