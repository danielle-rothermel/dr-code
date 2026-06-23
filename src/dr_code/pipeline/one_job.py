"""One-job generated-code evaluation API."""

from __future__ import annotations

from pydantic import Field

from dr_code.datasets.humaneval_loader import get_task
from dr_code.models.attempts import AttemptRecord
from dr_code.models.base import FrozenModel
from dr_code.models.humaneval import HumanEvalPlusTask
from dr_code.models.outcomes import ParseOutcome, TestOutcome
from dr_code.parsing.adapter import parse_attempt
from dr_code.testing.adapter import test_parsed_sample


class OneJobEvalRequest(FrozenModel):
    """Input for evaluating one generated-code output."""

    run_id: str | None = None
    task_id: str
    decoder_input: str
    raw_output: str
    decode_model: str | None = None
    encode_model: str | None = None
    decode_profile_id: str | None = None
    encode_profile_id: str | None = None
    metadata: dict[str, str | int | float | None] = Field(default_factory=dict)


class OneJobEvalResult(FrozenModel):
    """Parse and test outcomes for one generated-code output."""

    attempt: AttemptRecord
    parse_outcome: ParseOutcome
    test_outcome: TestOutcome


def evaluate_generated_code(
    request: OneJobEvalRequest,
    *,
    task: HumanEvalPlusTask | None = None,
    timeout_seconds: float | None = None,
) -> OneJobEvalResult:
    """Parse and test one generated-code output against HumanEval+."""
    resolved_task = task or get_task(request.task_id)
    attempt = AttemptRecord.from_bottleneck_output(
        run_id=request.run_id,
        task_id=resolved_task.task_id,
        decoder_input=request.decoder_input,
        raw_output=request.raw_output,
        decode_model=request.decode_model,
        encode_model=request.encode_model,
        decode_profile_id=request.decode_profile_id,
        encode_profile_id=request.encode_profile_id,
        extra=request.metadata,
    )
    parse_outcome = parse_attempt(attempt)
    test_outcome = test_parsed_sample(
        attempt,
        parse_outcome,
        task=resolved_task,
        timeout_seconds=timeout_seconds,
    )
    return OneJobEvalResult(
        attempt=attempt,
        parse_outcome=parse_outcome,
        test_outcome=test_outcome,
    )
