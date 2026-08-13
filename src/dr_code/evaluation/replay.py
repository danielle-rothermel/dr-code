from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias

from dr_exec import ExecutionPoolConfig, Executor
from dr_store import ArtifactBundlePublication, ObjectStore
from pydantic import Field, model_validator

from dr_code.core.models import FrozenModel
from dr_code.evaluation.batch import (
    AttemptLimits,
    CandidateJobBudget,
    EvalBatchRequest,
    EvalBatchResult,
    ProjectionRequest,
    RecordPlacement,
    RunGrade,
    SampleData,
    SampleWithCandidatesData,
    ShardLimits,
    SlotData,
    WindowLimits,
    _evaluate_batch_with_replay,
)
from dr_code.evaluation.bundle import RestoredEvalAttempt
from dr_code.evaluation.id import (
    EvalAttemptId,
    EvalRuntimeId,
    EvalSample,
    EvalSampleAuxiliaryArtifact,
)
from dr_code.evaluation.plan import EvalPlan
from dr_code.evaluation.records import (
    AttemptCompleteness,
    EvaluatedSampleRecord,
    ReplayMode,
    ReplaySource,
    SampleEvalRecord,
)
from dr_code.evaluation.validation import validate_eval_attempt_graph
from dr_code.metrics.engine.engine import (
    _bind_questions,
    _plan_candidate_metrics,
)
from dr_code.preprocessing import bind_preprocessing
from dr_code.trace import (
    Absent,
    ExternalPreprocessingTraceProducer,
    PreprocessingTraceProducer,
    TextArtifact,
    WiringError,
    deserialize_trace,
)

if TYPE_CHECKING:
    from dr_code.caching import WindowedExecutionCache


class ReplayUnavailable(FrozenModel):
    kind: Literal["unavailable"] = "unavailable"
    source: ReplaySource
    reason: str = Field(min_length=1)


class ReplayReady(FrozenModel):
    kind: Literal["ready"] = "ready"
    source: ReplaySource
    request: EvalBatchRequest

    @model_validator(mode="after")
    def validate_new_attempt(self) -> ReplayReady:
        if self.source.attempt == self.request.attempt:
            raise ValueError(
                "replay requires a new evaluation attempt identity"
            )
        return self


ReplayPreflight: TypeAlias = Annotated[
    ReplayUnavailable | ReplayReady,
    Field(discriminator="kind"),
]


def preflight_replay(
    restored: RestoredEvalAttempt,
    mode: ReplayMode,
    /,
    *,
    attempt: EvalAttemptId,
    runtime: EvalRuntimeId,
    cache_namespace: str,
    run_grade: RunGrade,
    record_placement: RecordPlacement,
    projections: tuple[ProjectionRequest, ...],
    attempt_limits: AttemptLimits,
    window_limits: WindowLimits,
    shard_limits: ShardLimits,
    job_budget: CandidateJobBudget,
) -> ReplayPreflight:
    """Validate whole-attempt evidence and build one current batch request."""

    source = ReplaySource(attempt=restored.attempt.identity, mode=mode)
    validate_eval_attempt_graph(restored.attempt, restored.samples)
    if restored.attempt.completeness is not AttemptCompleteness.COMPLETE:
        return ReplayUnavailable(
            source=source,
            reason="whole-attempt replay requires a complete source attempt",
        )
    if attempt == restored.attempt.identity:
        raise ValueError("replay requires a new evaluation attempt identity")

    support_error = _support_error(restored, mode)
    if support_error is not None:
        return ReplayUnavailable(source=source, reason=support_error)

    inputs = tuple(
        _replay_input(record, restored.attempt.plan, mode=mode)
        for record in restored.samples
    )
    request = EvalBatchRequest(
        attempt=attempt,
        plan=restored.attempt.plan,
        runtime=runtime,
        cache_namespace=cache_namespace,
        run_grade=run_grade,
        inputs=inputs,
        record_placement=record_placement,
        projections=projections,
        attempt_limits=attempt_limits,
        window_limits=window_limits,
        shard_limits=shard_limits,
        job_budget=job_budget,
    )
    return ReplayReady(source=source, request=request)


async def replay_eval_attempt(
    ready: ReplayReady,
    /,
    *,
    executor: Executor,
    execution_cache: WindowedExecutionCache,
    object_store: ObjectStore | None,
    publication: ArtifactBundlePublication | None,
    pool_config: ExecutionPoolConfig,
) -> EvalBatchResult:
    """Execute a ready replay through the standalone batch path."""

    return await _evaluate_batch_with_replay(
        ready.request,
        executor=executor,
        execution_cache=execution_cache,
        object_store=object_store,
        publication=publication,
        pool_config=pool_config,
        replay=ready.source,
    )


def _support_error(
    restored: RestoredEvalAttempt, mode: ReplayMode
) -> str | None:
    try:
        _bind_questions(restored.attempt.plan.procedure.metrics)
    except (TypeError, ValueError, WiringError) as error:
        return f"recorded metrics definition is unsupported: {error}"
    if mode is ReplayMode.SAMPLES:
        try:
            bound_producer = bind_preprocessing(
                restored.attempt.plan.procedure.preprocessing
            ).producer
        except (KeyError, TypeError, ValueError) as error:
            return f"recorded preprocessing definition is unsupported: {error}"
        if not isinstance(bound_producer, PreprocessingTraceProducer):
            return "recorded preprocessing implementation is unsupported"
        expected = bound_producer.definition
        for record in restored.samples:
            producer = record.trace.producer
            if not isinstance(
                producer,
                PreprocessingTraceProducer
                | ExternalPreprocessingTraceProducer,
            ):
                return (
                    "source trace has no replayable preprocessing definition"
                )
            if producer.definition != expected:
                raise ValueError(
                    "source trace preprocessing does not match the attempt plan"
                )

    for record in restored.samples:
        try:
            _sample_from_record(record, restored.attempt.plan)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "source sample evidence is inconsistent"
            ) from error
        if mode is ReplayMode.MATERIALIZED_CANDIDATES:
            producer = record.trace.producer
            if not isinstance(
                producer,
                PreprocessingTraceProducer
                | ExternalPreprocessingTraceProducer,
            ):
                return "source trace has no materialization definition"
            if isinstance(record, EvaluatedSampleRecord) and any(
                candidate.identity.preprocessing != producer.definition
                for candidate in record.candidates
            ):
                raise ValueError(
                    "recorded candidate preprocessing does not match its trace"
                )
            replay_input = _replay_input(
                record,
                restored.attempt.plan,
                mode=ReplayMode.MATERIALIZED_CANDIDATES,
            )
            assert isinstance(replay_input.data, SampleWithCandidatesData)
            from dr_code.evaluation._batch import _frozen_candidate_trace

            trace = _frozen_candidate_trace(replay_input.data)
            try:
                plans = tuple(
                    _plan_candidate_metrics(
                        restored.attempt.plan.procedure.metrics,
                        trace,
                        candidate,
                    )
                    for candidate in replay_input.data.candidates
                )
            except (TypeError, ValueError, WiringError) as error:
                return f"recorded candidate evidence is unsupported: {error}"
            if any(not plan.suites for plan in plans):
                return "recorded candidate evidence cannot build every evaluator suite"
        elif isinstance(record, EvaluatedSampleRecord):
            try:
                plans = tuple(
                    _plan_candidate_metrics(
                        restored.attempt.plan.procedure.metrics,
                        deserialize_trace(record.trace),
                        candidate,
                    )
                    for candidate in record.candidates
                )
            except (TypeError, ValueError, WiringError) as error:
                return f"recorded sample evidence is unsupported: {error}"
            if any(not plan.suites for plan in plans):
                return "recorded sample evidence cannot build every evaluator suite"
    return None


def _replay_input(
    record: SampleEvalRecord,
    plan: EvalPlan,
    *,
    mode: ReplayMode,
) -> SlotData:
    sample = _sample_from_record(record, plan)
    if mode is ReplayMode.SAMPLES:
        return SlotData(
            slot=record.slot,
            data=SampleData(sample=sample),
        )

    producer = record.trace.producer
    assert isinstance(
        producer,
        PreprocessingTraceProducer | ExternalPreprocessingTraceProducer,
    )
    return SlotData(
        slot=record.slot,
        data=SampleWithCandidatesData(
            sample=sample,
            preprocessing=producer.definition,
            candidates=(
                record.candidates
                if isinstance(record, EvaluatedSampleRecord)
                else ()
            ),
        ),
    )


def _sample_from_record(
    record: SampleEvalRecord, plan: EvalPlan
) -> EvalSample:
    raw_input = record.trace.values.get("input")
    if not isinstance(raw_input, TextArtifact):
        raise ValueError("source trace input is not a text artifact")
    step_names = {
        step.instance_name for step in plan.procedure.preprocessing.steps
    }
    auxiliary: list[EvalSampleAuxiliaryArtifact] = []
    for key, value in record.trace.values.items():
        if key in {"input", "output"} or key in step_names:
            continue
        if isinstance(value, Absent):
            raise ValueError(
                f"source auxiliary trace value {key!r} is not an artifact"
            )
        auxiliary.append(
            EvalSampleAuxiliaryArtifact(trace_key=key, artifact=value)
        )
    return EvalSample(
        metadata=record.sample,
        raw_input=raw_input,
        auxiliary_artifacts=tuple(auxiliary),
    )


__all__ = [
    "ReplayMode",
    "ReplayPreflight",
    "ReplayReady",
    "ReplaySource",
    "ReplayUnavailable",
    "preflight_replay",
    "replay_eval_attempt",
]
