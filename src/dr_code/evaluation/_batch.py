from __future__ import annotations

import asyncio
import hashlib
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from typing import Protocol, TypeAlias, TypeVar
from uuid import uuid5

from dr_exec import (
    CancelToken,
    CompletedExecution,
    ExecutionJob,
    ExecutionPool,
    ExecutionPoolConfig,
    ExecutionSubmission,
    Executor,
    FixedPoolCapacity,
    JobId,
)
from dr_serialize import Sha256Digest, canonical_json_bytes

from dr_code.caching import WindowedExecutionCache
from dr_code.caching.execution_cache import CachedExecutionObservation
from dr_code.caching.preprocess_batch import preprocess_batch
from dr_code.evaluation.aggregation import (
    AggregationInput,
    AggregationOk,
    AggregationResult,
    AggregationSlot,
    aggregate,
)
from dr_code.evaluation.batch import (
    EvaluationBatchRequest,
    EvaluationInput,
    FrozenCandidateEvaluationInput,
    ProjectionKind,
    SampleEvaluationInput,
)
from dr_code.evaluation.execution import (
    build_candidate_execution_job,
    candidate_execution_cache_key,
    executed_candidate_record,
    reused_candidate_record,
)
from dr_code.evaluation.identity import (
    EvaluationCandidateIdentity,
    MaterializedEvaluationCandidate,
)
from dr_code.evaluation.records import (
    AttemptCompleteness,
    AttemptLimitExhaustion,
    AttemptLimitKind,
    AttemptValidity,
    CandidateExecutionOutcome,
    CandidateExecutionRecord,
    EvaluationMemberRecord,
    EvaluatedSampleRecord,
    ExecutorExecutionFailure,
    HarnessExecutionFailure,
    NoCandidatesSampleRecord,
    PreprocessingAbsentSampleRecord,
    SampleEvaluationRecord,
)
from dr_code.evaluation.references import (
    EvidenceReference,
    StoredRecordReference,
)
from dr_code.evaluation.score import EvaluationCoordinate, Score
from dr_code.humaneval.job import HumanEvalCandidateJobRequest
from dr_code.metrics import (
    MeasuredRecord,
    MetricValue,
    MetricValueCoordinate,
    OperatorFailureRecord,
)
from dr_code.metrics.engine.engine import (
    _CandidateMetricPlan,
    _plan_candidate_metrics,
)
from dr_code.preprocessing import BoundPreprocessingRunner, bind_preprocessing
from dr_code.trace import (
    OUTPUT_KEY,
    Absent,
    CodeArtifact,
    CodeCandidateSetArtifact,
    ExternalPreprocessingTraceProducer,
    InspectedCodeCandidateSetArtifact,
    Trace,
    PreprocessingTraceProducer,
    serialize_trace,
)


@dataclass(frozen=True, slots=True)
class _AggregationResultProjectionDraft:
    result: AggregationResult


@dataclass(frozen=True, slots=True)
class _ScoreProjectionDraft:
    score: Score


@dataclass(frozen=True, slots=True)
class _PendingCacheObservation:
    request_key: str
    outcome: CandidateExecutionOutcome


@dataclass(frozen=True, slots=True)
class _EvaluationBatchAssembly:
    members: tuple[EvaluationMemberRecord, ...]
    completeness: AttemptCompleteness
    validity: AttemptValidity
    limit_exhaustion: AttemptLimitExhaustion | None
    aggregation: _AggregationResultProjectionDraft | None
    score: _ScoreProjectionDraft | None


class _RecordPlacementSink(Protocol):
    async def place(
        self, record: SampleEvaluationRecord, /
    ) -> EvidenceReference: ...

    async def finish(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _PreparedInput:
    input: EvaluationInput
    trace: Trace
    absence: Absent | None
    candidates: tuple[MaterializedEvaluationCandidate, ...]
    metric_plans: tuple[_CandidateMetricPlan, ...]


@dataclass(frozen=True, slots=True)
class _CandidateWork:
    index: int
    request: HumanEvalCandidateJobRequest
    request_key: str
    job_id: JobId


@dataclass(frozen=True, slots=True)
class _ExecutionWindow:
    records: tuple[CandidateExecutionRecord, ...]
    pending: tuple[_PendingCacheObservation, ...]
    admitted_jobs: int
    exhaustion: AttemptLimitExhaustion | None


@dataclass(frozen=True, slots=True)
class _GlobalCandidateWork:
    sample_index: int
    candidate_index: int
    work: _CandidateWork


@dataclass(frozen=True, slots=True)
class _GlobalExecutionResult:
    by_sample: dict[int, _ExecutionWindow]
    exhaustion: AttemptLimitExhaustion | None


_RunWindow: TypeAlias = Callable[
    [Sequence[_CandidateWork]],
    Awaitable[dict[int, CompletedExecution]],
]
_WindowValue = TypeVar("_WindowValue")


async def _evaluate_batch_assembly(
    request: EvaluationBatchRequest,
    /,
    *,
    executor: Executor,
    execution_cache: WindowedExecutionCache,
    pool_config: ExecutionPoolConfig,
    placement_sink: _RecordPlacementSink,
) -> _EvaluationBatchAssembly:
    async with ExecutionPool(executor=executor, config=pool_config) as pool:

        async def run_window(
            work: Sequence[_CandidateWork],
        ) -> dict[int, CompletedExecution]:
            async def submissions() -> AsyncIterator[
                ExecutionSubmission[_CandidateWork]
            ]:
                for item in work:
                    yield ExecutionSubmission(
                        job=build_candidate_execution_job(
                            item.job_id,
                            item.request,
                            request.job_budget,
                        ),
                        context=item,
                    )

            completed: dict[int, CompletedExecution] = {}
            async for completion in pool.run_stream(submissions()):
                completed[completion.context.index] = (
                    completion.completed_execution
                )
            return completed

        return await _assemble(
            request,
            execution_cache=execution_cache,
            run_window=run_window,
            placement_sink=placement_sink,
            pool_config=pool_config,
        )


async def _evaluate_durable_partition_assembly(
    request: EvaluationBatchRequest,
    /,
    *,
    executor: Executor,
    execution_cache: WindowedExecutionCache,
    placement_sink: _RecordPlacementSink,
) -> _EvaluationBatchAssembly:
    async def run_window(
        work: Sequence[_CandidateWork],
    ) -> dict[int, CompletedExecution]:
        completed: dict[int, CompletedExecution] = {}
        for item in work:
            job = build_candidate_execution_job(
                item.job_id,
                item.request,
                request.job_budget,
            )
            completed[item.index] = await _run_durable_job(executor, job)
        return completed

    return await _assemble(
        request,
        execution_cache=execution_cache,
        run_window=run_window,
        placement_sink=placement_sink,
        pool_config=ExecutionPoolConfig(),
    )


async def _run_durable_job(
    executor: Executor,
    job: ExecutionJob,
) -> CompletedExecution:
    cancellation = CancelToken()
    running = asyncio.create_task(
        asyncio.to_thread(executor.run, job, cancellation=cancellation)
    )
    try:
        return await asyncio.shield(running)
    except asyncio.CancelledError as cancelled:
        cancellation.cancel()
        while not running.done():
            try:
                await asyncio.shield(running)
            except asyncio.CancelledError:
                continue
            except BaseException as cleanup_error:
                raise cancelled from cleanup_error
        try:
            running.result()
        except BaseException as cleanup_error:
            raise cancelled from cleanup_error
        raise cancelled


def _preprocessing_worker_count(config: ExecutionPoolConfig) -> int | None:
    """Give preprocessing the width the caller asked evaluation to use."""

    capacity = config.capacity
    if isinstance(capacity, FixedPoolCapacity):
        return capacity.max_active_jobs
    return None


async def _assemble(
    request: EvaluationBatchRequest,
    *,
    execution_cache: WindowedExecutionCache,
    run_window: _RunWindow,
    placement_sink: _RecordPlacementSink,
    pool_config: ExecutionPoolConfig,
) -> _EvaluationBatchAssembly:
    runner = (
        bind_preprocessing(request.plan.procedure.preprocessing)
        if any(
            isinstance(item, SampleEvaluationInput) for item in request.inputs
        )
        else None
    )
    traces_by_text: dict[str, Trace] = {}
    if runner is not None:
        sample_texts = [
            item.sample.raw_input.text
            for item in request.inputs
            if isinstance(item, SampleEvaluationInput)
        ]
        if sample_texts:
            traces_by_text = await preprocess_batch(
                sample_texts,
                definition=request.plan.procedure.preprocessing,
                worker_count=_preprocessing_worker_count(pool_config),
            )
    prepared_inputs, prepare_exhaustion = _prepare_inputs(
        request,
        runner=runner,
        traces_by_text=traces_by_text,
    )
    execution_result = _GlobalExecutionResult(by_sample={}, exhaustion=None)
    if prepare_exhaustion is None:
        execution_result = await _execute_batch_candidates_globally(
            request,
            prepared_inputs,
            execution_cache=execution_cache,
            run_window=run_window,
        )
    exhaustion = prepare_exhaustion or execution_result.exhaustion
    return await _place_prepared_inputs(
        request,
        prepared_inputs,
        execution_result.by_sample,
        exhaustion=exhaustion,
        execution_cache=execution_cache,
        placement_sink=placement_sink,
    )


def _prepare_inputs(
    request: EvaluationBatchRequest,
    *,
    runner: BoundPreprocessingRunner | None,
    traces_by_text: Mapping[str, Trace] | None = None,
) -> tuple[tuple[_PreparedInput, ...], AttemptLimitExhaustion | None]:
    prepared_inputs: list[_PreparedInput] = []
    materialized_count = 0
    projected_rows = _terminal_projection_reserve(request)
    exhaustion: AttemptLimitExhaustion | None = None
    precomputed_traces = traces_by_text or {}

    for input_window in _windows(
        request.inputs,
        request.window_limits.max_preprocessing_slots,
    ):
        prepared_window = tuple(
            _prepare_input(
                request,
                item,
                runner=runner,
                traces_by_text=precomputed_traces,
            )
            for item in input_window
        )
        for prepared in prepared_window:
            discovered = len(prepared.candidates)
            observed_materialized = materialized_count + discovered
            if (
                observed_materialized
                > request.attempt_limits.max_materialized_candidates
            ):
                exhaustion = AttemptLimitExhaustion(
                    limit=AttemptLimitKind.MATERIALIZED_CANDIDATES,
                    configured=request.attempt_limits.max_materialized_candidates,
                    observed=observed_materialized,
                )
                break

            contribution = _projection_row_contribution(request, prepared)
            if (
                projected_rows + contribution
                > request.attempt_limits.max_projection_rows
            ):
                exhaustion = AttemptLimitExhaustion(
                    limit=AttemptLimitKind.PROJECTION_ROWS,
                    configured=request.attempt_limits.max_projection_rows,
                    observed=projected_rows + contribution,
                )
                break

            materialized_count = observed_materialized
            projected_rows += contribution
            prepared_inputs.append(prepared)
        if exhaustion is not None:
            break

    return tuple(prepared_inputs), exhaustion


async def _place_prepared_inputs(
    request: EvaluationBatchRequest,
    prepared_inputs: Sequence[_PreparedInput],
    execution_by_sample: Mapping[int, _ExecutionWindow],
    *,
    exhaustion: AttemptLimitExhaustion | None,
    execution_cache: WindowedExecutionCache,
    placement_sink: _RecordPlacementSink,
) -> _EvaluationBatchAssembly:
    members: list[EvaluationMemberRecord] = []
    aggregation_slots: list[AggregationSlot] = []
    score_values: list[MetricValue] = []
    invalid = False
    retained_bytes = 0

    for sample_index, prepared in enumerate(prepared_inputs):
        if prepared.absence is not None:
            record: SampleEvaluationRecord = PreprocessingAbsentSampleRecord(
                slot=prepared.input.slot,
                sample=prepared.input.sample.metadata,
                trace=serialize_trace(prepared.trace),
                absence=prepared.absence,
            )
            execution = _ExecutionWindow((), (), 0, None)
        elif not prepared.candidates:
            record = NoCandidatesSampleRecord(
                slot=prepared.input.slot,
                sample=prepared.input.sample.metadata,
                trace=serialize_trace(prepared.trace),
            )
            execution = _ExecutionWindow((), (), 0, None)
        else:
            if (
                exhaustion is not None
                and sample_index not in execution_by_sample
            ):
                break
            execution = execution_by_sample[sample_index]
            if execution.exhaustion is not None:
                exhaustion = execution.exhaustion
                break
            metric_records = tuple(
                metric
                for plan, candidate_record in zip(
                    prepared.metric_plans,
                    execution.records,
                    strict=True,
                )
                for metric in plan.records(candidate_record.outcome)
            )
            record = EvaluatedSampleRecord(
                slot=prepared.input.slot,
                sample=prepared.input.sample.metadata,
                trace=serialize_trace(prepared.trace),
                candidates=prepared.candidates,
                executions=execution.records,
                metrics=metric_records,
            )

        encoded_size = len(
            canonical_json_bytes(record.model_dump(mode="json"))
        )
        observed_bytes = retained_bytes + encoded_size
        if observed_bytes > request.attempt_limits.max_retained_evidence_bytes:
            exhaustion = AttemptLimitExhaustion(
                limit=AttemptLimitKind.RETAINED_EVIDENCE_BYTES,
                configured=request.attempt_limits.max_retained_evidence_bytes,
                observed=observed_bytes,
            )
            break

        reference = await placement_sink.place(record)
        members.append(
            EvaluationMemberRecord(
                slot=record.slot,
                sample=record.sample.identity,
                record=reference,
            )
        )
        if isinstance(reference, StoredRecordReference):
            for item in execution.pending:
                try:
                    await execution_cache.put(
                        item.request_key,
                        CachedExecutionObservation(
                            source_record=reference,
                            outcome=item.outcome,
                        ),
                    )
                finally:
                    execution_cache.discard(item.request_key)
        retained_bytes = observed_bytes
        invalid = invalid or _record_is_invalid(record)
        _extend_aggregate_state(
            request,
            record,
            aggregation_slots=aggregation_slots,
            score_values=score_values,
        )
        if exhaustion is not None:
            break

    await placement_sink.finish()
    members.extend(
        EvaluationMemberRecord(
            slot=item.slot,
            sample=item.sample.metadata.identity,
            record=None,
        )
        for item in request.inputs[len(members) :]
    )
    completeness = (
        AttemptCompleteness.COMPLETE
        if all(member.record is not None for member in members)
        else AttemptCompleteness.PARTIAL
    )
    validity = (
        AttemptValidity.INVALID
        if completeness is AttemptCompleteness.PARTIAL or invalid
        else AttemptValidity.VALID
    )
    aggregation, score = _terminal_projection_drafts(
        request,
        aggregation_slots=aggregation_slots,
        score_values=score_values,
    )
    return _EvaluationBatchAssembly(
        members=tuple(members),
        completeness=completeness,
        validity=validity,
        limit_exhaustion=exhaustion,
        aggregation=aggregation,
        score=score,
    )


def _prepare_input(
    request: EvaluationBatchRequest,
    item: EvaluationInput,
    *,
    runner: BoundPreprocessingRunner | None,
    traces_by_text: Mapping[str, Trace] | None = None,
) -> _PreparedInput:
    if isinstance(item, FrozenCandidateEvaluationInput):
        trace = _frozen_candidate_trace(item)
        candidates = item.candidates
        absence = None
    else:
        assert runner is not None
        text = item.sample.raw_input.text
        precomputed = (traces_by_text or {}).get(text)
        raw_trace = (
            precomputed
            if precomputed is not None
            else runner.run(item.sample.raw_input)
        )
        trace = Trace(
            values={
                **dict(raw_trace.values),
                **{
                    auxiliary.trace_key: auxiliary.artifact
                    for auxiliary in item.sample.auxiliary_artifacts
                },
            },
            producer=raw_trace.producer,
            step_facts=raw_trace.step_facts,
        )
        output = trace.value(OUTPUT_KEY)
        absence = output if isinstance(output, Absent) else None
        candidates = (
            () if absence is not None else _materialize_candidates(item, trace)
        )
    plans = tuple(
        _plan_candidate_metrics(
            request.plan.procedure.metrics,
            trace,
            candidate,
        )
        for candidate in candidates
    )
    if any(not plan.suites for plan in plans):
        raise ValueError(
            "every materialized candidate requires at least one valid code_test suite"
        )
    return _PreparedInput(
        input=item,
        trace=trace,
        absence=absence,
        candidates=candidates,
        metric_plans=plans,
    )


def _frozen_candidate_trace(item: FrozenCandidateEvaluationInput) -> Trace:
    return Trace(
        values={
            "input": item.sample.raw_input,
            "output": (
                item.candidates[0].source
                if item.candidates
                else item.sample.raw_input
            ),
            **{
                auxiliary.trace_key: auxiliary.artifact
                for auxiliary in item.sample.auxiliary_artifacts
            },
        },
        producer=ExternalPreprocessingTraceProducer(
            definition=item.preprocessing
        ),
    )


def _materialize_candidates(
    item: EvaluationInput,
    trace: Trace,
) -> tuple[MaterializedEvaluationCandidate, ...]:
    output = trace.value(OUTPUT_KEY)
    if isinstance(output, CodeArtifact):
        sources = (output,)
    elif isinstance(output, CodeCandidateSetArtifact):
        sources = tuple(
            CodeArtifact(source=candidate.source)
            for candidate in output.candidates
        )
    elif isinstance(output, InspectedCodeCandidateSetArtifact):
        sources = tuple(
            CodeArtifact(source=candidate.candidate.source)
            for candidate in output.candidates
        )
    else:
        return ()
    producer = trace.producer
    if not isinstance(
        producer,
        PreprocessingTraceProducer | ExternalPreprocessingTraceProducer,
    ):
        raise ValueError(
            "candidate materialization requires a preprocessing trace producer"
        )
    preprocessing = producer.definition
    return tuple(
        MaterializedEvaluationCandidate(
            identity=EvaluationCandidateIdentity(
                sample=item.sample.metadata.identity,
                preprocessing=preprocessing,
                candidate_ordinal=ordinal,
            ),
            source=source,
            source_sha256=Sha256Digest(
                hashlib.sha256(source.source.encode("utf-8")).hexdigest()
            ),
        )
        for ordinal, source in enumerate(sources)
    )


async def _execute_batch_candidates_globally(
    request: EvaluationBatchRequest,
    prepared_inputs: Sequence[_PreparedInput],
    *,
    execution_cache: WindowedExecutionCache,
    run_window: _RunWindow,
) -> _GlobalExecutionResult:
    flat_work: list[_GlobalCandidateWork] = []
    candidates_per_sample: dict[int, int] = {}
    global_index = 0
    for sample_index, prepared in enumerate(prepared_inputs):
        if prepared.absence is not None or not prepared.candidates:
            continue
        candidates_per_sample[sample_index] = len(prepared.candidates)
        for candidate_index, (candidate, plan) in enumerate(
            zip(prepared.candidates, prepared.metric_plans, strict=True)
        ):
            flat_work.append(
                _GlobalCandidateWork(
                    sample_index=sample_index,
                    candidate_index=candidate_index,
                    work=_candidate_work(
                        request,
                        candidate,
                        plan,
                        global_index,
                    ),
                )
            )
            global_index += 1

    records_by_sample: dict[int, dict[int, CandidateExecutionRecord]] = {}
    pending_by_sample: dict[int, list[_PendingCacheObservation]] = {}
    admitted_by_sample: dict[int, int] = {}
    admitted = 0
    exhaustion: AttemptLimitExhaustion | None = None

    for window in _windows(flat_work, request.window_limits.max_cache_keys):
        await execution_cache.prefetch(
            item.work.request_key for item in window
        )
        try:
            misses: list[_GlobalCandidateWork] = []
            for item in window:
                observation = execution_cache.get(item.work.request_key)
                if observation is None:
                    misses.append(item)
                else:
                    sample_records = records_by_sample.setdefault(
                        item.sample_index,
                        {},
                    )
                    sample_records[item.candidate_index] = (
                        reused_candidate_record(
                            item.work.request,
                            observation.source_record,
                            observation.outcome,
                            budget=request.job_budget,
                            runtime=request.runtime,
                            cache_namespace=request.cache_namespace,
                        )
                    )
            observed_admissions = admitted + len(misses)
            if observed_admissions > request.attempt_limits.max_admitted_jobs:
                exhaustion = AttemptLimitExhaustion(
                    limit=AttemptLimitKind.ADMITTED_JOBS,
                    configured=request.attempt_limits.max_admitted_jobs,
                    observed=observed_admissions,
                )
                break
            for admitted_window in _windows(
                misses,
                request.window_limits.max_admitted_jobs,
            ):
                work_items = tuple(item.work for item in admitted_window)
                completed = await run_window(work_items)
                for item in admitted_window:
                    record = executed_candidate_record(
                        item.work.request,
                        completed[item.work.index],
                        budget=request.job_budget,
                        runtime=request.runtime,
                        cache_namespace=request.cache_namespace,
                    )
                    sample_records = records_by_sample.setdefault(
                        item.sample_index,
                        {},
                    )
                    sample_records[item.candidate_index] = record
                    pending_by_sample.setdefault(item.sample_index, []).append(
                        _PendingCacheObservation(
                            request_key=item.work.request_key,
                            outcome=record.outcome,
                        )
                    )
                    admitted_by_sample[item.sample_index] = (
                        admitted_by_sample.get(item.sample_index, 0) + 1
                    )
                admitted += len(admitted_window)
        finally:
            for item in window:
                execution_cache.discard(item.work.request_key)
        if exhaustion is not None:
            break

    by_sample: dict[int, _ExecutionWindow] = {}
    for sample_index, candidate_count in candidates_per_sample.items():
        sample_records = records_by_sample.get(sample_index, {})
        if len(sample_records) != candidate_count:
            if exhaustion is not None:
                continue
            raise RuntimeError(
                "global candidate execution left a sample incomplete"
            )
        by_sample[sample_index] = _ExecutionWindow(
            records=tuple(
                sample_records[candidate_index]
                for candidate_index in range(candidate_count)
            ),
            pending=tuple(pending_by_sample.get(sample_index, ())),
            admitted_jobs=admitted_by_sample.get(sample_index, 0),
            exhaustion=None,
        )

    return _GlobalExecutionResult(by_sample=by_sample, exhaustion=exhaustion)


def _candidate_work(
    request: EvaluationBatchRequest,
    candidate: MaterializedEvaluationCandidate,
    plan: _CandidateMetricPlan,
    index: int,
) -> _CandidateWork:
    job_request = HumanEvalCandidateJobRequest(
        candidate=candidate,
        suites=plan.suites,
    )
    request_key = candidate_execution_cache_key(
        job_request,
        request.job_budget,
        request.cache_namespace,
    )
    return _CandidateWork(
        index=index,
        request=job_request,
        request_key=request_key,
        job_id=JobId(uuid5(request.attempt.attempt_id, request_key)),
    )


def _record_is_invalid(record: SampleEvaluationRecord) -> bool:
    if not isinstance(record, EvaluatedSampleRecord):
        return False
    return any(
        isinstance(
            execution.outcome,
            HarnessExecutionFailure | ExecutorExecutionFailure,
        )
        for execution in record.executions
    ) or any(
        isinstance(metric, OperatorFailureRecord) for metric in record.metrics
    )


def _projection_row_contribution(
    request: EvaluationBatchRequest,
    prepared: _PreparedInput,
) -> int:
    requested = {projection.kind for projection in request.projections}
    return (
        (1 if ProjectionKind.EVALUATION_SAMPLES in requested else 0)
        + (
            len(prepared.candidates)
            if ProjectionKind.MATERIALIZED_CANDIDATES in requested
            else 0
        )
        + (
            len(prepared.candidates)
            * len(request.plan.procedure.metrics.questions)
            if ProjectionKind.METRIC_RECORDS in requested
            else 0
        )
    )


def _terminal_projection_reserve(request: EvaluationBatchRequest) -> int:
    requested = {projection.kind for projection in request.projections}
    return int(ProjectionKind.AGGREGATION_RESULTS in requested) + int(
        ProjectionKind.SCORES in requested
    )


def _extend_aggregate_state(
    request: EvaluationBatchRequest,
    record: SampleEvaluationRecord,
    *,
    aggregation_slots: list[AggregationSlot],
    score_values: list[MetricValue],
) -> None:
    if not isinstance(record, EvaluatedSampleRecord):
        return
    question_count = len(request.plan.procedure.metrics.questions)
    for candidate_index, candidate in enumerate(record.candidates):
        candidate_metrics = record.metrics[
            candidate_index * question_count : (candidate_index + 1)
            * question_count
        ]
        aggregate_record = next(
            (
                metric
                for metric in candidate_metrics
                if metric.identity.question
                == request.plan.aggregation.question
            ),
            None,
        )
        aggregation_slots.append(
            AggregationSlot(
                candidate=candidate.identity,
                record=aggregate_record,
            )
        )
        if isinstance(aggregate_record, MeasuredRecord):
            score_values.extend(
                value
                for value in aggregate_record.values
                if value.name == request.plan.aggregation.value
            )


def _terminal_projection_drafts(
    request: EvaluationBatchRequest,
    *,
    aggregation_slots: Sequence[AggregationSlot],
    score_values: Sequence[MetricValue],
) -> tuple[
    _AggregationResultProjectionDraft | None,
    _ScoreProjectionDraft | None,
]:
    aggregation_draft = None
    score_draft = None
    if aggregation_slots:
        result = aggregate(
            AggregationInput(
                policy=request.plan.aggregation,
                slots=tuple(aggregation_slots),
            )
        )
        aggregation_draft = _AggregationResultProjectionDraft(result=result)
        score = _score_from_aggregation(request, result, score_values)
        if score is not None:
            score_draft = _ScoreProjectionDraft(score=score)
    return aggregation_draft, score_draft


def _score_from_aggregation(
    request: EvaluationBatchRequest,
    result: AggregationResult,
    values: Sequence[MetricValue],
) -> Score | None:
    if not isinstance(result, AggregationOk) or not values:
        return None
    units = {value.unit for value in values}
    if len(units) != 1:
        return None
    return Score(
        name=request.plan.aggregation.value,
        value=result.value,
        unit=next(iter(units)),
        evaluation=EvaluationCoordinate(
            plan_id=request.plan.plan_id,
            version=request.plan.version,
            task_set=request.plan.task_set.coordinate,
            repeat_plan=request.plan.repeat_plan.coordinate,
        ),
        sources=(
            MetricValueCoordinate(
                question=request.plan.aggregation.question,
                value=request.plan.aggregation.value,
            ),
        ),
    )


def _windows(
    values: Sequence[_WindowValue],
    size: int,
) -> Iterator[Sequence[_WindowValue]]:
    return (
        values[start : start + size] for start in range(0, len(values), size)
    )
