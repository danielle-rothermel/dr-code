from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from enum import StrEnum, UNIQUE, verify
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Annotated, Final, Literal, Self, TypeAlias

from dr_exec import ExecutionPoolConfig, Executor
from dr_store import ArtifactBundlePublication, ObjectStore
from pydantic import Field, PositiveInt, field_validator, model_validator

from dr_code.core.models import FrozenModel
from dr_code.evaluation.id import (
    EvalAttemptId,
    EvalRuntimeId,
    EvalSample,
    EvalSlotId,
    MaterializedEvalCandidate,
)
from dr_code.evaluation.plan import EvalPlan
from dr_code.trace import PreprocessingDefinitionCoordinate, Trace

if TYPE_CHECKING:
    from dr_code.caching import WindowedExecutionCache
    from dr_code.evaluation._batch import _EvalBatchAssembly
    from dr_code.evaluation.bundle import _RecordPlacement
    from dr_code.evaluation.records import (
        EvalAttemptRecord,
        ReplaySource,
    )
    from dr_code.evaluation.projections import ProjectionRow


class AttemptLimits(FrozenModel):
    max_slots: PositiveInt
    max_materialized_candidates: PositiveInt
    max_admitted_jobs: PositiveInt
    max_retained_evidence_bytes: PositiveInt
    max_projection_rows: PositiveInt


class WindowLimits(FrozenModel):
    max_preprocessing_slots: PositiveInt
    max_cache_keys: PositiveInt
    max_admitted_jobs: PositiveInt
    max_record_assemblies: PositiveInt
    max_projection_rows: PositiveInt


class ShardLimits(FrozenModel):
    max_records: PositiveInt
    max_uncompressed_bytes: PositiveInt


CANDIDATE_STREAM_HEAD_BYTES: Final = 536_870_912
CANDIDATE_PAYLOAD_OUTPUT_BYTES: Final = 2 * CANDIDATE_STREAM_HEAD_BYTES


class CandidateJobBudget(FrozenModel):
    # Wall time and input bytes stay required: their right value is the
    # caller's workload, and a silent default would hide a wedged candidate.
    wall_time_ns: PositiveInt
    input_bytes: PositiveInt
    # Retention defaults keep a failing candidate's evidence readable rather
    # than clipped, so reading the failure never depends on the caller's
    # having named a bound.
    payload_output_bytes: PositiveInt = CANDIDATE_PAYLOAD_OUTPUT_BYTES
    stdout_head_bytes: PositiveInt = CANDIDATE_STREAM_HEAD_BYTES
    stderr_head_bytes: PositiveInt = CANDIDATE_STREAM_HEAD_BYTES

    @model_validator(mode="after")
    def validate_output_retention(self) -> Self:
        if (
            self.stdout_head_bytes + self.stderr_head_bytes
            != self.payload_output_bytes
        ):
            raise ValueError(
                "stdout_head_bytes + stderr_head_bytes must equal "
                "payload_output_bytes"
            )
        return self


class SampleEvalInput(FrozenModel):
    kind: Literal["sample"] = "sample"
    slot: EvalSlotId
    sample: EvalSample


class FrozenCandidateEvalInput(FrozenModel):
    kind: Literal["frozen_candidates"] = "frozen_candidates"
    slot: EvalSlotId
    sample: EvalSample
    preprocessing: PreprocessingDefinitionCoordinate
    candidates: tuple[MaterializedEvalCandidate, ...]

    @model_validator(mode="after")
    def validate_candidates(self) -> Self:
        identities = tuple(candidate.identity for candidate in self.candidates)
        if any(
            identity.sample != self.sample.metadata.identity
            for identity in identities
        ):
            raise ValueError(
                "frozen candidate sample identities must match the input sample"
            )
        if any(
            identity.preprocessing != self.preprocessing
            for identity in identities
        ):
            raise ValueError(
                "frozen candidate preprocessing identities must match the input"
            )
        if len(set(identities)) != len(identities):
            raise ValueError("frozen candidate identities must be unique")
        if tuple(
            identity.candidate_ordinal for identity in identities
        ) != tuple(range(len(identities))):
            raise ValueError(
                "frozen candidates must preserve contiguous materialization order"
            )
        return self


EvalInput: TypeAlias = Annotated[
    SampleEvalInput | FrozenCandidateEvalInput,
    Field(discriminator="kind"),
]


@verify(UNIQUE)
class RecordPlacement(StrEnum):
    # Never build persisted payloads by iterating this closed vocabulary.

    BUNDLE_LOCAL = "bundle_local"
    OBJECT_STORE = "object_store"


@verify(UNIQUE)
class RunGrade(StrEnum):
    # Never build persisted payloads by iterating this closed vocabulary.

    TRIAL = "trial"
    SELECTION = "selection"


@verify(UNIQUE)
class ProjectionKind(StrEnum):
    # Never build persisted payloads by iterating this closed vocabulary.

    EVAL_SAMPLES = "evaluation_samples"
    MATERIALIZED_CANDIDATES = "materialized_candidates"
    METRIC_RECORDS = "metric_records"
    AGGREGATION_RESULTS = "aggregation_results"
    SCORES = "scores"


class ProjectionRequest(FrozenModel):
    kind: ProjectionKind
    definition_version: Literal[2] = 2


class EvalBatchRequest(FrozenModel):
    attempt: EvalAttemptId
    plan: EvalPlan
    runtime: EvalRuntimeId
    cache_namespace: str = Field(min_length=1)
    # Required, never defaulted: a silent grade would let a trial outcome
    # serve a selection-grade run from the same cache key.
    run_grade: RunGrade
    inputs: tuple[EvalInput, ...] = Field(min_length=1)
    record_placement: RecordPlacement
    projections: tuple[ProjectionRequest, ...]
    attempt_limits: AttemptLimits
    window_limits: WindowLimits
    shard_limits: ShardLimits
    job_budget: CandidateJobBudget
    # The caller's deliberate re-run: skip execution-cache lookup for this
    # request's generations so every candidate re-executes and the fresh
    # outcome is the one recorded and offered to persistence. Persisted
    # bindings are first-writer-wins, so this bypasses lookup without
    # replacing an entry already stored under the same key.
    fresh: bool = False

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if len(self.inputs) > self.attempt_limits.max_slots:
            raise ValueError("input slot count exceeds max_slots")

        frozen_candidate_count = sum(
            len(item.candidates)
            for item in self.inputs
            if isinstance(item, FrozenCandidateEvalInput)
        )
        if (
            frozen_candidate_count
            > self.attempt_limits.max_materialized_candidates
        ):
            raise ValueError(
                "frozen candidate count exceeds max_materialized_candidates"
            )

        slots = tuple(item.slot for item in self.inputs)
        samples = tuple(item.sample.metadata.identity for item in self.inputs)
        if len(set(slots)) != len(slots):
            raise ValueError("evaluation input slots must be unique")
        if len(set(samples)) != len(samples):
            raise ValueError("evaluation input samples must be unique")
        if any(
            item.slot.task_id != item.sample.metadata.task_id
            for item in self.inputs
        ):
            raise ValueError("input sample task_id must match its slot")
        preprocessing_keys = {
            step.instance_name
            for step in self.plan.procedure.preprocessing.steps
        }
        for item in self.inputs:
            collisions = preprocessing_keys & {
                auxiliary.trace_key
                for auxiliary in item.sample.auxiliary_artifacts
            }
            if collisions:
                raise ValueError(
                    "sample auxiliary trace keys collide with preprocessing "
                    "steps: " + ", ".join(sorted(collisions))
                )

        expected_slots = self.plan.ordered_slots()
        positions = {slot: index for index, slot in enumerate(expected_slots)}
        try:
            input_positions = tuple(positions[slot] for slot in slots)
        except KeyError as error:
            raise ValueError(
                "every evaluation input slot must belong to the evaluation plan"
            ) from error
        if input_positions != tuple(sorted(input_positions)):
            raise ValueError("evaluation inputs must preserve plan slot order")

        if len({request.kind for request in self.projections}) != len(
            self.projections
        ):
            raise ValueError("projection requests must be unique")

        window_attempt_pairs = (
            (
                self.window_limits.max_preprocessing_slots,
                self.attempt_limits.max_slots,
                "max_preprocessing_slots",
            ),
            (
                self.window_limits.max_cache_keys,
                self.attempt_limits.max_materialized_candidates,
                "max_cache_keys",
            ),
            (
                self.window_limits.max_admitted_jobs,
                self.attempt_limits.max_admitted_jobs,
                "max_admitted_jobs",
            ),
            (
                self.window_limits.max_record_assemblies,
                self.attempt_limits.max_slots,
                "max_record_assemblies",
            ),
            (
                self.window_limits.max_projection_rows,
                self.attempt_limits.max_projection_rows,
                "max_projection_rows",
            ),
        )
        for window, attempt, name in window_attempt_pairs:
            if window > attempt:
                raise ValueError(
                    f"window {name} must not exceed its attempt limit"
                )
        return self


class EvalProjectionReference(FrozenModel):
    kind: ProjectionKind
    definition_version: Literal[2] = 2
    source_attempt: EvalAttemptId
    artifact_name: str

    @field_validator("artifact_name")
    @classmethod
    def validate_artifact_name(cls, artifact_name: str) -> str:
        path = PurePosixPath(artifact_name)
        if (
            not artifact_name
            or artifact_name.startswith("/")
            or path.as_posix() != artifact_name
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError(
                "artifact_name must be a normalized relative artifact name"
            )
        return artifact_name


class EvalBatchResult(FrozenModel):
    attempt: EvalAttemptRecord
    projections: tuple[EvalProjectionReference, ...]
    bundle_path: Path | None

    def __init__(self, **data: object) -> None:
        if not type(self).__pydantic_complete__:
            from dr_code.evaluation.records import EvalAttemptRecord

            type(self).model_rebuild(
                _types_namespace={"EvalAttemptRecord": EvalAttemptRecord}
            )
        super().__init__(**data)


async def evaluate_batch(
    request: EvalBatchRequest,
    /,
    *,
    executor: Executor,
    execution_cache: WindowedExecutionCache,
    object_store: ObjectStore | None,
    publication: ArtifactBundlePublication | None,
    pool_config: ExecutionPoolConfig,
    preprocessed_traces: Mapping[str, Trace] | None = None,
) -> EvalBatchResult:
    """Evaluate one bounded standalone attempt and optionally publish it.

    `preprocessed_traces` lets a caller that already ran the pooled
    preprocessing leg over this request's sample texts hand those traces in, so
    the corpus is preprocessed once per attempt instead of twice. The traces
    must come from `request.plan.procedure.preprocessing`; a sample text absent
    from the mapping is preprocessed in process while the batch prepares it.
    """

    return await _evaluate_batch_with_replay(
        request,
        executor=executor,
        execution_cache=execution_cache,
        object_store=object_store,
        publication=publication,
        pool_config=pool_config,
        preprocessed_traces=preprocessed_traces,
        replay=None,
    )


async def _evaluate_batch_with_replay(
    request: EvalBatchRequest,
    /,
    *,
    executor: Executor,
    execution_cache: WindowedExecutionCache,
    object_store: ObjectStore | None,
    publication: ArtifactBundlePublication | None,
    pool_config: ExecutionPoolConfig,
    preprocessed_traces: Mapping[str, Trace] | None = None,
    replay: ReplaySource | None,
) -> EvalBatchResult:
    """Run the standalone path with attempt provenance fixed before publish."""

    from dr_code.evaluation._batch import _evaluate_batch_assembly

    async def assemble(
        placement: _RecordPlacement,
    ) -> _EvalBatchAssembly:
        return await _evaluate_batch_assembly(
            request,
            executor=executor,
            execution_cache=execution_cache,
            pool_config=pool_config,
            placement_sink=placement,
            preprocessed_traces=preprocessed_traces,
        )

    return await _evaluate(
        request,
        execution_cache=execution_cache,
        object_store=object_store,
        publication=publication,
        assemble=assemble,
        replay=replay,
    )


async def evaluate_durable_partition(
    request: EvalBatchRequest,
    /,
    *,
    executor: Executor,
    execution_cache: WindowedExecutionCache,
    object_store: ObjectStore | None,
    publication: ArtifactBundlePublication | None,
) -> EvalBatchResult:
    """Evaluate one durable partition serially without constructing a pool."""

    from dr_code.evaluation._batch import _evaluate_durable_partition_assembly

    async def assemble(
        placement: _RecordPlacement,
    ) -> _EvalBatchAssembly:
        return await _evaluate_durable_partition_assembly(
            request,
            executor=executor,
            execution_cache=execution_cache,
            placement_sink=placement,
        )

    return await _evaluate(
        request,
        execution_cache=execution_cache,
        object_store=object_store,
        publication=publication,
        assemble=assemble,
        replay=None,
    )


async def _evaluate(
    request: EvalBatchRequest,
    *,
    execution_cache: WindowedExecutionCache,
    object_store: ObjectStore | None,
    publication: ArtifactBundlePublication | None,
    assemble: Callable[[_RecordPlacement], Awaitable[_EvalBatchAssembly]],
    replay: ReplaySource | None,
) -> EvalBatchResult:
    from dr_code.evaluation._batch import _EvalBatchAssembly
    from dr_code.evaluation.bundle import (
        _PROJECTION_ARTIFACT_NAMES,
        _RecordPlacement,
        _publish_bundle,
        _write_projection_artifact,
        ProjectionArtifactHeader,
    )
    from dr_code.evaluation.records import EvalAttemptRecord

    if request.record_placement is RecordPlacement.BUNDLE_LOCAL:
        if publication is None:
            raise ValueError("bundle-local placement requires publication")
    elif object_store is None:
        raise ValueError("object-store placement requires object_store")
    if request.projections and publication is None:
        raise ValueError("requested projections require publication")

    placement = _RecordPlacement(
        request,
        publication=publication,
        object_store=object_store,
    )
    assembly = await assemble(placement)
    if not isinstance(assembly, _EvalBatchAssembly):
        raise TypeError("assembly returned an unsupported value")
    await placement.validate_bundle_reference_closure()
    attempt = EvalAttemptRecord(
        identity=request.attempt,
        plan=request.plan,
        runtime=request.runtime,
        cache_namespace=request.cache_namespace,
        members=assembly.members,
        completeness=assembly.completeness,
        validity=assembly.validity,
        limit_exhaustion=assembly.limit_exhaustion,
        replay=replay,
    )
    projections = tuple(
        EvalProjectionReference(
            kind=projection.kind,
            source_attempt=request.attempt,
            artifact_name=_PROJECTION_ARTIFACT_NAMES[projection.kind],
        )
        for projection in request.projections
    )
    if publication is not None:
        for projection in request.projections:
            header = ProjectionArtifactHeader(
                source_attempt=request.attempt,
                kind=projection.kind,
            )
            await _write_projection_artifact(
                publication,
                header,
                _projection_rows(
                    request,
                    placement=placement,
                    assembly=assembly,
                    kind=projection.kind,
                ),
                max_resident_rows=request.window_limits.max_projection_rows,
            )
        await asyncio.to_thread(
            _publish_bundle,
            publication,
            attempt=attempt,
            projections=projections,
        )
    return EvalBatchResult(
        attempt=attempt,
        projections=projections,
        bundle_path=None if publication is None else publication.path,
    )


async def _projection_rows(
    request: EvalBatchRequest,
    *,
    placement: object,
    assembly: object,
    kind: ProjectionKind,
) -> AsyncIterator[ProjectionRow]:
    from dr_code.evaluation._batch import _EvalBatchAssembly
    from dr_code.evaluation.bundle import _RecordPlacement
    from dr_code.evaluation.projections import (
        AggregationResultProjectionRow,
        EvalSampleProjectionRow,
        MaterializedCandidateProjectionRow,
        MetricRecordProjectionRow,
        ScoreProjectionRow,
    )
    from dr_code.evaluation.records import EvaluatedSampleRecord
    from dr_code.metrics import MeasuredRecord

    if not isinstance(placement, _RecordPlacement) or not isinstance(
        assembly, _EvalBatchAssembly
    ):
        raise TypeError("unsupported projection assembly")
    if kind is ProjectionKind.AGGREGATION_RESULTS:
        if assembly.aggregation is not None:
            yield AggregationResultProjectionRow(
                source_attempt=request.attempt,
                policy=request.plan.aggregation,
                result=assembly.aggregation.result,
            )
        return
    if kind is ProjectionKind.SCORES:
        if assembly.score is not None:
            yield ScoreProjectionRow(
                source_attempt=request.attempt,
                score=assembly.score.score,
            )
        return

    question_count = len(request.plan.procedure.metrics.questions)
    async for reference, record in placement.iter_records():
        if kind is ProjectionKind.EVAL_SAMPLES:
            yield EvalSampleProjectionRow(
                source_attempt=request.attempt,
                slot=record.slot,
                sample=record.sample,
                status=record.status,
                record=reference,
            )
            continue
        if not isinstance(record, EvaluatedSampleRecord):
            continue
        for candidate_index, candidate in enumerate(record.candidates):
            if kind is ProjectionKind.MATERIALIZED_CANDIDATES:
                yield MaterializedCandidateProjectionRow(
                    source_attempt=request.attempt,
                    candidate=candidate.identity,
                    source_sha256=candidate.source_sha256,
                    sample_record=reference,
                )
                continue
            metrics = record.metrics[
                candidate_index * question_count : (candidate_index + 1)
                * question_count
            ]
            for metric in metrics:
                yield MetricRecordProjectionRow(
                    source_attempt=request.attempt,
                    candidate=candidate.identity,
                    question=metric.identity.question,
                    status=metric.status,
                    values=(
                        metric.values
                        if isinstance(metric, MeasuredRecord)
                        else ()
                    ),
                    sample_record=reference,
                )


__all__ = [
    "AttemptLimits",
    "CandidateJobBudget",
    "EvalBatchRequest",
    "EvalBatchResult",
    "EvalInput",
    "EvalProjectionReference",
    "FrozenCandidateEvalInput",
    "ProjectionKind",
    "ProjectionRequest",
    "RecordPlacement",
    "RunGrade",
    "SampleEvalInput",
    "ShardLimits",
    "WindowLimits",
    "evaluate_batch",
    "evaluate_durable_partition",
]
