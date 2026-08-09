from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from enum import StrEnum, UNIQUE, verify
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Annotated, Literal, Self, TypeAlias

from dr_exec import ExecutionPoolConfig, Executor
from dr_store import ArtifactBundlePublication, ObjectStore
from pydantic import Field, PositiveInt, field_validator, model_validator

from dr_code.core.models import FrozenModel
from dr_code.evaluation.identity import (
    EvaluationAttemptIdentity,
    EvaluationRuntimeIdentity,
    EvaluationSample,
    EvaluationSlotIdentity,
    MaterializedEvaluationCandidate,
)
from dr_code.evaluation.plan import EvaluationPlan
from dr_code.trace import PreprocessingDefinitionCoordinate

if TYPE_CHECKING:
    from dr_code.caching import WindowedExecutionCache
    from dr_code.evaluation._batch import _EvaluationBatchAssembly
    from dr_code.evaluation.bundle import _RecordPlacement
    from dr_code.evaluation.records import (
        EvaluationAttemptRecord,
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


class CandidateJobBudget(FrozenModel):
    wall_time_ns: PositiveInt
    input_bytes: PositiveInt
    payload_output_bytes: PositiveInt
    stdout_head_bytes: PositiveInt
    stderr_head_bytes: PositiveInt

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


class SampleEvaluationInput(FrozenModel):
    kind: Literal["sample"] = "sample"
    slot: EvaluationSlotIdentity
    sample: EvaluationSample


class FrozenCandidateEvaluationInput(FrozenModel):
    kind: Literal["frozen_candidates"] = "frozen_candidates"
    slot: EvaluationSlotIdentity
    sample: EvaluationSample
    preprocessing: PreprocessingDefinitionCoordinate
    candidates: tuple[MaterializedEvaluationCandidate, ...]

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


EvaluationInput: TypeAlias = Annotated[
    SampleEvaluationInput | FrozenCandidateEvaluationInput,
    Field(discriminator="kind"),
]


@verify(UNIQUE)
class RecordPlacement(StrEnum):
    # Never build persisted payloads by iterating this closed vocabulary.

    BUNDLE_LOCAL = "bundle_local"
    OBJECT_STORE = "object_store"


@verify(UNIQUE)
class ProjectionKind(StrEnum):
    # Never build persisted payloads by iterating this closed vocabulary.

    EVALUATION_SAMPLES = "evaluation_samples"
    MATERIALIZED_CANDIDATES = "materialized_candidates"
    METRIC_RECORDS = "metric_records"
    AGGREGATION_RESULTS = "aggregation_results"
    SCORES = "scores"


class ProjectionRequest(FrozenModel):
    kind: ProjectionKind
    definition_version: Literal[1] = 1


class EvaluationBatchRequest(FrozenModel):
    attempt: EvaluationAttemptIdentity
    plan: EvaluationPlan
    runtime: EvaluationRuntimeIdentity
    cache_namespace: str = Field(min_length=1)
    inputs: tuple[EvaluationInput, ...] = Field(min_length=1)
    record_placement: RecordPlacement
    projections: tuple[ProjectionRequest, ...]
    attempt_limits: AttemptLimits
    window_limits: WindowLimits
    shard_limits: ShardLimits
    job_budget: CandidateJobBudget

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if len(self.inputs) > self.attempt_limits.max_slots:
            raise ValueError("input slot count exceeds max_slots")

        frozen_candidate_count = sum(
            len(item.candidates)
            for item in self.inputs
            if isinstance(item, FrozenCandidateEvaluationInput)
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

        expected_slots = tuple(
            EvaluationSlotIdentity(
                task_set=self.plan.task_set.coordinate,
                repeat_plan=self.plan.repeat_plan.coordinate,
                task_id=task_id,
                repeat_index=repeat_index,
            )
            for task_id in self.plan.task_set.selected
            for repeat_index in range(self.plan.repeat_plan.repeats)
        )
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


class EvaluationProjectionReference(FrozenModel):
    kind: ProjectionKind
    definition_version: Literal[1] = 1
    source_attempt: EvaluationAttemptIdentity
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


class EvaluationBatchResult(FrozenModel):
    attempt: EvaluationAttemptRecord
    projections: tuple[EvaluationProjectionReference, ...]
    bundle_path: Path | None

    def __init__(self, **data: object) -> None:
        if not type(self).__pydantic_complete__:
            from dr_code.evaluation.records import EvaluationAttemptRecord

            type(self).model_rebuild(
                _types_namespace={
                    "EvaluationAttemptRecord": EvaluationAttemptRecord
                }
            )
        super().__init__(**data)


async def evaluate_batch(
    request: EvaluationBatchRequest,
    /,
    *,
    executor: Executor,
    execution_cache: WindowedExecutionCache,
    object_store: ObjectStore | None,
    publication: ArtifactBundlePublication | None,
    pool_config: ExecutionPoolConfig,
) -> EvaluationBatchResult:
    """Evaluate one bounded standalone attempt and optionally publish it."""

    return await _evaluate_batch_with_replay(
        request,
        executor=executor,
        execution_cache=execution_cache,
        object_store=object_store,
        publication=publication,
        pool_config=pool_config,
        replay=None,
    )


async def _evaluate_batch_with_replay(
    request: EvaluationBatchRequest,
    /,
    *,
    executor: Executor,
    execution_cache: WindowedExecutionCache,
    object_store: ObjectStore | None,
    publication: ArtifactBundlePublication | None,
    pool_config: ExecutionPoolConfig,
    replay: ReplaySource | None,
) -> EvaluationBatchResult:
    """Run the standalone path with attempt provenance fixed before publish."""

    from dr_code.evaluation._batch import _evaluate_batch_assembly

    async def assemble(
        placement: _RecordPlacement,
    ) -> _EvaluationBatchAssembly:
        return await _evaluate_batch_assembly(
            request,
            executor=executor,
            execution_cache=execution_cache,
            pool_config=pool_config,
            placement_sink=placement,
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
    request: EvaluationBatchRequest,
    /,
    *,
    executor: Executor,
    execution_cache: WindowedExecutionCache,
    object_store: ObjectStore | None,
    publication: ArtifactBundlePublication | None,
) -> EvaluationBatchResult:
    """Evaluate one durable partition serially without constructing a pool."""

    from dr_code.evaluation._batch import _evaluate_durable_partition_assembly

    async def assemble(
        placement: _RecordPlacement,
    ) -> _EvaluationBatchAssembly:
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
    request: EvaluationBatchRequest,
    *,
    execution_cache: WindowedExecutionCache,
    object_store: ObjectStore | None,
    publication: ArtifactBundlePublication | None,
    assemble: Callable[
        [_RecordPlacement], Awaitable[_EvaluationBatchAssembly]
    ],
    replay: ReplaySource | None,
) -> EvaluationBatchResult:
    from dr_code.evaluation._batch import _EvaluationBatchAssembly
    from dr_code.evaluation.bundle import (
        _PROJECTION_ARTIFACT_NAMES,
        _RecordPlacement,
        _publish_bundle,
        _write_projection_artifact,
        ProjectionArtifactHeader,
    )
    from dr_code.evaluation.records import EvaluationAttemptRecord

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
    if not isinstance(assembly, _EvaluationBatchAssembly):
        raise TypeError("assembly returned an unsupported value")
    await placement.validate_bundle_reference_closure()
    attempt = EvaluationAttemptRecord(
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
        EvaluationProjectionReference(
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
    return EvaluationBatchResult(
        attempt=attempt,
        projections=projections,
        bundle_path=None if publication is None else publication.path,
    )


async def _projection_rows(
    request: EvaluationBatchRequest,
    *,
    placement: object,
    assembly: object,
    kind: ProjectionKind,
) -> AsyncIterator[ProjectionRow]:
    from dr_code.evaluation._batch import _EvaluationBatchAssembly
    from dr_code.evaluation.bundle import _RecordPlacement
    from dr_code.evaluation.projections import (
        AggregationResultProjectionRow,
        EvaluationSampleProjectionRow,
        MaterializedCandidateProjectionRow,
        MetricRecordProjectionRow,
        ScoreProjectionRow,
    )
    from dr_code.evaluation.records import EvaluatedSampleRecord
    from dr_code.metrics import MeasuredRecord

    if not isinstance(placement, _RecordPlacement) or not isinstance(
        assembly, _EvaluationBatchAssembly
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
        if kind is ProjectionKind.EVALUATION_SAMPLES:
            yield EvaluationSampleProjectionRow(
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
    "EvaluationBatchRequest",
    "EvaluationBatchResult",
    "EvaluationInput",
    "EvaluationProjectionReference",
    "FrozenCandidateEvaluationInput",
    "ProjectionKind",
    "ProjectionRequest",
    "RecordPlacement",
    "SampleEvaluationInput",
    "ShardLimits",
    "WindowLimits",
    "evaluate_batch",
    "evaluate_durable_partition",
]
