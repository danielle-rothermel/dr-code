from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from uuid import UUID

from dr_serialize import Jsonable, Sha256Digest, build_identity_document
from dr_store import CacheEntry, CacheHit, EvictStatus, ObjectReference

from _candidate_job_builders import candidate_job_task
from dr_code.caching import WindowedExecutionCache
from dr_code.evaluation import (
    AggregationPolicy,
    AggregationStatistic,
    AttemptLimits,
    BundleRecordReference,
    CandidateJobBudget,
    DatasetCoordinate,
    EvalAttemptId,
    EvalBatchRequest,
    EvalCandidateId,
    EvalProcedure,
    EvalRuntimeId,
    EvalSample,
    EvalSampleAuxiliaryArtifact,
    EvalSampleId,
    EvalSampleMetadata,
    EvalSlotId,
    EvalSourceId,
    GeneratedSampleProvenance,
    MaterializedEvalCandidate,
    PreprocessMode,
    ProjectionKind,
    ProjectionRequest,
    RecordPlacement,
    RunGrade,
    SampleData,
    SampleWithCandidatesData,
    ShardLimits,
    SlotData,
    StoredRecordReference,
    SamplingPlan,
    SamplingPlanCoordinate,
    TaskSet,
    TaskSetCoordinate,
    WindowLimits,
)
from dr_code.evaluation.records import SampleEvalRecord
from dr_code.humaneval.settings import CodeTestSettings
from dr_code.metrics import (
    MetricName,
    MetricQuestion,
    MetricQuestionCoordinate,
    MetricsDefinition,
)
from dr_code.metrics.coordinates import question_settings
from dr_code.preprocessing import (
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
    bind_preprocessing,
)
from dr_code.trace import CodeArtifact, JsonArtifact, TextArtifact


class BatchStore:
    def __init__(self) -> None:
        self.records: dict[str, tuple[str, Jsonable]] = {}
        self.get_calls: list[tuple[str, ...]] = []

    async def get_many(
        self,
        keys: Iterable[str],
        *,
        schema: str,
    ) -> dict[str, CacheHit | None]:
        requested = tuple(keys)
        self.get_calls.append(requested)
        return {
            key: (
                CacheHit(record=stored[1])
                if (stored := self.records.get(key)) is not None
                and stored[0] == schema
                else None
            )
            for key in requested
        }

    async def put_many(
        self,
        entries: Mapping[str, CacheEntry],
    ) -> dict[str, ObjectReference]:
        # Bindings are append-only first-writer-wins, matching dr-store: a
        # later entry for a bound key keeps the stored winner and returns it.
        for key, entry in entries.items():
            self.records.setdefault(key, (entry.schema, entry.record))
        return {
            key: ObjectReference.for_record(*self.records[key])
            for key in entries
        }

    async def evict_bindings(
        self,
        keys: Iterable[str],
    ) -> dict[str, EvictStatus]:
        distinct = tuple(dict.fromkeys(keys))
        statuses: dict[str, EvictStatus] = {}
        for key in distinct:
            if key in self.records:
                del self.records[key]
                statuses[key] = EvictStatus.EVICTED
            else:
                statuses[key] = EvictStatus.ABSENT
        return statuses


class MemoryPlacement:
    def __init__(self) -> None:
        self.records: list[SampleEvalRecord] = []

    async def place(self, record: SampleEvalRecord, /):  # type: ignore[no-untyped-def]
        reference = BundleRecordReference(
            artifact_name="sample-records-00000000.jsonl",
            record_index=len(self.records),
            record_sha256=Sha256Digest(
                hashlib.sha256(record.model_dump_json().encode()).hexdigest()
            ),
            schema="dr-code/sample-evaluation-record-v1",
            schema_version=1,
        )
        self.records.append(record)
        return reference

    async def finish(self) -> None:
        return None


class StoredMemoryPlacement(MemoryPlacement):
    async def place(self, record: SampleEvalRecord, /):  # type: ignore[no-untyped-def]
        self.records.append(record)
        return StoredRecordReference(
            reference=ObjectReference.for_record(
                "dr-code/sample-evaluation-record-v1",
                record.model_dump(mode="json"),
            ),
            schema_version=1,
        )


TASK_ID = candidate_job_task().task_id


def runtime() -> EvalRuntimeId:
    return EvalRuntimeId(
        document=build_identity_document(
            schema="tests/evaluation-runtime",
            schema_version=1,
            payload={"runtime": "batch"},
        )
    )


def sample(
    index: int,
    *,
    text: str = "def observed_load_count(_x):\n    return 1\n",
) -> EvalSample:
    task = candidate_job_task()
    source_reference = StoredRecordReference(
        reference=ObjectReference.for_record(
            "tests/input",
            {"input": index},
        ),
        schema_version=1,
    )
    return EvalSample(
        metadata=EvalSampleMetadata(
            identity=EvalSampleId(sample_id=f"sample-{index}"),
            task_id=TASK_ID,
            provenance=GeneratedSampleProvenance(
                source_identity=EvalSourceId(
                    namespace="tests",
                    value=f"input-{index}",
                ),
                source_reference=source_reference,
                generation_id=f"generation-{index}",
            ),
        ),
        raw_input=TextArtifact(text=text),
        auxiliary_artifacts=(
            EvalSampleAuxiliaryArtifact(
                trace_key="task",
                artifact=JsonArtifact(payload=task.model_dump(mode="json")),
            ),
        ),
    )


def sample_data(
    index: int,
    *,
    text: str = "def observed_load_count(_x):\n    return 1\n",
) -> SampleData:
    return SampleData(sample=sample(index, text=text))


def sample_with_candidates_data(
    index: int,
) -> SampleWithCandidatesData:
    selected_sample = sample(index)
    preprocessing = bind_preprocessing(
        EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION
    ).producer.definition
    source = CodeArtifact(
        source="def observed_load_count(_x):\n    return 1\n"
    )
    return SampleWithCandidatesData(
        sample=selected_sample,
        preprocessing=preprocessing,
        candidates=(
            MaterializedEvalCandidate(
                identity=EvalCandidateId(
                    sample=selected_sample.metadata.identity,
                    preprocessing=preprocessing,
                    candidate_ordinal=0,
                ),
                source=source,
                source_sha256=Sha256Digest(
                    hashlib.sha256(source.source.encode()).hexdigest()
                ),
            ),
        ),
    )


def slot_data(
    slot: EvalSlotId, data: SampleData | SampleWithCandidatesData
) -> SlotData:
    return SlotData(slot=slot, data=data)


def request(
    count: int = 1,
    *,
    preprocess_mode: PreprocessMode,
    attempt_limits: AttemptLimits | None = None,
    window_limits: WindowLimits | None = None,
    inputs: tuple[SlotData, ...] | None = None,
    texts: tuple[str, ...] | None = None,
    projections: tuple[ProjectionKind, ...] = tuple(ProjectionKind),
) -> EvalBatchRequest:
    settings = CodeTestSettings()
    question = MetricQuestion(
        metric=MetricName.CODE_TEST,
        on="output",
        settings=settings,
    )
    question_coordinate = MetricQuestionCoordinate(
        metric=MetricName.CODE_TEST,
        on_key="output",
        settings=question_settings(settings),
    )
    dataset = DatasetCoordinate(dataset_id="tests", version="1")
    task_set = TaskSet(
        coordinate=TaskSetCoordinate(
            task_set_id="batch",
            version="1",
            dataset=dataset,
        ),
        population=(TASK_ID,),
        selected=(TASK_ID,),
    )
    sampling_plan = SamplingPlan(
        coordinate=SamplingPlanCoordinate(
            sampling_plan_id="batch",
            version="1",
        ),
        task_count=1,
        task_num_samples=(count,),
    )
    if inputs is None:
        inputs = tuple(
            slot_data(
                EvalSlotId(
                    task_set=task_set.coordinate,
                    sampling_plan=sampling_plan.coordinate,
                    task_id=TASK_ID,
                    sample_index=index,
                ),
                SampleData(
                    sample=(
                        sample(index)
                        if texts is None
                        else sample(index, text=texts[index])
                    )
                ),
            )
            for index in range(count)
        )
    return EvalBatchRequest(
        attempt=EvalAttemptId(attempt_id=UUID(int=1)),
        plan={
            "plan_id": "batch",
            "version": "1",
            "task_set": task_set,
            "sampling_plan": sampling_plan,
            "procedure": EvalProcedure(
                preprocessing=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
                metrics=MetricsDefinition(
                    definition_id="batch",
                    version="0",
                    questions=(question,),
                ),
            ),
            "aggregation": AggregationPolicy(
                question=question_coordinate,
                value="passed_count",
                statistic=AggregationStatistic.MEAN,
            ),
        },
        runtime=runtime(),
        cache_namespace="tests/batch",
        run_grade=RunGrade.TRIAL,
        inputs=inputs,
        record_placement=RecordPlacement.BUNDLE_LOCAL,
        projections=tuple(
            ProjectionRequest(kind=kind) for kind in projections
        ),
        attempt_limits=attempt_limits
        or AttemptLimits(
            max_slots=max(1, count),
            max_materialized_candidates=max(1, count * 4),
            max_admitted_jobs=max(1, count * 4),
            max_retained_evidence_bytes=10_000_000,
            max_projection_rows=max(5, count * 5 + 2),
        ),
        window_limits=window_limits
        or WindowLimits(
            max_preprocessing_slots=1,
            max_cache_keys=1,
            max_admitted_jobs=1,
            max_record_assemblies=1,
            max_projection_rows=5,
        ),
        shard_limits=ShardLimits(
            max_records=10,
            max_uncompressed_bytes=10_000_000,
        ),
        job_budget=CandidateJobBudget(
            wall_time_ns=5_000_000_000,
            input_bytes=2_097_152,
            payload_output_bytes=2_097_152,
            stdout_head_bytes=1_048_576,
            stderr_head_bytes=1_048_576,
        ),
        preprocess_mode=preprocess_mode,
    )


def frozen_input(index: int, slot: EvalSlotId) -> SlotData:
    return slot_data(slot, sample_with_candidates_data(index))


def cache(store: BatchStore, *, resident: int = 4) -> WindowedExecutionCache:
    return WindowedExecutionCache(
        store,
        runtime=runtime(),
        max_resident_entries=resident,
        max_pending_checkpoint_entries=resident,
    )
