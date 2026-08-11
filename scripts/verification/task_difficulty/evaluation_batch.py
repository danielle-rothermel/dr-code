"""Build and export task-difficulty evaluation batches via evaluate_batch."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Final
from uuid import NAMESPACE_URL, uuid5

import polars as pl
from dr_exec import (
    BudgetAxis,
    BudgetExceededOutcome,
    Budgets,
    CompletedExecution,
    ContainmentProfile,
    EnvGrant,
    ExecutionJob,
    Executor,
    ExitedOutcome,
    FailureOwner,
    FiniteByteLimit,
    FiniteDurationLimit,
    FiniteOutput,
    JobId,
    OutputOverflowPolicy,
    PayloadRetentionBudget,
    ProtocolFailedOutcome,
    SignaledOutcome,
    StreamRetentionBudget,
    UntrustedPythonTarget,
)
from dr_serialize import Sha256Digest, build_identity_document
from dr_store import ObjectReference, ObjectStore

from dr_code.evaluation import (
    AggregationPolicy,
    AggregationStatistic,
    AttemptCompleteness,
    AttemptLimits,
    CandidateJobBudget,
    CorpusSampleProvenance,
    DatasetCoordinate,
    EvaluationAttemptIdentity,
    EvaluationBatchRequest,
    EvaluationCandidateIdentity,
    EvaluationProcedure,
    EvaluationReadLimits,
    EvaluationRuntimeIdentity,
    EvaluationSample,
    EvaluationSampleAuxiliaryArtifact,
    EvaluationSampleIdentity,
    EvaluationSampleMetadata,
    EvaluationSlotIdentity,
    EvaluationSourceIdentity,
    FrozenCandidateEvaluationInput,
    MaterializedEvaluationCandidate,
    MetricRecordProjectionRow,
    ProjectionKind,
    ProjectionRequest,
    RecordPlacement,
    RepeatPlan,
    RepeatPlanCoordinate,
    StoredRecordReference,
    TaskSet,
    TaskSetCoordinate,
    WindowLimits,
    restore_evaluation_attempt,
    evaluate_batch,
    read_evaluation_projection,
)
from dr_code.evaluation.batch import EvaluationBatchResult, ShardLimits
from dr_code.evaluation.plan import EvaluationPlan
from dr_code.humaneval import HumanEvalTask, parse_humaneval_dataset
from dr_code.humaneval.sampling import load_humaneval_rows
from dr_code.humaneval.settings import CodeTestSettings
from dr_code.metrics import (
    MetricName,
    MetricQuestion,
    MetricQuestionCoordinate,
    MetricsDefinition,
    RecordStatus,
)
from dr_code.metrics.coordinates import question_settings
from dr_code.preprocessing import (
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
    bind_preprocessing,
)
from dr_code.trace import (
    CodeArtifact,
    JsonArtifact,
    PreprocessingDefinitionCoordinate,
    PreprocessingTraceProducer,
    TextArtifact,
)

from workflow_settings import EvaluationSettings

_WORKFLOW_METRICS_DEFINITION_ID: Final = (
    "directional-humaneval-task-difficulty"
)
_WORKFLOW_METRICS_DEFINITION_VERSION: Final = "0"
_WORKFLOW_CACHE_NAMESPACE: Final = "directional-humaneval-task-difficulty"
_WORKFLOW_PLAN_ID: Final = "directional-humaneval-task-difficulty"
_WORKFLOW_PLAN_VERSION: Final = "1"
_TASK_SET_ID: Final = "directional-humaneval-balanced"
_REPEAT_PLAN_ID: Final = "directional-humaneval-balanced"
_DATASET_ID: Final = "directional-humaneval-generation-corpus"
_DATASET_VERSION: Final = "1"
_CORPUS_ROW_SCHEMA: Final = "dr-code/generation-corpus-row-v1"
_ATTEMPT_NAMESPACE: Final = NAMESPACE_URL
_EXECUTION_REQUEST_SCHEMA: Final = "dr-code/python-execution-request"
_EXECUTION_REQUEST_SCHEMA_VERSION: Final = 1
_EXECUTION_ENVIRONMENT: Final[dict[str, str]] = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
}
_PROBE_TIMEOUT_SECONDS: Final = 10.0
_NANOSECONDS_PER_SECOND: Final = 1_000_000_000
_MAX_INPUT_BYTES: Final = 2_097_152
_MAX_STREAM_BYTES: Final = 536_870_912
_RUNTIME_PROBE_SOURCE: Final = """\
import json
import platform
import sys

def dr_exec_main(request, emit):
    import numpy
    print(json.dumps({
        "implementation": platform.python_implementation(),
        "numpy_version": numpy.__version__,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
    }, sort_keys=True, separators=(",", ":")))
"""
_CELL_SORT_COLUMNS = ("generation_mode", "budget_mode", "model_key")


def _exhaustive_preprocessing_coordinate() -> (
    PreprocessingDefinitionCoordinate
):
    bound = bind_preprocessing(EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION)
    producer = bound.producer
    if not isinstance(producer, PreprocessingTraceProducer):
        raise TypeError("expected registered preprocessing producer")
    return producer.definition


def _metric_int(values: dict[str, object], key: str) -> int:
    raw = values.get(key)
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    return 0


def candidate_job_budget(timeout_seconds: float) -> CandidateJobBudget:
    if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
        raise ValueError("timeout_seconds must be finite and positive")
    timeout_nanoseconds = timeout_seconds * _NANOSECONDS_PER_SECOND
    if not math.isfinite(timeout_nanoseconds):
        raise ValueError("timeout_seconds is too large to represent")
    return CandidateJobBudget(
        wall_time_ns=math.ceil(timeout_nanoseconds),
        input_bytes=_MAX_INPUT_BYTES,
        payload_output_bytes=2 * _MAX_STREAM_BYTES,
        stdout_head_bytes=_MAX_STREAM_BYTES,
        stderr_head_bytes=_MAX_STREAM_BYTES,
    )


def settings_fingerprint(
    *,
    settings: EvaluationSettings,
    manifest_sha256: str,
    selected_sample_path: Path,
) -> str:
    sample_digest = hashlib.sha256(
        selected_sample_path.read_bytes()
    ).hexdigest()
    payload = {
        "manifest_sha256": manifest_sha256,
        "selected_sample_sha256": sample_digest,
        "timeout_seconds": settings.timeout_seconds,
        "worker_count": settings.worker_count,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def attempt_identity(fingerprint: str) -> EvaluationAttemptIdentity:
    return EvaluationAttemptIdentity(
        attempt_id=uuid5(_ATTEMPT_NAMESPACE, fingerprint.encode())
    )


def load_humaneval_tasks(
    snapshot_path: Path,
    task_ids: tuple[str, ...],
) -> dict[str, HumanEvalTask]:
    selected = set(task_ids)
    rows = [
        row
        for row in load_humaneval_rows(snapshot_path=snapshot_path)
        if str(row["task_id"]) in selected
    ]
    tasks = {task.task_id: task for task in parse_humaneval_dataset(rows)}
    missing = selected.difference(tasks)
    if missing:
        raise ValueError(
            "HumanEval snapshot is missing tasks: "
            + ", ".join(sorted(missing))
        )
    return tasks


def _samples_per_task(selected: pl.DataFrame) -> int:
    counts = selected.group_by("task_id").len().get_column("len")
    unique = counts.unique().to_list()
    if len(unique) != 1:
        raise ValueError(
            "selected sample must contain the same number of generations "
            f"for every task; observed counts {sorted(unique)}"
        )
    return int(unique[0])


def _ordered_rows(selected: pl.DataFrame) -> list[dict[str, object]]:
    task_ids = selected.get_column("task_id").unique().sort().to_list()
    repeats = _samples_per_task(selected)
    rows_by_task: dict[str, list[dict[str, object]]] = {}
    for task_id in task_ids:
        task_rows = selected.filter(pl.col("task_id") == task_id).sort(
            list(_CELL_SORT_COLUMNS)
        )
        if task_rows.height != repeats:
            raise ValueError(
                f"task {task_id!r} has {task_rows.height} rows, expected "
                f"{repeats}"
            )
        rows_by_task[task_id] = list(task_rows.iter_rows(named=True))
    ordered: list[dict[str, object]] = []
    for task_id in task_ids:
        ordered.extend(rows_by_task[task_id])
    return ordered


def _corpus_provenance(
    row: dict[str, object],
    *,
    manifest_sha256: str,
) -> CorpusSampleProvenance:
    sample_id = str(row["sample_id"])
    return CorpusSampleProvenance(
        source_identity=EvaluationSourceIdentity(
            namespace=_WORKFLOW_CACHE_NAMESPACE,
            value=sample_id,
        ),
        source_reference=StoredRecordReference(
            reference=ObjectReference.for_record(
                _CORPUS_ROW_SCHEMA,
                {
                    "manifest_sha256": manifest_sha256,
                    "sample_id": sample_id,
                },
            ),
            schema_version=1,
        ),
        dataset=DatasetCoordinate(
            dataset_id=_DATASET_ID,
            version=_DATASET_VERSION,
        ),
        row_id=sample_id,
    )


def _materialized_candidates(
    row: dict[str, object],
    *,
    preprocessing: PreprocessingDefinitionCoordinate,
) -> tuple[MaterializedEvaluationCandidate, ...]:
    sample_id = str(row["sample_id"])
    candidates = row["code_candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(
            f"selected sample {sample_id!r} has no code candidates"
        )
    materialized: list[MaterializedEvaluationCandidate] = []
    for ordinal, source in enumerate(candidates):
        if not isinstance(source, str):
            raise TypeError("candidate source must be a string")
        materialized.append(
            MaterializedEvaluationCandidate(
                identity=EvaluationCandidateIdentity(
                    sample=EvaluationSampleIdentity(sample_id=sample_id),
                    preprocessing=preprocessing,
                    candidate_ordinal=ordinal,
                ),
                source=CodeArtifact(source=source),
                source_sha256=Sha256Digest(
                    hashlib.sha256(source.encode("utf-8")).hexdigest()
                ),
            )
        )
    return tuple(materialized)


def build_evaluation_plan(
    task_ids: tuple[str, ...],
    *,
    repeats: int,
) -> EvaluationPlan:
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
    dataset = DatasetCoordinate(
        dataset_id=_DATASET_ID,
        version=_DATASET_VERSION,
    )
    task_set = TaskSet(
        coordinate=TaskSetCoordinate(
            task_set_id=_TASK_SET_ID,
            version=_WORKFLOW_PLAN_VERSION,
            dataset=dataset,
        ),
        population=task_ids,
        selected=task_ids,
    )
    repeat_plan = RepeatPlan(
        coordinate=RepeatPlanCoordinate(
            repeat_plan_id=_REPEAT_PLAN_ID,
            version=_WORKFLOW_PLAN_VERSION,
        ),
        task_count=len(task_ids),
        repeats=repeats,
    )
    return EvaluationPlan(
        plan_id=_WORKFLOW_PLAN_ID,
        version=_WORKFLOW_PLAN_VERSION,
        task_set=task_set,
        repeat_plan=repeat_plan,
        procedure=EvaluationProcedure(
            preprocessing=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
            metrics=MetricsDefinition(
                definition_id=_WORKFLOW_METRICS_DEFINITION_ID,
                version=_WORKFLOW_METRICS_DEFINITION_VERSION,
                questions=(question,),
            ),
        ),
        aggregation=AggregationPolicy(
            question=question_coordinate,
            value="passed_count",
            statistic=AggregationStatistic.MEAN,
        ),
    )


def _compute_limits(
    *,
    slot_count: int,
    candidate_count: int,
    worker_count: int,
) -> tuple[AttemptLimits, WindowLimits, ShardLimits]:
    admitted = max(worker_count, 1)
    attempt = AttemptLimits(
        max_slots=max(1, slot_count),
        max_materialized_candidates=max(1, candidate_count),
        max_admitted_jobs=max(1, candidate_count),
        max_retained_evidence_bytes=max(
            10_000_000,
            candidate_count * 512_000,
        ),
        max_projection_rows=max(
            5,
            candidate_count + slot_count + 4,
        ),
    )
    window = WindowLimits(
        max_preprocessing_slots=1,
        max_cache_keys=min(
            attempt.max_materialized_candidates,
            max(admitted * 4, admitted, 1),
        ),
        max_admitted_jobs=admitted,
        max_record_assemblies=1,
        max_projection_rows=min(
            attempt.max_projection_rows,
            max(admitted * 8, 8),
        ),
    )
    shard = ShardLimits(
        max_records=max(10, slot_count),
        max_uncompressed_bytes=max(10_000_000, slot_count * 512_000),
    )
    return attempt, window, shard


def build_task_difficulty_batch_request(
    selected: pl.DataFrame,
    *,
    snapshot_path: Path,
    manifest_sha256: str,
    settings: EvaluationSettings,
    runtime: EvaluationRuntimeIdentity,
    attempt: EvaluationAttemptIdentity,
) -> EvaluationBatchRequest:
    ordered_rows = _ordered_rows(selected)
    task_ids = tuple(
        dict.fromkeys(str(row["task_id"]) for row in ordered_rows)
    )
    repeats = _samples_per_task(selected)
    plan = build_evaluation_plan(task_ids, repeats=repeats)
    tasks = load_humaneval_tasks(snapshot_path, task_ids)
    preprocessing = _exhaustive_preprocessing_coordinate()

    inputs: list[FrozenCandidateEvaluationInput] = []
    total_candidates = 0
    for task_id in task_ids:
        task = tasks[task_id]
        task_rows = [
            row for row in ordered_rows if str(row["task_id"]) == task_id
        ]
        for repeat_index, row in enumerate(task_rows):
            decoder_output = row.get("decoder_output")
            raw_text = (
                decoder_output
                if isinstance(decoder_output, str) and decoder_output.strip()
                else ""
            )
            sample = EvaluationSample(
                metadata=EvaluationSampleMetadata(
                    identity=EvaluationSampleIdentity(
                        sample_id=str(row["sample_id"])
                    ),
                    task_id=task_id,
                    provenance=_corpus_provenance(
                        row,
                        manifest_sha256=manifest_sha256,
                    ),
                ),
                raw_input=TextArtifact(text=raw_text),
                auxiliary_artifacts=(
                    EvaluationSampleAuxiliaryArtifact(
                        trace_key="task",
                        artifact=JsonArtifact(
                            payload=task.model_dump(mode="json")
                        ),
                    ),
                ),
            )
            candidates = _materialized_candidates(
                row,
                preprocessing=preprocessing,
            )
            total_candidates += len(candidates)
            inputs.append(
                FrozenCandidateEvaluationInput(
                    slot=EvaluationSlotIdentity(
                        task_set=plan.task_set.coordinate,
                        repeat_plan=plan.repeat_plan.coordinate,
                        task_id=task_id,
                        repeat_index=repeat_index,
                    ),
                    sample=sample,
                    preprocessing=preprocessing,
                    candidates=candidates,
                )
            )

    attempt_limits, window_limits, shard_limits = _compute_limits(
        slot_count=len(inputs),
        candidate_count=total_candidates,
        worker_count=settings.worker_count,
    )
    return EvaluationBatchRequest(
        attempt=attempt,
        plan=plan,
        runtime=runtime,
        cache_namespace=_WORKFLOW_CACHE_NAMESPACE,
        inputs=tuple(inputs),
        record_placement=RecordPlacement.OBJECT_STORE,
        projections=(ProjectionRequest(kind=ProjectionKind.METRIC_RECORDS),),
        attempt_limits=attempt_limits,
        window_limits=window_limits,
        shard_limits=shard_limits,
        job_budget=candidate_job_budget(settings.timeout_seconds),
    )


def build_preflight_batch_request_for_task(
    task: HumanEvalTask,
    *,
    settings: EvaluationSettings,
    runtime: EvaluationRuntimeIdentity,
    manifest_sha256: str,
) -> EvaluationBatchRequest:
    """Build a one-sample batch for runtime preflight on a known task."""

    preprocessing = _exhaustive_preprocessing_coordinate()
    plan = build_evaluation_plan((task.task_id,), repeats=1)
    sample = EvaluationSample(
        metadata=EvaluationSampleMetadata(
            identity=EvaluationSampleIdentity(sample_id="runtime-preflight"),
            task_id=task.task_id,
            provenance=_corpus_provenance(
                {"sample_id": "runtime-preflight"},
                manifest_sha256=manifest_sha256,
            ),
        ),
        raw_input=TextArtifact(text=""),
        auxiliary_artifacts=(
            EvaluationSampleAuxiliaryArtifact(
                trace_key="task",
                artifact=JsonArtifact(payload=task.model_dump(mode="json")),
            ),
        ),
    )
    candidates = _materialized_candidates(
        {
            "sample_id": "runtime-preflight",
            "code_candidates": [task.ground_truth_code],
        },
        preprocessing=preprocessing,
    )
    return EvaluationBatchRequest(
        attempt=EvaluationAttemptIdentity(
            attempt_id=uuid5(_ATTEMPT_NAMESPACE, b"runtime-preflight")
        ),
        plan=plan,
        runtime=runtime,
        cache_namespace=_WORKFLOW_CACHE_NAMESPACE,
        inputs=(
            FrozenCandidateEvaluationInput(
                slot=EvaluationSlotIdentity(
                    task_set=plan.task_set.coordinate,
                    repeat_plan=plan.repeat_plan.coordinate,
                    task_id=task.task_id,
                    repeat_index=0,
                ),
                sample=sample,
                preprocessing=preprocessing,
                candidates=candidates,
            ),
        ),
        record_placement=RecordPlacement.OBJECT_STORE,
        projections=(ProjectionRequest(kind=ProjectionKind.METRIC_RECORDS),),
        attempt_limits=AttemptLimits(
            max_slots=1,
            max_materialized_candidates=max(1, len(candidates)),
            max_admitted_jobs=max(1, len(candidates)),
            max_retained_evidence_bytes=10_000_000,
            max_projection_rows=8,
        ),
        window_limits=WindowLimits(
            max_preprocessing_slots=1,
            max_cache_keys=max(1, len(candidates)),
            max_admitted_jobs=1,
            max_record_assemblies=1,
            max_projection_rows=8,
        ),
        shard_limits=ShardLimits(
            max_records=4,
            max_uncompressed_bytes=10_000_000,
        ),
        job_budget=candidate_job_budget(settings.timeout_seconds),
    )


def evaluation_read_limits(
    *,
    sample_count: int,
    candidate_count: int,
) -> EvaluationReadLimits:
    from dr_store import BundleReadLimits

    return EvaluationReadLimits(
        bundle=BundleReadLimits(
            manifest_max_bytes=1 << 20,
            manifest_max_depth=64,
            max_artifacts=max(32, sample_count // 50 + 16),
            max_bytes_per_artifact=1 << 30,
            max_total_artifact_bytes=4 << 30,
        ),
        max_sample_records=max(1, sample_count, candidate_count),
        max_object_reads=max(1, sample_count),
        max_reference_depth=10,
    )


def _metric_values_by_name(
    row: MetricRecordProjectionRow,
) -> dict[str, object]:
    return {value.name: value.value for value in row.values}


def _projection_row_to_record(
    row: MetricRecordProjectionRow,
    *,
    selected_by_sample: dict[str, dict[str, object]],
    runtime_identity_json: str,
    settings: EvaluationSettings,
    source_by_sample_and_ordinal: dict[tuple[str, int], str],
) -> dict[str, object]:
    sample_id = row.candidate.sample.sample_id
    candidate_index = row.candidate.candidate_ordinal
    sample_row = selected_by_sample[sample_id]
    identity_values = {
        "metric_schema_version": 0,
        "metric_name": str(row.question.metric),
        "metric_version": "0",
        "metrics_definition_id": _WORKFLOW_METRICS_DEFINITION_ID,
        "metrics_definition_version": _WORKFLOW_METRICS_DEFINITION_VERSION,
        "runtime_identity": runtime_identity_json,
        "evaluation_worker_count": settings.worker_count,
        "evaluation_timeout_seconds": settings.timeout_seconds,
    }
    if row.status is RecordStatus.OPERATOR_FAILURE:
        return {
            **identity_values,
            "sample_id": sample_id,
            "task_id": sample_row["task_id"],
            "generation_mode": sample_row["generation_mode"],
            "budget_mode": sample_row["budget_mode"],
            "model_key": sample_row["model_key"],
            "candidate_index": candidate_index,
            "candidate_source": source_by_sample_and_ordinal.get(
                (sample_id, candidate_index),
                "",
            ),
            "metric_status": "operator_failure",
            "candidate_passed": None,
            "failure_type": "OperatorFailure",
            "failure_message": "metric operator failure",
        }
    values = _metric_values_by_name(row)
    passed_count = _metric_int(values, "passed_count")
    total_cases = _metric_int(values, "total_cases")
    coverage_complete = bool(values.get("coverage_complete", False))
    return {
        **identity_values,
        "sample_id": sample_id,
        "task_id": sample_row["task_id"],
        "generation_mode": sample_row["generation_mode"],
        "budget_mode": sample_row["budget_mode"],
        "model_key": sample_row["model_key"],
        "candidate_index": candidate_index,
        "candidate_source": source_by_sample_and_ordinal.get(
            (sample_id, candidate_index),
            "",
        ),
        "metric_status": "measured",
        "total_cases": total_cases,
        "passed_count": passed_count,
        "failed_count": _metric_int(values, "failed_count"),
        "error_count": _metric_int(values, "error_count"),
        "timeout_count": _metric_int(values, "timeout_count"),
        "coverage_complete": coverage_complete,
        "function_count": _metric_int(values, "function_count"),
        "best_function_name": values.get("best_function_name"),
        "candidate_passed": (
            coverage_complete and passed_count == total_cases
        ),
        "failure_type": None,
        "failure_message": None,
    }


def export_candidate_results(
    bundle_path: Path,
    selected: pl.DataFrame,
    output_path: Path,
    *,
    settings: EvaluationSettings,
    runtime_identity_json: str,
    limits: EvaluationReadLimits | None = None,
    object_store: ObjectStore | None = None,
) -> pl.DataFrame:
    candidate_count = int(selected.get_column("candidate_count").sum())
    if limits is None:
        limits = evaluation_read_limits(
            sample_count=selected.height,
            candidate_count=candidate_count,
        )
    _, rows = read_evaluation_projection(
        bundle_path,
        ProjectionKind.METRIC_RECORDS,
        limits=limits,
    )
    selected_by_sample = {
        str(row["sample_id"]): row for row in selected.iter_rows(named=True)
    }
    source_by_sample_and_ordinal: dict[tuple[str, int], str] = {}
    for row in selected.iter_rows(named=True):
        sample_id = str(row["sample_id"])
        candidates = row["code_candidates"]
        if isinstance(candidates, list):
            for index, source in enumerate(candidates):
                if isinstance(source, str):
                    source_by_sample_and_ordinal[(sample_id, index)] = source
    metric_rows = [
        row for row in rows if isinstance(row, MetricRecordProjectionRow)
    ]
    records = [
        _projection_row_to_record(
            row,
            selected_by_sample=selected_by_sample,
            runtime_identity_json=runtime_identity_json,
            settings=settings,
            source_by_sample_and_ordinal=source_by_sample_and_ordinal,
        )
        for row in metric_rows
    ]
    frame = pl.DataFrame(records, infer_schema_length=None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_path)
    return frame


async def bundle_is_complete(
    bundle_path: Path,
    *,
    object_store: ObjectStore | None,
    limits: EvaluationReadLimits,
) -> bool:
    if not bundle_path.is_dir():
        return False
    manifest_path = bundle_path / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        restored = await restore_evaluation_attempt(
            bundle_path,
            object_store=object_store,
            limits=limits,
        )
    except Exception:
        return False
    return restored.attempt.completeness is AttemptCompleteness.COMPLETE


def runtime_identity_from_executor(
    executor: Executor,
) -> EvaluationRuntimeIdentity:
    runtime = getattr(executor, "runtime", None)
    if runtime is not None:
        return EvaluationRuntimeIdentity(document=runtime.describe().id_doc)
    return EvaluationRuntimeIdentity(
        document=build_identity_document(
            schema="dr-code/task-difficulty-runtime",
            schema_version=1,
            payload={"executor": type(executor).__name__},
        )
    )


def runtime_identity_with_packages(
    base: EvaluationRuntimeIdentity,
    packages: dict[str, object],
) -> EvaluationRuntimeIdentity:
    return EvaluationRuntimeIdentity(
        document=build_identity_document(
            schema="dr-code/task-difficulty-runtime",
            schema_version=1,
            payload={
                "runtime": base.document.to_json_dict(),
                "packages": packages,
            },
        )
    )


def runtime_identity_json(runtime: EvaluationRuntimeIdentity) -> str:
    return json.dumps(
        runtime.document.to_json_dict(),
        sort_keys=True,
        separators=(",", ":"),
    )


def _build_python_execution_job(
    *,
    driver_source: str,
    input_json: str,
    timeout_seconds: float,
) -> ExecutionJob:
    payload = json.loads(input_json)
    timeout_nanoseconds = timeout_seconds * _NANOSECONDS_PER_SECOND
    return ExecutionJob(
        job_id=JobId(uuid5(_ATTEMPT_NAMESPACE, driver_source.encode())),
        target=UntrustedPythonTarget(
            driver_source=driver_source,
            request=build_identity_document(
                schema=_EXECUTION_REQUEST_SCHEMA,
                schema_version=_EXECUTION_REQUEST_SCHEMA_VERSION,
                payload=payload,
            ),
            containment_profile=ContainmentProfile.PROCESS_BOUNDARY_ONLY,
        ),
        env=EnvGrant.fixed(_EXECUTION_ENVIRONMENT),
        budgets=Budgets(
            wall_time=FiniteDurationLimit(
                max_ns=math.ceil(timeout_nanoseconds)
            ),
            input_bytes=FiniteByteLimit(max_bytes=_MAX_INPUT_BYTES),
            payload_output=FiniteOutput(
                max_bytes=2 * _MAX_STREAM_BYTES,
                overflow_policy=OutputOverflowPolicy.FAIL,
                retention=PayloadRetentionBudget(
                    stdout=StreamRetentionBudget(
                        head_bytes=_MAX_STREAM_BYTES,
                        tail_bytes=0,
                    ),
                    stderr=StreamRetentionBudget(
                        head_bytes=_MAX_STREAM_BYTES,
                        tail_bytes=0,
                    ),
                ),
            ),
        ),
    )


def _stream_text(payload_stream: object, label: str) -> str:
    head = getattr(payload_stream, "head", b"")
    if not isinstance(head, (bytes, bytearray)):
        raise TypeError(f"{label} head must be bytes")
    return bytes(head).decode("utf-8", errors="replace")


def _interpret_completed_execution(
    execution: CompletedExecution,
) -> tuple[int, str, str]:
    result = execution.result
    outcome = result.outcome
    attribution = result.attribution
    if isinstance(outcome, ExitedOutcome):
        return (
            outcome.exit_code,
            _stream_text(result.payload_outputs.stdout, "stdout"),
            _stream_text(result.payload_outputs.stderr, "stderr"),
        )
    if isinstance(outcome, SignaledOutcome):
        raise RuntimeError(
            "execution died on signal "
            f"{outcome.signal_number}: "
            + _stream_text(result.payload_outputs.stderr, "stderr")
        )
    if isinstance(outcome, BudgetExceededOutcome):
        if outcome.axis is BudgetAxis.WALL_TIME:
            raise RuntimeError("execution exceeded its wall-clock budget")
        raise RuntimeError("execution exceeded its payload output budget")
    if isinstance(outcome, ProtocolFailedOutcome):
        if attribution.owner is FailureOwner.PAYLOAD:
            raise RuntimeError(
                "execution ended before completing its protected protocol "
                f"({outcome.failure_code}): "
                + _stream_text(result.payload_outputs.stderr, "stderr")
            )
    raise RuntimeError(
        "execution produced no payload-owned outcome: "
        f"{outcome.kind} attributed to {attribution.owner}"
        + (f" ({attribution.detail})" if attribution.detail else "")
    )


def probe_runtime_packages(executor: Executor) -> dict[str, object]:
    completed = executor.run(
        _build_python_execution_job(
            driver_source=_RUNTIME_PROBE_SOURCE,
            input_json="{}",
            timeout_seconds=_PROBE_TIMEOUT_SECONDS,
        )
    )
    returncode, stdout, stderr = _interpret_completed_execution(completed)
    if returncode != 0:
        raise RuntimeError(
            "evaluation runtime dependency probe failed: " + stderr.strip()
        )
    try:
        package_identity = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "evaluation runtime dependency probe returned invalid JSON"
        ) from exc
    if not isinstance(package_identity, dict):
        raise RuntimeError(
            "evaluation runtime dependency probe returned non-object JSON"
        )
    return package_identity


def load_run_manifest(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("run manifest must be a JSON object")
    return payload


def write_run_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def manifest_matches(
    stored: dict[str, object] | None,
    *,
    fingerprint: str,
) -> bool:
    if stored is None:
        return False
    return stored.get("settings_fingerprint") == fingerprint


__all__ = [
    "EvaluationBatchResult",
    "attempt_identity",
    "build_preflight_batch_request_for_task",
    "build_task_difficulty_batch_request",
    "bundle_is_complete",
    "candidate_job_budget",
    "evaluate_batch",
    "evaluation_read_limits",
    "export_candidate_results",
    "load_run_manifest",
    "manifest_matches",
    "probe_runtime_packages",
    "runtime_identity_from_executor",
    "runtime_identity_json",
    "runtime_identity_with_packages",
    "settings_fingerprint",
    "write_run_manifest",
]
