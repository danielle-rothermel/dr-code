"""Resumable, content-bound HumanEval+ evaluation of corpus candidates."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sqlite3
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait,
)
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, cast

import pyarrow as pa
import pyarrow.parquet as pq
from dr_exec import (
    EXECUTOR_IDENTITY,
    Attribution,
    BatchRequest,
    BatchResult,
    Budgets,
    ContainmentProfile,
    EnvironmentGrant,
    ExecutorFailure,
    Outcome,
    PythonRuntime,
    Records,
    run_untrusted_python,
)
from pydantic import ValidationError

from dr_code.corpus.preprocessing_artifacts import (
    PROJECTED_ARTIFACT_SCHEMAS,
    file_sha256,
)
from dr_code.corpus.coordinate_validation import (
    CoordinateValidationError,
    validate_evaluation_coordinates,
    validate_preprocessing_coordinates,
)
from dr_code.corpus.candidate_evaluation_contract import (
    CANDIDATE_EVALUATION_COORDINATE_FIELDS,
    CANDIDATE_EVALUATION_MANIFEST_FIELDS,
    CANDIDATE_EVALUATION_SCHEMA_VERSION,
    CANDIDATE_RESULT_FACT_FIELDS,
    MEMBERSHIP_SCHEMA,
    RESULTS_SCHEMA,
    CandidateEvaluationContractError,
    candidate_evaluation_identity,
    candidate_evaluation_key,
    canonical_candidate_result,
    preprocessing_run_identity,
)
from dr_code.corpus.preprocessing_contract import (
    PREPROCESSING_INPUT_FIELDS,
    PREPROCESSING_MANIFEST_FIELDS,
    PREPROCESSING_MANIFEST_SCHEMA_VERSION,
)
from dr_code.corpus.preprocessing_run import (
    CorpusRunError,
    validate_preprocessing_derivation,
    validate_preprocessing_relations,
)
from dr_code.corpus.evaluation_generation import (
    EvaluationGeneration,
    EvaluationGenerationError,
    MANIFEST_FILENAME,
    MEMBERSHIP_FILENAME,
    RESULTS_FILENAME,
    StagedCurrentSwitch,
    publish_generation_directory,
    publish_staged_current_switch,
    resolve_current_generation,
    staged_current_switch,
    staged_generation_directory,
    validate_captured_generation,
    validate_evaluation_root,
)
from dr_code.corpus.evaluation_relations import (
    EvaluationRelationsError,
    validate_evaluation_relations,
)
from dr_code.corpus.runtime_provenance import (
    checkout_source_tree_sha256,
    installed_environment_provenance,
)
from dr_code.corpus.stable_files import StableFile, stable_files
from dr_code.corpus.output_paths import (
    UnsafeOutputPathError,
    validate_output_path,
    validate_owned_tree,
)
from dr_code.eval import (
    EvaluationProcedureConfig,
    EvaluationProcedureDefinition,
    MetricExtractionConfig,
    MetricExtractionDefinition,
    MetricQuestionBinding,
    OperatorCoordinates,
    PreprocessingConfig,
    humaneval_task_identity,
)
from dr_code.eval.facts import MetricRecord, RecordStatus
from dr_code.eval.immutable_json import thaw_json
from dr_code.eval.resolved_versions import resolved_operator_identity
from dr_code.humaneval.batch_runner import (
    HUMANEVAL_ENVIRONMENT,
    HUMANEVAL_PROFILE,
    HUMANEVAL_RUNTIME,
    PRODUCTION_EXECUTOR,
    BatchExecutor,
    evaluate_human_eval_code,
)
from dr_code.humaneval.profiles import (
    DEFAULT_HUMANEVAL_TIMEOUT_SECONDS,
    HUMANEVAL_METRICS_PROFILE_ID,
    HUMANEVAL_METRICS_PROFILE_VERSION,
)
from dr_code.humaneval.sampling import (
    DEFAULT_HUMAN_EVAL_DATASET_NAME,
    DEFAULT_HUMAN_EVAL_DATASET_SPLIT,
    DEFAULT_HUMAN_EVAL_HF_REVISION,
    DEFAULT_HUMAN_EVAL_SNAPSHOT_SHA256,
    load_human_eval_snapshot_rows_bytes,
)
from dr_code.humaneval.task import HumanEvalTask, parse_human_eval_dataset
from dr_code.metrics import MetricName, extract_metrics
from dr_code.metrics.operators.code_test import CodeTestResult
from dr_code.preprocessing.candidate_identity import candidate_id_for_source
from dr_code.trace import (
    CodeArtifact,
    JsonArtifact,
    Trace,
    TraceProducer,
)

STATE_FILENAME: Final = "candidate_evaluation.sqlite3"
RECORDS_DIRNAME: Final = "execution_records"
SCHEMA_VERSION: Final = CANDIDATE_EVALUATION_SCHEMA_VERSION
RUNNER_IDENTITY: Final = EXECUTOR_IDENTITY
"""The production runner identity is dr-exec's executor identity.

This is the *executor* half of the runner identity — spawn/lifecycle/capture
provenance. It is machine-invariant by design (it changes only on a dr-exec
version bump), so it never substitutes for the cross-machine runtime identity
(``runtime_provenance``), which answers a different question and is folded in
separately.
"""

# The infrastructure-failure vocabulary persisted in ``failure_type`` is
# dr-exec's attribution literals: a run that produced a result carries its
# ``Attribution``; a raised ``ExecutorFailure`` (no result to attribute) is
# recorded as the ``executor`` attribution literal. dr-code never derives these
# from field or class names — the literals are dr-exec's pinned contract.
_RETRIABLE_ATTRIBUTIONS: Final = frozenset(
    {Attribution.CHANNEL, Attribution.MACHINE}
)
"""Attributions of a produced run that are transient infrastructure faults.

A protocol-channel corruption (``channel``) or a non-ENOENT spawn failure
(``machine``, e.g. EACCES) can clear on a retry. ``absence`` (a missing
interpreter, ENOENT) is *not* transient and is never retried; ``budget`` and
``payload`` outcomes are scored against the candidate, never infrastructure.
"""
_LEASE_SECONDS: Final = 300.0
_PREFLIGHT_TIMEOUT_SECONDS: Final = 10.0
_ADMISSION_BATCH_SIZE: Final = 1_000
_METRICS_PROFILE: Final = (
    f"{HUMANEVAL_METRICS_PROFILE_ID}@{HUMANEVAL_METRICS_PROFILE_VERSION}"
)
_FACT_FIELDS: Final = CANDIDATE_RESULT_FACT_FIELDS
_RESULT_EVIDENCE_SCHEMA: Final = "dr_code.corpus.candidate_result_evidence"
_RESULT_EVIDENCE_SCHEMA_VERSION: Final = 1
_WORK_COLUMNS: Final = (
    "evaluation_key",
    "task_id",
    "task_identity",
    "source_sha256",
    "candidate_source",
    "status",
    "attempt_count",
    "record_status",
    "failure_type",
    "failure_message",
    "values_json",
    "completed_at",
    "reused_from_manifest_sha256",
    "owner_lease_id",
    "result_evidence_sha256",
)


class CandidateEvaluationError(ValueError):
    """Corpus, state, reuse, or output violates the evaluator contract."""


@dataclass(frozen=True, slots=True)
class EvaluationArtifacts:
    output_dir: Path
    membership_path: Path
    results_path: Path
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class _StagedEvaluationArtifacts:
    membership_path: Path
    results_path: Path
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class _EvaluationConfig:
    preprocessing_config: PreprocessingConfig
    metric_definition: MetricExtractionDefinition
    metric_config: MetricExtractionConfig
    procedure_config: EvaluationProcedureConfig
    trace_producer: TraceProducer
    operator: OperatorCoordinates
    question_identity_hash: str
    operator_name: str
    operator_version: str


@dataclass(frozen=True, slots=True)
class _Work:
    evaluation_key: str
    task_id: str
    task_identity: str
    source_sha256: str
    candidate_source: str


@dataclass(frozen=True, slots=True)
class _InfrastructureFailure:
    """An execution failure the evaluator handles outside candidate scoring.

    ``failure_type`` is a dr-exec attribution literal (never a class or field
    name). ``retriable`` is decided at classification time from the two
    adjudicated sources: a produced run's channel/machine attribution and a
    raised ``ExecutorFailure`` are retriable; ``absence`` is a non-transient
    infrastructure failure that terminates immediately.
    """

    failure_type: str
    failure_message: str
    retriable: bool


class _AttributionObservingExecutor:
    """Wraps the injected batch executor to record each run's outcome.

    The metrics lane scores an executor/channel/machine/absence outcome as
    candidate case data (a ``MEASURED`` record). The corpus evaluator needs the
    raw attribution to decide infrastructure retries, so this wrapper captures
    the most recent run outcome; ``_measure_work`` reads it after
    ``extract_metrics`` returns and reclassifies channel/machine/absence runs as
    infrastructure failures. The wrapper adds nothing run-varying: it only
    observes, delegating every call unchanged to the real executor.
    """

    __slots__ = ("_executor", "_last_outcome")

    def __init__(self, executor: BatchExecutor) -> None:
        self._executor = executor
        self._last_outcome: Outcome | None = None

    @property
    def last_outcome(self) -> Outcome | None:
        return self._last_outcome

    def run_batch(
        self,
        request: BatchRequest,
        *,
        profile: ContainmentProfile,
        budgets: Budgets,
        records: Records,
        runtime: PythonRuntime,
        environment: EnvironmentGrant,
    ) -> BatchResult:
        result = self._executor.run_batch(
            request,
            profile=profile,
            budgets=budgets,
            records=records,
            runtime=runtime,
            environment=environment,
        )
        self._last_outcome = result.run.outcome
        return result


@dataclass(frozen=True, slots=True)
class _ValidatedPreprocessingRun:
    coordinates: dict[str, object]
    config: PreprocessingConfig


@dataclass(frozen=True, slots=True)
class _ReuseSource:
    manifest_path: Path
    membership_path: Path
    results_path: Path
    manifest_sha256: str
    membership_sha256: str
    results_sha256: str
    membership_rows: int
    result_rows: int

    def descriptor(self) -> dict[str, object]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "candidate_membership_sha256": self.membership_sha256,
            "candidate_results_sha256": self.results_sha256,
            "membership_rows": self.membership_rows,
            "result_rows": self.result_rows,
        }


_REUSE_COORDINATES: Final = (
    "schema_version",
    "dataset",
    "snapshot_sha256",
    "metric_extraction_definition_ref",
    "metric_extraction_config",
    "metric_extraction_definition_identity",
    "metric_extraction_config_identity",
    "evaluation_procedure_definition_ref",
    "evaluation_procedure_config",
    "evaluation_procedure_definition_identity",
    "evaluation_procedure_config_identity",
    "trace_producer",
    "operator_coordinates",
    "question_identity_hash",
    "operator_name",
    "operator_version",
    "runner_identity",
    "runtime_identity",
    "host_runtime",
    "installed_environment",
    "trusted_source_sha256",
)
_MANIFEST_FIELDS: Final = CANDIDATE_EVALUATION_MANIFEST_FIELDS


def humaneval_metric_definition() -> MetricExtractionDefinition:
    """Return the canonical facts-first metric declaration."""

    return MetricExtractionDefinition(
        definition_id=HUMANEVAL_METRICS_PROFILE_ID,
        version=HUMANEVAL_METRICS_PROFILE_VERSION,
        questions=(
            MetricQuestionBinding(
                metric=MetricName.CODE_TEST,
                on="candidate",
                settings=(
                    ("task_key", "task"),
                    (
                        "timeout_seconds",
                        DEFAULT_HUMANEVAL_TIMEOUT_SECONDS,
                    ),
                ),
            ),
        ),
    )


def evaluate_preprocessing_candidates(
    *,
    preprocessing_run: Path | str,
    corpus_path: Path | str,
    output_dir: Path | str,
    snapshot_path: Path | str,
    max_workers: int = 4,
    max_infrastructure_retries: int = 2,
    executor: BatchExecutor | None = None,
    runner_identity: str | None = None,
    reuse_results_from: Sequence[Path | str] = (),
) -> EvaluationArtifacts:
    """Evaluate deduplicated work while retaining every corpus occurrence."""

    if max_workers < 1:
        raise CandidateEvaluationError("max_workers must be at least 1")
    if max_infrastructure_retries < 0:
        raise CandidateEvaluationError(
            "max_infrastructure_retries must be non-negative"
        )
    resolved_runner_identity = _resolve_runner_identity(
        executor, runner_identity
    )
    runner = PRODUCTION_EXECUTOR if executor is None else executor
    run_dir = Path(preprocessing_run).expanduser().resolve(strict=True)
    corpus_file = Path(corpus_path).expanduser().resolve(strict=True)
    snapshot_file = Path(snapshot_path).expanduser().resolve(strict=True)
    requested_destination = Path(output_dir).expanduser()
    try:
        requested_destination = validate_output_path(
            requested_destination,
            label="candidate evaluation output root",
        )
        validate_owned_tree(
            requested_destination,
            label="candidate evaluation output root",
        )
        validate_evaluation_root(requested_destination)
    except (EvaluationGenerationError, UnsafeOutputPathError) as exc:
        raise CandidateEvaluationError(str(exc)) from exc
    destination = requested_destination
    host_runtime = _host_runtime_coordinates()
    installed_environment = installed_environment_provenance()
    trusted_source = _trusted_source_fingerprints()
    runtime_identity = _runtime_identity(
        runner_identity=resolved_runner_identity,
        host_runtime=host_runtime,
        installed_environment=installed_environment,
        trusted_source_sha256=trusted_source,
    )
    capture_paths: dict[str, Path | str] = {
        "corpus": corpus_file,
        "snapshot": snapshot_file,
        "preprocessing_manifest": run_dir / "manifest.json",
        **{
            f"preprocessing_{relation}": run_dir / f"{relation}.parquet"
            for relation in PROJECTED_ARTIFACT_SCHEMAS
        },
    }
    reuse_generations = []
    for index, raw_path in enumerate(reuse_results_from):
        requested_reuse = Path(raw_path).expanduser()
        if requested_reuse.resolve(strict=True) == destination:
            raise CandidateEvaluationError(
                "evaluation output cannot reuse results from itself"
            )
        try:
            generation = resolve_current_generation(requested_reuse)
        except EvaluationGenerationError as exc:
            raise CandidateEvaluationError(str(exc)) from exc
        reuse_generations.append(generation)
        capture_paths.update(
            {
                f"reuse_{index}_manifest": generation.manifest_path,
                f"reuse_{index}_membership": generation.membership_path,
                f"reuse_{index}_results": generation.results_path,
            }
        )
    try:
        with stable_files(capture_paths) as captured:
            for index, generation in enumerate(reuse_generations):
                validate_captured_generation(
                    generation,
                    manifest_sha256=captured[f"reuse_{index}_manifest"].sha256,
                    membership_sha256=captured[
                        f"reuse_{index}_membership"
                    ].sha256,
                    results_sha256=captured[f"reuse_{index}_results"].sha256,
                )
            return _evaluate_captured_inputs(
                captured=captured,
                reuse_count=len(reuse_generations),
                destination=destination,
                runner=runner,
                production_runner=executor is None,
                runner_identity=resolved_runner_identity,
                runtime_identity=runtime_identity,
                host_runtime=host_runtime,
                installed_environment=installed_environment,
                trusted_source=trusted_source,
                max_workers=max_workers,
                max_infrastructure_retries=max_infrastructure_retries,
            )
    except (ValueError, OSError, pa.ArrowException) as exc:
        if isinstance(exc, CandidateEvaluationError):
            raise
        raise CandidateEvaluationError(
            f"candidate evaluation input capture failed: {exc}"
        ) from exc


def _evaluate_captured_inputs(
    *,
    captured: Mapping[str, StableFile],
    reuse_count: int,
    destination: Path,
    runner: BatchExecutor,
    production_runner: bool,
    runner_identity: str,
    runtime_identity: str,
    host_runtime: Mapping[str, object],
    installed_environment: Mapping[str, object],
    trusted_source: Mapping[str, str],
    max_workers: int,
    max_infrastructure_retries: int,
) -> EvaluationArtifacts:
    preprocessing_relations = {
        relation: captured[f"preprocessing_{relation}"]
        for relation in PROJECTED_ARTIFACT_SCHEMAS
    }
    tasks = _load_tasks(captured["snapshot"])
    preprocessing_coordinates = _validate_preprocessing_run(
        manifest_file=captured["preprocessing_manifest"],
        relations=preprocessing_relations,
        corpus_file=captured["corpus"],
        installed_environment=installed_environment,
        snapshot_tasks=tasks,
    )
    config = _evaluation_config(preprocessing_coordinates.config)
    _validate_evaluation_config(config)
    base_coordinates = _immutable_coordinates(
        preprocessing_coordinates=preprocessing_coordinates.coordinates,
        preprocessing_config=preprocessing_coordinates.config,
        corpus_file=captured["corpus"],
        snapshot_file=captured["snapshot"],
        config=config,
        runner_identity=runner_identity,
        runtime_identity=runtime_identity,
        host_runtime=host_runtime,
        installed_environment=installed_environment,
        trusted_source=trusted_source,
        max_infrastructure_retries=max_infrastructure_retries,
    )
    reuse_sources = _load_reuse_sources(
        captured,
        reuse_count=reuse_count,
        expected=base_coordinates,
        preprocessing_config=preprocessing_coordinates.config,
    )
    destination.mkdir(parents=True, exist_ok=True)
    records_dir = destination / RECORDS_DIRNAME
    records_dir.mkdir(exist_ok=True)
    records = Records.directory(records_dir)
    if production_runner:
        _preflight_production(
            tasks=tasks, config=config, runner=runner, records=records
        )
    task_identities = {
        task_id: humaneval_task_identity(task)
        for task_id, task in tasks.items()
    }
    immutable = {
        **base_coordinates,
        "reuse_result_sources": [
            source.descriptor() for source in reuse_sources
        ],
    }
    connection = _open_state(destination / STATE_FILENAME)
    lease_id = uuid.uuid4().hex
    try:
        _acquire_lease(connection, lease_id)
        with _lease_heartbeat(connection, lease_id) as stop_heartbeat:
            _initialize_state(connection, immutable, lease_id=lease_id)
            _prepare_memberships(
                connection=connection,
                lease_id=lease_id,
                corpus_file=captured["corpus"].path,
                candidates_file=preprocessing_relations["candidates"].path,
                results_file=preprocessing_relations["results"].path,
                tasks=tasks,
                task_identities=task_identities,
                config=config,
                runtime_identity=runtime_identity,
                runner_identity=runner_identity,
            )
            _validate_completed_result_evidence(connection)
            _recover_work(
                connection,
                lease_id=lease_id,
                max_infrastructure_retries=max_infrastructure_retries,
            )
            _reuse_completed_results(
                connection,
                lease_id=lease_id,
                reuse_sources=reuse_sources,
                task_identities=task_identities,
                config=config,
                runtime_identity=runtime_identity,
                runner_identity=runner_identity,
            )
            _run_pending_work(
                connection=connection,
                lease_id=lease_id,
                tasks=tasks,
                config=config,
                max_workers=max_workers,
                max_infrastructure_retries=max_infrastructure_retries,
                runner=runner,
                records=records,
            )
            return _export_artifacts(
                connection,
                destination,
                immutable,
                config=config,
                corpus_path=captured["corpus"].path,
                candidates_path=preprocessing_relations["candidates"].path,
                lease_id=lease_id,
                reuse_sources=reuse_sources,
                stop_heartbeat=stop_heartbeat,
            )
    finally:
        _release_lease(connection, lease_id)
        connection.close()


def _evaluation_config(
    preprocessing: PreprocessingConfig,
) -> _EvaluationConfig:
    preprocessing = PreprocessingConfig.model_validate(
        preprocessing.model_dump(mode="python")
    )
    metric_definition = humaneval_metric_definition()
    metric_config = metric_definition.materialize()
    if (
        len(metric_definition.questions) != 1
        or len(metric_config.resolved_operator_versions) != 1
    ):
        raise CandidateEvaluationError(
            "HumanEval candidate evaluation requires exactly one question"
        )
    question = metric_definition.questions[0]
    # The materialized config's question carries the concrete, defaults-filled
    # settings (e.g. the code_test ``budgets`` block); the operator identity
    # hash folds those in, so the persisted operator coordinates must too.
    concrete_question = metric_config.questions[0]
    (
        question_identity_hash,
        operator_name,
        operator_version,
        operator_implementation_hash,
    ) = metric_config.resolved_operator_versions[0]
    expected_resolution = (
        question.identity_hash(),
        question.metric,
        *resolved_operator_identity(question.metric),
    )
    if (
        question_identity_hash,
        operator_name,
        operator_version,
        operator_implementation_hash,
    ) != expected_resolution:
        raise CandidateEvaluationError(
            "HumanEval candidate evaluation has stale question/operator "
            "coordinates"
        )
    procedure = EvaluationProcedureDefinition(
        definition_id="humaneval-candidate-evaluation",
        version="1",
    ).materialize(
        preprocessing=preprocessing,
        metric_extraction=metric_config,
    )
    return _EvaluationConfig(
        preprocessing_config=preprocessing,
        metric_definition=metric_definition,
        metric_config=metric_config,
        procedure_config=procedure,
        trace_producer=TraceProducer(
            producer_id=preprocessing.definition_ref.definition_id,
            version=preprocessing.definition_ref.version,
            definition_hash=preprocessing.definition_ref.identity_hash,
            preprocessing_config_hash=preprocessing.config_identity_hash,
            implementation_hash=preprocessing.implementation_hash,
        ),
        operator=OperatorCoordinates(
            name=operator_name,
            version=operator_version,
            implementation_hash=operator_implementation_hash,
            settings=tuple(concrete_question.settings_dict().items()),
        ),
        question_identity_hash=question_identity_hash,
        operator_name=operator_name,
        operator_version=operator_version,
    )


def _validate_evaluation_config(config: _EvaluationConfig) -> None:
    definition = config.metric_definition
    metric_config = config.metric_config
    if (
        len(definition.questions) != 1
        or len(metric_config.resolved_operator_versions) != 1
    ):
        raise CandidateEvaluationError(
            "HumanEval candidate evaluation requires exactly one question"
        )
    question = definition.questions[0]
    expected_resolution = (
        question.identity_hash(),
        question.metric,
        *resolved_operator_identity(question.metric),
    )
    expected_coordinates = expected_resolution[:3]
    if (
        metric_config.definition_ref.identity_hash
        != definition.identity_hash()
        or metric_config.resolved_operator_versions[0] != expected_resolution
        or (
            config.question_identity_hash,
            config.operator_name,
            config.operator_version,
        )
        != expected_coordinates
        or config.procedure_config.metric_extraction_config_hash
        != metric_config.config_identity_hash
        or config.procedure_config.preprocessing_config_hash
        != config.preprocessing_config.config_identity_hash
        or config.trace_producer
        != TraceProducer(
            producer_id=config.preprocessing_config.definition_ref.definition_id,
            version=config.preprocessing_config.definition_ref.version,
            definition_hash=(
                config.preprocessing_config.definition_ref.identity_hash
            ),
            preprocessing_config_hash=(
                config.preprocessing_config.config_identity_hash
            ),
            implementation_hash=(
                config.preprocessing_config.implementation_hash
            ),
        )
        or config.operator.question_identity_hash(on_key=question.on)
        != config.question_identity_hash
    ):
        raise CandidateEvaluationError(
            "HumanEval candidate evaluation has stale or forged "
            "question/operator coordinates"
        )


def _validate_preprocessing_run(
    *,
    manifest_file: StableFile,
    relations: Mapping[str, StableFile],
    corpus_file: StableFile,
    installed_environment: Mapping[str, object],
    snapshot_tasks: Mapping[str, HumanEvalTask],
) -> _ValidatedPreprocessingRun:
    manifest = _read_json_object(manifest_file.path, "preprocessing manifest")
    if manifest.get("schema_version") != PREPROCESSING_MANIFEST_SCHEMA_VERSION:
        raise CandidateEvaluationError(
            "preprocessing run requires schema_version 3"
        )
    if manifest.get("complete") is not True:
        raise CandidateEvaluationError("preprocessing run is not complete")
    if set(manifest) != PREPROCESSING_MANIFEST_FIELDS:
        raise CandidateEvaluationError(
            "preprocessing manifest schema does not match schema_version 3"
        )
    if manifest.get("installed_environment") != installed_environment:
        raise CandidateEvaluationError(
            "preprocessing run installed environment does not match the "
            "current environment"
        )
    input_coordinates = manifest["input"]
    if (
        not isinstance(input_coordinates, dict)
        or set(input_coordinates) != PREPROCESSING_INPUT_FIELDS
    ):
        raise CandidateEvaluationError(
            "preprocessing manifest input coordinates are invalid"
        )
    _validate_corpus_results_before_state(
        corpus_file.path,
        relations["results"].path,
        snapshot_tasks=snapshot_tasks,
    )
    _validate_corpus_input_coordinates(
        corpus_file,
        cast(dict[str, object], input_coordinates),
    )
    try:
        preprocessing_config = validate_preprocessing_coordinates(manifest)
    except CoordinateValidationError as exc:
        raise CandidateEvaluationError(str(exc)) from exc
    recorded_hashes = manifest.get("relation_sha256")
    if not isinstance(recorded_hashes, dict) or set(recorded_hashes) != set(
        PROJECTED_ARTIFACT_SCHEMAS
    ):
        raise CandidateEvaluationError(
            "preprocessing manifest has incomplete relation hashes"
        )
    relation_coordinates: dict[str, dict[str, object]] = {}
    totals = manifest.get("relation_totals")
    if not isinstance(totals, dict):
        raise CandidateEvaluationError(
            "preprocessing manifest has no relation totals"
        )
    for relation, schema in PROJECTED_ARTIFACT_SCHEMAS.items():
        stable = relations[relation]
        path = stable.path
        try:
            parquet = pq.ParquetFile(path)
        except (OSError, pa.ArrowException) as exc:
            raise CandidateEvaluationError(
                f"preprocessing relation is invalid: {relation}"
            ) from exc
        if not parquet.schema_arrow.equals(schema):
            raise CandidateEvaluationError(
                f"preprocessing relation schema mismatch: {relation}"
            )
        actual_hash = stable.sha256
        recorded_hash = recorded_hashes.get(relation)
        if not _is_sha256(recorded_hash) or recorded_hash != actual_hash:
            raise CandidateEvaluationError(
                f"preprocessing relation hash mismatch: {relation}"
            )
        if totals.get(relation) != parquet.metadata.num_rows:
            raise CandidateEvaluationError(
                f"preprocessing relation count mismatch: {relation}"
            )
        relation_coordinates[relation] = {
            "sha256": actual_hash,
            "rows": parquet.metadata.num_rows,
        }
    try:
        outcomes = validate_preprocessing_relations(
            input_parquet=pq.ParquetFile(corpus_file.path),
            results_path=relations["results"].path,
            candidates_path=relations["candidates"].path,
            step_facts_path=relations["step_facts"].path,
            rejections_path=relations["rejections"].path,
        )
        validate_preprocessing_derivation(
            input_parquet=pq.ParquetFile(corpus_file.path),
            results_path=relations["results"].path,
            candidates_path=relations["candidates"].path,
            step_facts_path=relations["step_facts"].path,
            rejections_path=relations["rejections"].path,
            preprocessing_config=preprocessing_config,
        )
    except CorpusRunError as exc:
        raise CandidateEvaluationError(
            f"preprocessing relations are inconsistent: {exc}"
        ) from exc
    if manifest.get("outcome_totals") != dict(sorted(outcomes.items())):
        raise CandidateEvaluationError(
            "preprocessing outcome totals do not match relations"
        )
    try:
        identity = preprocessing_run_identity(manifest)
    except CandidateEvaluationContractError as exc:
        raise CandidateEvaluationError(str(exc)) from exc
    return _ValidatedPreprocessingRun(
        coordinates={
            "identity": identity,
            "relations": relation_coordinates,
        },
        config=preprocessing_config,
    )


def _validate_corpus_input_coordinates(
    corpus_file: StableFile,
    recorded: Mapping[str, object],
) -> None:
    """Bind evaluation to the exact Parquet input used for preprocessing."""

    try:
        parquet = pq.ParquetFile(corpus_file.path)
    except (OSError, pa.ArrowException) as exc:
        raise CandidateEvaluationError(
            "evaluation corpus is not valid Parquet"
        ) from exc
    actual = {
        "sha256": corpus_file.sha256,
        "size": corpus_file.size,
        "schema_hex": parquet.schema_arrow.serialize().to_pybytes().hex(),
        "expected_rows": parquet.metadata.num_rows,
        "expected_row_groups": parquet.num_row_groups,
        "row_groups": [
            {
                "index": index,
                "rows": parquet.metadata.row_group(index).num_rows,
                "total_byte_size": (
                    parquet.metadata.row_group(index).total_byte_size
                ),
            }
            for index in range(parquet.num_row_groups)
        ],
    }
    mismatches = [
        name for name, value in actual.items() if recorded.get(name) != value
    ]
    if mismatches:
        raise CandidateEvaluationError(
            "evaluation corpus does not match preprocessing input coordinate(s): "
            + ", ".join(mismatches)
        )


def _immutable_coordinates(
    *,
    preprocessing_coordinates: Mapping[str, object],
    preprocessing_config: PreprocessingConfig,
    corpus_file: StableFile,
    snapshot_file: StableFile,
    config: _EvaluationConfig,
    runner_identity: str,
    runtime_identity: str,
    host_runtime: Mapping[str, object],
    installed_environment: Mapping[str, object],
    trusted_source: Mapping[str, str],
    max_infrastructure_retries: int,
) -> dict[str, object]:
    metric_definition = config.metric_definition
    procedure_ref = config.procedure_config.definition_ref
    coordinates: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "preprocessing_run": dict(preprocessing_coordinates),
        "corpus_sha256": corpus_file.sha256,
        "snapshot_sha256": snapshot_file.sha256,
        "dataset": {
            "dataset_id": DEFAULT_HUMAN_EVAL_DATASET_NAME,
            "split": DEFAULT_HUMAN_EVAL_DATASET_SPLIT,
            "hf_revision": DEFAULT_HUMAN_EVAL_HF_REVISION,
        },
        "metric_extraction_definition_ref": (
            config.metric_config.definition_ref.model_dump(mode="json")
        ),
        # Thaw the frozen JSON in concrete question settings (e.g. the
        # code_test ``budgets`` block): pydantic's json mode leaves
        # ``FrozenJsonDict`` values unserializable, so the canonical
        # coordinate JSON needs the plain-dict form, which round-trips
        # back through ``MetricExtractionConfig.model_validate``.
        "metric_extraction_config": thaw_json(
            config.metric_config.model_dump(mode="python")
        ),
        "metric_extraction_definition_identity": (
            metric_definition.identity_hash()
        ),
        "metric_extraction_config_identity": (
            config.metric_config.config_identity_hash
        ),
        "evaluation_procedure_definition_ref": (
            procedure_ref.model_dump(mode="json")
        ),
        "evaluation_procedure_config": (
            config.procedure_config.model_dump(mode="json")
        ),
        "evaluation_procedure_definition_identity": (
            procedure_ref.identity_hash
        ),
        "evaluation_procedure_config_identity": (
            config.procedure_config.config_identity_hash
        ),
        "trace_producer": config.trace_producer.model_dump(mode="json"),
        "operator_coordinates": config.operator.model_dump(mode="json"),
        "question_identity_hash": config.question_identity_hash,
        "operator_name": config.operator_name,
        "operator_version": config.operator_version,
        "metrics_profile": _METRICS_PROFILE,
        "runner_identity": runner_identity,
        "runtime_identity": runtime_identity,
        "host_runtime": dict(host_runtime),
        "installed_environment": dict(installed_environment),
        "trusted_source_sha256": dict(trusted_source),
        "max_infrastructure_retries": max_infrastructure_retries,
    }
    try:
        validate_evaluation_coordinates(
            coordinates,
            preprocessing_config=preprocessing_config,
        )
    except CoordinateValidationError as exc:
        raise CandidateEvaluationError(str(exc)) from exc
    return {
        **coordinates,
        "evaluation_identity": _evaluation_identity(coordinates),
    }


def _evaluation_identity(coordinates: Mapping[str, object]) -> str:
    try:
        return candidate_evaluation_identity(coordinates)
    except CandidateEvaluationContractError as exc:
        raise CandidateEvaluationError(str(exc)) from exc


def _load_reuse_sources(
    captured: Mapping[str, StableFile],
    *,
    reuse_count: int,
    expected: Mapping[str, object],
    preprocessing_config: PreprocessingConfig,
) -> tuple[_ReuseSource, ...]:
    result: list[_ReuseSource] = []
    identities: set[tuple[str, str, str]] = set()
    for index in range(reuse_count):
        manifest_file = captured[f"reuse_{index}_manifest"]
        membership_file = captured[f"reuse_{index}_membership"]
        results_file = captured[f"reuse_{index}_results"]
        manifest_path = manifest_file.path
        membership_path = membership_file.path
        results_path = results_file.path
        manifest = _read_json_object(
            manifest_path, "candidate evaluation manifest"
        )
        if manifest.get("complete") is not True:
            raise CandidateEvaluationError(
                f"reuse source is not complete: {manifest_path}"
            )
        for coordinate in _REUSE_COORDINATES:
            if coordinate not in manifest or _canonical_json(
                manifest[coordinate]
            ) != _canonical_json(expected[coordinate]):
                raise CandidateEvaluationError(
                    "reuse source has incompatible coordinate "
                    f"{coordinate!r}: {manifest_path}"
                )
        if set(manifest) != _MANIFEST_FIELDS:
            raise CandidateEvaluationError(
                f"reuse source manifest schema mismatch: {manifest_path}"
            )
        recorded_evaluation_identity = manifest["evaluation_identity"]
        evaluation_coordinates = {
            field: manifest[field]
            for field in CANDIDATE_EVALUATION_COORDINATE_FIELDS
        }
        try:
            validate_evaluation_coordinates(
                evaluation_coordinates,
                preprocessing_config=preprocessing_config,
            )
        except CoordinateValidationError as exc:
            raise CandidateEvaluationError(str(exc)) from exc
        if not _is_sha256(
            recorded_evaluation_identity
        ) or recorded_evaluation_identity != _evaluation_identity(
            evaluation_coordinates
        ):
            raise CandidateEvaluationError(
                f"reuse source has invalid evaluation identity: {manifest_path}"
            )
        membership_rows = _nonnegative_int(
            manifest.get("membership_rows"), "reuse membership_rows"
        )
        result_rows = _nonnegative_int(
            manifest.get("result_rows"), "reuse result_rows"
        )
        membership_sha = _validate_exported_parquet(
            membership_file,
            MEMBERSHIP_SCHEMA,
            expected_rows=membership_rows,
            recorded_sha=manifest.get("candidate_membership_sha256"),
        )
        results_sha = _validate_exported_parquet(
            results_file,
            RESULTS_SCHEMA,
            expected_rows=result_rows,
            recorded_sha=manifest.get("candidate_results_sha256"),
        )
        source = _ReuseSource(
            manifest_path=manifest_path,
            membership_path=membership_path,
            results_path=results_path,
            manifest_sha256=manifest_file.sha256,
            membership_sha256=membership_sha,
            results_sha256=results_sha,
            membership_rows=membership_rows,
            result_rows=result_rows,
        )
        identity = (
            source.manifest_sha256,
            source.membership_sha256,
            source.results_sha256,
        )
        if identity in identities:
            raise CandidateEvaluationError(
                f"duplicate reuse source: {manifest_file.source_path.parent}"
            )
        identities.add(identity)
        result.append(source)
    return tuple(result)


def _load_tasks(snapshot_file: StableFile) -> dict[str, HumanEvalTask]:
    try:
        rows = load_human_eval_snapshot_rows_bytes(
            snapshot_file.path.read_bytes(),
            expected_snapshot_sha256=DEFAULT_HUMAN_EVAL_SNAPSHOT_SHA256,
        )
        tasks = parse_human_eval_dataset(rows)
    except Exception as exc:
        raise CandidateEvaluationError(
            "HumanEval+ snapshot does not match the pinned dataset coordinates"
        ) from exc
    result = {task.task_id: task for task in tasks}
    if len(result) != len(tasks):
        raise CandidateEvaluationError(
            "snapshot contains duplicate task_id values"
        )
    return result


def _prepare_memberships(
    *,
    connection: sqlite3.Connection,
    lease_id: str,
    corpus_file: Path,
    candidates_file: Path,
    results_file: Path,
    tasks: Mapping[str, HumanEvalTask],
    task_identities: Mapping[str, str],
    config: _EvaluationConfig,
    runtime_identity: str,
    runner_identity: str,
) -> None:
    _stage_validated_corpus_results(
        connection,
        corpus_file=corpus_file,
        results_file=results_file,
        lease_id=lease_id,
    )
    temporary_statements = (
        "DROP TABLE IF EXISTS temp.seen_membership",
        "DROP TABLE IF EXISTS temp.seen_candidate_id",
        "DROP TABLE IF EXISTS temp.seen_candidate_index",
        """CREATE TEMP TABLE seen_membership(
            sample_id TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            candidate_index INTEGER NOT NULL,
            PRIMARY KEY(sample_id, candidate_id, candidate_index)
        )""",
        """CREATE TEMP TABLE seen_candidate_id(
            sample_id TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            PRIMARY KEY(sample_id, candidate_id)
        )""",
        """CREATE TEMP TABLE seen_candidate_index(
            sample_id TEXT NOT NULL,
            candidate_index INTEGER NOT NULL,
            PRIMARY KEY(sample_id, candidate_index)
        )""",
    )
    with _lease_transaction(connection, lease_id):
        for statement in temporary_statements:
            connection.execute(statement)
    parquet = pq.ParquetFile(candidates_file)
    columns = (
        "sample_id",
        "candidate_id",
        "candidate_index",
        "cleaned_source",
        "source_sha256",
    )
    for batch in parquet.iter_batches(batch_size=1_000, columns=list(columns)):
        memberships: list[tuple[object, ...]] = []
        work: dict[str, _Work] = {}
        for row in pa.Table.from_batches([batch]).to_pylist():
            sample_id = _string(row["sample_id"], "candidate sample_id")
            candidate_id = _string(row["candidate_id"], "candidate_id")
            candidate_index = _nonnegative_int(
                row["candidate_index"], "candidate_index"
            )
            source = _string(row["cleaned_source"], "cleaned_source")
            source_sha = _string(row["source_sha256"], "source_sha256")
            if source_sha != _sha256_text(source):
                raise CandidateEvaluationError(
                    f"invalid source hash for {sample_id!r}/{candidate_id!r}"
                )
            if candidate_id != candidate_id_for_source(source):
                raise CandidateEvaluationError(
                    f"candidate_id is not content-derived for {sample_id!r}"
                )
            sample = connection.execute(
                """SELECT task_id, source_kind
                     FROM staged_corpus
                    WHERE sample_id = ?""",
                (sample_id,),
            ).fetchone()
            if sample is None:
                raise CandidateEvaluationError(
                    f"candidate sample_id is absent from corpus: {sample_id!r}"
                )
            task_id, source_kind = sample
            if task_id not in tasks:
                raise CandidateEvaluationError(
                    f"corpus task_id is absent from snapshot: {task_id!r}"
                )
            for table, values in (
                (
                    "seen_membership",
                    (sample_id, candidate_id, candidate_index),
                ),
                ("seen_candidate_id", (sample_id, candidate_id)),
                ("seen_candidate_index", (sample_id, candidate_index)),
            ):
                placeholders = ", ".join("?" for _ in values)
                try:
                    connection.execute(
                        f"INSERT INTO {table} VALUES ({placeholders})", values
                    )
                except sqlite3.IntegrityError as exc:
                    raise CandidateEvaluationError(
                        f"duplicate candidate coordinate in {table}"
                    ) from exc
            task_identity = task_identities[task_id]
            key = _evaluation_key(
                task_id=task_id,
                task_identity=task_identity,
                source_sha256=source_sha,
                config=config,
                runtime_identity=runtime_identity,
                runner_identity=runner_identity,
            )
            memberships.append(
                (
                    sample_id,
                    candidate_id,
                    candidate_index,
                    task_id,
                    task_identity,
                    source_kind,
                    source_sha,
                    key,
                )
            )
            proposed = _Work(
                evaluation_key=key,
                task_id=task_id,
                task_identity=task_identity,
                source_sha256=source_sha,
                candidate_source=source,
            )
            existing = work.get(key)
            if existing is not None and existing != proposed:
                raise CandidateEvaluationError("evaluation key collision")
            work.setdefault(key, proposed)
        _upsert_memberships_and_work(
            connection, memberships, work, lease_id=lease_id
        )
        _heartbeat(connection, lease_id)
    _validate_candidate_counts(connection)
    with _lease_transaction(connection, lease_id):
        _validate_persisted_membership_exact(connection)
        _validate_persisted_work_exact(connection)


def _stage_validated_corpus_results(
    connection: sqlite3.Connection,
    *,
    corpus_file: Path,
    results_file: Path,
    lease_id: str | None,
    snapshot_tasks: Mapping[str, HumanEvalTask] | None = None,
) -> None:
    statements = (
        "DROP TABLE IF EXISTS temp.staged_corpus",
        "DROP TABLE IF EXISTS temp.staged_preprocessing_result",
        """CREATE TEMP TABLE staged_corpus(
            sample_id TEXT PRIMARY KEY NOT NULL,
            task_id TEXT NOT NULL,
            source_kind TEXT,
            raw_output_sha256 TEXT
        )""",
        """CREATE TEMP TABLE staged_preprocessing_result(
            sample_id TEXT PRIMARY KEY NOT NULL,
            raw_output_sha256 TEXT,
            final_candidate_count INTEGER NOT NULL
        )""",
    )
    with _admission_transaction(connection, lease_id):
        for statement in statements:
            connection.execute(statement)
    _stage_corpus(connection, corpus_file, lease_id=lease_id)
    _stage_preprocessing_results(connection, results_file, lease_id=lease_id)
    if snapshot_tasks is not None:
        _validate_staged_task_ids(
            connection,
            snapshot_tasks=snapshot_tasks,
            lease_id=lease_id,
        )
    missing = connection.execute(
        """SELECT sample_id FROM staged_corpus
           EXCEPT
           SELECT sample_id FROM staged_preprocessing_result
           LIMIT 1"""
    ).fetchone()
    extra = connection.execute(
        """SELECT sample_id FROM staged_preprocessing_result
           EXCEPT
           SELECT sample_id FROM staged_corpus
           LIMIT 1"""
    ).fetchone()
    if missing is not None or extra is not None:
        raise CandidateEvaluationError(
            "preprocessing results do not exactly match corpus sample membership"
        )
    mismatch = connection.execute(
        """SELECT corpus.sample_id
             FROM staged_corpus AS corpus
             JOIN staged_preprocessing_result AS result USING(sample_id)
            WHERE corpus.raw_output_sha256 IS NOT result.raw_output_sha256
            LIMIT 1"""
    ).fetchone()
    if mismatch is not None:
        raise CandidateEvaluationError(
            "corpus decoder_output does not match preprocessing results "
            f"for sample_id {mismatch[0]!r}"
        )


def _validate_corpus_results_before_state(
    corpus_file: Path,
    results_file: Path,
    *,
    snapshot_tasks: Mapping[str, HumanEvalTask] | None = None,
) -> None:
    """Validate cross-file coordinates in a private disk-backed database."""

    connection = sqlite3.connect("", isolation_level=None)
    try:
        connection.execute("PRAGMA temp_store=FILE")
        _stage_validated_corpus_results(
            connection,
            corpus_file=corpus_file,
            results_file=results_file,
            lease_id=None,
            snapshot_tasks=snapshot_tasks,
        )
    finally:
        connection.close()


def _validate_staged_task_ids(
    connection: sqlite3.Connection,
    *,
    snapshot_tasks: Mapping[str, HumanEvalTask],
    lease_id: str | None,
) -> None:
    with _admission_transaction(connection, lease_id):
        connection.execute("DROP TABLE IF EXISTS temp.snapshot_task")
        connection.execute(
            "CREATE TEMP TABLE snapshot_task(task_id TEXT PRIMARY KEY NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO snapshot_task(task_id) VALUES (?)",
            ((task_id,) for task_id in snapshot_tasks),
        )
    missing = connection.execute(
        """SELECT DISTINCT corpus.task_id
             FROM staged_corpus AS corpus
             LEFT JOIN snapshot_task AS snapshot USING(task_id)
            WHERE snapshot.task_id IS NULL
            LIMIT 1"""
    ).fetchone()
    if missing is not None:
        raise CandidateEvaluationError(
            f"corpus task_id is absent from snapshot: {missing[0]!r}"
        )


@contextmanager
def _admission_transaction(
    connection: sqlite3.Connection,
    lease_id: str | None,
) -> Iterator[None]:
    if lease_id is not None:
        with _lease_transaction(connection, lease_id):
            yield
        return
    connection.execute("BEGIN")
    try:
        yield
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def _stage_corpus(
    connection: sqlite3.Connection,
    path: Path,
    *,
    lease_id: str | None,
) -> None:
    parquet = pq.ParquetFile(path)
    required = {"sample_id", "decoder_output", "task_id"}
    if missing := required - set(parquet.schema_arrow.names):
        raise CandidateEvaluationError(
            "corpus is missing required column(s): "
            + ", ".join(sorted(missing))
        )
    source_column = (
        "source_kind" if "source_kind" in parquet.schema_arrow.names else None
    )
    columns = ["sample_id", "decoder_output", "task_id"]
    if source_column:
        columns.append(source_column)
    for batch in parquet.iter_batches(
        batch_size=_ADMISSION_BATCH_SIZE,
        columns=columns,
    ):
        staged: list[tuple[object, ...]] = []
        for row in pa.Table.from_batches([batch]).to_pylist():
            sample_id = _string(row["sample_id"], "corpus sample_id")
            task_id = _string(row["task_id"], "corpus task_id")
            decoder_output = row["decoder_output"]
            if decoder_output is not None and not isinstance(
                decoder_output, str
            ):
                raise CandidateEvaluationError(
                    "corpus decoder_output must be a string or null"
                )
            source_kind = row.get(source_column) if source_column else None
            if source_kind is not None and not isinstance(source_kind, str):
                raise CandidateEvaluationError(
                    "corpus source_kind must be a string or null"
                )
            staged.append(
                (
                    sample_id,
                    task_id,
                    source_kind,
                    (
                        None
                        if decoder_output is None
                        else _sha256_text(decoder_output)
                    ),
                )
            )
        try:
            with _admission_transaction(connection, lease_id):
                connection.executemany(
                    """INSERT INTO staged_corpus(
                           sample_id, task_id, source_kind, raw_output_sha256
                       ) VALUES (?, ?, ?, ?)""",
                    staged,
                )
        except sqlite3.IntegrityError as exc:
            raise CandidateEvaluationError(
                "corpus contains duplicate sample_id values"
            ) from exc


def _stage_preprocessing_results(
    connection: sqlite3.Connection,
    path: Path,
    *,
    lease_id: str | None,
) -> None:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=_ADMISSION_BATCH_SIZE,
        columns=[
            "sample_id",
            "raw_output_sha256",
            "final_candidate_count",
        ],
    ):
        staged: list[tuple[object, ...]] = []
        for row in pa.Table.from_batches([batch]).to_pylist():
            sample_id = _string(row["sample_id"], "result sample_id")
            raw_output_sha256 = row["raw_output_sha256"]
            if raw_output_sha256 is not None and not _is_sha256(
                raw_output_sha256
            ):
                raise CandidateEvaluationError(
                    "preprocessing result raw_output_sha256 must be a SHA-256 "
                    "or null"
                )
            staged.append(
                (
                    sample_id,
                    raw_output_sha256,
                    _nonnegative_int(
                        row["final_candidate_count"],
                        "final_candidate_count",
                    ),
                )
            )
        try:
            with _admission_transaction(connection, lease_id):
                connection.executemany(
                    """INSERT INTO staged_preprocessing_result(
                           sample_id, raw_output_sha256, final_candidate_count
                       ) VALUES (?, ?, ?)""",
                    staged,
                )
        except sqlite3.IntegrityError as exc:
            raise CandidateEvaluationError(
                "preprocessing results contain duplicate sample_id values"
            ) from exc


def _open_state(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=30.0, isolation_level=None)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA temp_store=FILE")
    return connection


def _initialize_state(
    connection: sqlite3.Connection,
    immutable: Mapping[str, object],
    *,
    lease_id: str,
) -> None:
    statements = (
        """CREATE TABLE IF NOT EXISTS metadata(
            key TEXT PRIMARY KEY NOT NULL,
            value_json TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS work(
            evaluation_key TEXT PRIMARY KEY NOT NULL,
            task_id TEXT NOT NULL,
            task_identity TEXT NOT NULL,
            source_sha256 TEXT NOT NULL,
            candidate_source TEXT NOT NULL,
            status TEXT NOT NULL
                CHECK(status IN ('pending', 'running', 'completed')),
            attempt_count INTEGER NOT NULL DEFAULT 0,
            record_status TEXT,
            failure_type TEXT,
            failure_message TEXT,
            values_json TEXT,
            completed_at TEXT,
            reused_from_manifest_sha256 TEXT,
            owner_lease_id TEXT,
            result_evidence_sha256 TEXT
        )""",
        """CREATE INDEX IF NOT EXISTS work_pending_claim
            ON work(status, evaluation_key)""",
        """CREATE TABLE IF NOT EXISTS membership(
            sample_id TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            candidate_index INTEGER NOT NULL,
            task_id TEXT NOT NULL,
            task_identity TEXT NOT NULL,
            source_kind TEXT,
            source_sha256 TEXT NOT NULL,
            evaluation_key TEXT NOT NULL REFERENCES work(evaluation_key),
            PRIMARY KEY(sample_id, candidate_id, candidate_index)
        )""",
    )
    with _lease_transaction(connection, lease_id):
        for statement in statements:
            connection.execute(statement)
        actual_work_columns = tuple(
            row[1] for row in connection.execute("PRAGMA table_info(work)")
        )
        if actual_work_columns != _WORK_COLUMNS:
            if actual_work_columns == _WORK_COLUMNS[:-1]:
                raise CandidateEvaluationError(
                    "evaluation state predates canonical result evidence; "
                    "resume is fail-closed and requires a fresh state database"
                )
            raise CandidateEvaluationError(
                "evaluation state work schema is incompatible"
            )
        encoded = {
            key: _canonical_json(value) for key, value in immutable.items()
        }
        existing = dict(
            connection.execute("SELECT key, value_json FROM metadata")
        )
        if existing and existing != encoded:
            raise CandidateEvaluationError(
                "evaluation state is incompatible with requested coordinates"
            )
        if not existing:
            connection.executemany(
                "INSERT INTO metadata(key, value_json) VALUES (?, ?)",
                encoded.items(),
            )


def _upsert_memberships_and_work(
    connection: sqlite3.Connection,
    memberships: list[tuple[object, ...]],
    work: Mapping[str, _Work],
    *,
    lease_id: str,
) -> None:
    with _lease_transaction(connection, lease_id):
        for item in work.values():
            existing = connection.execute(
                """SELECT task_id, task_identity, source_sha256,
                          candidate_source
                     FROM work WHERE evaluation_key = ?""",
                (item.evaluation_key,),
            ).fetchone()
            identity = (
                item.task_id,
                item.task_identity,
                item.source_sha256,
                item.candidate_source,
            )
            if existing is not None and existing != identity:
                raise CandidateEvaluationError(
                    "persisted work conflicts with evaluation identity"
                )
            connection.execute(
                """INSERT INTO work(
                       evaluation_key, task_id, task_identity, source_sha256,
                       candidate_source, status
                   ) VALUES (?, ?, ?, ?, ?, 'pending')
                   ON CONFLICT(evaluation_key) DO NOTHING""",
                (item.evaluation_key, *identity),
            )
        for values in memberships:
            existing = connection.execute(
                """SELECT task_id, task_identity, source_kind, source_sha256,
                          evaluation_key
                     FROM membership
                    WHERE sample_id = ? AND candidate_id = ?
                      AND candidate_index = ?""",
                values[:3],
            ).fetchone()
            if existing is not None and existing != values[3:]:
                raise CandidateEvaluationError(
                    "persisted membership conflicts with candidate occurrence"
                )
            connection.execute(
                """INSERT INTO membership(
                       sample_id, candidate_id, candidate_index, task_id,
                       task_identity, source_kind, source_sha256, evaluation_key
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(sample_id, candidate_id, candidate_index)
                   DO NOTHING""",
                values,
            )


def _validate_candidate_counts(connection: sqlite3.Connection) -> None:
    mismatch = connection.execute(
        """SELECT expected.sample_id
             FROM staged_preprocessing_result AS expected
             LEFT JOIN (
                 SELECT sample_id, COUNT(*) AS candidate_count,
                        MAX(candidate_index) AS maximum_index
                   FROM seen_candidate_index
                  GROUP BY sample_id
             ) AS observed USING(sample_id)
            WHERE COALESCE(observed.candidate_count, 0)
                      != expected.final_candidate_count
               OR (
                   expected.final_candidate_count > 0
                   AND observed.maximum_index
                       != expected.final_candidate_count - 1
               )
            LIMIT 1"""
    ).fetchone()
    if mismatch is not None:
        raise CandidateEvaluationError(
            "candidate rows do not match final_candidate_count for "
            f"{mismatch[0]!r}"
        )


def _validate_persisted_membership_exact(
    connection: sqlite3.Connection,
) -> None:
    missing = connection.execute(
        """SELECT sample_id, candidate_id, candidate_index FROM seen_membership
           EXCEPT
           SELECT sample_id, candidate_id, candidate_index FROM membership
           LIMIT 1"""
    ).fetchone()
    extra = connection.execute(
        """SELECT sample_id, candidate_id, candidate_index FROM membership
           EXCEPT
           SELECT sample_id, candidate_id, candidate_index FROM seen_membership
           LIMIT 1"""
    ).fetchone()
    if missing is not None or extra is not None:
        raise CandidateEvaluationError(
            "persisted membership does not exactly match preprocessing candidates"
        )


def _validate_persisted_work_exact(connection: sqlite3.Connection) -> None:
    orphan = connection.execute(
        """SELECT evaluation_key FROM work
            WHERE NOT EXISTS (
                SELECT 1 FROM membership
                 WHERE membership.evaluation_key = work.evaluation_key
            )
            LIMIT 1"""
    ).fetchone()
    if orphan is not None:
        raise CandidateEvaluationError(
            "persisted work contains an evaluation absent from membership"
        )


def _acquire_lease(connection: sqlite3.Connection, lease_id: str) -> None:
    now = time.time()
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS evaluator_lease(
                   singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                   lease_id TEXT NOT NULL,
                   heartbeat_at REAL NOT NULL
               )"""
        )
        row = connection.execute(
            """SELECT lease_id, heartbeat_at FROM evaluator_lease
               WHERE singleton = 1"""
        ).fetchone()
        if row is not None and now - row[1] < _LEASE_SECONDS:
            raise CandidateEvaluationError(
                "candidate evaluation state is owned by a live evaluator"
            )
        connection.execute("DELETE FROM evaluator_lease WHERE singleton = 1")
        connection.execute(
            """INSERT INTO evaluator_lease(singleton, lease_id, heartbeat_at)
               VALUES (1, ?, ?)""",
            (lease_id, now),
        )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def _heartbeat(connection: sqlite3.Connection, lease_id: str) -> None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        _validate_lease_owner(connection, lease_id)
        updated = connection.execute(
            """UPDATE evaluator_lease SET heartbeat_at = ?
               WHERE singleton = 1 AND lease_id = ?""",
            (time.time(), lease_id),
        )
        if updated.rowcount != 1:
            raise CandidateEvaluationError(
                "candidate evaluation lease was lost"
            )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def _validate_lease_owner(
    connection: sqlite3.Connection, lease_id: str
) -> None:
    owner = connection.execute(
        """SELECT heartbeat_at FROM evaluator_lease
            WHERE singleton = 1 AND lease_id = ?""",
        (lease_id,),
    ).fetchone()
    if owner is None or time.time() - owner[0] >= _LEASE_SECONDS:
        raise CandidateEvaluationError("candidate evaluation lease was lost")


@contextmanager
def _lease_transaction(
    connection: sqlite3.Connection,
    lease_id: str,
) -> Iterator[None]:
    """Fence a bounded persistent mutation on the exact active lease."""

    connection.execute("BEGIN IMMEDIATE")
    try:
        _validate_lease_owner(connection, lease_id)
        connection.execute(
            """UPDATE evaluator_lease SET heartbeat_at = ?
               WHERE singleton = 1 AND lease_id = ?""",
            (time.time(), lease_id),
        )
        yield
        _validate_lease_owner(connection, lease_id)
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def _release_lease(connection: sqlite3.Connection, lease_id: str) -> None:
    try:
        with connection:
            connection.execute(
                """DELETE FROM evaluator_lease
                   WHERE singleton = 1 AND lease_id = ?""",
                (lease_id,),
            )
    except sqlite3.Error:
        pass


def _result_evidence_sha256(
    *,
    evaluation_key: str,
    task_id: str,
    task_identity: str,
    source_sha256: str,
    candidate_source: str,
    record_status: str,
    failure_type: str | None,
    failure_message: str | None,
    values_json: str | None,
) -> str:
    values: dict[str, object] | None
    if values_json is None:
        values = None
    else:
        try:
            decoded = json.loads(
                values_json,
                parse_constant=_reject_nonfinite_json_number,
            )
        except json.JSONDecodeError as exc:
            raise CandidateEvaluationError(
                "persisted fact values are invalid JSON"
            ) from exc
        if not isinstance(decoded, dict):
            raise CandidateEvaluationError(
                "persisted fact values must be an object"
            )
        if values_json != _canonical_json(decoded):
            raise CandidateEvaluationError(
                "persisted fact values must be canonical JSON"
            )
        values = decoded
    payload = {
        "schema": _RESULT_EVIDENCE_SCHEMA,
        "schema_version": _RESULT_EVIDENCE_SCHEMA_VERSION,
        "work_identity": {
            "evaluation_key": evaluation_key,
            "task_id": task_id,
            "task_identity": task_identity,
            "source_sha256": source_sha256,
            "candidate_source": candidate_source,
        },
        "result": {
            "work_status": "completed",
            "record_status": record_status,
            "failure_type": failure_type,
            "failure_message": failure_message,
            "values": values,
        },
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _validate_completed_result_evidence(
    connection: sqlite3.Connection,
) -> None:
    rows = connection.execute(
        """SELECT evaluation_key, task_id, task_identity, source_sha256,
                  candidate_source, status, record_status, failure_type,
                  failure_message, values_json, result_evidence_sha256
             FROM work ORDER BY evaluation_key"""
    )
    for row in rows:
        status = row[5]
        evidence = row[10]
        if status != "completed":
            if evidence is not None:
                raise CandidateEvaluationError(
                    "incomplete work must not carry result evidence"
                )
            continue
        if not isinstance(row[6], str) or not row[6]:
            raise CandidateEvaluationError(
                "completed work has no record status evidence"
            )
        expected = _result_evidence_sha256(
            evaluation_key=row[0],
            task_id=row[1],
            task_identity=row[2],
            source_sha256=row[3],
            candidate_source=row[4],
            record_status=row[6],
            failure_type=row[7],
            failure_message=row[8],
            values_json=row[9],
        )
        if evidence != expected:
            raise CandidateEvaluationError(
                f"completed result evidence mismatch for {row[0]}"
            )


def _recover_work(
    connection: sqlite3.Connection,
    *,
    lease_id: str,
    max_infrastructure_retries: int,
) -> None:
    with _lease_transaction(connection, lease_id):
        connection.execute(
            """UPDATE work
                  SET status = 'pending', owner_lease_id = NULL
                WHERE status = 'running' AND attempt_count <= ?""",
            (max_infrastructure_retries,),
        )
        while True:
            abandoned = connection.execute(
                """SELECT evaluation_key, task_id, task_identity,
                          source_sha256, candidate_source,
                          COALESCE(
                              failure_type, 'AbandonedInfrastructureAttempt'
                          ),
                          COALESCE(
                              failure_message,
                              'infrastructure attempt was abandoned after its '
                              || 'retry budget was exhausted'
                          )
                     FROM work
                    WHERE status = 'running' AND attempt_count > ?
                    ORDER BY evaluation_key LIMIT 1""",
                (max_infrastructure_retries,),
            ).fetchone()
            if abandoned is None:
                break
            (
                evaluation_key,
                task_id,
                task_identity,
                source_sha256,
                candidate_source,
                failure_type,
                failure_message,
            ) = abandoned
            evidence = _result_evidence_sha256(
                evaluation_key=evaluation_key,
                task_id=task_id,
                task_identity=task_identity,
                source_sha256=source_sha256,
                candidate_source=candidate_source,
                record_status="infrastructure_failure",
                failure_type=failure_type,
                failure_message=failure_message,
                values_json=None,
            )
            connection.execute(
                """UPDATE work
                      SET status = 'completed',
                          record_status = 'infrastructure_failure',
                          failure_type = ?,
                          failure_message = ?,
                          values_json = NULL,
                          completed_at = ?,
                          owner_lease_id = NULL,
                          result_evidence_sha256 = ?
                    WHERE evaluation_key = ? AND status = 'running'
                      AND attempt_count > ?""",
                (
                    failure_type,
                    failure_message,
                    _timestamp(),
                    evidence,
                    evaluation_key,
                    max_infrastructure_retries,
                ),
            )


def _reuse_completed_results(
    connection: sqlite3.Connection,
    *,
    lease_id: str,
    reuse_sources: Sequence[_ReuseSource],
    task_identities: Mapping[str, str],
    config: _EvaluationConfig,
    runtime_identity: str,
    runner_identity: str,
) -> None:
    for source in reuse_sources:
        parquet = pq.ParquetFile(source.results_path)
        for batch in parquet.iter_batches(batch_size=1_000):
            with _lease_transaction(connection, lease_id):
                for row in pa.Table.from_batches([batch]).to_pylist():
                    imported = _validated_reuse_result(
                        row,
                        task_identities=task_identities,
                        config=config,
                        runtime_identity=runtime_identity,
                        runner_identity=runner_identity,
                        source_path=source.results_path,
                    )
                    if imported is None:
                        continue
                    key, identity, values = imported
                    target = connection.execute(
                        """SELECT task_id, task_identity, source_sha256,
                                  candidate_source, status, record_status,
                                  failure_type, failure_message, values_json,
                                  reused_from_manifest_sha256
                             FROM work WHERE evaluation_key = ?""",
                        (key,),
                    ).fetchone()
                    if target is None:
                        continue
                    if target[:4] != identity:
                        raise CandidateEvaluationError(
                            f"reuse result conflicts with target work: {key}"
                        )
                    value_fields = values[1:]
                    if target[4] == "pending":
                        evidence = _result_evidence_sha256(
                            evaluation_key=key,
                            task_id=identity[0],
                            task_identity=identity[1],
                            source_sha256=identity[2],
                            candidate_source=identity[3],
                            record_status=value_fields[0],
                            failure_type=value_fields[1],
                            failure_message=value_fields[2],
                            values_json=value_fields[3],
                        )
                        connection.execute(
                            """UPDATE work
                                  SET status = 'completed', record_status = ?,
                                      failure_type = ?, failure_message = ?,
                                      values_json = ?, completed_at = ?,
                                      reused_from_manifest_sha256 = ?,
                                      result_evidence_sha256 = ?
                                WHERE evaluation_key = ?
                                  AND status = 'pending'""",
                            (
                                *value_fields,
                                _timestamp(),
                                source.manifest_sha256,
                                evidence,
                                key,
                            ),
                        )
                    elif target[4] == "completed" and target[9] is not None:
                        if target[5:9] != value_fields:
                            raise CandidateEvaluationError(
                                f"reuse sources conflict for work: {key}"
                            )
            _heartbeat(connection, lease_id)


def _validated_reuse_result(
    row: Mapping[str, object],
    *,
    task_identities: Mapping[str, str],
    config: _EvaluationConfig,
    runtime_identity: str,
    runner_identity: str,
    source_path: Path,
) -> (
    tuple[
        str,
        tuple[str, str, str, str],
        tuple[str, str, None, None, str],
    ]
    | None
):
    key = _string(row.get("evaluation_key"), "reuse evaluation_key")
    task_id = _string(row.get("task_id"), "reuse task_id")
    task_identity = _string(row.get("task_identity"), "reuse task_identity")
    source = _string(row.get("cleaned_source"), "reuse cleaned_source")
    source_sha = _string(row.get("source_sha256"), "reuse source_sha256")
    if (
        source_sha != _sha256_text(source)
        or task_identities.get(task_id) != task_identity
    ):
        raise CandidateEvaluationError(
            f"reuse result has invalid content identity: {source_path}"
        )
    expected_key = _evaluation_key(
        task_id=task_id,
        task_identity=task_identity,
        source_sha256=source_sha,
        config=config,
        runtime_identity=runtime_identity,
        runner_identity=runner_identity,
    )
    if key != expected_key:
        raise CandidateEvaluationError(
            f"reuse result has invalid evaluation key: {source_path}"
        )
    expected_coordinates = {
        "metric_extraction_config_identity": (
            config.metric_config.config_identity_hash
        ),
        "evaluation_procedure_config_identity": (
            config.procedure_config.config_identity_hash
        ),
        "runtime_identity": runtime_identity,
        "runner_identity": runner_identity,
        "metrics_profile": _METRICS_PROFILE,
        "question_identity_hash": config.question_identity_hash,
        "operator_name": config.operator_name,
        "operator_version": config.operator_version,
    }
    if any(
        row.get(name) != value for name, value in expected_coordinates.items()
    ):
        raise CandidateEvaluationError(
            f"reuse result has incompatible coordinates: {source_path}"
        )
    status = _string(row.get("record_status"), "reuse record_status")
    if status != RecordStatus.MEASURED.value:
        return None
    facts = {name: row.get(name) for name in _FACT_FIELDS}
    values_json = _canonical_json(facts)
    reconstructed = _result_row(
        (
            key,
            task_id,
            task_identity,
            source_sha,
            source,
            status,
            row.get("failure_type"),
            row.get("failure_message"),
            values_json,
        ),
        config=config,
        runtime_identity=runtime_identity,
        runner_identity=runner_identity,
    )
    if any(
        reconstructed[field.name] != row.get(field.name)
        for field in RESULTS_SCHEMA
    ):
        raise CandidateEvaluationError(
            f"reuse result fields are inconsistent: {source_path}"
        )
    return (
        key,
        (task_id, task_identity, source_sha, source),
        (
            key,
            status,
            None,
            None,
            values_json,
        ),
    )


def _run_pending_work(
    *,
    connection: sqlite3.Connection,
    lease_id: str,
    tasks: Mapping[str, HumanEvalTask],
    config: _EvaluationConfig,
    max_workers: int,
    max_infrastructure_retries: int,
    runner: BatchExecutor,
    records: Records,
) -> None:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: dict[
            Future[
                tuple[str, MetricRecord | None, _InfrastructureFailure | None]
            ],
            str,
        ] = {}
        while True:
            while len(futures) < max_workers:
                work = _claim_next_work(connection, lease_id)
                if work is None:
                    break
                future = executor.submit(
                    _measure_work,
                    work,
                    tasks[work.task_id],
                    config,
                    runner,
                    records,
                )
                futures[future] = work.evaluation_key
            if not futures:
                return
            completed, _pending = wait(
                futures,
                timeout=_LEASE_SECONDS / 3,
                return_when=FIRST_COMPLETED,
            )
            if not completed:
                _heartbeat(connection, lease_id)
                continue
            for future in completed:
                futures.pop(future)
                key, record, infrastructure_failure = future.result()
                _complete_work(
                    connection,
                    key,
                    record,
                    infrastructure_failure,
                    lease_id=lease_id,
                    max_infrastructure_retries=max_infrastructure_retries,
                )
            _heartbeat(connection, lease_id)


def _claim_next_work(
    connection: sqlite3.Connection, lease_id: str
) -> _Work | None:
    with _lease_transaction(connection, lease_id):
        row = connection.execute(
            """SELECT evaluation_key, task_id, task_identity, source_sha256,
                      candidate_source
                 FROM work WHERE status = 'pending'
                ORDER BY evaluation_key LIMIT 1"""
        ).fetchone()
        if row is None:
            return None
        claimed = connection.execute(
            """UPDATE work
                  SET status = 'running',
                      attempt_count = attempt_count + 1,
                      owner_lease_id = ?
                WHERE evaluation_key = ? AND status = 'pending'
                  AND EXISTS (
                      SELECT 1 FROM evaluator_lease
                       WHERE singleton = 1 AND lease_id = ?
                  )""",
            (lease_id, row[0], lease_id),
        )
        if claimed.rowcount == 1:
            return _Work(*row)
        _validate_lease_owner(connection, lease_id)
        return None


def _measure_work(
    work: _Work,
    task: HumanEvalTask,
    config: _EvaluationConfig,
    runner: BatchExecutor,
    records: Records,
) -> tuple[str, MetricRecord | None, _InfrastructureFailure | None]:
    candidate = CodeArtifact(source=work.candidate_source)
    trace = Trace(
        values={
            "input": candidate,
            "output": candidate,
            "candidate": candidate,
            "task": JsonArtifact(payload=task.model_dump(mode="json")),
        },
        producer=config.trace_producer,
    )
    observing = _AttributionObservingExecutor(runner)
    try:
        metric_records = extract_metrics(
            trace,
            metric_extraction=config.metric_config,
            evaluation_procedure=config.procedure_config,
            executor=observing,
            records=records,
        )
    except ExecutorFailure as exc:
        # A failure with no result to attribute: retriable infrastructure,
        # recorded under the ``executor`` attribution literal.
        return (
            work.evaluation_key,
            None,
            _InfrastructureFailure(
                failure_type=Attribution.EXECUTOR.value,
                failure_message=str(exc),
                retriable=True,
            ),
        )
    infrastructure_failure = _infrastructure_failure_from_outcome(
        observing.last_outcome
    )
    if infrastructure_failure is not None:
        # The metrics lane scored this run's channel/machine/absence outcome as
        # candidate case data; the corpus evaluator reclassifies it as an
        # infrastructure failure and discards the scored record.
        return work.evaluation_key, None, infrastructure_failure
    if len(metric_records) != 1:
        raise CandidateEvaluationError(
            "HumanEval candidate metric produced an unexpected record count"
        )
    record = metric_records[0]
    if (
        record.question != config.operator_name
        or record.question_identity_hash != config.question_identity_hash
        or record.on_key != "candidate"
        or record.evaluation_procedure_config_hash
        != config.procedure_config.config_identity_hash
        or record.trace_producer != config.trace_producer
        or record.operator != config.operator
        or any(
            fact.lineage.question_identity_hash
            != config.question_identity_hash
            or fact.lineage.operator != config.operator_name
            or fact.lineage.operator_version != config.operator_version
            or fact.lineage.evaluation_procedure_config_hash
            != config.procedure_config.config_identity_hash
            for fact in record.facts
        )
    ):
        raise CandidateEvaluationError(
            "HumanEval candidate metric returned forged question/operator "
            "coordinates"
        )
    return work.evaluation_key, record, None


def _infrastructure_failure_from_outcome(
    outcome: Outcome | None,
) -> _InfrastructureFailure | None:
    """Classify a produced run's outcome for infrastructure retry.

    Channel and machine attributions are transient infrastructure faults and
    retriable. An absence attribution (missing interpreter, ENOENT) is a
    non-transient infrastructure failure that terminates immediately — never
    retried. Every other attribution (payload, budget, executor-as-data) is a
    candidate-observable outcome the metrics lane already scored; it is not an
    infrastructure failure here.
    """
    if outcome is None:
        return None
    attribution = outcome.attribution
    if attribution in _RETRIABLE_ATTRIBUTIONS:
        return _InfrastructureFailure(
            failure_type=attribution.value,
            failure_message=(
                f"execution failed with {attribution.value} attribution"
            ),
            retriable=True,
        )
    if attribution is Attribution.ABSENCE:
        return _InfrastructureFailure(
            failure_type=attribution.value,
            failure_message=(
                "execution interpreter was absent; not a transient failure"
            ),
            retriable=False,
        )
    return None


def _complete_work(
    connection: sqlite3.Connection,
    evaluation_key: str,
    record: MetricRecord | None,
    infrastructure_failure: _InfrastructureFailure | None,
    *,
    lease_id: str,
    max_infrastructure_retries: int,
) -> None:
    measured_values: str | None = None
    if record is not None and record.status is RecordStatus.MEASURED:
        try:
            values = CodeTestResult.model_validate(
                record.fact_values(), strict=True
            ).to_values()
        except ValidationError as exc:
            raise CandidateEvaluationError(
                f"measured candidate result facts are invalid: {exc}"
            ) from exc
        measured_values = _canonical_json(values)
    with _lease_transaction(connection, lease_id):
        ownership = connection.execute(
            """SELECT attempt_count, task_id, task_identity, source_sha256,
                      candidate_source
                 FROM work
                WHERE evaluation_key = ? AND status = 'running'
                  AND owner_lease_id = ?
                  AND EXISTS (
                      SELECT 1 FROM evaluator_lease
                       WHERE singleton = 1 AND lease_id = ?
                  )""",
            (evaluation_key, lease_id, lease_id),
        ).fetchone()
        if ownership is None:
            raise CandidateEvaluationError(
                "lost ownership while completing evaluation work"
            )
        identity = ownership[1:]
        if infrastructure_failure is not None:
            attempts = ownership[0]
            failure_type = infrastructure_failure.failure_type
            failure_message = infrastructure_failure.failure_message
            if (
                infrastructure_failure.retriable
                and attempts <= max_infrastructure_retries
            ):
                updated = connection.execute(
                    """UPDATE work SET status = 'pending',
                                      record_status = NULL,
                                      failure_type = ?,
                                      failure_message = ?,
                                      values_json = NULL,
                                      result_evidence_sha256 = NULL,
                                      owner_lease_id = NULL
                        WHERE evaluation_key = ? AND status = 'running'
                          AND owner_lease_id = ?
                          AND EXISTS (
                              SELECT 1 FROM evaluator_lease
                               WHERE singleton = 1 AND lease_id = ?
                          )""",
                    (
                        failure_type,
                        failure_message,
                        evaluation_key,
                        lease_id,
                        lease_id,
                    ),
                )
            else:
                evidence = _result_evidence_sha256(
                    evaluation_key=evaluation_key,
                    task_id=identity[0],
                    task_identity=identity[1],
                    source_sha256=identity[2],
                    candidate_source=identity[3],
                    record_status="infrastructure_failure",
                    failure_type=failure_type,
                    failure_message=failure_message,
                    values_json=None,
                )
                updated = connection.execute(
                    """UPDATE work SET status = 'completed',
                                      record_status = 'infrastructure_failure',
                                      failure_type = ?,
                                      failure_message = ?,
                                      values_json = NULL,
                                      completed_at = ?,
                                      result_evidence_sha256 = ?,
                                      owner_lease_id = NULL
                        WHERE evaluation_key = ? AND status = 'running'
                          AND owner_lease_id = ?
                          AND EXISTS (
                              SELECT 1 FROM evaluator_lease
                               WHERE singleton = 1 AND lease_id = ?
                          )""",
                    (
                        failure_type,
                        failure_message,
                        _timestamp(),
                        evidence,
                        evaluation_key,
                        lease_id,
                        lease_id,
                    ),
                )
        else:
            if record is None:
                raise CandidateEvaluationError(
                    "completed work has neither a record nor an "
                    "infrastructure failure"
                )
            evidence = _result_evidence_sha256(
                evaluation_key=evaluation_key,
                task_id=identity[0],
                task_identity=identity[1],
                source_sha256=identity[2],
                candidate_source=identity[3],
                record_status=record.status.value,
                failure_type=record.failure_type,
                failure_message=record.failure_message,
                values_json=measured_values,
            )
            updated = connection.execute(
                """UPDATE work SET status = 'completed', record_status = ?,
                                  failure_type = ?, failure_message = ?,
                                  values_json = ?, completed_at = ?,
                                  result_evidence_sha256 = ?,
                                  owner_lease_id = NULL
                    WHERE evaluation_key = ? AND status = 'running'
                      AND owner_lease_id = ?
                      AND EXISTS (
                          SELECT 1 FROM evaluator_lease
                           WHERE singleton = 1 AND lease_id = ?
                      )""",
                (
                    record.status.value,
                    record.failure_type,
                    record.failure_message,
                    measured_values,
                    _timestamp(),
                    evidence,
                    evaluation_key,
                    lease_id,
                    lease_id,
                ),
            )
        if updated.rowcount != 1:
            raise CandidateEvaluationError(
                "lost ownership while completing evaluation work"
            )


def _export_artifacts(
    connection: sqlite3.Connection,
    output_dir: Path,
    immutable: Mapping[str, object],
    *,
    config: _EvaluationConfig,
    corpus_path: Path,
    candidates_path: Path,
    lease_id: str,
    reuse_sources: Sequence[_ReuseSource],
    stop_heartbeat: Callable[[], None],
) -> EvaluationArtifacts:
    _validate_completed_result_evidence(connection)
    with staged_generation_directory(output_dir) as staging:
        staged = _StagedEvaluationArtifacts(
            membership_path=staging / MEMBERSHIP_FILENAME,
            results_path=staging / RESULTS_FILENAME,
            manifest_path=staging / MANIFEST_FILENAME,
        )
        _validate_lease_owner(connection, lease_id)
        outstanding = connection.execute(
            "SELECT COUNT(*) FROM work WHERE status != 'completed'"
        ).fetchone()[0]
        if outstanding:
            raise CandidateEvaluationError("cannot export incomplete state")
        membership_rows = connection.execute(
            "SELECT COUNT(*) FROM membership"
        ).fetchone()[0]
        result_rows = connection.execute(
            "SELECT COUNT(*) FROM work"
        ).fetchone()[0]
        reused_counts = dict(
            connection.execute(
                """SELECT reused_from_manifest_sha256, COUNT(*)
                         FROM work
                        WHERE reused_from_manifest_sha256 IS NOT NULL
                        GROUP BY reused_from_manifest_sha256"""
            )
        )
        metric_identity = immutable["metric_extraction_config_identity"]
        procedure_identity = immutable["evaluation_procedure_config_identity"]
        runtime_identity = str(immutable["runtime_identity"])
        runner_identity = str(immutable["runner_identity"])
        _write_query_parquet(
            staged.membership_path,
            MEMBERSHIP_SCHEMA,
            connection.execute(
                """SELECT sample_id, candidate_id, candidate_index,
                              task_id, task_identity, source_kind,
                              source_sha256, evaluation_key
                         FROM membership
                        ORDER BY sample_id, candidate_index, candidate_id"""
            ),
            lambda row: _membership_row(
                row,
                config=config,
                metric_identity=str(metric_identity),
                procedure_identity=str(procedure_identity),
                runtime_identity=runtime_identity,
                runner_identity=runner_identity,
            ),
        )
        _heartbeat(connection, lease_id)
        _write_query_parquet(
            staged.results_path,
            RESULTS_SCHEMA,
            connection.execute(
                """SELECT evaluation_key, task_id, task_identity,
                              source_sha256, candidate_source, record_status,
                              failure_type, failure_message, values_json
                         FROM work ORDER BY evaluation_key"""
            ),
            lambda row: _result_row(
                row,
                config=config,
                runtime_identity=runtime_identity,
                runner_identity=runner_identity,
            ),
        )
        _heartbeat(connection, lease_id)
        try:
            validate_evaluation_relations(
                corpus_path=corpus_path,
                candidates_path=candidates_path,
                membership_path=staged.membership_path,
                results_path=staged.results_path,
                coordinates=immutable,
            )
        except EvaluationRelationsError as exc:
            raise CandidateEvaluationError(str(exc)) from exc
        reuse_provenance = [
            {
                **source.descriptor(),
                "reused_result_rows": reused_counts.get(
                    source.manifest_sha256, 0
                ),
            }
            for source in reuse_sources
        ]
        manifest = {
            **immutable,
            "membership_rows": membership_rows,
            "result_rows": result_rows,
            "candidate_membership_sha256": file_sha256(staged.membership_path),
            "candidate_results_sha256": file_sha256(staged.results_path),
            "record_status_totals": dict(
                connection.execute(
                    """SELECT record_status, COUNT(*) FROM work
                           GROUP BY record_status ORDER BY record_status"""
                )
            ),
            "reused_result_rows": sum(reused_counts.values()),
            "reused_result_rows_by_source": reuse_provenance,
            "complete": True,
        }
        _write_text(
            staged.manifest_path,
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        _heartbeat(connection, lease_id)
        try:
            generation = publish_generation_directory(output_dir, staging)
            with staged_current_switch(
                output_dir, generation
            ) as staged_switch:
                stop_heartbeat()
                generation = _terminal_switch_current(
                    connection,
                    lease_id=lease_id,
                    staged_switch=staged_switch,
                )
        except EvaluationGenerationError as exc:
            raise CandidateEvaluationError(str(exc)) from exc
        return EvaluationArtifacts(
            output_dir=output_dir,
            membership_path=generation.membership_path,
            results_path=generation.results_path,
            manifest_path=generation.manifest_path,
        )


@contextmanager
def _lease_heartbeat(
    connection: sqlite3.Connection, lease_id: str
) -> Iterator[Callable[[], None]]:
    _heartbeat(connection, lease_id)
    database_path = next(
        (
            Path(row[2])
            for row in connection.execute("PRAGMA database_list")
            if row[1] == "main"
        ),
        None,
    )
    if database_path is None:
        raise CandidateEvaluationError(
            "candidate evaluation state has no main database"
        )
    stopped = threading.Event()
    failed = threading.Event()
    errors: list[BaseException] = []

    def maintain() -> None:
        heartbeat_connection: sqlite3.Connection | None = None
        try:
            heartbeat_connection = sqlite3.connect(database_path, timeout=30.0)
            while not _wait_for_lease_heartbeat(stopped):
                _heartbeat(heartbeat_connection, lease_id)
        except BaseException as exc:
            errors.append(exc)
            failed.set()
        finally:
            if heartbeat_connection is not None:
                heartbeat_connection.close()

    def validate() -> None:
        if failed.is_set():
            raise CandidateEvaluationError(
                "candidate evaluation lease was lost during export"
            ) from errors[0]

    heartbeat = threading.Thread(
        target=maintain,
        name=f"candidate-evaluation-heartbeat-{lease_id}",
        daemon=True,
    )
    heartbeat.start()
    joined = False

    def stop_and_validate() -> None:
        nonlocal joined
        if not joined:
            stopped.set()
            heartbeat.join()
            joined = True
        validate()

    try:
        yield stop_and_validate
    finally:
        stop_and_validate()


def _terminal_switch_current(
    connection: sqlite3.Connection,
    *,
    lease_id: str,
    staged_switch: StagedCurrentSwitch,
) -> EvaluationGeneration:
    """Fence the fast pointer switch; durable CURRENT is the commit point."""

    connection.execute("BEGIN IMMEDIATE")
    switched = False
    try:
        _validate_lease_owner(connection, lease_id)
        updated = connection.execute(
            """UPDATE evaluator_lease SET heartbeat_at = ?
               WHERE singleton = 1 AND lease_id = ?""",
            (time.time(), lease_id),
        )
        if updated.rowcount != 1:
            raise CandidateEvaluationError(
                "candidate evaluation lease was lost"
            )
        generation = publish_staged_current_switch(staged_switch)
        switched = True
        connection.commit()
        return generation
    except BaseException:
        if not switched:
            connection.rollback()
            raise
        try:
            connection.commit()
        except sqlite3.Error:
            pass
        return staged_switch.generation


def _wait_for_lease_heartbeat(stopped: threading.Event) -> bool:
    return stopped.wait(_LEASE_SECONDS / 3)


def _result_row(
    row: tuple[object, ...],
    *,
    config: _EvaluationConfig,
    runtime_identity: str,
    runner_identity: str,
) -> dict[str, object]:
    values_json = row[8]
    values = json.loads(values_json) if isinstance(values_json, str) else {}
    if not isinstance(values, dict):
        raise CandidateEvaluationError(
            "persisted fact values must be an object"
        )
    try:
        result = canonical_candidate_result(
            task_id=str(row[1]),
            task_identity=str(row[2]),
            cleaned_source=str(row[4]),
            source_sha256=str(row[3]),
            question_identity_hash=config.question_identity_hash,
            operator_name=config.operator_name,
            operator_version=config.operator_version,
            trace_producer=config.trace_producer,
            operator=config.operator,
            metric_extraction_config_identity=(
                config.metric_config.config_identity_hash
            ),
            evaluation_procedure_config_identity=(
                config.procedure_config.config_identity_hash
            ),
            runtime_identity=runtime_identity,
            runner_identity=runner_identity,
            metrics_profile=_METRICS_PROFILE,
            record_status=row[5],
            failure_type=row[6],
            failure_message=row[7],
            facts={name: values.get(name) for name in _FACT_FIELDS},
        )
    except CandidateEvaluationContractError as exc:
        raise CandidateEvaluationError(str(exc)) from exc
    if result["evaluation_key"] != row[0]:
        raise CandidateEvaluationError(
            "persisted evaluation key is not canonical"
        )
    return result


def _membership_row(
    row: tuple[object, ...],
    *,
    config: _EvaluationConfig,
    metric_identity: str,
    procedure_identity: str,
    runtime_identity: str,
    runner_identity: str,
) -> dict[str, object]:
    return {
        "sample_id": row[0],
        "candidate_id": row[1],
        "candidate_index": row[2],
        "task_id": row[3],
        "task_identity": row[4],
        "source_kind": row[5],
        "source_sha256": row[6],
        "evaluation_key": row[7],
        "question_identity_hash": config.question_identity_hash,
        "operator_name": config.operator_name,
        "operator_version": config.operator_version,
        "trace_producer_json": _canonical_json(
            config.trace_producer.model_dump(mode="json")
        ),
        "operator_coordinates_json": _canonical_json(
            config.operator.model_dump(mode="json")
        ),
        "metric_extraction_config_identity": metric_identity,
        "evaluation_procedure_config_identity": procedure_identity,
        "runtime_identity": runtime_identity,
        "runner_identity": runner_identity,
    }


def _evaluation_key(
    *,
    task_id: str,
    task_identity: str,
    source_sha256: str,
    config: _EvaluationConfig,
    runtime_identity: str,
    runner_identity: str,
) -> str:
    try:
        return candidate_evaluation_key(
            task_id=task_id,
            task_identity=task_identity,
            source_sha256=source_sha256,
            question_identity_hash=config.question_identity_hash,
            operator_name=config.operator_name,
            operator_version=config.operator_version,
            metric_extraction_config_identity=(
                config.metric_config.config_identity_hash
            ),
            evaluation_procedure_config_identity=(
                config.procedure_config.config_identity_hash
            ),
            runtime_identity=runtime_identity,
            runner_identity=runner_identity,
        )
    except CandidateEvaluationContractError as exc:
        raise CandidateEvaluationError(str(exc)) from exc


def _resolve_runner_identity(
    executor: BatchExecutor | None, requested: str | None
) -> str:
    if executor is None:
        if requested not in (None, RUNNER_IDENTITY):
            raise CandidateEvaluationError(
                f"production runner identity must be {RUNNER_IDENTITY!r}"
            )
        return RUNNER_IDENTITY
    if not requested:
        raise CandidateEvaluationError(
            "an injected executor requires an explicit runner_identity"
        )
    if requested == RUNNER_IDENTITY:
        raise CandidateEvaluationError(
            "an injected executor cannot claim the production runner identity"
        )
    return requested


def _runtime_identity(
    *,
    runner_identity: str,
    host_runtime: Mapping[str, object],
    installed_environment: Mapping[str, object],
    trusted_source_sha256: Mapping[str, str],
) -> str:
    return _sha256_text(
        _canonical_json(
            {
                "runner_identity": runner_identity,
                "host_runtime": host_runtime,
                "installed_environment": installed_environment,
                "trusted_source_sha256": trusted_source_sha256,
            }
        )
    )


def _host_runtime_coordinates() -> dict[str, object]:
    return {
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "byteorder": sys.byteorder,
    }


def _trusted_source_fingerprints() -> dict[str, str]:
    return {"checkout_source_tree": checkout_source_tree_sha256()}


def _preflight_production(
    *,
    tasks: Mapping[str, HumanEvalTask],
    config: _EvaluationConfig,
    runner: BatchExecutor,
    records: Records,
) -> None:
    if not tasks:
        raise CandidateEvaluationError("HumanEval+ snapshot contains no tasks")
    try:
        numpy_probe = run_untrusted_python(
            "import numpy\nprint(numpy.__version__)\n",
            profile=HUMANEVAL_PROFILE,
            budgets=Budgets(wall_clock=_PREFLIGHT_TIMEOUT_SECONDS),
            records=records,
            runtime=HUMANEVAL_RUNTIME,
            environment=HUMANEVAL_ENVIRONMENT,
        )
    except ExecutorFailure as exc:
        raise CandidateEvaluationError(
            "production execution preflight could not start"
        ) from exc
    if (
        numpy_probe.outcome.attribution is not Attribution.PAYLOAD
        or numpy_probe.returncode != 0
    ):
        raise CandidateEvaluationError(
            "production execution preflight could not import NumPy"
        )
    _ = config
    for task_id in sorted(tasks):
        task = tasks[task_id]
        try:
            evaluation = evaluate_human_eval_code(
                task=task,
                candidate_code=task.ground_truth_code,
                timeout_seconds=_PREFLIGHT_TIMEOUT_SECONDS,
                executor=runner,
                records=records,
            )
        except Exception as exc:
            raise CandidateEvaluationError(
                f"production HumanEval preflight failed for {task_id}"
            ) from exc
        counts = evaluation.status_counts
        if (
            not evaluation.coverage_complete
            or counts.get("passed", 0) != evaluation.total_cases
        ):
            raise CandidateEvaluationError(
                f"production HumanEval preflight did not pass for {task_id}"
            )


def _validate_exported_parquet(
    stable_file: StableFile,
    schema: pa.Schema,
    *,
    expected_rows: int,
    recorded_sha: object,
) -> str:
    path = stable_file.path
    try:
        parquet = pq.ParquetFile(path)
    except (OSError, pa.ArrowException) as exc:
        raise CandidateEvaluationError(
            f"reuse artifact is invalid: {path}"
        ) from exc
    if not parquet.schema_arrow.equals(schema):
        raise CandidateEvaluationError(
            f"reuse artifact schema mismatch: {path}"
        )
    if parquet.metadata.num_rows != expected_rows:
        raise CandidateEvaluationError(
            f"reuse artifact row count mismatch: {path}"
        )
    actual = stable_file.sha256
    if not _is_sha256(recorded_sha) or recorded_sha != actual:
        raise CandidateEvaluationError(f"reuse artifact hash mismatch: {path}")
    return actual


def _write_query_parquet(
    path: Path,
    schema: pa.Schema,
    rows: sqlite3.Cursor,
    transform: Callable[[tuple[object, ...]], dict[str, object]],
) -> None:
    try:
        with pq.ParquetWriter(
            path,
            schema,
            compression="zstd",
            use_dictionary=False,
            write_statistics=True,
            version="2.6",
        ) as writer:
            batch: list[dict[str, object]] = []
            for row in rows:
                batch.append(transform(row))
                if len(batch) >= 10_000:
                    writer.write_table(
                        pa.Table.from_pylist(batch, schema=schema)
                    )
                    batch.clear()
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_number,
        )
    except (OSError, json.JSONDecodeError, CandidateEvaluationError) as exc:
        raise CandidateEvaluationError(f"{label} is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise CandidateEvaluationError(f"{label} must be a JSON object")
    return value


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CandidateEvaluationError(
                f"JSON object contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _reject_nonfinite_json_number(value: str) -> object:
    raise CandidateEvaluationError(
        f"JSON contains non-finite number {value!r}"
    )


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise CandidateEvaluationError(f"{field} must be a non-empty string")
    return value


def _nonnegative_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CandidateEvaluationError(
            f"{field} must be a non-negative integer"
        )
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "CandidateEvaluationError",
    "EvaluationArtifacts",
    "MANIFEST_FILENAME",
    "MEMBERSHIP_FILENAME",
    "MEMBERSHIP_SCHEMA",
    "RESULTS_FILENAME",
    "RESULTS_SCHEMA",
    "RUNNER_IDENTITY",
    "STATE_FILENAME",
    "evaluate_preprocessing_candidates",
    "humaneval_metric_definition",
]
