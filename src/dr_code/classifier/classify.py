"""Concurrent, resumable failure classification orchestration."""

from __future__ import annotations

import fcntl
import hashlib
import os
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import BinaryIO, Final, Iterator, cast

from dr_code.classifier.aggregation import (
    AGGREGATION_VERSION,
    ItemAggregate,
    RepeatFailure,
    RepeatFailureKind,
    RepeatOutcome,
    RepeatPhase,
    aggregate_repeats,
)
from dr_code.classifier.extraction import (
    EXTRACTION_VERSION,
    FailureItem,
    stream_failures,
)
from dr_code.classifier.lane import (
    Lane,
    LaneTransportError,
    lane_policy,
    lane_policy_identity,
    parse_label_response,
)
from dr_code.classifier.prompt import (
    CORRECTION_ATTEMPTS,
    MAX_SOURCE_CHARS,
    MAX_EVIDENCE_CHARS,
    MAX_METADATA_CHARS,
    MAX_TASK_CONTEXT_CHARS,
    PROMPT_TEMPLATE_VERSION,
    PROMPT_VERSION,
    correction_prompt,
    prompt_template_identity,
)
from dr_code.classifier.records import (
    DETAIL_ARTIFACT_VERSION,
    DETAIL_SCHEMA_VERSION,
    AggregateRecord,
    ClassifierConfigRecord,
    ClassifierExperimentRecord,
    ExperimentHeaderRecord,
    ItemIdentityRecord,
    ItemRecord,
    RepeatFailureRecord,
    RepeatRecord,
    ResumeIdentityRecord,
    RunScopeRecord,
    SelectionPolicyRecord,
    canonical_record_bytes,
    experiment_identity,
    identity_key,
    read_artifact_stream,
    write_records_atomic,
)
from dr_code.classifier.taxonomy import (
    TAXONOMY_VERSION,
    FailureFamily,
    taxonomy_identity,
)
from dr_code.corpus.run_descriptor import RunDescriptor
from dr_code.viewer.analytics import ViewerAnalytics
from dr_code.viewer.domain import (
    TaskAnnotation,
    TaskAnnotationOrigin,
    TaskAnnotationPublicationIntent,
    TaskAnnotationProvenance,
    TaskIdentity,
    validate_task_annotation,
)

MAX_CONCURRENCY: Final = 8
MIXED_CATEGORY: Final = "mixed"
MACHINE_PRODUCER: Final = "dr_code.failure_classifier"
MAX_AUDIT_DETAIL_CHARS: Final = 512


@dataclass(frozen=True, slots=True)
class ClassificationSummary:
    experiment_identity: str
    run_id: str
    dataset_id: str
    provider: str
    model: str
    repeats: int
    taxonomy_version: str
    prompt_version: str
    parse_total: int
    parse_selected: int
    test_total: int
    test_selected: int
    resumed: int
    classified: int
    repeat_failures: int
    mean_agreement: float | None
    label_counts: dict[str, dict[str, int]]
    tasks_written: int
    tasks_protected: int
    tasks_removed: int
    details_path: Path
    details_sha256: str


class _InputSpool:
    """Disk-backed, fully authenticated provider-input snapshot."""

    def __init__(self, parent: Path) -> None:
        descriptor, raw_path = tempfile.mkstemp(
            prefix=".classification-inputs.",
            suffix=".sqlite3",
            dir=parent,
        )
        os.close(descriptor)
        self._path = Path(raw_path)
        self._database = sqlite3.connect(self._path)
        self._database.execute(
            "CREATE TABLE inputs ("
            "position INTEGER PRIMARY KEY,"
            "identity BLOB NOT NULL UNIQUE,"
            "family TEXT NOT NULL,"
            "sample_id TEXT NOT NULL,"
            "candidate_id TEXT,"
            "evaluation_key TEXT,"
            "dataset_id TEXT NOT NULL,"
            "task_id TEXT,"
            "task_identity TEXT,"
            "rendered_input TEXT NOT NULL"
            ")"
        )

    def __enter__(self) -> _InputSpool:
        return self

    def __exit__(self, *exc_info: object) -> None:
        try:
            self._database.close()
        finally:
            self._path.unlink(missing_ok=True)

    def add(
        self,
        item: FailureItem,
        identity: ResumeIdentityRecord,
    ) -> None:
        try:
            self._database.execute(
                "INSERT INTO inputs("
                "identity, family, sample_id, candidate_id, evaluation_key, "
                "dataset_id, task_id, task_identity, rendered_input"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    identity_key(identity),
                    item.family.value,
                    item.sample_id,
                    item.candidate_id,
                    item.evaluation_key,
                    item.dataset_id,
                    item.task_id,
                    item.task_identity,
                    item.rendered_input,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                "duplicate extracted classification identity"
            ) from exc

    def iter_items(
        self,
    ) -> Iterator[tuple[FailureItem, ResumeIdentityRecord]]:
        cursor = self._database.execute(
            "SELECT identity, family, sample_id, candidate_id, "
            "evaluation_key, dataset_id, task_id, task_identity, "
            "rendered_input FROM inputs ORDER BY position"
        )
        while rows := cursor.fetchmany(128):
            for row in rows:
                identity = ResumeIdentityRecord.model_validate_json(
                    cast(bytes, row[0]),
                    strict=True,
                )
                yield (
                    FailureItem(
                        family=FailureFamily(cast(str, row[1])),
                        sample_id=cast(str, row[2]),
                        candidate_id=cast(str | None, row[3]),
                        evaluation_key=cast(str | None, row[4]),
                        dataset_id=cast(str, row[5]),
                        task_id=cast(str | None, row[6]),
                        task_identity=cast(str | None, row[7]),
                        rendered_input=cast(str, row[8]),
                    ),
                    identity,
                )

    def iter_task_identities(
        self,
    ) -> Iterator[tuple[str, str, str]]:
        cursor = self._database.execute(
            "SELECT DISTINCT dataset_id, task_id, task_identity "
            "FROM inputs WHERE task_id IS NOT NULL "
            "ORDER BY dataset_id, task_id, task_identity"
        )
        while rows := cursor.fetchmany(128):
            for dataset_id, task_id, task_identity in rows:
                yield (
                    cast(str, dataset_id),
                    cast(str, task_id),
                    cast(str, task_identity),
                )

    def selected_count(self, family: FailureFamily) -> int:
        row = self._database.execute(
            "SELECT count(*) FROM inputs WHERE family = ?",
            (family.value,),
        ).fetchone()
        assert row is not None
        return cast(int, row[0])


class _RecordSpool:
    """Disk-backed identity index and deterministic record spool."""

    def __init__(self, parent: Path) -> None:
        descriptor, raw_path = tempfile.mkstemp(
            prefix=".classification-records.",
            suffix=".sqlite3",
            dir=parent,
        )
        os.close(descriptor)
        self._path = Path(raw_path)
        self._database = sqlite3.connect(self._path)
        self._database.execute(
            "CREATE TABLE records ("
            "identity BLOB PRIMARY KEY,"
            "payload BLOB NOT NULL,"
            "task_id TEXT,"
            "task_identity TEXT"
            ")"
        )
        self._database.execute(
            "CREATE TABLE selected (identity BLOB PRIMARY KEY)"
        )

    def __enter__(self) -> _RecordSpool:
        return self

    def __exit__(self, *exc_info: object) -> None:
        try:
            self._database.close()
        finally:
            self._path.unlink(missing_ok=True)

    def clear(self) -> None:
        self._database.execute("DELETE FROM records")
        self._database.execute("DELETE FROM selected")

    def add_existing(self, record: ItemRecord) -> None:
        try:
            self._database.execute(
                "INSERT INTO records("
                "identity, payload, task_id, task_identity"
                ") VALUES (?, ?, ?, ?)",
                (
                    identity_key(record.identity),
                    canonical_record_bytes(record),
                    record.identity.item.task_id,
                    record.identity.item.task_identity,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                "duplicate classification record identity"
            ) from exc

    def select(
        self,
        identity: ResumeIdentityRecord,
        *,
        resume: bool,
    ) -> ItemRecord | None:
        key = identity_key(identity)
        try:
            self._database.execute(
                "INSERT INTO selected(identity) VALUES (?)",
                (key,),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                "duplicate extracted classification identity"
            ) from exc
        if not resume:
            return None
        row = self._database.execute(
            "SELECT payload FROM records WHERE identity = ?",
            (key,),
        ).fetchone()
        return (
            _record_from_payload(cast(bytes, row[0]))
            if row is not None
            else None
        )

    def add_completed(self, record: ItemRecord) -> None:
        key = identity_key(record.identity)
        try:
            self._database.execute(
                "INSERT INTO records("
                "identity, payload, task_id, task_identity"
                ") VALUES (?, ?, ?, ?)",
                (
                    key,
                    canonical_record_bytes(record),
                    record.identity.item.task_id,
                    record.identity.item.task_identity,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                "duplicate completed classification identity"
            ) from exc

    def select_all(self) -> None:
        self._database.execute(
            "INSERT INTO selected(identity) SELECT identity FROM records"
        )

    def iter_selected(self) -> Iterator[ItemRecord]:
        cursor = self._database.execute(
            "SELECT r.payload "
            "FROM selected AS s "
            "JOIN records AS r USING (identity) "
            "ORDER BY s.identity"
        )
        while rows := cursor.fetchmany(128):
            for (payload,) in rows:
                yield _record_from_payload(cast(bytes, payload))

    def iter_selected_by_task(self) -> Iterator[ItemRecord]:
        cursor = self._database.execute(
            "SELECT r.payload "
            "FROM selected AS s "
            "JOIN records AS r USING (identity) "
            "WHERE r.task_id IS NOT NULL "
            "ORDER BY r.task_id, r.task_identity, s.identity"
        )
        while rows := cursor.fetchmany(128):
            for (payload,) in rows:
                yield _record_from_payload(cast(bytes, payload))

    def selected_count(self) -> int:
        row = self._database.execute(
            "SELECT count(*) FROM selected"
        ).fetchone()
        assert row is not None
        return cast(int, row[0])

    def validate_complete(self) -> None:
        row = self._database.execute(
            "SELECT count(*) "
            "FROM selected AS s "
            "LEFT JOIN records AS r USING (identity) "
            "WHERE r.identity IS NULL"
        ).fetchone()
        assert row is not None
        if cast(int, row[0]):
            raise RuntimeError(
                "classification record spool has incomplete selected items"
            )


def _record_from_payload(payload: bytes) -> ItemRecord:
    return ItemRecord.model_validate_json(payload, strict=True)


def classify_one_repeat(
    lane: Lane,
    family: FailureFamily,
    prompt: str,
) -> RepeatOutcome:
    """Classify once, with one correction only for an invalid response."""
    try:
        raw = lane.complete(prompt)
    except LaneTransportError as exc:
        return _transport_failure(exc)
    try:
        response = parse_label_response(raw, family)
    except ValueError as first_error:
        primary_validation_failure = _bounded_audit_detail(str(first_error))
        try:
            raw = lane.complete(
                correction_prompt(prompt, primary_validation_failure)
            )
        except LaneTransportError as exc:
            return _transport_failure(
                exc,
                phase=RepeatPhase.CORRECTION,
                primary_validation_failure=primary_validation_failure,
            )
        try:
            response = parse_label_response(raw, family)
        except ValueError as second_error:
            return RepeatOutcome(
                label=None,
                rationale=None,
                failure=RepeatFailure(
                    kind=RepeatFailureKind.INVALID_RESPONSE,
                    detail=_bounded_audit_detail(str(second_error)),
                ),
                phase=RepeatPhase.CORRECTION,
                attempt=2,
                corrected=False,
                primary_validation_failure=primary_validation_failure,
            )
        return RepeatOutcome(
            label=response.label,
            rationale=response.rationale,
            phase=RepeatPhase.CORRECTION,
            attempt=2,
            corrected=True,
            primary_validation_failure=primary_validation_failure,
        )
    return RepeatOutcome(
        label=response.label,
        rationale=response.rationale,
    )


def classify_item(
    lane: Lane,
    item: FailureItem,
    *,
    repeats: int,
) -> tuple[ItemAggregate, tuple[RepeatOutcome, ...]]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    outcomes = tuple(
        classify_one_repeat(lane, item.family, item.rendered_input)
        for _ in range(repeats)
    )
    return aggregate_repeats(outcomes), outcomes


def run_classification(
    analytics: ViewerAnalytics,
    descriptor: RunDescriptor,
    lane: Lane,
    *,
    details_path: Path,
    repeats: int = 5,
    parse_limit: int | None = 300,
    test_limit: int | None = 100,
    concurrency: int = 4,
    force: bool = False,
) -> ClassificationSummary:
    """Classify selected failures, checkpoint details, then write rollups."""
    requested_details_path = details_path.expanduser()
    details_path = requested_details_path.resolve()
    _validate_details_output_path(details_path)
    if requested_details_path.is_symlink():
        raise ValueError(
            "classification details path must not be a symbolic link"
        )
    with _classification_output_lock(details_path):
        with analytics.classification_serialization():
            return _run_classification_locked(
                analytics,
                descriptor,
                lane,
                details_path=details_path,
                repeats=repeats,
                parse_limit=parse_limit,
                test_limit=test_limit,
                concurrency=concurrency,
                force=force,
            )


def _run_classification_locked(
    analytics: ViewerAnalytics,
    descriptor: RunDescriptor,
    lane: Lane,
    *,
    details_path: Path,
    repeats: int,
    parse_limit: int | None,
    test_limit: int | None,
    concurrency: int,
    force: bool,
) -> ClassificationSummary:
    """Private classifier workflow; caller owns output then database locks."""
    descriptor = analytics.require_registered_descriptor(descriptor)
    _validate_run_configuration(
        descriptor=descriptor,
        lane=lane,
        repeats=repeats,
        concurrency=concurrency,
    )
    experiment = build_classification_experiment(
        descriptor,
        lane,
        repeats=repeats,
        parse_limit=parse_limit,
        test_limit=test_limit,
    )
    experiment_sha256 = experiment_identity(experiment)
    with (
        _InputSpool(details_path.parent) as inputs,
        _RecordSpool(details_path.parent) as spool,
    ):
        parse_total, test_total = _freeze_classification_inputs(
            analytics,
            descriptor,
            experiment_sha256=experiment_sha256,
            repeats=repeats,
            parse_limit=parse_limit,
            test_limit=test_limit,
            inputs=inputs,
        )
        return _classify_with_spool(
            analytics,
            descriptor,
            lane,
            details_path=details_path,
            repeats=repeats,
            concurrency=concurrency,
            force=force,
            experiment=experiment,
            experiment_sha256=experiment_sha256,
            parse_total=parse_total,
            test_total=test_total,
            inputs=inputs,
            spool=spool,
        )


def _freeze_classification_inputs(  # noqa: PLR0913
    analytics: ViewerAnalytics,
    descriptor: RunDescriptor,
    *,
    experiment_sha256: str,
    repeats: int,
    parse_limit: int | None,
    test_limit: int | None,
    inputs: _InputSpool,
) -> tuple[int, int]:
    """Freeze and authenticate the complete selected scope before lane work."""
    extracted = stream_failures(
        analytics,
        descriptor.run_id,
        parse_limit=parse_limit,
        test_limit=test_limit,
    )
    for item in extracted.items:
        identity = _resume_identity(
            experiment_sha256,
            repeats,
            item,
            dataset_id=descriptor.dataset_id,
        )
        inputs.add(item, identity)
    expected_parse = (
        extracted.parse_total
        if parse_limit is None
        else min(parse_limit, extracted.parse_total)
    )
    expected_test = (
        extracted.test_total
        if test_limit is None
        else min(test_limit, extracted.test_total)
    )
    if inputs.selected_count(FailureFamily.PARSE) != expected_parse:
        raise RuntimeError(
            "parse classification pagination did not freeze the exact "
            "selected population"
        )
    if inputs.selected_count(FailureFamily.TEST) != expected_test:
        raise RuntimeError(
            "test classification pagination did not freeze the exact "
            "selected population"
        )
    analytics.validate_classification_task_scope(
        descriptor.run_id,
        inputs.iter_task_identities(),
    )
    return extracted.parse_total, extracted.test_total


def _classify_with_spool(  # noqa: PLR0913
    analytics: ViewerAnalytics,
    descriptor: RunDescriptor,
    lane: Lane,
    *,
    details_path: Path,
    repeats: int,
    concurrency: int,
    force: bool,
    experiment: ClassifierExperimentRecord,
    experiment_sha256: str,
    parse_total: int,
    test_total: int,
    inputs: _InputSpool,
    spool: _RecordSpool,
) -> ClassificationSummary:
    _recover_task_annotation_publication(
        analytics,
        details_path,
    )
    captured = _load_artifact_capture(
        details_path,
        on_record=spool.add_existing,
    )
    artifact = captured[0] if captured is not None else None
    if (
        artifact is not None
        and artifact.experiment_identity != experiment_sha256
    ):
        raise ValueError(
            "classification details path belongs to a different experiment"
        )
    staged_path = _staged_artifact_path(details_path)
    staged_capture = None
    if staged_path.exists():
        spool.clear()
        staged_capture = _load_artifact_capture(
            staged_path,
            on_record=spool.add_existing,
        )
    staged_artifact = staged_capture[0] if staged_capture is not None else None
    if (
        staged_artifact is not None
        and staged_artifact.experiment_identity != experiment_sha256
    ):
        raise ValueError(
            "classification checkpoint belongs to a different experiment"
        )
    if force:
        _unlink_and_fsync(staged_path)
        spool.clear()
    resume = not force and (
        staged_artifact is not None or artifact is not None
    )
    resumed = 0
    classified = 0
    parse_selected = inputs.selected_count(FailureFamily.PARSE)
    test_selected = inputs.selected_count(FailureFamily.TEST)
    next_checkpoint_size = 1

    def classify_pending(
        item: FailureItem, identity: ResumeIdentityRecord
    ) -> ItemRecord:
        aggregate, outcomes = classify_item(lane, item, repeats=repeats)
        return _item_record(identity, aggregate, outcomes)

    try:
        items = inputs.iter_items()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            exhausted = False
            while not exhausted:
                pending: list[tuple[FailureItem, ResumeIdentityRecord]] = []
                while len(pending) < concurrency:
                    try:
                        item, identity = next(items)
                    except StopIteration:
                        exhausted = True
                        break
                    record = spool.select(identity, resume=resume)
                    if record is None:
                        pending.append((item, identity))
                        classified += 1
                    else:
                        resumed += 1
                futures: tuple[Future[ItemRecord], ...] = tuple(
                    executor.submit(classify_pending, item, identity)
                    for item, identity in pending
                )
                if futures:
                    done, _ = wait(futures)
                else:
                    done = set()
                for future in done:
                    record = future.result()
                    spool.add_completed(record)
                spool.validate_complete()
                selected_count = spool.selected_count()
                if done and selected_count >= next_checkpoint_size:
                    write_records_atomic(
                        staged_path,
                        experiment,
                        spool.iter_selected(),
                        records_are_sorted=True,
                    )
                    while next_checkpoint_size <= selected_count:
                        next_checkpoint_size *= 2

        spool.validate_complete()
        write_records_atomic(
            staged_path,
            experiment,
            spool.iter_selected(),
            records_are_sorted=True,
        )
        _, details_sha256 = _authenticate_completed_artifact(
            staged_path,
            experiment_sha256=experiment_sha256,
            records=spool.iter_selected(),
        )
    except BaseException:
        if force:
            _unlink_and_fsync(staged_path)
        raise

    intent = TaskAnnotationPublicationIntent(
        producer=MACHINE_PRODUCER,
        experiment_identity=experiment_sha256,
        output_path=str(details_path),
        staged_path=str(staged_path),
        prior_sha256=_file_sha256_or_none(details_path),
        intended_sha256=details_sha256,
    )
    analytics.begin_task_annotation_publication(intent)
    _publish_staged_artifact(staged_path, details_path)
    tasks_written, tasks_protected, tasks_removed = _finalize_task_rollups(
        analytics,
        spool.iter_selected_by_task,
        experiment=experiment,
        experiment_sha256=experiment_sha256,
        details_sha256=details_sha256,
        intent=intent,
    )
    family_counts = _family_label_counts(spool.iter_selected())
    return ClassificationSummary(
        experiment_identity=experiment_sha256,
        run_id=descriptor.run_id,
        dataset_id=descriptor.dataset_id,
        provider=lane.provider,
        model=lane.model,
        repeats=repeats,
        taxonomy_version=TAXONOMY_VERSION,
        prompt_version=PROMPT_VERSION,
        parse_total=parse_total,
        parse_selected=parse_selected,
        test_total=test_total,
        test_selected=test_selected,
        resumed=resumed,
        classified=classified,
        repeat_failures=sum(
            record.aggregate.failed_repeats for record in spool.iter_selected()
        ),
        mean_agreement=_mean_agreement_records(spool.iter_selected()),
        label_counts=family_counts,
        tasks_written=tasks_written,
        tasks_protected=tasks_protected,
        tasks_removed=tasks_removed,
        details_path=details_path,
        details_sha256=details_sha256,
    )


def build_classification_experiment(
    descriptor: RunDescriptor,
    lane: Lane,
    *,
    repeats: int,
    parse_limit: int | None,
    test_limit: int | None,
) -> ClassifierExperimentRecord:
    """Build the sole typed identity owner for output and resume behavior."""
    _validate_selection_limit(parse_limit, "parse_limit")
    _validate_selection_limit(test_limit, "test_limit")
    return ClassifierExperimentRecord(
        run=_run_scope(descriptor),
        config=_classifier_config(lane, repeats),
        selection=SelectionPolicyRecord(
            parse_limit=parse_limit,
            test_limit=test_limit,
        ),
    )


def _validate_run_configuration(
    *,
    descriptor: RunDescriptor,
    lane: Lane,
    repeats: int,
    concurrency: int,
) -> None:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if not 1 <= concurrency <= MAX_CONCURRENCY:
        raise ValueError(
            f"concurrency must be between 1 and {MAX_CONCURRENCY}"
        )
    if not descriptor.dataset_id or descriptor.dataset_id.strip() != (
        descriptor.dataset_id
    ):
        raise ValueError("descriptor dataset_id must be nonblank and trimmed")
    for value, name in (
        (lane.provider, "provider"),
        (lane.model, "model"),
    ):
        if not value or value.strip() != value:
            raise ValueError(f"{name} must be nonblank and trimmed")


def _run_scope(descriptor: RunDescriptor) -> RunScopeRecord:
    return RunScopeRecord(
        run_id=descriptor.run_id,
        dataset_id=descriptor.dataset_id,
        corpus_sha256=descriptor.corpus_sha256,
        preprocessing_manifest_sha256=(
            descriptor.preprocessing_manifest_sha256
        ),
        preprocessing_identity=descriptor.preprocessing_identity,
        preprocessing_schema_version=descriptor.preprocessing_schema_version,
        definition_id=descriptor.definition_id,
        definition_version=descriptor.definition_version,
        definition_identity=descriptor.definition_identity,
        evaluation_manifest_sha256=descriptor.evaluation_manifest_sha256,
        evaluation_generation_id=descriptor.evaluation_generation_id,
        evaluation_pointer_sha256=descriptor.evaluation_pointer_sha256,
        evaluation_identity=descriptor.evaluation_identity,
    )


def _classifier_config(lane: Lane, repeats: int) -> ClassifierConfigRecord:
    transport = lane_policy(lane)
    transport_settings = dict(transport.transport)
    executable_value = transport_settings.get("executable")
    timeout_value = transport_settings.get("timeout_seconds")
    executable = (
        executable_value if isinstance(executable_value, str) else None
    )
    timeout_seconds = (
        float(timeout_value)
        if isinstance(timeout_value, (int, float))
        and not isinstance(timeout_value, bool)
        else None
    )
    return ClassifierConfigRecord(
        artifact_version=DETAIL_ARTIFACT_VERSION,
        schema_version=DETAIL_SCHEMA_VERSION,
        extraction_version=EXTRACTION_VERSION,
        aggregation_version=AGGREGATION_VERSION,
        taxonomy_version=TAXONOMY_VERSION,
        taxonomy_identity=taxonomy_identity(),
        prompt_version=PROMPT_VERSION,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
        prompt_template_identity=prompt_template_identity(),
        prompt_max_evidence_chars=MAX_EVIDENCE_CHARS,
        prompt_max_input_chars=MAX_SOURCE_CHARS,
        prompt_max_task_context_chars=MAX_TASK_CONTEXT_CHARS,
        prompt_max_metadata_chars=MAX_METADATA_CHARS,
        prompt_correction_attempts=CORRECTION_ATTEMPTS,
        provider=lane.provider,
        model=lane.model,
        lane_policy_identity=lane_policy_identity(transport),
        lane_adapter=transport.adapter,
        lane_executable=executable,
        lane_timeout_seconds=timeout_seconds,
        repeats=repeats,
    )


def _resume_identity(
    experiment_sha256: str,
    repeats: int,
    item: FailureItem,
    *,
    dataset_id: str,
) -> ResumeIdentityRecord:
    if item.dataset_id != dataset_id:
        raise ValueError(
            "classification input dataset_id does not match run descriptor"
        )
    return ResumeIdentityRecord(
        experiment_identity=experiment_sha256,
        repeats=repeats,
        item=ItemIdentityRecord(
            family=item.family,
            sample_id=item.sample_id,
            candidate_id=item.candidate_id,
            evaluation_key=item.evaluation_key,
            task_id=item.task_id,
            task_identity=item.task_identity,
            rendered_input_sha256=hashlib.sha256(
                item.rendered_input.encode("utf-8")
            ).hexdigest(),
        ),
    )


def _item_record(
    identity: ResumeIdentityRecord,
    aggregate: ItemAggregate,
    outcomes: tuple[RepeatOutcome, ...],
) -> ItemRecord:
    return ItemRecord(
        identity=identity,
        aggregate=AggregateRecord(
            label=aggregate.label,
            agreement=aggregate.agreement,
            tie=aggregate.tie,
            successful_repeats=aggregate.successful_repeats,
            failed_repeats=aggregate.failed_repeats,
            label_counts=dict(aggregate.label_counts),
        ),
        repeats=tuple(
            RepeatRecord(
                index=index,
                label=outcome.label,
                rationale=outcome.rationale,
                failure=(
                    RepeatFailureRecord(
                        kind=outcome.failure.kind,
                        detail=outcome.failure.detail,
                    )
                    if outcome.failure is not None
                    else None
                ),
                phase=outcome.phase,
                attempt=outcome.attempt,
                corrected=outcome.corrected,
                primary_validation_failure=(
                    outcome.primary_validation_failure
                ),
            )
            for index, outcome in enumerate(outcomes)
        ),
    )


def _transport_failure(
    error: LaneTransportError,
    *,
    phase: RepeatPhase = RepeatPhase.PRIMARY,
    primary_validation_failure: str | None = None,
) -> RepeatOutcome:
    return RepeatOutcome(
        label=None,
        rationale=None,
        failure=RepeatFailure(
            kind=RepeatFailureKind.TRANSPORT,
            detail=error.kind.value,
        ),
        phase=phase,
        attempt=2 if phase is RepeatPhase.CORRECTION else 1,
        corrected=False,
        primary_validation_failure=primary_validation_failure,
    )


def _bounded_audit_detail(detail: str) -> str:
    detail = detail.strip()
    if not detail:
        return "invalid response"
    marker = "...[audit truncated]"
    if len(detail) > MAX_AUDIT_DETAIL_CHARS:
        return detail[: MAX_AUDIT_DETAIL_CHARS - len(marker)] + marker
    return detail


def _family_label_counts(
    records: Iterable[ItemRecord],
) -> dict[str, dict[str, int]]:
    counts: dict[FailureFamily, Counter[str]] = {
        family: Counter() for family in FailureFamily
    }
    for record in records:
        label = record.aggregate.label
        if label is not None:
            counts[record.identity.item.family][label] += 1
    return {
        family.value: dict(sorted(counts[family].items()))
        for family in FailureFamily
    }


def _mean_agreement_records(records: Iterable[ItemRecord]) -> float | None:
    total = 0.0
    count = 0
    for record in records:
        agreement = record.aggregate.agreement
        if agreement is not None:
            total += agreement
            count += 1
    return total / count if count else None


def _finalize_task_rollups(
    analytics: ViewerAnalytics,
    records: Callable[[], Iterable[ItemRecord]],
    *,
    experiment: ClassifierExperimentRecord,
    experiment_sha256: str,
    details_sha256: str,
    intent: TaskAnnotationPublicationIntent,
) -> tuple[int, int, int]:
    dataset_id = experiment.run.dataset_id

    def scope() -> Iterator[TaskIdentity]:
        for (task_id, task_identity), _ in groupby(
            records(),
            key=_record_task_key,
        ):
            yield TaskIdentity(
                dataset_id=dataset_id,
                task_id=task_id,
                task_identity=task_identity,
            )

    def annotations() -> Iterator[TaskAnnotation]:
        config = experiment.config
        run = experiment.run
        for (task_id, task_identity), task_records in groupby(
            records(),
            key=_record_task_key,
        ):
            category_counts: Counter[tuple[FailureFamily, str]] = Counter()
            namespace_counts: dict[FailureFamily, Counter[str]] = {
                family: Counter() for family in FailureFamily
            }
            agreement_total = 0.0
            agreement_count = 0
            for record in task_records:
                label = record.aggregate.label
                if label is None:
                    continue
                family = record.identity.item.family
                category_counts[(family, label)] += 1
                namespace_counts[family][label] += 1
                agreement = record.aggregate.agreement
                if agreement is not None:
                    agreement_total += agreement
                    agreement_count += 1
            if not category_counts:
                continue
            category = _dominant_category(category_counts)
            label_counts = {
                family.value: dict(sorted(namespace_counts[family].items()))
                for family in FailureFamily
            }
            provenance = TaskAnnotationProvenance(
                model=config.model,
                taxonomy_version=config.taxonomy_version,
                repeats=config.repeats,
                agreement=(
                    agreement_total / agreement_count
                    if agreement_count
                    else None
                ),
                extra={
                    "details_sha256": details_sha256,
                    "experiment_identity": experiment_sha256,
                    "label_counts": label_counts,
                    "producer": MACHINE_PRODUCER,
                    "prompt_version": config.prompt_version,
                    "provider": config.provider,
                    "task_identity": task_identity,
                    "run": {
                        "corpus_sha256": run.corpus_sha256,
                        "dataset_id": run.dataset_id,
                        "evaluation_manifest_sha256": (
                            run.evaluation_manifest_sha256
                        ),
                        "preprocessing_manifest_sha256": (
                            run.preprocessing_manifest_sha256
                        ),
                        "run_id": run.run_id,
                    },
                    "schema_version": config.schema_version,
                    "selection": experiment.selection.model_dump(mode="json"),
                },
            )
            yield validate_task_annotation(
                identity=TaskIdentity(
                    dataset_id=dataset_id,
                    task_id=task_id,
                    task_identity=task_identity,
                ),
                origin=TaskAnnotationOrigin.MACHINE,
                category=category,
                note=_rollup_note(label_counts),
                tags=(),
                provenance=provenance,
            )

    result = analytics.finalize_task_annotation_publication(
        scope(),
        annotations(),
        intent=intent,
    )
    return result.written, result.protected, result.removed


def _record_task_key(record: ItemRecord) -> tuple[str, str]:
    return (
        cast(str, record.identity.item.task_id),
        cast(str, record.identity.item.task_identity),
    )


def _dominant_category(
    counts: Counter[tuple[FailureFamily, str]],
) -> str:
    top_count = max(counts.values())
    winners = sorted(
        category for category, count in counts.items() if count == top_count
    )
    return winners[0][1] if len(winners) == 1 else MIXED_CATEGORY


def _rollup_note(counts: dict[str, dict[str, int]]) -> str:
    parts = [
        f"{family}:{label}={count}"
        for family, labels in sorted(counts.items())
        for label, count in sorted(labels.items())
    ]
    return "Failure classifications: " + ", ".join(parts)


def _validate_selection_limit(value: int | None, label: str) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value < 1
    ):
        raise ValueError(f"{label} must be a positive integer or null")


def _validate_details_output_path(path: Path) -> None:
    name = path.name
    if name.startswith(".") and name.endswith((".publication", ".lock")):
        raise ValueError(
            "classification details path basename is reserved for internal "
            "publication state"
        )


@contextmanager
def _classification_output_lock(
    path: Path,
) -> Iterator[None]:
    """Hold the one canonical lock for a classification output path."""
    path = path.expanduser().resolve()
    _validate_details_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _output_lock_path(path)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _staged_artifact_path(path: Path) -> Path:
    return path.parent / f".{path.name}.publication"


def _output_lock_path(path: Path) -> Path:
    return path.parent / f".{path.name}.lock"


def _recover_task_annotation_publication(
    analytics: ViewerAnalytics,
    details_path: Path,
) -> None:
    """Resolve only the two unambiguous persisted publication states."""
    intent = analytics.get_task_annotation_publication_intent(
        str(details_path)
    )
    if intent is None:
        return
    staged_path = _staged_artifact_path(details_path)
    if (
        intent.producer != MACHINE_PRODUCER
        or intent.output_path != str(details_path)
        or intent.staged_path != str(staged_path)
    ):
        raise RuntimeError(
            "pending classification publication intent has invalid ownership "
            "or paths; machine rollups remain suppressed"
        )
    _reject_symlink(details_path, "classification output")
    _reject_symlink(staged_path, "classification staged artifact")
    output_sha256 = _file_sha256_or_none(details_path)
    staged_sha256 = _file_sha256_or_none(staged_path)

    if (
        output_sha256 == intent.prior_sha256
        and staged_sha256 == intent.intended_sha256
    ):
        with _RecordSpool(details_path.parent) as spool:
            if intent.prior_sha256 is not None:
                _authenticate_completed_artifact(
                    details_path,
                    experiment_sha256=intent.experiment_identity,
                    expected_sha256=intent.prior_sha256,
                    on_record=spool.add_existing,
                )
                spool.clear()
            _authenticate_completed_artifact(
                staged_path,
                experiment_sha256=intent.experiment_identity,
                expected_sha256=intent.intended_sha256,
                on_record=spool.add_existing,
            )
        analytics.abort_task_annotation_publication(intent)
        _unlink_and_fsync(staged_path)
        return

    if output_sha256 == intent.intended_sha256 and staged_sha256 is None:
        with _RecordSpool(details_path.parent) as spool:
            header, _ = _authenticate_completed_artifact(
                details_path,
                experiment_sha256=intent.experiment_identity,
                expected_sha256=intent.intended_sha256,
                on_record=spool.add_existing,
            )
            spool.select_all()
            _finalize_task_rollups(
                analytics,
                spool.iter_selected_by_task,
                experiment=header.experiment,
                experiment_sha256=intent.experiment_identity,
                details_sha256=intent.intended_sha256,
                intent=intent,
            )
        return

    raise RuntimeError(
        "ambiguous classification publication evidence: expected either "
        "the prior output plus intended staged artifact, or the intended "
        "output with no staged artifact; machine rollups remain suppressed"
    )


def _authenticate_completed_artifact(
    path: Path,
    *,
    experiment_sha256: str,
    records: Iterable[ItemRecord] | None = None,
    expected_sha256: str | None = None,
    on_record: Callable[[ItemRecord], None] | None = None,
) -> tuple[ExperimentHeaderRecord, str]:
    _reject_symlink(path, "classification artifact")
    expected_records = iter(records) if records is not None else None

    def authenticate_record(record: ItemRecord) -> None:
        if expected_records is not None:
            try:
                expected = next(expected_records)
            except StopIteration as exc:
                raise RuntimeError(
                    "classification staged artifact contains unexpected "
                    "records"
                ) from exc
            if record != expected:
                raise RuntimeError(
                    "classification staged artifact is not the complete "
                    "intended record set"
                )
        if on_record is not None:
            on_record(record)

    try:
        with path.open("rb") as stream:
            header, actual_sha256 = _parse_artifact_stream(
                stream,
                on_record=authenticate_record,
            )
            os.fsync(stream.fileno())
    except FileNotFoundError:
        raise RuntimeError("classification artifact disappeared")
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise RuntimeError(
            "classification artifact content changed during publication"
        )
    if header.experiment_identity != experiment_sha256:
        raise RuntimeError(
            "classification artifact does not match publication experiment"
        )
    if expected_records is not None:
        try:
            next(expected_records)
        except StopIteration:
            pass
        else:
            raise RuntimeError(
                "classification staged artifact is not the complete intended "
                "record set"
            )
    return header, actual_sha256


def _load_artifact_capture(
    path: Path,
    *,
    on_record: Callable[[ItemRecord], None],
) -> tuple[ExperimentHeaderRecord, str] | None:
    """Hash and parse one stable stream into a caller-owned record sink."""
    _reject_symlink(path, "classification artifact")
    try:
        with path.open("rb") as stream:
            return _parse_artifact_stream(stream, on_record=on_record)
    except FileNotFoundError:
        return None


def _parse_artifact_stream(
    stream: BinaryIO,
    *,
    on_record: Callable[[ItemRecord], None],
) -> tuple[ExperimentHeaderRecord, str]:
    digest = hashlib.sha256()

    def lines() -> Iterator[bytes]:
        for line in stream:
            digest.update(line)
            yield line

    header = read_artifact_stream(lines(), on_record=on_record)
    return header, digest.hexdigest()


def _publish_staged_artifact(staged: Path, destination: Path) -> None:
    os.replace(staged, destination)
    _fsync_directory(destination.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unlink_and_fsync(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _fsync_directory(path.parent)


def _reject_symlink(path: Path, label: str) -> None:
    if path.is_symlink():
        raise RuntimeError(f"{label} must not be a symbolic link")


def _file_sha256_or_none(path: Path) -> str | None:
    try:
        return _file_sha256(path)
    except FileNotFoundError:
        return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = (
    "MAX_CONCURRENCY",
    "MIXED_CATEGORY",
    "ClassificationSummary",
    "build_classification_experiment",
    "classify_item",
    "classify_one_repeat",
    "run_classification",
)
