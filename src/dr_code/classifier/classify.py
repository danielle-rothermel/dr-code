"""Orchestrate repeated classification, aggregation, and persistence.

This module is lane-agnostic: it takes any object satisfying the
:class:`~dr_code.classifier.lane.Lane` protocol, so tests inject a mock lane and
the pilot injects :class:`~dr_code.classifier.lane.PiLane`.

Outputs are additive:

* a deterministic per-example JSONL artifact (one :class:`ItemRecord` per line);
* a per-task rollup written through ``put_task_annotation`` with
  ``origin=machine`` and provenance carrying model/taxonomy_version/repeats/
  mean-agreement plus ``extra`` (per-label counts, run ref, detail path).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from dr_code.classifier.aggregation import (
    ItemAggregate,
    RepeatOutcome,
    aggregate_repeats,
    mean_agreement,
)
from dr_code.classifier.extraction import (
    FailureItem,
    extract_parse_failures,
    extract_test_failures,
)
from dr_code.classifier.lane import (
    REPARSE_INSTRUCTION,
    Lane,
    LaneTransportError,
    parse_label_response,
)
from dr_code.classifier.prompt import build_prompt
from dr_code.classifier.records import ItemRecord, RepeatRecord
from dr_code.classifier.taxonomy import (
    TAXONOMY_VERSION,
    FailureKind,
    OTHER_LABEL,
    is_valid_label,
)
from dr_code.viewer.analytics import ViewerAnalytics
from dr_code.viewer.domain import (
    AnnotationOrigin,
    RunDescriptor,
    TaskAnnotationProvenance,
)


MAX_CONCURRENCY = 4


@dataclass(frozen=True, slots=True)
class ClassificationSummary:
    """Descriptive statistics from one classification run."""

    run_id: str
    parse_total: int
    parse_classified: int
    test_total: int
    test_classified: int
    skipped: int
    repeats: int
    taxonomy_version: str
    model: str
    lane: str
    mean_agreement: float | None
    min_agreement: float | None
    label_distribution: dict[str, int]
    typed_failures: int
    tasks_written: int
    detail_path: Path


def classify_one_repeat(lane: Lane, prompt: str) -> RepeatOutcome:
    """Run one repeat with a single reparse retry on a malformed reply."""
    try:
        raw = lane.complete(prompt)
    except (LaneTransportError, OSError) as exc:
        return RepeatOutcome(
            label=None, rationale=None, failure_reason=f"transport: {exc}"
        )
    try:
        response = parse_label_response(raw)
    except ValueError:
        # One reparse attempt: re-ask for strict JSON only.
        retry_prompt = f"{prompt}\n\n{REPARSE_INSTRUCTION}"
        try:
            raw = lane.complete(retry_prompt)
            response = parse_label_response(raw)
        except (LaneTransportError, OSError) as exc:
            return RepeatOutcome(
                label=None,
                rationale=None,
                failure_reason=f"transport-on-reparse: {exc}",
            )
        except ValueError as exc:
            return RepeatOutcome(
                label=None,
                rationale=None,
                failure_reason=f"malformed-response: {exc}",
            )
    return RepeatOutcome(label=response.label, rationale=response.rationale)


def classify_item(
    lane: Lane, item: FailureItem, *, repeats: int
) -> tuple[ItemAggregate, list[RepeatOutcome]]:
    """Classify one failure item over ``repeats`` repeats and aggregate.

    Labels outside the taxonomy are coerced to a typed failure rather than
    trusted, so an off-taxonomy string never becomes a stored label.
    """
    prompt = build_prompt(item.kind, item.text)
    outcomes: list[RepeatOutcome] = []
    for _ in range(repeats):
        outcome = classify_one_repeat(lane, prompt)
        if outcome.label is not None and not is_valid_label(
            item.kind, outcome.label
        ):
            outcome = RepeatOutcome(
                label=None,
                rationale=outcome.rationale,
                failure_reason=f"off-taxonomy-label: {outcome.label}",
            )
        outcomes.append(outcome)
    return aggregate_repeats(outcomes), outcomes


def _to_record(
    item: FailureItem,
    aggregate: ItemAggregate,
    outcomes: list[RepeatOutcome],
    *,
    repeats: int,
    model: str,
    lane_name: str,
) -> ItemRecord:
    return ItemRecord(
        item_id=item.item_id,
        kind=item.kind.value,
        sample_id=item.sample_id,
        dataset_id=item.dataset_id,
        task_id=item.task_id,
        failure_code=item.failure_code,
        failed_step=item.failed_step,
        taxonomy_version=TAXONOMY_VERSION,
        model=model,
        lane=lane_name,
        repeats=repeats,
        majority_label=aggregate.majority_label,
        agreement=aggregate.agreement,
        tie=aggregate.tie,
        successful_repeats=aggregate.successful_repeats,
        failed_repeats=aggregate.failed_repeats,
        label_counts=aggregate.label_counts,
        repeat_records=[
            RepeatRecord(
                index=index,
                label=outcome.label,
                rationale=outcome.rationale,
                failure_reason=outcome.failure_reason,
            )
            for index, outcome in enumerate(outcomes)
        ],
    )


def _load_existing(
    detail_path: Path, *, taxonomy_version: str
) -> dict[str, ItemRecord]:
    """Load prior records at the same taxonomy_version for resumption."""
    if not detail_path.is_file():
        return {}
    existing: dict[str, ItemRecord] = {}
    for line in detail_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = ItemRecord.model_validate_json(line)
        if record.taxonomy_version == taxonomy_version:
            existing[record.item_id] = record
    return existing


def _write_details(detail_path: Path, records: list[ItemRecord]) -> None:
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda record: record.item_id)
    lines = [
        record.model_dump_json(exclude_none=False) for record in ordered
    ]
    detail_path.write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
    )


def run_classification(
    analytics: ViewerAnalytics,
    descriptor: RunDescriptor,
    lane: Lane,
    *,
    detail_path: Path,
    repeats: int = 5,
    parse_limit: int | None = 300,
    test_limit: int | None = 100,
    include_tests: bool = True,
    concurrency: int = MAX_CONCURRENCY,
    force: bool = False,
) -> ClassificationSummary:
    """Classify a run's failures and persist details + per-task rollups."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    concurrency = max(1, min(concurrency, MAX_CONCURRENCY))
    connection = analytics._database.connection  # noqa: SLF001 - internal seam
    parse_items, parse_total = extract_parse_failures(
        connection, descriptor, limit=parse_limit
    )
    test_items: list[FailureItem] = []
    test_total = 0
    if include_tests:
        test_items, test_total = extract_test_failures(
            connection, descriptor, limit=test_limit
        )
    items = [*parse_items, *test_items]

    existing = (
        {} if force else _load_existing(
            detail_path, taxonomy_version=TAXONOMY_VERSION
        )
    )
    pending = [item for item in items if item.item_id not in existing]
    skipped = len(items) - len(pending)

    new_records: dict[str, ItemRecord] = {}

    def classify(item: FailureItem) -> tuple[str, ItemRecord]:
        aggregate, outcomes = classify_item(lane, item, repeats=repeats)
        record = _to_record(
            item,
            aggregate,
            outcomes,
            repeats=repeats,
            model=lane.model,
            lane_name=lane.name,
        )
        return item.item_id, record

    if pending:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for item_id, record in pool.map(classify, pending):
                new_records[item_id] = record

    all_records = {**existing, **new_records}
    ordered_records = [all_records[item.item_id] for item in items]
    _write_details(detail_path, list(all_records.values()))

    tasks_written = _write_task_rollups(
        analytics,
        descriptor,
        ordered_records,
        detail_path=detail_path,
        repeats=repeats,
        model=lane.model,
    )

    aggregates = [
        _aggregate_from_record(record) for record in ordered_records
    ]
    label_distribution = _label_distribution(ordered_records)
    typed_failures = sum(record.failed_repeats for record in ordered_records)
    agreements = [a.agreement for a in aggregates if a.agreement is not None]

    return ClassificationSummary(
        run_id=descriptor.run_id,
        parse_total=parse_total,
        parse_classified=len(parse_items),
        test_total=test_total,
        test_classified=len(test_items),
        skipped=skipped,
        repeats=repeats,
        taxonomy_version=TAXONOMY_VERSION,
        model=lane.model,
        lane=lane.name,
        mean_agreement=mean_agreement(aggregates),
        min_agreement=min(agreements) if agreements else None,
        label_distribution=label_distribution,
        typed_failures=typed_failures,
        tasks_written=tasks_written,
        detail_path=detail_path,
    )


def _aggregate_from_record(record: ItemRecord) -> ItemAggregate:
    return ItemAggregate(
        majority_label=record.majority_label,
        agreement=record.agreement,
        tie=record.tie,
        successful_repeats=record.successful_repeats,
        failed_repeats=record.failed_repeats,
        label_counts=dict(record.label_counts),
    )


def _label_distribution(records: list[ItemRecord]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        counter[record.majority_label or "<no-label>"] += 1
    return dict(sorted(counter.items()))


def _write_task_rollups(
    analytics: ViewerAnalytics,
    descriptor: RunDescriptor,
    records: list[ItemRecord],
    *,
    detail_path: Path,
    repeats: int,
    model: str,
) -> int:
    """Write one machine task annotation per task with parse failures."""
    by_task: dict[tuple[str, str], list[ItemRecord]] = defaultdict(list)
    for record in records:
        if record.kind != FailureKind.PARSE.value:
            continue
        if record.dataset_id is None or record.task_id is None:
            continue
        by_task[(record.dataset_id, record.task_id)].append(record)

    written = 0
    for (dataset_id, task_id), task_records in sorted(by_task.items()):
        # Skip tasks whose every failure produced only typed lane failures: no
        # label was earned, so we do not write a fabricated machine annotation.
        if all(record.majority_label is None for record in task_records):
            continue
        counts: Counter[str] = Counter()
        agreements: list[float] = []
        for record in task_records:
            counts[record.majority_label or "<no-label>"] += 1
            if record.agreement is not None:
                agreements.append(record.agreement)
        category = _dominant_category(counts)
        note = _counts_note(len(task_records), counts)
        task_agreement = (
            sum(agreements) / len(agreements) if agreements else None
        )
        provenance = TaskAnnotationProvenance(
            model=model,
            taxonomy_version=TAXONOMY_VERSION,
            repeats=repeats,
            agreement=task_agreement,
            extra={
                "per_label_counts": dict(sorted(counts.items())),
                "run_ref": descriptor.run_id,
                "item_details_path": str(detail_path),
                "failure_kind": FailureKind.PARSE.value,
            },
        )
        analytics.put_task_annotation(
            dataset_id,
            task_id,
            origin=AnnotationOrigin.MACHINE,
            category=category,
            note=note,
            provenance=provenance,
        )
        written += 1
    return written


def _dominant_category(counts: Counter[str]) -> str:
    if not counts:
        return OTHER_LABEL
    labels = {label for label in counts if label != "<no-label>"}
    if not labels:
        return OTHER_LABEL
    top = max(counts[label] for label in labels)
    winners = [label for label in labels if counts[label] == top]
    if len(winners) > 1:
        return "mixed"
    return winners[0]


def _counts_note(total: int, counts: Counter[str]) -> str:
    parts = [
        f"{count} {label}"
        for label, count in sorted(
            counts.items(), key=lambda kv: (-kv[1], kv[0])
        )
    ]
    plural = "failure" if total == 1 else "failures"
    return f"{total} {plural}: " + ", ".join(parts)


__all__ = (
    "MAX_CONCURRENCY",
    "ClassificationSummary",
    "classify_item",
    "classify_one_repeat",
    "run_classification",
)
