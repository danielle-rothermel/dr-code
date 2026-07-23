from __future__ import annotations

import json

from dr_code.classifier.classify import classify_item, run_classification
from dr_code.classifier.extraction import FailureItem
from dr_code.classifier.records import ItemRecord
from dr_code.classifier.taxonomy import TAXONOMY_VERSION, FailureKind
from dr_code.viewer.analytics import ViewerAnalytics
from dr_code.viewer.database import ViewerDatabase
from dr_code.viewer.domain import AnnotationOrigin
from viewer.helpers import write_bundle


class FixedLane:
    """A mock lane that always labels by the item's sample_id mapping."""

    name = "mock-lane"
    model = "mock-model"

    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        for sample_id, label in self._mapping.items():
            if sample_id in prompt:
                return json.dumps({"label": label, "rationale": "fixed"})
        return "unclassifiable"


def _analytics(tmp_path):
    descriptor = write_bundle(tmp_path / "bundle")
    database = ViewerDatabase(":memory:")
    analytics = ViewerAnalytics(database, [descriptor])
    return analytics, descriptor


_LABELS = {
    "This is prose.": "prose-no-code",
    "Alternate prose.": "prose-no-code",
    "Unclassified prose.": "truncated-output",
    "def broken(": "syntax-error-other",
    "answer = 42": "expression-not-function",
    # test failure text (fixture failure_message is None; outcome carried)
    "failed": "wrong-algorithm",
}


def test_classify_item_majority_over_repeats() -> None:
    item = FailureItem(
        item_id="parse:x",
        kind=FailureKind.PARSE,
        sample_id="x",
        dataset_id="Task",
        task_id="Task/1",
        failure_code="no_code_candidates",
        failed_step="extract_candidates",
        text="This is prose.",
    )
    lane = FixedLane({"This is prose.": "prose-no-code"})
    aggregate, outcomes = classify_item(lane, item, repeats=5)
    assert aggregate.majority_label == "prose-no-code"
    assert aggregate.agreement == 1.0
    assert len(outcomes) == 5


def test_run_classification_writes_details_and_task_rollups(
    tmp_path,
) -> None:
    analytics, descriptor = _analytics(tmp_path)
    lane = FixedLane(_LABELS)
    detail_path = tmp_path / "details.jsonl"

    summary = run_classification(
        analytics,
        descriptor,
        lane,
        detail_path=detail_path,
        repeats=3,
    )

    assert summary.parse_total == 5
    assert summary.parse_classified == 5
    assert summary.test_classified == 1
    assert summary.typed_failures == 0
    assert summary.mean_agreement == 1.0
    # Detail JSONL: one line per item (5 parse + 1 test).
    lines = detail_path.read_text().splitlines()
    assert len(lines) == 6
    records = [ItemRecord.model_validate_json(line) for line in lines]
    assert all(r.taxonomy_version == TAXONOMY_VERSION for r in records)
    parse_records = [r for r in records if r.kind == "parse"]
    assert len(parse_records) == 5


def test_task_rollup_round_trips_origin_and_provenance(tmp_path) -> None:
    analytics, descriptor = _analytics(tmp_path)
    lane = FixedLane(_LABELS)
    detail_path = tmp_path / "details.jsonl"
    run_classification(
        analytics, descriptor, lane, detail_path=detail_path, repeats=3
    )

    # Task/2 has three parse failures, all prose-no-code except one
    # truncated-output -> dominant is prose-no-code.
    annotation = analytics.get_task_annotation("Task", "Task/2")
    assert annotation is not None
    assert annotation.origin is AnnotationOrigin.MACHINE
    assert annotation.category == "prose-no-code"
    assert annotation.note.startswith("3 failures:")
    prov = annotation.provenance
    assert prov is not None
    assert prov.model == "mock-model"
    assert prov.taxonomy_version == TAXONOMY_VERSION
    assert prov.repeats == 3
    assert prov.agreement == 1.0
    assert prov.extra is not None
    assert prov.extra["run_ref"] == descriptor.run_id
    assert prov.extra["item_details_path"] == str(detail_path)
    assert prov.extra["per_label_counts"] == {
        "prose-no-code": 2,
        "truncated-output": 1,
    }


def test_export_task_annotations_shows_machine_rows(tmp_path) -> None:
    analytics, descriptor = _analytics(tmp_path)
    lane = FixedLane(_LABELS)
    run_classification(
        analytics,
        descriptor,
        lane,
        detail_path=tmp_path / "details.jsonl",
        repeats=3,
    )
    exported = analytics.export_task_annotations()
    machine = [row for row in exported if row["origin"] == "machine"]
    assert machine
    for row in machine:
        provenance = json.loads(row["provenance"])
        assert provenance["taxonomy_version"] == TAXONOMY_VERSION
        assert provenance["model"] == "mock-model"


def test_resume_skips_items_at_same_taxonomy_version(tmp_path) -> None:
    analytics, descriptor = _analytics(tmp_path)
    detail_path = tmp_path / "details.jsonl"
    first = run_classification(
        analytics,
        descriptor,
        FixedLane(_LABELS),
        detail_path=detail_path,
        repeats=3,
    )
    assert first.skipped == 0

    second_lane = FixedLane(_LABELS)
    second = run_classification(
        analytics,
        descriptor,
        second_lane,
        detail_path=detail_path,
        repeats=3,
    )
    # Everything was already classified: nothing re-called.
    assert second.skipped == 6
    assert second_lane.calls == 0


def test_force_reclassifies_everything(tmp_path) -> None:
    analytics, descriptor = _analytics(tmp_path)
    detail_path = tmp_path / "details.jsonl"
    run_classification(
        analytics,
        descriptor,
        FixedLane(_LABELS),
        detail_path=detail_path,
        repeats=3,
    )
    forced_lane = FixedLane(_LABELS)
    forced = run_classification(
        analytics,
        descriptor,
        forced_lane,
        detail_path=detail_path,
        repeats=3,
        force=True,
    )
    assert forced.skipped == 0
    assert forced_lane.calls > 0


def test_typed_failures_are_recorded_not_fabricated(tmp_path) -> None:
    analytics, descriptor = _analytics(tmp_path)

    class BrokenLane:
        name = "broken"
        model = "broken-model"

        def complete(self, prompt: str) -> str:
            return "never valid json"

    detail_path = tmp_path / "details.jsonl"
    summary = run_classification(
        analytics,
        descriptor,
        BrokenLane(),
        detail_path=detail_path,
        repeats=2,
        include_tests=False,
    )
    # 5 parse items x 2 repeats all failed.
    assert summary.typed_failures == 10
    # No majority labels -> no task rollups written.
    assert summary.tasks_written == 0
    records = [
        ItemRecord.model_validate_json(line)
        for line in detail_path.read_text().splitlines()
    ]
    for record in records:
        assert record.majority_label is None
        assert record.agreement is None
        assert all(r.label is None for r in record.repeat_records)


def test_off_taxonomy_label_becomes_a_typed_failure(tmp_path) -> None:
    analytics, descriptor = _analytics(tmp_path)

    class InventiveLane:
        name = "inventive"
        model = "inventive-model"

        def complete(self, prompt: str) -> str:
            return json.dumps(
                {"label": "totally-made-up", "rationale": "nope"}
            )

    detail_path = tmp_path / "details.jsonl"
    summary = run_classification(
        analytics,
        descriptor,
        InventiveLane(),
        detail_path=detail_path,
        repeats=2,
        include_tests=False,
    )
    assert summary.typed_failures == 10
    records = [
        ItemRecord.model_validate_json(line)
        for line in detail_path.read_text().splitlines()
    ]
    reasons = [
        r.failure_reason
        for record in records
        for r in record.repeat_records
    ]
    assert all(
        reason is not None and "off-taxonomy" in reason for reason in reasons
    )
