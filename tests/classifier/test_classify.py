from __future__ import annotations

import json
import hashlib
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

import dr_code.classifier.classify as classify_module
from dr_code.classifier.classify import (
    MIXED_CATEGORY,
    _dominant_category,
    build_classification_experiment,
    classify_one_repeat,
    run_classification,
)
from dr_code.classifier.extraction import FailureStream
from dr_code.classifier.lane import (
    LanePolicy,
    LaneTransportError,
    TransportFailureKind,
)
from dr_code.classifier.records import experiment_identity, load_records
from dr_code.classifier.taxonomy import FailureFamily
from dr_code.viewer.analytics import ViewerAnalytics
from dr_code.viewer.database import ViewerDatabase
from dr_code.viewer.domain import (
    MachineTaskAnnotationWriteOutcome,
    TaskIdentity,
    TaskAnnotationProvenance,
)
from viewer.helpers import write_bundle


class FixedLane:
    provider = "fake-provider"
    model = "fake-model"
    policy = LanePolicy(adapter="test-fixed-lane-v1")

    def __init__(
        self,
        *,
        delays: bool = False,
        parse_label: str = "prose-no-code",
        test_label: str = "wrong-algorithm",
        invalid: bool = False,
        fail_after: int | None = None,
    ) -> None:
        self.calls = 0
        self._lock = threading.Lock()
        self.delays = delays
        self.parse_label = parse_label
        self.test_label = test_label
        self.invalid = invalid
        self.fail_after = fail_after

    def complete(self, prompt: str) -> str:
        with self._lock:
            self.calls += 1
            call = self.calls
        if self.fail_after is not None and call > self.fail_after:
            raise RuntimeError("injected lane failure")
        if self.delays:
            time.sleep((call % 3) * 0.002)
        if self.invalid:
            return "invalid"
        label = (
            self.test_label
            if "Classify one test failure." in prompt
            else self.parse_label
        )
        return json.dumps({"label": label, "rationale": "fixed"})


class SimulatedCrash(BaseException):
    pass


def _setup(tmp_path, *, dataset_id: str = "org/benchmark"):
    descriptor = write_bundle(
        tmp_path / "bundle",
        dataset_id=dataset_id,
        task_namespace="Independent",
    )
    analytics = ViewerAnalytics(ViewerDatabase(":memory:"), [descriptor])
    return analytics, descriptor


def _fixture_task_identity(task_id: str) -> str:
    """Match the deterministic task identity recorded by ``write_bundle``."""
    return hashlib.sha256(task_id.encode()).hexdigest()


def test_run_writes_both_families_and_structured_provenance(tmp_path) -> None:
    analytics, descriptor = _setup(tmp_path)
    details = tmp_path / "details.jsonl"
    summary = run_classification(
        analytics,
        descriptor,
        FixedLane(),
        details_path=details,
        repeats=3,
    )
    assert summary.parse_total == 5
    assert summary.test_total == 1
    assert summary.classified == 6
    assert summary.mean_agreement == 1
    assert summary.label_counts == {
        "parse": {"prose-no-code": 5},
        "test": {"wrong-algorithm": 1},
    }
    assert len(load_records(details)) == 6

    test_annotation = analytics.get_task_annotation(
        "org/benchmark",
        "Independent/6",
        _fixture_task_identity("Independent/6"),
    )
    assert test_annotation is not None
    assert test_annotation.category == "wrong-algorithm"
    provenance = test_annotation.provenance
    assert isinstance(provenance, TaskAnnotationProvenance)
    assert provenance.model == "fake-model"
    assert provenance.repeats == 3
    assert provenance.agreement == 1
    assert provenance.extra["provider"] == "fake-provider"
    assert provenance.extra["label_counts"] == {
        "parse": {},
        "test": {"wrong-algorithm": 1},
    }
    assert provenance.extra["details_sha256"] == summary.details_sha256
    assert provenance.extra["run"]["dataset_id"] == "org/benchmark"
    assert [
        item["identity"]["task_id"]
        for item in analytics.export_task_annotations()
    ] == ["Independent/6"]
    encoded = json.dumps(
        analytics.export_task_annotations(),
        ensure_ascii=False,
        sort_keys=True,
    )
    assert str(tmp_path) not in encoded
    assert "timestamp" not in encoded


def test_parse_only_items_without_task_identity_are_output_only(
    tmp_path,
) -> None:
    descriptor = write_bundle(
        tmp_path / "bundle",
        dataset_id="org/benchmark",
        task_namespace="Independent",
        with_evaluation=False,
    )
    analytics = ViewerAnalytics(ViewerDatabase(":memory:"), [descriptor])
    details = tmp_path / "details.jsonl"
    lane = FixedLane()

    summary = run_classification(
        analytics,
        descriptor,
        lane,
        details_path=details,
        repeats=1,
    )

    assert lane.calls == 5
    assert summary.tasks_written == 0
    assert details.exists()
    assert not classify_module._staged_artifact_path(details).exists()


def test_human_is_atomically_protected_and_machine_is_refreshed(
    tmp_path,
) -> None:
    analytics, descriptor = _setup(tmp_path)
    human = analytics.put_task_annotation(
        "org/benchmark",
        "Independent/6",
        _fixture_task_identity("Independent/6"),
        category="human-choice",
    )
    first = run_classification(
        analytics,
        descriptor,
        FixedLane(),
        details_path=tmp_path / "details.jsonl",
        repeats=1,
    )
    assert first.tasks_protected == 1
    assert (
        analytics.get_task_annotation(
            "org/benchmark",
            "Independent/6",
            _fixture_task_identity("Independent/6"),
        )
        == human
    )

    machine_task = "Independent/6"
    assert analytics.delete_task_annotation(
        "org/benchmark",
        machine_task,
        _fixture_task_identity(machine_task),
    )
    prior = analytics.put_machine_task_annotation(
        "org/benchmark",
        machine_task,
        _fixture_task_identity(machine_task),
        category="other",
        provenance=TaskAnnotationProvenance(
            model="old",
            taxonomy_version="old",
            repeats=1,
            agreement=1,
        ),
    )
    assert prior.outcome is MachineTaskAnnotationWriteOutcome.WRITTEN
    second = run_classification(
        analytics,
        descriptor,
        FixedLane(test_label="wrong-edge-case"),
        details_path=tmp_path / "details.jsonl",
        repeats=1,
        force=True,
    )
    assert second.tasks_protected == 0
    refreshed = analytics.get_task_annotation(
        "org/benchmark",
        machine_task,
        _fixture_task_identity(machine_task),
    )
    assert refreshed is not None
    assert refreshed.category == "wrong-edge-case"
    assert refreshed.provenance is not None
    assert refreshed.provenance.model == "fake-model"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda descriptor, lane, repeats: (
            replace(descriptor, corpus_sha256="f" * 64),
            lane,
            repeats,
        ),
        lambda descriptor, lane, repeats: (
            replace(
                descriptor,
                preprocessing_manifest_sha256="e" * 64,
            ),
            lane,
            repeats,
        ),
        lambda descriptor, lane, repeats: (
            replace(
                descriptor,
                evaluation_manifest_sha256="d" * 64,
            ),
            lane,
            repeats,
        ),
        lambda descriptor, lane, repeats: (
            descriptor,
            replace(lane, provider="provider-2"),
            repeats,
        ),
        lambda descriptor, lane, repeats: (
            descriptor,
            replace(lane, model="model-2"),
            repeats,
        ),
        lambda descriptor, lane, repeats: (
            descriptor,
            lane,
            repeats + 1,
        ),
    ],
)
def test_resume_requires_every_run_and_config_coordinate(
    tmp_path,
    mutate: Callable,
) -> None:
    analytics, descriptor = _setup(tmp_path)

    from dataclasses import dataclass

    @dataclass(frozen=True, slots=True)
    class Lane:
        provider: str = "provider"
        model: str = "model"
        calls: list[int] | None = None
        policy: LanePolicy = LanePolicy(adapter="test-resume-lane-v1")

        def complete(self, prompt: str) -> str:
            assert self.calls is not None
            self.calls.append(1)
            label = (
                "wrong-algorithm"
                if "Classify one test failure." in prompt
                else "prose-no-code"
            )
            return json.dumps({"label": label, "rationale": "fixed"})

    details = tmp_path / "details.jsonl"
    run_classification(
        analytics,
        descriptor,
        Lane(calls=[]),
        details_path=details,
        repeats=1,
    )
    changed_descriptor, changed_lane, changed_repeats = mutate(
        descriptor, Lane(calls=[]), 1
    )
    before = details.read_bytes()
    with pytest.raises(
        ValueError,
        match="different experiment|registered immutable coordinates",
    ):
        run_classification(
            analytics,
            changed_descriptor,
            changed_lane,
            details_path=details,
            repeats=changed_repeats,
        )
    assert changed_lane.calls == []
    assert details.read_bytes() == before


def test_prompt_and_taxonomy_changes_invalidate_resume(
    tmp_path, monkeypatch
) -> None:
    analytics, descriptor = _setup(tmp_path)
    details = tmp_path / "details.jsonl"
    run_classification(
        analytics,
        descriptor,
        FixedLane(),
        details_path=details,
        repeats=1,
    )
    monkeypatch.setattr(classify_module, "PROMPT_VERSION", "failure-prompt-v5")
    with pytest.raises(ValueError, match="different experiment"):
        run_classification(
            analytics,
            descriptor,
            FixedLane(),
            details_path=details,
            repeats=1,
        )
    monkeypatch.setattr(
        classify_module, "TAXONOMY_VERSION", "failure-taxonomy-v2"
    )
    with pytest.raises(ValueError, match="different experiment"):
        run_classification(
            analytics,
            descriptor,
            FixedLane(),
            details_path=details,
            repeats=1,
        )


def test_injected_lane_policy_is_required_and_distinguishes_generation(
    tmp_path,
) -> None:
    analytics, descriptor = _setup(tmp_path)

    class MissingPolicyLane:
        provider = "provider"
        model = "model"

        def complete(self, prompt: str) -> str:
            raise AssertionError("lane must be rejected before completion")

    with pytest.raises(ValueError, match="typed canonical policy"):
        run_classification(
            analytics,
            descriptor,
            MissingPolicyLane(),  # type: ignore[arg-type]
            details_path=tmp_path / "missing-policy.jsonl",
            repeats=1,
        )

    class TemperatureLane:
        provider = "provider"
        model = "model"

        def __init__(self, temperature: float) -> None:
            self.policy = LanePolicy(
                adapter="test-temperature-lane-v1",
                generation=(("temperature", temperature),),
            )

        def complete(self, prompt: str) -> str:
            raise AssertionError("identity construction does not complete")

    cold = build_classification_experiment(
        descriptor,
        TemperatureLane(0.0),
        repeats=1,
        parse_limit=1,
        test_limit=1,
    )
    warm = build_classification_experiment(
        descriptor,
        TemperatureLane(0.7),
        repeats=1,
        parse_limit=1,
        test_limit=1,
    )

    assert cold.config.lane_adapter == warm.config.lane_adapter
    assert cold.config.lane_policy_identity != warm.config.lane_policy_identity
    assert experiment_identity(cold) != experiment_identity(warm)


def test_output_lock_is_lexical_and_exposes_no_forgeable_token(
    tmp_path,
) -> None:
    analytics, descriptor = _setup(tmp_path)
    details = (tmp_path / "details.jsonl").resolve()

    with pytest.raises(TypeError, match="_output_lock"):
        run_classification(
            analytics,
            descriptor,
            FixedLane(),
            details_path=details,
            repeats=1,
            _output_lock=object(),  # type: ignore[call-arg]
        )

    mutable_path = [details]
    with classify_module._classification_output_lock(details) as token:
        mutable_path[0] = tmp_path / "other.jsonl"
        assert token is None
    assert not hasattr(classify_module, "_ClassificationOutputLock")
    assert "_classification_output_lock" not in classify_module.__all__


def test_rendered_input_change_invalidates_only_changed_item(
    tmp_path, monkeypatch
) -> None:
    analytics, descriptor = _setup(tmp_path)
    details = tmp_path / "details.jsonl"
    run_classification(
        analytics,
        descriptor,
        FixedLane(),
        details_path=details,
        repeats=1,
    )
    original = analytics.parse_failures_for_classification

    def changed(run_id: str, *, limit: int | None = None):
        page = original(run_id, limit=limit)
        items = list(page.items)
        items[0] = replace(
            items[0],
            decoder_output=items[0].decoder_output + "\nchanged",
        )
        return replace(page, items=tuple(items))

    monkeypatch.setattr(
        analytics, "parse_failures_for_classification", changed
    )
    lane = FixedLane()
    summary = run_classification(
        analytics,
        descriptor,
        lane,
        details_path=details,
        repeats=1,
    )
    assert summary.resumed == 5
    assert summary.classified == 1
    assert lane.calls == 1


def test_concurrency_completion_order_cannot_change_bytes(tmp_path) -> None:
    analytics, descriptor = _setup(tmp_path)
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    run_classification(
        analytics,
        descriptor,
        FixedLane(delays=True),
        details_path=first_path,
        repeats=2,
        concurrency=4,
    )
    run_classification(
        analytics,
        descriptor,
        FixedLane(delays=True),
        details_path=second_path,
        repeats=2,
        concurrency=2,
    )
    assert first_path.read_bytes() == second_path.read_bytes()


def test_atomic_checkpoint_resumes_after_injected_write_failure(
    tmp_path, monkeypatch
) -> None:
    analytics, descriptor = _setup(tmp_path)
    details = tmp_path / "details.jsonl"
    original_write = classify_module.write_records_atomic
    calls = 0

    def fail_second(path, experiment, records, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected")
        original_write(path, experiment, records, **kwargs)

    monkeypatch.setattr(classify_module, "write_records_atomic", fail_second)
    with pytest.raises(OSError, match="injected"):
        run_classification(
            analytics,
            descriptor,
            FixedLane(),
            details_path=details,
            repeats=1,
            concurrency=1,
        )
    assert not details.exists()
    checkpoint = load_records(
        classify_module._staged_artifact_path(details.resolve())
    )
    assert len(checkpoint) == 1

    monkeypatch.setattr(
        classify_module, "write_records_atomic", original_write
    )
    resumed = run_classification(
        analytics,
        descriptor,
        FixedLane(),
        details_path=details,
        repeats=1,
        concurrency=1,
    )
    assert resumed.resumed == 1
    assert len(load_records(details)) == 6


def test_checkpoints_stream_spooled_records_in_geometric_batches(
    tmp_path,
    monkeypatch,
) -> None:
    analytics, descriptor = _setup(tmp_path)
    details = tmp_path / "details.jsonl"
    original_write = classify_module.write_records_atomic
    checkpoint_sizes: list[int] = []
    streamed_inputs: list[bool] = []

    def record_write(path, experiment, records, **kwargs):
        streamed_inputs.append(not isinstance(records, tuple | list))
        count = 0

        def observed():
            nonlocal count
            for record in records:
                count += 1
                yield record

        original_write(path, experiment, observed(), **kwargs)
        checkpoint_sizes.append(count)

    monkeypatch.setattr(classify_module, "write_records_atomic", record_write)
    run_classification(
        analytics,
        descriptor,
        FixedLane(),
        details_path=details,
        repeats=1,
        concurrency=2,
    )

    assert checkpoint_sizes == [2, 4, 6]
    assert streamed_inputs == [True, True, True]


def test_pending_classifications_never_materialize_beyond_concurrency(
    tmp_path,
    monkeypatch,
) -> None:
    analytics, descriptor = _setup(tmp_path)
    batch_sizes: list[int] = []

    class DeferredFuture:
        def __init__(self, function, args) -> None:
            self.function = function
            self.args = args

        def result(self):
            return self.function(*self.args)

    class DeferredExecutor:
        def __init__(self, *, max_workers: int) -> None:
            assert max_workers == 2

        def __enter__(self):
            return self

        def __exit__(self, *exc_info) -> None:
            return None

        def submit(self, function, *args):
            return DeferredFuture(function, args)

    def observe_wait(futures):
        batch_sizes.append(len(futures))
        return set(futures), set()

    monkeypatch.setattr(
        classify_module,
        "ThreadPoolExecutor",
        DeferredExecutor,
    )
    monkeypatch.setattr(classify_module, "wait", observe_wait)

    run_classification(
        analytics,
        descriptor,
        FixedLane(),
        details_path=tmp_path / "details.jsonl",
        repeats=1,
        concurrency=2,
    )

    assert batch_sizes == [2, 2, 2]


def test_all_failed_items_never_create_machine_rollups(tmp_path) -> None:
    analytics, descriptor = _setup(tmp_path)

    class InvalidLane:
        provider = "provider"
        model = "model"
        policy = LanePolicy(adapter="test-invalid-lane-v1")

        def complete(self, prompt: str) -> str:
            return "invalid"

    summary = run_classification(
        analytics,
        descriptor,
        InvalidLane(),
        details_path=tmp_path / "details.jsonl",
        repeats=2,
    )
    assert summary.repeat_failures == 12
    assert summary.tasks_written == 0
    assert summary.tasks_protected == 0
    assert summary.label_counts == {"parse": {}, "test": {}}
    assert analytics.export_task_annotations() == []


def test_rollup_tie_counts_shared_labels_by_failure_family() -> None:
    assert (
        _dominant_category(
            Counter(
                {
                    (FailureFamily.PARSE, "other"): 1,
                    (FailureFamily.TEST, "other"): 1,
                }
            )
        )
        == MIXED_CATEGORY
    )


def test_rollup_unique_namespaced_winner_uses_plain_label() -> None:
    assert (
        _dominant_category(
            Counter(
                {
                    (FailureFamily.PARSE, "other"): 2,
                    (FailureFamily.TEST, "other"): 1,
                }
            )
        )
        == "other"
    )


def test_summary_preserves_empty_test_namespace(tmp_path) -> None:
    descriptor = write_bundle(
        tmp_path / "bundle",
        dataset_id="org/benchmark",
        task_namespace="Independent",
    )
    analytics = ViewerAnalytics(ViewerDatabase(":memory:"), [descriptor])
    summary = run_classification(
        analytics,
        descriptor,
        FixedLane(test_label="not-a-label"),
        details_path=tmp_path / "details.jsonl",
        repeats=1,
    )
    assert summary.label_counts == {
        "parse": {"prose-no-code": 5},
        "test": {},
    }


def test_blank_transport_detail_becomes_stable_nonblank_failure() -> None:
    class BlankTransportLane:
        provider = "provider"
        model = "model"

        def complete(self, prompt: str) -> str:
            raise LaneTransportError(
                TransportFailureKind.OPERATING_SYSTEM,
                " \t ",
            )

    outcome = classify_one_repeat(
        BlankTransportLane(),
        FailureFamily.PARSE,
        "prompt",
    )

    assert outcome.failure is not None
    assert outcome.failure.detail == "operating_system"


def test_cross_experiment_explicit_path_rejects_force_without_lane_or_write(
    tmp_path,
) -> None:
    analytics, descriptor = _setup(tmp_path)
    details = tmp_path / "details.jsonl"
    run_classification(
        analytics,
        descriptor,
        FixedLane(),
        details_path=details,
        repeats=1,
    )
    before = details.read_bytes()
    changed = FixedLane()
    changed.provider = "other-provider"

    with pytest.raises(ValueError, match="different experiment"):
        run_classification(
            analytics,
            descriptor,
            changed,
            details_path=details,
            repeats=1,
            force=True,
        )

    assert changed.calls == 0
    assert details.read_bytes() == before


def test_force_authenticates_foreign_staged_checkpoint_before_discard(
    tmp_path,
) -> None:
    analytics, descriptor = _setup(tmp_path)
    details = tmp_path / "details.jsonl"
    provider_a = FixedLane(fail_after=1)
    with pytest.raises(RuntimeError, match="injected lane failure"):
        run_classification(
            analytics,
            descriptor,
            provider_a,
            details_path=details,
            repeats=1,
            concurrency=1,
        )
    staged = classify_module._staged_artifact_path(details.resolve())
    before = staged.read_bytes()
    provider_b = FixedLane()
    provider_b.provider = "provider-b"

    with pytest.raises(ValueError, match="checkpoint.*different experiment"):
        run_classification(
            analytics,
            descriptor,
            provider_b,
            details_path=details,
            repeats=1,
            concurrency=1,
            force=True,
        )

    assert provider_b.calls == 0
    assert staged.read_bytes() == before


@pytest.mark.parametrize(
    "case",
    ("stage-alias", "lock-alias", "canonical-relative", "symlink-alias"),
)
def test_reserved_internal_output_paths_are_rejected_before_mutation(
    tmp_path,
    monkeypatch,
    case,
) -> None:
    analytics, descriptor = _setup(tmp_path)
    reserved_name = (
        ".details.jsonl.lock"
        if case == "lock-alias"
        else ".details.jsonl.publication"
    )
    reserved = tmp_path / reserved_name
    reserved.write_bytes(b"other output must survive")
    monkeypatch.chdir(tmp_path)
    if case == "canonical-relative":
        details = Path("sub/../.details.jsonl.publication")
    elif case == "symlink-alias":
        alias = tmp_path / "details-alias.jsonl"
        alias.symlink_to(reserved.name)
        details = alias
    else:
        details = reserved
    lane = FixedLane()

    with pytest.raises(
        ValueError,
        match="classification details path basename is reserved",
    ):
        run_classification(
            analytics,
            descriptor,
            lane,
            details_path=details,
            repeats=1,
            force=True,
        )

    assert lane.calls == 0
    assert reserved.read_bytes() == b"other output must survive"
    assert not (tmp_path / f".{reserved_name}.lock").exists()


def test_forced_failure_preserves_prior_complete_artifact(tmp_path) -> None:
    analytics, descriptor = _setup(tmp_path)
    details = tmp_path / "details.jsonl"
    run_classification(
        analytics,
        descriptor,
        FixedLane(),
        details_path=details,
        repeats=1,
        concurrency=1,
    )
    before = details.read_bytes()

    with pytest.raises(RuntimeError, match="injected lane failure"):
        run_classification(
            analytics,
            descriptor,
            FixedLane(fail_after=1),
            details_path=details,
            repeats=1,
            concurrency=1,
            force=True,
        )

    assert details.read_bytes() == before
    assert not classify_module._staged_artifact_path(
        details.resolve()
    ).exists()
    assert (
        analytics.get_task_annotation_publication_intent(
            str(details.resolve())
        )
        is None
    )


def test_duplicate_extracted_identity_rejected_before_lane_or_write(
    tmp_path,
    monkeypatch,
) -> None:
    analytics, descriptor = _setup(tmp_path)
    details = tmp_path / "details.jsonl"
    original = classify_module.stream_failures(
        analytics,
        descriptor.run_id,
        parse_limit=300,
        test_limit=100,
    )
    first = next(original.items)
    monkeypatch.setattr(
        classify_module,
        "stream_failures",
        lambda *args, **kwargs: FailureStream(
            items=iter((first, first)),
            parse_total=2,
            test_total=0,
        ),
    )
    lane = FixedLane()

    with pytest.raises(ValueError, match="duplicate extracted"):
        run_classification(
            analytics,
            descriptor,
            lane,
            details_path=details,
            repeats=1,
        )

    assert lane.calls == 0
    assert not details.exists()


def test_complete_selected_scope_is_frozen_before_first_provider_call(
    tmp_path,
    monkeypatch,
) -> None:
    analytics, descriptor = _setup(tmp_path)
    original = classify_module.stream_failures(
        analytics,
        descriptor.run_id,
        parse_limit=300,
        test_limit=100,
    )
    items = tuple(original.items)
    extraction_complete = False
    stream_calls = 0

    def frozen_items():
        nonlocal extraction_complete
        yield from items
        extraction_complete = True

    def one_stream(*args, **kwargs):
        nonlocal stream_calls
        stream_calls += 1
        return FailureStream(
            items=frozen_items(),
            parse_total=original.parse_total,
            test_total=original.test_total,
        )

    class ObservedLane(FixedLane):
        def complete(self, prompt: str) -> str:
            assert extraction_complete
            return super().complete(prompt)

    monkeypatch.setattr(classify_module, "stream_failures", one_stream)

    run_classification(
        analytics,
        descriptor,
        ObservedLane(),
        details_path=tmp_path / "details.jsonl",
        repeats=1,
    )

    assert stream_calls == 1


def test_classifying_one_run_preserves_other_registered_run(
    tmp_path,
) -> None:
    first = write_bundle(
        tmp_path / "first",
        run_id="first",
        dataset_id="org/benchmark",
        task_namespace="First",
    )
    second = write_bundle(
        tmp_path / "second",
        run_id="second",
        dataset_id="org/benchmark",
        task_namespace="Second",
    )
    with ViewerDatabase(":memory:") as database:
        first_analytics = ViewerAnalytics(database, [first])
        run_classification(
            first_analytics,
            first,
            FixedLane(),
            details_path=tmp_path / "first.jsonl",
            repeats=1,
            parse_limit=1,
            test_limit=1,
        )
        second_analytics = ViewerAnalytics(database, [second])
        run_classification(
            second_analytics,
            second,
            FixedLane(),
            details_path=tmp_path / "second.jsonl",
            repeats=1,
            parse_limit=1,
            test_limit=1,
        )

        first_identity = TaskIdentity(
            dataset_id="org/benchmark",
            task_id="First/6",
            task_identity=_fixture_task_identity("First/6"),
        )
        second_identity = TaskIdentity(
            dataset_id="org/benchmark",
            task_id="Second/6",
            task_identity=_fixture_task_identity("Second/6"),
        )
        assert database.task_is_registered(first_identity)
        assert database.task_is_registered(second_identity)


def test_all_failed_force_removes_owned_machine_rollups_but_not_human(
    tmp_path,
) -> None:
    analytics, descriptor = _setup(tmp_path)
    details = tmp_path / "details.jsonl"
    first = run_classification(
        analytics,
        descriptor,
        FixedLane(),
        details_path=details,
        repeats=1,
    )
    assert first.tasks_written == 1
    human = analytics.put_task_annotation(
        "org/benchmark",
        "Independent/5",
        _fixture_task_identity("Independent/5"),
        category="human",
    )

    failed = run_classification(
        analytics,
        descriptor,
        FixedLane(invalid=True),
        details_path=details,
        repeats=1,
        force=True,
    )

    assert failed.tasks_removed == 1
    assert failed.tasks_protected == 0
    assert (
        analytics.get_task_annotation(
            "org/benchmark",
            "Independent/5",
            _fixture_task_identity("Independent/5"),
        )
        == human
    )
    assert (
        analytics.get_task_annotation(
            "org/benchmark",
            "Independent/6",
            _fixture_task_identity("Independent/6"),
        )
        is None
    )


def test_capped_all_failed_cleanup_is_exact_and_experiment_owned(
    tmp_path,
) -> None:
    analytics, descriptor = _setup(tmp_path)
    run_classification(
        analytics,
        descriptor,
        FixedLane(),
        details_path=tmp_path / "full.jsonl",
        repeats=1,
    )
    capped = tmp_path / "capped.jsonl"
    run_classification(
        analytics,
        descriptor,
        FixedLane(),
        details_path=capped,
        repeats=1,
        parse_limit=1,
        test_limit=1,
    )

    failed = run_classification(
        analytics,
        descriptor,
        FixedLane(invalid=True),
        details_path=capped,
        repeats=1,
        parse_limit=1,
        test_limit=1,
        force=True,
    )

    assert failed.tasks_removed == 1
    assert (
        analytics.get_task_annotation(
            "org/benchmark",
            "Independent/6",
            _fixture_task_identity("Independent/6"),
        )
        is None
    )
    assert analytics.export_task_annotations() == []


@pytest.mark.parametrize("has_prior_publication", [False, True])
def test_crash_before_replace_aborts_on_reopen_without_exposing_machine_rows(
    tmp_path,
    monkeypatch,
    has_prior_publication,
) -> None:
    descriptor = write_bundle(
        tmp_path / "bundle",
        dataset_id="org/benchmark",
        task_namespace="Independent",
    )
    database_path = tmp_path / "viewer.duckdb"
    details = (tmp_path / "details.jsonl").resolve()
    with ViewerDatabase(database_path) as database:
        analytics = ViewerAnalytics(database, [descriptor])
        prior_bytes = None
        if has_prior_publication:
            run_classification(
                analytics,
                descriptor,
                FixedLane(),
                details_path=details,
                repeats=1,
            )
            prior_bytes = details.read_bytes()
            human = analytics.put_task_annotation(
                "org/benchmark",
                "Independent/6",
                _fixture_task_identity("Independent/6"),
                category="human",
            )
        original_begin = analytics.begin_task_annotation_publication

        def persist_then_fail(intent):
            original_begin(intent)
            raise RuntimeError("injected after intent commit")

        monkeypatch.setattr(
            analytics,
            "begin_task_annotation_publication",
            persist_then_fail,
        )
        with pytest.raises(RuntimeError, match="after intent commit"):
            run_classification(
                analytics,
                descriptor,
                FixedLane(parse_label="syntax-error-other"),
                details_path=details,
                repeats=1,
                force=True,
            )

        staged = classify_module._staged_artifact_path(details)
        assert staged.exists()
        assert (
            details.read_bytes() if details.exists() else None
        ) == prior_bytes
        intent = analytics.get_task_annotation_publication_intent(str(details))
        assert intent is not None
        exported = analytics.export_task_annotations()
        if has_prior_publication:
            assert exported == [
                {
                    "identity": {
                        "dataset_id": "org/benchmark",
                        "task_id": "Independent/6",
                        "task_identity": _fixture_task_identity(
                            "Independent/6"
                        ),
                    },
                    "origin": "human",
                    "category": "human",
                    "note": None,
                    "tags": [],
                    "provenance": None,
                }
            ]
            assert (
                analytics.get_task_annotation(
                    "org/benchmark",
                    "Independent/6",
                    _fixture_task_identity("Independent/6"),
                )
                == human
            )
        else:
            assert exported == []

        monkeypatch.setattr(
            analytics,
            "begin_task_annotation_publication",
            original_begin,
        )

    with ViewerDatabase(database_path) as reopened_database:
        reopened = ViewerAnalytics(reopened_database, [descriptor])
        recovered = run_classification(
            reopened,
            descriptor,
            FixedLane(parse_label="syntax-error-other"),
            details_path=details,
            repeats=1,
            force=True,
        )
        assert (
            reopened.get_task_annotation_publication_intent(str(details))
            is None
        )
        assert not staged.exists()
        machine = [
            item
            for item in reopened.export_task_annotations()
            if item["origin"] == "machine"
        ]
        if has_prior_publication:
            assert machine == []
            assert (
                reopened.get_task_annotation(
                    "org/benchmark",
                    "Independent/6",
                    _fixture_task_identity("Independent/6"),
                )
                == human
            )
        else:
            assert machine
            assert {
                item["provenance"]["extra"]["details_sha256"]
                for item in machine
            } == {recovered.details_sha256}


def test_crash_after_replace_finishes_pending_rollups_on_reopen(
    tmp_path,
    monkeypatch,
) -> None:
    descriptor = write_bundle(
        tmp_path / "bundle",
        dataset_id="org/benchmark",
        task_namespace="Independent",
    )
    database_path = tmp_path / "viewer.duckdb"
    details = (tmp_path / "details.jsonl").resolve()
    with ViewerDatabase(database_path) as database:
        analytics = ViewerAnalytics(database, [descriptor])
        first = run_classification(
            analytics,
            descriptor,
            FixedLane(),
            details_path=details,
            repeats=1,
        )
        original_publish = classify_module._publish_staged_artifact

        def publish_then_crash(staged, destination):
            original_publish(staged, destination)
            raise SimulatedCrash

        monkeypatch.setattr(
            classify_module,
            "_publish_staged_artifact",
            publish_then_crash,
        )
        with pytest.raises(SimulatedCrash):
            run_classification(
                analytics,
                descriptor,
                FixedLane(parse_label="syntax-error-other"),
                details_path=details,
                repeats=1,
                force=True,
            )

        intended_sha256 = hashlib.sha256(details.read_bytes()).hexdigest()
        assert intended_sha256 != first.details_sha256
        assert not classify_module._staged_artifact_path(details).exists()
        assert (
            analytics.get_task_annotation_publication_intent(str(details))
            is not None
        )
        assert analytics.export_task_annotations() == []
        monkeypatch.setattr(
            classify_module,
            "_publish_staged_artifact",
            original_publish,
        )

    with ViewerDatabase(database_path) as reopened_database:
        reopened = ViewerAnalytics(reopened_database, [descriptor])
        lane = FixedLane()
        recovered = run_classification(
            reopened,
            descriptor,
            lane,
            details_path=details,
            repeats=1,
        )

        assert lane.calls == 0
        assert recovered.details_sha256 == intended_sha256
        assert (
            reopened.get_task_annotation_publication_intent(str(details))
            is None
        )
        machine = [
            item
            for item in reopened.export_task_annotations()
            if item["origin"] == "machine"
        ]
        assert machine
        assert {
            item["provenance"]["extra"]["details_sha256"] for item in machine
        } == {intended_sha256}


def test_recovery_rejects_artifact_swapped_between_hash_and_parse(
    tmp_path,
    monkeypatch,
) -> None:
    descriptor = write_bundle(
        tmp_path / "bundle",
        dataset_id="org/benchmark",
        task_namespace="Independent",
    )
    database_path = tmp_path / "viewer.duckdb"
    details = (tmp_path / "details.jsonl").resolve()
    with ViewerDatabase(database_path) as database:
        analytics = ViewerAnalytics(database, [descriptor])
        run_classification(
            analytics,
            descriptor,
            FixedLane(),
            details_path=details,
            repeats=1,
        )
        prior_bytes = details.read_bytes()
        original_publish = classify_module._publish_staged_artifact

        def publish_then_crash(staged, destination):
            original_publish(staged, destination)
            raise SimulatedCrash

        monkeypatch.setattr(
            classify_module,
            "_publish_staged_artifact",
            publish_then_crash,
        )
        with pytest.raises(SimulatedCrash):
            run_classification(
                analytics,
                descriptor,
                FixedLane(parse_label="syntax-error-other"),
                details_path=details,
                repeats=1,
                force=True,
            )
        monkeypatch.setattr(
            classify_module,
            "_publish_staged_artifact",
            original_publish,
        )
        original_hash = classify_module._file_sha256_or_none
        swapped = False

        def hash_then_swap(path):
            nonlocal swapped
            digest = original_hash(path)
            if path == details and not swapped:
                details.write_bytes(prior_bytes)
                swapped = True
            return digest

        monkeypatch.setattr(
            classify_module,
            "_file_sha256_or_none",
            hash_then_swap,
        )
        lane = FixedLane()
        with pytest.raises(RuntimeError, match="content changed"):
            run_classification(
                analytics,
                descriptor,
                lane,
                details_path=details,
                repeats=1,
            )

        assert lane.calls == 0
        assert analytics.get_task_annotation_publication_intent(str(details))


def test_valid_relative_path_recovers_pre_replace_intent(
    tmp_path,
    monkeypatch,
) -> None:
    descriptor = write_bundle(
        tmp_path / "bundle",
        dataset_id="org/benchmark",
        task_namespace="Independent",
    )
    database_path = tmp_path / "viewer.duckdb"
    monkeypatch.chdir(tmp_path)
    relative_details = Path("relative/details.jsonl")
    canonical_details = (tmp_path / relative_details).resolve()
    with ViewerDatabase(database_path) as database:
        analytics = ViewerAnalytics(database, [descriptor])
        original_begin = analytics.begin_task_annotation_publication

        def persist_then_fail(intent):
            original_begin(intent)
            raise RuntimeError("injected after intent commit")

        monkeypatch.setattr(
            analytics,
            "begin_task_annotation_publication",
            persist_then_fail,
        )
        with pytest.raises(RuntimeError, match="after intent commit"):
            run_classification(
                analytics,
                descriptor,
                FixedLane(),
                details_path=relative_details,
                repeats=1,
            )
        assert not canonical_details.exists()
        assert classify_module._staged_artifact_path(
            canonical_details
        ).exists()
        assert (
            analytics.get_task_annotation_publication_intent(
                str(canonical_details)
            )
            is not None
        )
        monkeypatch.setattr(
            analytics,
            "begin_task_annotation_publication",
            original_begin,
        )

    with ViewerDatabase(database_path) as reopened_database:
        reopened = ViewerAnalytics(reopened_database, [descriptor])
        recovered = run_classification(
            reopened,
            descriptor,
            FixedLane(),
            details_path=relative_details,
            repeats=1,
        )

        assert recovered.details_path == canonical_details
        assert canonical_details.exists()
        assert (
            reopened.get_task_annotation_publication_intent(
                str(canonical_details)
            )
            is None
        )


@pytest.mark.parametrize("prior_state", ("corrupt", "wrong-experiment"))
def test_pre_replace_recovery_authenticates_prior_before_abort(
    tmp_path,
    monkeypatch,
    prior_state,
) -> None:
    descriptor = write_bundle(
        tmp_path / "bundle",
        dataset_id="org/benchmark",
        task_namespace="Independent",
    )
    database_path = tmp_path / "viewer.duckdb"
    details = (tmp_path / "details.jsonl").resolve()
    with ViewerDatabase(database_path) as database:
        analytics = ViewerAnalytics(database, [descriptor])
        run_classification(
            analytics,
            descriptor,
            FixedLane(),
            details_path=details,
            repeats=1,
        )
        original_begin = analytics.begin_task_annotation_publication

        def persist_then_fail(intent):
            original_begin(intent)
            raise RuntimeError("injected after intent commit")

        monkeypatch.setattr(
            analytics,
            "begin_task_annotation_publication",
            persist_then_fail,
        )
        with pytest.raises(RuntimeError, match="after intent commit"):
            run_classification(
                analytics,
                descriptor,
                FixedLane(parse_label="syntax-error-other"),
                details_path=details,
                repeats=1,
                force=True,
            )
        monkeypatch.setattr(
            analytics,
            "begin_task_annotation_publication",
            original_begin,
        )
        staged = classify_module._staged_artifact_path(details)
        staged_bytes = staged.read_bytes()
        if prior_state == "corrupt":
            replacement = b"not a classification artifact\n"
        else:
            other_lane = FixedLane()
            other_lane.provider = "other-provider"
            other_details = tmp_path / "other.jsonl"
            with ViewerDatabase(":memory:") as other_database:
                other_analytics = ViewerAnalytics(
                    other_database,
                    [descriptor],
                )
                run_classification(
                    other_analytics,
                    descriptor,
                    other_lane,
                    details_path=other_details,
                    repeats=1,
                )
            replacement = other_details.read_bytes()
        details.write_bytes(replacement)
        replacement_sha256 = hashlib.sha256(replacement).hexdigest()
        database.connection.execute(
            """
            UPDATE task_annotation_publication_intents
            SET prior_sha256 = ?
            WHERE output_path = ?
            """,
            [replacement_sha256, str(details)],
        )

    with ViewerDatabase(database_path) as reopened_database:
        reopened = ViewerAnalytics(reopened_database, [descriptor])
        expected = (
            "invalid classification details"
            if prior_state == "corrupt"
            else "does not match publication experiment"
        )
        with pytest.raises((ValueError, RuntimeError), match=expected):
            run_classification(
                reopened,
                descriptor,
                FixedLane(),
                details_path=details,
                repeats=1,
            )

        assert (
            reopened.get_task_annotation_publication_intent(str(details))
            is not None
        )
        assert staged.read_bytes() == staged_bytes
        assert reopened.export_task_annotations() == []


@pytest.mark.parametrize("tamper", ["third-hash", "missing"])
def test_ambiguous_pending_publication_remains_fail_closed(
    tmp_path,
    monkeypatch,
    tamper,
) -> None:
    descriptor = write_bundle(
        tmp_path / "bundle",
        dataset_id="org/benchmark",
        task_namespace="Independent",
    )
    database_path = tmp_path / "viewer.duckdb"
    details = (tmp_path / "details.jsonl").resolve()
    with ViewerDatabase(database_path) as database:
        analytics = ViewerAnalytics(database, [descriptor])
        run_classification(
            analytics,
            descriptor,
            FixedLane(),
            details_path=details,
            repeats=1,
        )
        original_begin = analytics.begin_task_annotation_publication

        def persist_then_crash(intent):
            original_begin(intent)
            raise SimulatedCrash

        monkeypatch.setattr(
            analytics,
            "begin_task_annotation_publication",
            persist_then_crash,
        )
        with pytest.raises(SimulatedCrash):
            run_classification(
                analytics,
                descriptor,
                FixedLane(parse_label="syntax-error-other"),
                details_path=details,
                repeats=1,
                force=True,
            )
        monkeypatch.setattr(
            analytics,
            "begin_task_annotation_publication",
            original_begin,
        )
        staged = classify_module._staged_artifact_path(details)
        if tamper == "third-hash":
            staged.write_bytes(b"tampered")
        else:
            staged.unlink()

    with ViewerDatabase(database_path) as reopened_database:
        reopened = ViewerAnalytics(reopened_database, [descriptor])
        with pytest.raises(
            RuntimeError,
            match="ambiguous classification publication evidence",
        ):
            run_classification(
                reopened,
                descriptor,
                FixedLane(),
                details_path=details,
                repeats=1,
            )
        assert (
            reopened.get_task_annotation_publication_intent(str(details))
            is not None
        )
        assert reopened.export_task_annotations() == []


def test_concurrent_force_on_same_path_keeps_file_and_rollup_sha_consistent(
    tmp_path,
) -> None:
    descriptor = write_bundle(
        tmp_path / "bundle",
        dataset_id="org/benchmark",
        task_namespace="Independent",
    )
    database_path = tmp_path / "viewer.duckdb"
    details = tmp_path / "details.jsonl"
    with (
        ViewerDatabase(database_path) as first_database,
        ViewerDatabase(database_path) as second_database,
    ):
        first_analytics = ViewerAnalytics(first_database, [descriptor])
        second_analytics = ViewerAnalytics(second_database, [descriptor])
        barrier = threading.Barrier(2)

        def classify(
            analytics: ViewerAnalytics,
            lane: FixedLane,
        ):
            barrier.wait()
            return run_classification(
                analytics,
                descriptor,
                lane,
                details_path=details,
                repeats=1,
                concurrency=1,
                force=True,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(
                classify,
                first_analytics,
                FixedLane(delays=True),
            )
            second_future = executor.submit(
                classify,
                second_analytics,
                FixedLane(parse_label="syntax-error-other"),
            )
            first_summary = first_future.result()
            second_summary = second_future.result()

        actual_sha256 = hashlib.sha256(details.read_bytes()).hexdigest()
        assert first_summary.experiment_identity == (
            second_summary.experiment_identity
        )
        exported = first_analytics.export_task_annotations()
        machine = [item for item in exported if item["origin"] == "machine"]
        assert machine
        assert {
            item["provenance"]["extra"]["details_sha256"] for item in machine
        } == {actual_sha256}
