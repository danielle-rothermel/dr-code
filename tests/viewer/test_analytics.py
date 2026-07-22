from __future__ import annotations

from dataclasses import replace

import pytest

from dr_code.viewer.analytics import ViewerAnalytics
from dr_code.viewer.database import ViewerDatabase
from dr_code.viewer.domain import (
    IncompatibleRunsError,
    InvalidQueryError,
    Verdict,
)
from viewer.helpers import write_bundle


def test_waterfall_counts_and_stage_drilldowns_share_predicates(
    tmp_path,
) -> None:
    descriptor = write_bundle(tmp_path / "bundle")
    with ViewerDatabase(":memory:") as database:
        analytics = ViewerAnalytics(database, [descriptor])

        waterfall = analytics.waterfall(descriptor.run_id)

        assert [stage.stage_id for stage in waterfall.stages] == [
            "source",
            "output_present",
            "output_nonblank",
            "has_extracted_candidate",
            "has_compilable_candidate",
            "has_top_level_candidate",
            "has_tested_candidate",
            "has_passing_candidate",
        ]
        assert [stage.count for stage in waterfall.stages] == [
            9,
            8,
            7,
            4,
            3,
            2,
            2,
            1,
        ]
        assert [stage.denominator for stage in waterfall.stages] == [
            9,
            9,
            8,
            7,
            4,
            3,
            2,
            2,
        ]
        for stage in waterfall.stages:
            page = analytics.examples(
                descriptor.run_id, stage_id=stage.stage_id, limit=20
            )
            assert page.total == stage.count
            assert len(page.items) == stage.count


def test_each_nonzero_waterfall_loss_has_exact_drilldown(tmp_path) -> None:
    descriptor = write_bundle(tmp_path / "bundle")
    with ViewerDatabase(":memory:") as database:
        analytics = ViewerAnalytics(database, [descriptor])
        stages = analytics.waterfall(descriptor.run_id).stages

        for previous, current in zip(stages, stages[1:]):
            expected_loss = previous.count - current.count
            if expected_loss == 0:
                continue
            page = analytics.examples(
                descriptor.run_id,
                stage_id=f"lost:{current.stage_id}",
                limit=20,
            )
            assert page.total == expected_loss
            assert len(page.items) == expected_loss

        with pytest.raises(InvalidQueryError, match="no preceding-stage loss"):
            analytics.examples(descriptor.run_id, stage_id="lost:source")


def test_failures_are_terminal_nonblank_groups_with_exact_drilldown(
    tmp_path,
) -> None:
    descriptor = write_bundle(tmp_path / "bundle")
    with ViewerDatabase(":memory:") as database:
        analytics = ViewerAnalytics(database, [descriptor])

        failures = analytics.failures(descriptor.run_id)

        assert failures.total_count == 5
        assert {
            (group.failure_code, group.failed_step, group.cause, group.count)
            for group in failures.groups
        } == {
            ("no_code_candidates", "extract_candidates", "primary", 1),
            ("no_code_candidates", "extract_candidates", "alternate", 1),
            ("no_code_candidates", "extract_candidates", None, 1),
            (
                "no_compilable_candidate",
                "filter_compilable",
                "syntax",
                1,
            ),
            (
                "no_top_level_function_candidate",
                "filter_has_top_level_function",
                "no function",
                1,
            ),
        }
        for group in failures.groups:
            page = analytics.examples(
                descriptor.run_id,
                failure_code=group.failure_code,
                failed_step=group.failed_step,
                cause=group.cause,
                cause_is_null=group.cause is None,
            )
            assert page.total == group.count

        broad = analytics.examples(
            descriptor.run_id,
            failure_code="no_code_candidates",
            failed_step="extract_candidates",
        )
        assert broad.total == 3


def test_examples_are_stably_paginated_searchable_and_detailed(
    tmp_path,
) -> None:
    descriptor = write_bundle(tmp_path / "bundle")
    with ViewerDatabase(":memory:") as database:
        analytics = ViewerAnalytics(database, [descriptor])

        first = analytics.examples(descriptor.run_id, limit=2, offset=0)
        second = analytics.examples(descriptor.run_id, limit=2, offset=2)
        injection = analytics.examples(
            descriptor.run_id, search="%' OR TRUE --", limit=20
        )
        detail = analytics.example(descriptor.run_id, "compile-fail")

    assert first.total == second.total == 9
    assert [item.sample_id for item in (*first.items, *second.items)] == [
        "blank",
        "compile-fail",
        "fail",
        "missing",
    ]
    assert injection.total == 0
    assert detail.context["task_id"] == "Task/3"
    assert detail.outcome == "no_compilable_candidate"
    assert detail.raw_decoder_output == "def broken("
    assert detail.facts
    assert detail.rejections[0]["reason_code"] == "not_compilable"


def test_compatible_comparison_allows_different_definition_hashes(
    tmp_path,
) -> None:
    baseline = write_bundle(
        tmp_path / "baseline", run_id="baseline", definition_hash="a" * 128
    )
    candidate = write_bundle(
        tmp_path / "candidate",
        run_id="candidate",
        corpus_path=baseline.corpus_path,
        definition_hash="f" * 128,
    )
    with ViewerDatabase(":memory:") as database:
        analytics = ViewerAnalytics(database, [baseline, candidate])

        comparison = analytics.compare("baseline", "candidate")
        transition = analytics.examples(
            "baseline",
            compare_run_id="candidate",
            baseline_outcome="function_candidates_extracted",
            candidate_outcome="function_candidates_extracted",
        )

    assert sum(item.count for item in comparison.transitions) == 9
    assert all(stage.count_delta == 0 for stage in comparison.stages)
    assert comparison.stages[1].baseline_denominator_count == 9
    assert transition.total == 2


def test_comparison_rejects_different_evaluation_semantics(tmp_path) -> None:
    baseline = write_bundle(tmp_path / "baseline", run_id="baseline")
    candidate = write_bundle(
        tmp_path / "candidate",
        run_id="candidate",
        corpus_path=baseline.corpus_path,
    )
    assert candidate.evaluation_coordinates is not None
    changed_coordinates = dict(candidate.evaluation_coordinates)
    changed_coordinates["execution_fingerprint"] = "0" * 64
    candidate = replace(candidate, evaluation_coordinates=changed_coordinates)
    with ViewerDatabase(":memory:") as database:
        analytics = ViewerAnalytics(database, [baseline, candidate])

        with pytest.raises(
            IncompatibleRunsError, match="semantic coordinates"
        ):
            analytics.compare("baseline", "candidate")


def test_annotation_target_is_verified_and_appears_in_detail(tmp_path) -> None:
    descriptor = write_bundle(tmp_path / "bundle")
    with ViewerDatabase(":memory:") as database:
        analytics = ViewerAnalytics(database, [descriptor])
        before = analytics.example(descriptor.run_id, "no-code")
        assert before.decoder_output_sha256 is not None
        tag = analytics.create_tag("parser")

        annotation = analytics.put_annotation(
            descriptor.corpus_sha256,
            before.sample_id,
            before.decoder_output_sha256,
            verdict=Verdict.SHOULD_BE_PARSEABLE,
            note="extract this",
            tag_ids=[tag.tag_id],
        )
        after = analytics.example(descriptor.run_id, "no-code")

        with pytest.raises(InvalidQueryError, match="not present"):
            analytics.put_annotation(
                descriptor.corpus_sha256,
                before.sample_id,
                "0" * 64,
                verdict=Verdict.EXPECTED_NO_CODE,
            )
        export = analytics.export_annotations()

    assert after.annotation == annotation
    assert export[0]["tags"] == ["parser"]


def test_unchanged_output_reuses_annotation_in_another_run_list(
    tmp_path,
) -> None:
    baseline = write_bundle(tmp_path / "baseline", run_id="baseline")
    candidate = write_bundle(
        tmp_path / "candidate",
        run_id="candidate",
        corpus_path=baseline.corpus_path,
    )
    with ViewerDatabase(":memory:") as database:
        analytics = ViewerAnalytics(database, [baseline, candidate])
        example = analytics.example("baseline", "no-code")
        assert example.decoder_output_sha256 is not None
        analytics.put_annotation(
            baseline.corpus_sha256,
            example.sample_id,
            example.decoder_output_sha256,
            verdict=Verdict.EXPECTED_NO_CODE,
        )

        candidate_page = analytics.examples(
            "candidate",
            failure_code="no_code_candidates",
            failed_step="extract_candidates",
            cause="primary",
        )

    assert (
        candidate_page.items[0].annotation_verdict is Verdict.EXPECTED_NO_CODE
    )
