from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dr_code.corpus.run_descriptor import RunValidationError
from dr_code.viewer.analytics import ViewerAnalytics
from dr_code.viewer.database import ViewerDatabase
from dr_code.viewer.domain import (
    IncompatibleRunsError,
    InvalidQueryError,
    Verdict,
)
from viewer.helpers import CORPUS_ROWS, CORPUS_SCHEMA, write_bundle


def test_list_runs_preserves_registration_order(tmp_path) -> None:
    before = write_bundle(tmp_path / "before", run_id="before")
    after = write_bundle(
        tmp_path / "after",
        run_id="after",
        corpus_path=before.corpus_path,
    )

    with ViewerDatabase(":memory:") as database:
        analytics = ViewerAnalytics(database, [before, after])

        assert [run.label for run in analytics.list_runs()] == [
            "before",
            "after",
        ]


def test_registered_runs_query_viewer_owned_artifact_snapshots(
    tmp_path,
) -> None:
    descriptor = write_bundle(tmp_path / "bundle", with_evaluation=False)
    with ViewerDatabase(":memory:") as database:
        analytics = ViewerAnalytics(database, [descriptor])
        descriptor.results_path.write_bytes(b"mutated after registration")

        waterfall = analytics.waterfall(descriptor.run_id)

    assert waterfall.stages[0].count == 9


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


def test_review_examples_batch_preserves_exact_selection_and_page_order(
    tmp_path, monkeypatch
) -> None:
    descriptor = write_bundle(
        tmp_path / "bundle",
        no_code_causes=("shared", "shared", None),
    )
    with ViewerDatabase(":memory:") as database:
        analytics = ViewerAnalytics(database, [descriptor])
        selected = analytics.examples(
            descriptor.run_id,
            failure_code="no_code_candidates",
            failed_step="extract_candidates",
            cause="shared",
            limit=2,
        )
        annotated = analytics.example(descriptor.run_id, "no-code-alt")
        assert annotated.decoder_output_sha256 is not None
        tag = analytics.create_tag("batch annotation")
        analytics.put_annotation(
            descriptor.corpus_sha256,
            annotated.sample_id,
            annotated.decoder_output_sha256,
            verdict=None,
            note="loaded in batch",
            tag_ids=[tag.tag_id],
        )
        expected = tuple(
            analytics.example(descriptor.run_id, item.sample_id)
            for item in selected.items
        )

        joined_calls: list[tuple[str, ...]] = []
        relation_calls: list[tuple[str, ...]] = []
        annotation_calls: list[tuple[tuple[str, str], ...]] = []
        original_joined = analytics._joined_rows
        original_relations = analytics._relation_rows_by_sample
        original_annotations = database.get_annotations

        def counted_joined(descriptor, sample_ids):
            joined_calls.append(sample_ids)
            return original_joined(descriptor, sample_ids)

        def counted_relations(path, sample_ids, columns, order_by):
            relation_calls.append(sample_ids)
            return original_relations(path, sample_ids, columns, order_by)

        def counted_annotations(corpus_sha256, identities):
            values = tuple(identities)
            annotation_calls.append(values)
            return original_annotations(corpus_sha256, values)

        def reject_single_detail_load(*_args, **_kwargs):
            pytest.fail("review page must not call example() per item")

        monkeypatch.setattr(analytics, "_joined_rows", counted_joined)
        monkeypatch.setattr(
            analytics, "_relation_rows_by_sample", counted_relations
        )
        monkeypatch.setattr(database, "get_annotations", counted_annotations)
        monkeypatch.setattr(analytics, "example", reject_single_detail_load)

        actual = analytics.review_examples(
            descriptor.run_id,
            failure_code="no_code_candidates",
            failed_step="extract_candidates",
            cause="shared",
            limit=2,
        )
        actual_joined_calls = list(joined_calls)
        actual_relation_calls = list(relation_calls)
        actual_annotation_calls = list(annotation_calls)
        second_page = analytics.review_examples(
            descriptor.run_id,
            failure_code="no_code_candidates",
            failed_step="extract_candidates",
            cause="shared",
            limit=1,
            offset=1,
        )
        searched = analytics.review_examples(
            descriptor.run_id,
            failure_code="no_code_candidates",
            failed_step="extract_candidates",
            cause="shared",
            search="Alternate prose",
        )
        null_cause = analytics.review_examples(
            descriptor.run_id,
            failure_code="no_code_candidates",
            failed_step="extract_candidates",
            cause_is_null=True,
        )
        empty_cause = analytics.review_examples(
            descriptor.run_id,
            failure_code="no_code_candidates",
            failed_step="extract_candidates",
            cause="",
        )

    assert actual.items == expected
    assert actual.total == second_page.total == 2
    assert [item.sample_id for item in actual.items] == [
        "no-code",
        "no-code-alt",
    ]
    assert [item.sample_id for item in second_page.items] == ["no-code-alt"]
    assert [item.sample_id for item in searched.items] == ["no-code-alt"]
    assert [item.sample_id for item in null_cause.items] == ["no-code-null"]
    assert empty_cause.total == 0
    assert actual_joined_calls == [("no-code", "no-code-alt")]
    assert actual_relation_calls == [
        ("no-code", "no-code-alt"),
        ("no-code", "no-code-alt"),
        ("no-code", "no-code-alt"),
    ]
    assert len(actual_annotation_calls) == 1
    assert len(actual_annotation_calls[0]) == 2
    detail = actual.items[0]
    assert (
        detail.failure_code,
        detail.failed_step,
        detail.cause,
    ) == ("no_code_candidates", "extract_candidates", "shared")
    assert detail.raw_decoder_output == "This is prose."
    assert detail.context["task_id"] == "Task/2"
    assert detail.facts


def test_review_examples_requires_one_exact_cause_representation(
    tmp_path,
) -> None:
    descriptor = write_bundle(tmp_path / "bundle")
    with ViewerDatabase(":memory:") as database:
        analytics = ViewerAnalytics(database, [descriptor])

        with pytest.raises(InvalidQueryError, match="exactly one"):
            analytics.review_examples(
                descriptor.run_id,
                failure_code="no_code_candidates",
                failed_step="extract_candidates",
            )
        with pytest.raises(InvalidQueryError, match="exactly one"):
            analytics.review_examples(
                descriptor.run_id,
                failure_code="no_code_candidates",
                failed_step="extract_candidates",
                cause="primary",
                cause_is_null=True,
            )


def test_compatible_comparison_preserves_denominators(tmp_path) -> None:
    baseline = write_bundle(
        tmp_path / "baseline",
        run_id="baseline",
    )
    candidate = write_bundle(
        tmp_path / "candidate",
        run_id="candidate",
        corpus_path=baseline.corpus_path,
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
    assert [stage.stage_id for stage in comparison.stages] == [
        "lost:output_present",
        "lost:output_nonblank",
        "has_extracted_candidate",
        "has_compilable_candidate",
        "has_top_level_candidate",
        "has_passing_candidate",
    ]
    assert [stage.baseline_count for stage in comparison.stages] == [
        1,
        1,
        4,
        3,
        2,
        1,
    ]
    assert [
        stage.baseline_denominator_count for stage in comparison.stages
    ] == [9, 9, 7, 7, 7, 7]
    assert [
        stage.candidate_denominator_count for stage in comparison.stages
    ] == [9, 9, 7, 7, 7, 7]
    assert [stage.baseline_rate for stage in comparison.stages] == [
        1 / 9,
        1 / 9,
        4 / 7,
        3 / 7,
        2 / 7,
        1 / 7,
    ]
    assert transition.total == 2


def test_preprocessing_only_comparison_uses_available_stages(tmp_path) -> None:
    baseline = write_bundle(
        tmp_path / "baseline",
        run_id="baseline",
        with_evaluation=False,
    )
    candidate = write_bundle(
        tmp_path / "candidate",
        run_id="candidate",
        corpus_path=baseline.corpus_path,
        with_evaluation=False,
    )
    with ViewerDatabase(":memory:") as database:
        comparison = ViewerAnalytics(database, [baseline, candidate]).compare(
            "baseline", "candidate"
        )

    assert [stage.stage_id for stage in comparison.stages] == [
        "lost:output_present",
        "lost:output_nonblank",
        "has_extracted_candidate",
        "has_compilable_candidate",
        "has_top_level_candidate",
    ]


def test_candidate_origins_use_current_structured_contract(tmp_path) -> None:
    descriptor = write_bundle(tmp_path / "bundle")
    with ViewerDatabase(":memory:") as database:
        analytics = ViewerAnalytics(database, [descriptor])
        detail = analytics.example(descriptor.run_id, "pass")

    assert detail.candidates[0]["origins"] == [
        {
            "path": [
                {
                    "kind": "response_representation",
                    "details": {"name": "normalized_raw_response"},
                },
                {"kind": "unfenced_segment", "details": {"index": 0}},
                {"kind": "anchored_python_block", "details": {"index": 0}},
            ]
        }
    ]


def test_comparison_rejects_different_evaluation_semantics(tmp_path) -> None:
    baseline = write_bundle(tmp_path / "baseline", run_id="baseline")
    candidate = write_bundle(
        tmp_path / "candidate",
        run_id="candidate",
        corpus_path=baseline.corpus_path,
    )
    assert candidate.evaluation_coordinates is not None
    changed_coordinates = dict(candidate.evaluation_coordinates)
    changed_coordinates["runtime_identity"] = "0" * 64
    candidate = replace(candidate, evaluation_coordinates=changed_coordinates)
    with ViewerDatabase(":memory:") as database:
        analytics = ViewerAnalytics(database, [baseline, candidate])

        with pytest.raises(
            IncompatibleRunsError, match="semantic coordinates"
        ):
            analytics.compare("baseline", "candidate")


def test_comparison_ignores_only_run_instance_evaluation_fingerprints(
    tmp_path,
) -> None:
    baseline = write_bundle(tmp_path / "baseline", run_id="baseline")
    candidate = write_bundle(
        tmp_path / "candidate",
        run_id="candidate",
        corpus_path=baseline.corpus_path,
    )
    assert candidate.evaluation_coordinates is not None
    changed = dict(candidate.evaluation_coordinates)
    changed["preprocessing_run"] = {
        "identity": "0" * 64,
        "relations": {"results": {"sha256": "1" * 64, "rows": 9}},
    }
    changed["evaluation_identity"] = "2" * 64
    candidate = replace(
        candidate,
        evaluation_coordinates=changed,
        evaluation_generation_id="3" * 64,
        evaluation_manifest_sha256="4" * 64,
        evaluation_pointer_sha256="5" * 64,
    )
    with ViewerDatabase(":memory:") as database:
        comparison = ViewerAnalytics(database, [baseline, candidate]).compare(
            "baseline", "candidate"
        )

    assert comparison.baseline.run_id == "baseline"


def test_context_collisions_and_nested_nonfinite_values_survive(
    tmp_path,
) -> None:
    corpus = tmp_path / "corpus.parquet"
    nested_type = pa.struct(
        [pa.field("values", pa.list_(pa.float64()), nullable=False)]
    )
    schema = pa.schema(
        [
            *CORPUS_SCHEMA,
            pa.field("outcome", pa.string(), nullable=False),
            pa.field("raw_output_sha256", pa.string(), nullable=False),
            pa.field("nested", nested_type, nullable=False),
        ]
    )
    rows = [
        (
            *row,
            "corpus-outcome",
            "corpus-fingerprint",
            {"values": [float("nan"), float("inf"), float("-inf")]},
        )
        for row in CORPUS_ROWS
    ]
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    field.name: value
                    for field, value in zip(schema, row, strict=True)
                }
                for row in rows
            ],
            schema=schema,
        ),
        corpus,
    )
    descriptor = write_bundle(tmp_path / "bundle", corpus_path=corpus)

    with ViewerDatabase(":memory:") as database:
        detail = ViewerAnalytics(database, [descriptor]).example(
            descriptor.run_id, "pass"
        )

    assert detail.outcome == "function_candidates_extracted"
    assert detail.context["outcome"] == "corpus-outcome"
    assert detail.context["raw_output_sha256"] == "corpus-fingerprint"
    assert detail.context["nested"] == {
        "values": ["NaN", "Infinity", "-Infinity"]
    }


@pytest.mark.parametrize("mutation", ["orphan_candidate", "null_fingerprint"])
def test_relational_integrity_rejects_orphans_and_null_fingerprints(
    tmp_path, mutation: str
) -> None:
    descriptor = write_bundle(tmp_path / mutation, with_evaluation=False)
    if mutation == "orphan_candidate":
        path = descriptor.candidates_path
        table = pq.read_table(path)
        rows = table.to_pylist()
        rows.append({**rows[0], "sample_id": "orphan/sample"})
        message = "candidates contains sample_id absent"
    else:
        path = descriptor.results_path
        table = pq.read_table(path)
        rows = table.to_pylist()
        next(row for row in rows if row["sample_id"] == "pass")[
            "raw_output_sha256"
        ] = None
        message = "raw_output_sha256 does not match"
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    manifest_path = descriptor.preprocessing_manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relation = path.stem
    manifest["relation_totals"][relation] = len(rows)
    manifest["relation_sha256"][relation] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RunValidationError, match=message):
        type(descriptor).from_paths(
            label=descriptor.label,
            corpus_path=descriptor.corpus_path,
            preprocessing=manifest_path.parent,
        )


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


def test_delete_annotation_authenticates_target_before_mutation(
    tmp_path,
) -> None:
    descriptor = write_bundle(tmp_path / "bundle")
    stale_output = "0" * 64
    with ViewerDatabase(":memory:") as database:
        analytics = ViewerAnalytics(database, [descriptor])
        database.put_annotation(
            descriptor.corpus_sha256,
            "no-code",
            stale_output,
            verdict=Verdict.EXPECTED_NO_CODE,
            note="seeded stale row",
        )

        with pytest.raises(InvalidQueryError, match="not present"):
            analytics.delete_annotation(
                descriptor.corpus_sha256, "no-code", stale_output
            )

        assert (
            database.get_annotation(
                descriptor.corpus_sha256, "no-code", stale_output
            )
            is not None
        )
        valid = analytics.example(descriptor.run_id, "no-code")
        assert valid.decoder_output_sha256 is not None
        assert not analytics.delete_annotation(
            descriptor.corpus_sha256,
            valid.sample_id,
            valid.decoder_output_sha256,
        )


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
