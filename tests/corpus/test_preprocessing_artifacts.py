"""Contract coverage for preprocessing trace corpus projections."""

from __future__ import annotations

import hashlib

import pyarrow.parquet as pq
import pytest

from dr_code.corpus.preprocessing_artifacts import (
    AtomicProjectedPartWriter,
    PROJECTED_ARTIFACT_SCHEMAS,
    combine_projected_parts,
    project_preprocessing_result,
    write_projected_part,
)
from dr_code.preprocessing.definitions import (
    HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
)
from dr_code.preprocessing.runner import run_preprocessing
from dr_code.trace import (
    CandidateLineage,
    CandidateOrigin,
    CodeCandidateSetArtifact,
    TextArtifact,
    external_trace,
)


def _success_trace():
    first = "def first():\n    return 1\n"
    second = "async def second():\n    return 2\n"
    return external_trace(
        {
            "input": TextArtifact(text="raw"),
            "output": CodeCandidateSetArtifact(
                candidates=(first, second),
                lineage=(
                    CandidateLineage(
                        candidate_id="candidate-first",
                        origins=(
                            CandidateOrigin(
                                variant="normalized_raw_response",
                                strategy="fenced_blocks",
                            ),
                            CandidateOrigin(
                                variant="top_level_json_code",
                                strategy="markdown_wrapper",
                            ),
                        ),
                    ),
                    CandidateLineage(
                        candidate_id="candidate-second",
                        origins=(
                            CandidateOrigin(
                                variant="field_marker_code",
                                strategy="fenced_blocks",
                            ),
                        ),
                    ),
                ),
            ),
        },
        step_facts={
            "filter_compilable": {
                "survivors": [
                    {
                        "candidate_id": "candidate-first",
                        "parse_ok": True,
                        "parse_error": None,
                        "compile_ok": True,
                        "compile_error": None,
                        "compile_warnings": [],
                    },
                    {
                        "candidate_id": "candidate-second",
                        "parse_ok": True,
                        "parse_error": None,
                        "compile_ok": True,
                        "compile_error": None,
                        "compile_warnings": [],
                    },
                ]
            },
            "filter_has_top_level_function": {
                "survivors": [
                    {
                        "candidate_id": "candidate-first",
                        "top_level_function_count": 1,
                        "top_level_function_names": ["first"],
                        "top_level_async_function_names": [],
                    },
                    {
                        "candidate_id": "candidate-second",
                        "top_level_function_count": 1,
                        "top_level_function_names": ["second"],
                        "top_level_async_function_names": ["second"],
                    },
                ]
            },
            "return_all": {"outcome_code": "function_candidates_extracted"},
        },
    )


def test_missing_decoder_output_emits_only_result() -> None:
    projected = project_preprocessing_result("sample-1", None, None)

    assert projected.results == [
        {
            "sample_id": "sample-1",
            "decoder_output_presence": "missing",
            "raw_output_sha256": None,
            "outcome": "decoder_output_missing",
            "outcome_code": None,
            "failure_code": None,
            "failed_step": None,
            "cause": None,
            "propagated_through": None,
            "final_candidate_count": 0,
        }
    ]
    assert projected.candidates == []
    assert projected.step_facts == []
    assert projected.rejections == []


def test_blank_present_decoder_output_projects_official_absent_trace() -> None:
    trace = run_preprocessing(
        HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
        TextArtifact(text=" \n\t"),
    )
    projected = project_preprocessing_result("sample-blank", " \n\t", trace)

    assert projected.results[0]["decoder_output_presence"] == "present"
    assert projected.results[0]["failure_code"] == "decoder_output_blank"
    assert projected.results[0]["failed_step"] == "require_nonblank_text"
    assert projected.results[0]["final_candidate_count"] == 0
    assert projected.candidates == []
    assert projected.step_facts[0]["step_name"] == "require_nonblank_text"


def test_success_projects_multiple_candidates_origins_and_diagnostics() -> (
    None
):
    projected = project_preprocessing_result(
        "sample-ok", "raw", _success_trace()
    )

    assert (
        projected.results[0]["outcome_code"] == "function_candidates_extracted"
    )
    assert projected.results[0]["outcome"] == "function_candidates_extracted"
    assert projected.results[0]["final_candidate_count"] == 2
    assert (
        projected.results[0]["raw_output_sha256"]
        == hashlib.sha256(b"raw").hexdigest()
    )
    assert projected.candidates == [
        {
            "sample_id": "sample-ok",
            "candidate_index": 0,
            "candidate_id": "candidate-first",
            "cleaned_source": "def first():\n    return 1\n",
            "source_sha256": hashlib.sha256(
                b"def first():\n    return 1\n"
            ).hexdigest(),
            "origins": [
                {
                    "variant": "normalized_raw_response",
                    "strategy": "fenced_blocks",
                },
                {
                    "variant": "top_level_json_code",
                    "strategy": "markdown_wrapper",
                },
            ],
            "parse_ok": True,
            "parse_error": None,
            "compile_ok": True,
            "compile_error": None,
            "compile_warnings": [],
            "top_level_function_count": 1,
            "top_level_function_names": ["first"],
            "top_level_async_function_names": [],
        },
        {
            "sample_id": "sample-ok",
            "candidate_index": 1,
            "candidate_id": "candidate-second",
            "cleaned_source": "async def second():\n    return 2\n",
            "source_sha256": hashlib.sha256(
                b"async def second():\n    return 2\n"
            ).hexdigest(),
            "origins": [
                {
                    "variant": "field_marker_code",
                    "strategy": "fenced_blocks",
                }
            ],
            "parse_ok": True,
            "parse_error": None,
            "compile_ok": True,
            "compile_error": None,
            "compile_warnings": [],
            "top_level_function_count": 1,
            "top_level_function_names": ["second"],
            "top_level_async_function_names": ["second"],
        },
    ]


def test_rejections_are_mechanically_flattened_with_canonical_details() -> (
    None
):
    trace = external_trace(
        {
            "input": TextArtifact(text="raw"),
            "output": CodeCandidateSetArtifact(candidates=()),
        },
        step_facts={
            "filter": {
                "rejections": [
                    {
                        "candidate_id": "candidate-1",
                        "input_index": 3,
                        "reason_code": "not_compilable",
                        "compile_error": "SyntaxError: invalid syntax",
                        "parse_ok": False,
                    },
                    {"index": 4, "reason": "blank_or_whitespace"},
                ]
            },
            "complete": {"outcome_code": "empty_for_test"},
        },
    )

    projected = project_preprocessing_result("sample-rejection", "raw", trace)

    assert projected.rejections == [
        {
            "sample_id": "sample-rejection",
            "step_name": "filter",
            "candidate_id": "candidate-1",
            "input_index": 3,
            "reason_code": "not_compilable",
            "details_json": '{"compile_error":"SyntaxError: invalid syntax","parse_ok":false}',
        },
        {
            "sample_id": "sample-rejection",
            "step_name": "filter",
            "candidate_id": None,
            "input_index": 4,
            "reason_code": "blank_or_whitespace",
            "details_json": "{}",
        },
    ]


def test_projection_never_reparses_or_infers_candidate_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dr_code.code_analysis as code_analysis

    def fail_parse(*args: object, **kwargs: object) -> object:  # noqa: ARG001
        raise AssertionError("projector must not parse candidate source")

    monkeypatch.setattr(code_analysis.ast, "parse", fail_parse)

    projected = project_preprocessing_result(
        "sample-ok", "raw", _success_trace()
    )

    assert projected.candidates[0]["top_level_function_names"] == ["first"]


def test_parquet_schema_round_trip_and_streaming_combination(
    tmp_path,
) -> None:
    first = project_preprocessing_result("sample-a", "raw", _success_trace())
    second = project_preprocessing_result("sample-b", None, None)
    first_part = write_projected_part(tmp_path, "a", first)
    second_part = write_projected_part(tmp_path, "b", second)

    for relation_name, path in first_part.relation_paths.items():
        table = pq.read_table(path)
        assert table.schema.equals(PROJECTED_ARTIFACT_SCHEMAS[relation_name])

    combined_paths = combine_projected_parts(tmp_path, ("a", "b"))

    assert first_part.row_counts == {
        "results": 1,
        "candidates": 2,
        "step_facts": 3,
        "rejections": 0,
    }
    assert second_part.row_counts == {
        "results": 1,
        "candidates": 0,
        "step_facts": 0,
        "rejections": 0,
    }
    assert {
        relation_name: pq.read_table(path).num_rows
        for relation_name, path in combined_paths.items()
    } == {
        "results": 2,
        "candidates": 2,
        "step_facts": 3,
        "rejections": 0,
    }


def test_part_write_discards_unpublished_temp_shard(tmp_path) -> None:
    stale = tmp_path / "parts" / ".retry.tmp"
    stale.mkdir(parents=True)
    (stale / "partial.parquet").write_bytes(b"interrupted")

    part = write_projected_part(
        tmp_path,
        "retry",
        project_preprocessing_result("sample-a", None, None),
    )

    assert not stale.exists()
    assert part.relation_paths["results"].is_file()


def test_atomic_part_writer_appends_bounded_projections_and_writes_empty_relations(
    tmp_path,
) -> None:
    with AtomicProjectedPartWriter(tmp_path, "batched") as writer:
        writer.append(project_preprocessing_result("sample-a", None, None))
        writer.append(project_preprocessing_result("sample-b", None, None))

    part = writer.part

    assert part.row_counts == {
        "results": 2,
        "candidates": 0,
        "step_facts": 0,
        "rejections": 0,
    }
    for relation_name, path in part.relation_paths.items():
        table = pq.read_table(path)
        assert table.schema.equals(PROJECTED_ARTIFACT_SCHEMAS[relation_name])
    assert pq.read_table(part.relation_paths["results"]).num_rows == 2
    assert pq.read_table(part.relation_paths["candidates"]).num_rows == 0


def test_atomic_part_writer_aborts_unpublished_temp_shard(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="stop"):
        with AtomicProjectedPartWriter(tmp_path, "aborted") as writer:
            writer.append(project_preprocessing_result("sample-a", None, None))
            raise RuntimeError("stop")

    assert not (tmp_path / "parts" / ".aborted.tmp").exists()
    assert not (tmp_path / "parts" / "aborted").exists()
