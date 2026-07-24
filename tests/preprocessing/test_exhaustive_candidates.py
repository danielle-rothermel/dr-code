"""Regression coverage for exhaustive candidate extraction and provenance."""

from __future__ import annotations

import json

import pytest

from dr_code.preprocessing.steps.base import StepFailedError, StepOutput
from dr_code.preprocessing.steps.dedupe_candidates import DedupeCandidates
from dr_code.preprocessing.steps.extract_candidates import ExtractCandidates
from dr_code.preprocessing.steps.filter_nonblank_candidates import (
    FilterNonblankCandidates,
)
from dr_code.preprocessing.steps.require_nonblank_text import (
    RequireNonblankText,
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
    is_absent,
)


def _origin(variant: str, strategy: str) -> CandidateOrigin:
    return CandidateOrigin(variant=variant, strategy=strategy)


def _candidate_set(output: StepOutput) -> CodeCandidateSetArtifact:
    assert isinstance(output.value, CodeCandidateSetArtifact)
    return output.value


def test_extraction_collects_every_strategy_in_order_with_lineage() -> None:
    output = ExtractCandidates().apply(
        TextArtifact(text="def f():\n    return 1")
    )

    candidates = _candidate_set(output)
    assert candidates.candidates == (
        "def f():\n    return 1",
        "def f():\n    return 1",
    )
    assert [item.origins[0].strategy for item in candidates.lineage] == [
        "fenced_blocks",
        "markdown_wrapper",
    ]
    assert output.facts["candidate_count"] == 2
    assert output.facts["origins"] == [
        {
            "variant": "normalized_raw_response",
            "strategy": "fenced_blocks",
            "candidate_count": 1,
        },
        {
            "variant": "normalized_raw_response",
            "strategy": "markdown_wrapper",
            "candidate_count": 1,
        },
        {
            "variant": "normalized_raw_response",
            "strategy": "escaped_python",
            "candidate_count": 0,
        },
        {
            "variant": "normalized_raw_response",
            "strategy": "escaped_markdown_wrapper",
            "candidate_count": 0,
        },
    ]


def test_extraction_reads_json_string_and_top_level_code_object() -> None:
    string_output = ExtractCandidates().apply(
        TextArtifact(text=json.dumps("def from_string():\n    return 1"))
    )
    object_output = ExtractCandidates().apply(
        TextArtifact(
            text=json.dumps({"code": "def from_object():\n    return 2"})
        )
    )

    string_candidates = _candidate_set(string_output)
    object_candidates = _candidate_set(object_output)
    assert "decoded_whole_response_json_string" in {
        origin.variant
        for item in string_candidates.lineage
        for origin in item.origins
    }
    assert "top_level_json_code" in {
        origin.variant
        for item in object_candidates.lineage
        for origin in item.origins
    }


def test_oversized_json_number_becomes_a_normal_preprocessing_failure() -> (
    None
):
    trace = run_preprocessing(
        HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
        TextArtifact(text="9" * 5_000),
    )

    output = trace.value("output")
    assert is_absent(output)
    assert output.failure_code == "no_code_candidates"


def test_json_code_is_recovered_beside_an_oversized_number() -> None:
    raw = (
        '{"irrelevant":'
        + ("9" * 5_000)
        + ',"code":"def recovered():\\n    return 1"}'
    )

    output = run_preprocessing(
        HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
        TextArtifact(text=raw),
    ).value("output")

    assert isinstance(output, CodeCandidateSetArtifact)
    assert "def recovered():\n    return 1" in output.candidates


def test_extraction_reads_marker_payload_without_consuming_next_marker() -> (
    None
):
    output = ExtractCandidates().apply(
        TextArtifact(
            text=(
                "[[ ## code ## ]]\n"
                "def f():\n"
                "    return 1\n"
                "[[ ## complete ## ]]\n"
                "done"
            )
        )
    )

    candidates = _candidate_set(output)
    marker_candidates = [
        (candidate, lineage)
        for candidate, lineage in zip(
            candidates.candidates, candidates.lineage, strict=True
        )
        if lineage.origins[0].variant == "field_marker_code"
    ]
    assert marker_candidates
    assert all(
        "complete" not in candidate for candidate, _ in marker_candidates
    )


def test_indented_marker_payload_is_dedented_without_boundary_corruption() -> (
    None
):
    trace = run_preprocessing(
        HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
        TextArtifact(
            text=(
                "[[ ## code ## ]]\n"
                "    import math\n"
                "    def area(radius):\n"
                "        return math.pi * radius ** 2\n"
                "[[ ## complete ## ]]"
            )
        ),
    )

    output = trace.value("output")
    assert isinstance(output, CodeCandidateSetArtifact)
    assert any(
        candidate.startswith("import math\ndef area")
        for candidate in output.candidates
    )


def test_extraction_failure_has_code_and_structured_facts() -> None:
    with pytest.raises(StepFailedError) as raised:
        ExtractCandidates().apply(TextArtifact(text="just prose"))

    assert raised.value.failure_code == "no_code_candidates"
    assert raised.value.facts["candidate_count"] == 0
    assert raised.value.facts["origins"]


def test_require_nonblank_text_fails_blank_decoder_output() -> None:
    with pytest.raises(StepFailedError) as raised:
        RequireNonblankText().apply(TextArtifact(text=" \n\t"))

    assert raised.value.failure_code == "decoder_output_blank"
    assert raised.value.facts == {
        "text_character_count": 3,
        "is_nonblank": False,
    }


def test_filter_nonblank_removes_cleaned_blank_and_preserves_lineage() -> None:
    kept_origin = _origin("raw", "fenced_blocks")
    output = FilterNonblankCandidates().apply(
        CodeCandidateSetArtifact(
            candidates=("\n\t", "x = 1\n"),
            lineage=(
                CandidateLineage(
                    origins=(_origin("raw", "markdown_wrapper"),)
                ),
                CandidateLineage(origins=(kept_origin,)),
            ),
        )
    )

    candidates = _candidate_set(output)
    assert candidates.candidates == ("x = 1\n",)
    assert candidates.lineage == (CandidateLineage(origins=(kept_origin,)),)
    assert output.facts["rejections"] == [
        {"index": 0, "reason": "blank_or_whitespace"}
    ]


def test_filter_nonblank_fails_when_cleaning_exhausts_candidates() -> None:
    with pytest.raises(StepFailedError) as raised:
        FilterNonblankCandidates().apply(
            CodeCandidateSetArtifact(candidates=("  ", "\n"))
        )

    assert raised.value.failure_code == "no_nonblank_cleaned_candidate"
    assert raised.value.facts["output_candidate_count"] == 0


def test_dedupe_keeps_first_source_merges_origins_and_assigns_stable_id() -> (
    None
):
    first = _origin("normalized_raw_response", "fenced_blocks")
    duplicate = _origin("top_level_json_code", "markdown_wrapper")
    value = CodeCandidateSetArtifact(
        candidates=("x = 1\n", "x = 1\n", "y = 2\n"),
        lineage=(
            CandidateLineage(origins=(first,)),
            CandidateLineage(origins=(duplicate,)),
            CandidateLineage(origins=(_origin("raw", "fenced_blocks"),)),
        ),
    )

    output = DedupeCandidates().apply(value)
    repeated = DedupeCandidates().apply(value)

    candidates = _candidate_set(output)
    repeated_candidates = _candidate_set(repeated)
    assert candidates.candidates == ("x = 1\n", "y = 2\n")
    assert candidates.lineage[0].origins == (first, duplicate)
    assert candidates.lineage[0].candidate_id
    assert (
        candidates.lineage[0].candidate_id
        == repeated_candidates.lineage[0].candidate_id
    )
    assert output.facts["duplicate_groups"] == [
        {
            "candidate_id": candidates.lineage[0].candidate_id,
            "first_input_index": 0,
            "duplicate_input_indexes": [1],
            "merged_origins": [
                first.model_dump(mode="json"),
                duplicate.model_dump(mode="json"),
            ],
        }
    ]


def test_dedupe_fails_only_for_empty_input() -> None:
    with pytest.raises(StepFailedError) as raised:
        DedupeCandidates().apply(CodeCandidateSetArtifact(candidates=()))

    assert raised.value.failure_code == "no_candidates_to_dedupe"


def test_pipeline_dedupes_after_cleaning_and_retains_all_origins() -> None:
    trace = run_preprocessing(
        HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
        TextArtifact(text="def f():\n    return 1"),
    )
    deduped = trace.value("dedupe_candidates")

    assert isinstance(deduped, CodeCandidateSetArtifact)
    assert deduped.candidates == ("def f():\n    return 1",)
    assert deduped.lineage[0].candidate_id
    assert deduped.lineage[0].origins == (
        _origin("normalized_raw_response", "fenced_blocks"),
        _origin("normalized_raw_response", "markdown_wrapper"),
    )


def test_pipeline_supports_tilde_fences() -> None:
    trace = run_preprocessing(
        HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
        TextArtifact(text="~~~python\ndef add_one(x):\n    return x + 1\n~~~"),
    )
    output = trace.value("output")

    assert isinstance(output, CodeCandidateSetArtifact)
    assert output.candidates == ("def add_one(x):\n    return x + 1",)
