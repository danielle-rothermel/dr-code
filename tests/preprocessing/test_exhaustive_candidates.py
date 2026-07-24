"""Regression coverage for exhaustive candidate extraction and provenance."""

from __future__ import annotations

import json

import pytest

from dr_code.preprocessing.extraction import response_fragments
from dr_code.preprocessing.steps.base import StepFailedError, StepOutput
from dr_code.preprocessing.candidate_identity import candidate_id_for_source
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
    ExtractionOperation,
    IdentifiedCandidateSetArtifact,
    TextArtifact,
    is_absent,
)


def _origin(variant: str, strategy: str) -> CandidateOrigin:
    return CandidateOrigin(
        path=(
            ExtractionOperation(
                kind="test_origin",
                details={"variant": variant, "strategy": strategy},
            ),
        )
    )


def _candidate_set(output: StepOutput) -> CodeCandidateSetArtifact:
    assert isinstance(output.value, CodeCandidateSetArtifact)
    return output.value


def test_extraction_records_complete_ordered_lineage() -> None:
    output = ExtractCandidates().apply(
        TextArtifact(text="def f():\n    return 1")
    )

    candidates = _candidate_set(output)
    assert candidates.candidates == ("def f():\n    return 1",)
    assert [
        operation.kind for operation in candidates.lineage[0].origins[0].path
    ] == [
        "response_representation",
        "unfenced_segment",
        "anchored_python_block",
    ]
    assert output.facts["candidate_count"] == 1
    assert output.facts["paths"] == [
        candidates.lineage[0].origins[0].model_dump(mode="json")
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
        str(operation.details.get("name"))
        for item in string_candidates.lineage
        for origin in item.origins
        for operation in origin.path
        if operation.kind == "response_representation"
    }
    assert "top_level_json_code" in {
        str(operation.details.get("name"))
        for item in object_candidates.lineage
        for origin in item.origins
        for operation in origin.path
        if operation.kind == "response_representation"
    }


def test_oversized_json_number_becomes_a_normal_preprocessing_failure() -> (
    None
):
    trace = run_preprocessing(
        HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION.materialize(),
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
        HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION.materialize(),
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
        if any(
            operation.kind == "response_representation"
            and operation.details.get("name") == "field_marker_code"
            for operation in lineage.origins[0].path
        )
    ]
    assert marker_candidates
    assert all(
        "complete" not in candidate for candidate, _ in marker_candidates
    )


@pytest.mark.parametrize(
    "text",
    [
        "The inline marker [[ ## code ## ]] is prose.",
        'marker = "[[ ## code ## ]]"',
        "# [[ ## code ## ]]",
    ],
)
def test_inline_field_markers_do_not_create_provenance(text: str) -> None:
    fragments = response_fragments(text)

    assert all(
        operation.details.get("name") != "field_marker_code"
        for fragment in fragments
        for operation in fragment.path
    )


@pytest.mark.parametrize(
    "inline_marker",
    [
        "The inline marker [[ ## complete ## ]] is prose.",
        'marker = "[[ ## complete ## ]]"',
        "# [[ ## complete ## ]]",
    ],
)
def test_inline_field_markers_do_not_terminate_payload(
    inline_marker: str,
) -> None:
    fragments = response_fragments(
        f"[[ ## code ## ]]\nfirst\n{inline_marker}\nlast\n[[ ## complete ## ]]"
    )

    marker_fragment = next(
        fragment
        for fragment in fragments
        if fragment.path[0].details.get("name") == "field_marker_code"
    )
    assert marker_fragment.text == f"first\n{inline_marker}\nlast"


def test_indented_structural_markers_extract_crlf_payload() -> None:
    fragments = response_fragments(
        " \t[[ ## code ## ]] \t\r\n"
        "def f():\r\n"
        "    return 1\r\n"
        "\t[[ ## complete ## ]] \t\r\n"
        "ignored"
    )

    marker_fragment = next(
        fragment
        for fragment in fragments
        if fragment.path[0].details.get("name") == "field_marker_code"
    )
    assert marker_fragment.text == "def f():\r\n    return 1"


def test_indented_marker_payload_is_dedented_without_boundary_corruption() -> (
    None
):
    trace = run_preprocessing(
        HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION.materialize(),
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
    assert raised.value.facts["paths"] == []


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
            CodeCandidateSetArtifact(
                candidates=("  ", "\n"),
                lineage=(
                    CandidateLineage(origins=(_origin("raw", "first_blank"),)),
                    CandidateLineage(
                        origins=(_origin("raw", "second_blank"),)
                    ),
                ),
            )
        )

    assert raised.value.failure_code == "no_nonblank_cleaned_candidate"
    assert raised.value.facts["output_candidate_count"] == 0


def test_candidate_identity_is_stable_and_content_derived() -> None:
    source = "def f():\n    return 1"

    assert candidate_id_for_source(source) == candidate_id_for_source(source)
    assert candidate_id_for_source(source) != candidate_id_for_source(
        source + "\n"
    )


def test_pipeline_dedupes_after_cleaning_and_retains_all_origins() -> None:
    trace = run_preprocessing(
        HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION.materialize(),
        TextArtifact(text="def f():\n    return 1"),
    )
    deduped = trace.value("identify_candidates")

    assert isinstance(deduped, IdentifiedCandidateSetArtifact)
    assert tuple(item.source for item in deduped.candidates) == (
        "def f():\n    return 1",
    )
    assert deduped.candidates[0].lineage.candidate_id
    assert [
        operation.kind
        for operation in deduped.candidates[0].lineage.origins[0].path
    ] == [
        "response_representation",
        "unfenced_segment",
        "anchored_python_block",
    ]


def test_pipeline_supports_tilde_fences() -> None:
    trace = run_preprocessing(
        HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION.materialize(),
        TextArtifact(text="~~~python\ndef add_one(x):\n    return x + 1\n~~~"),
    )
    output = trace.value("output")

    assert isinstance(output, CodeCandidateSetArtifact)
    assert output.candidates == ("def add_one(x):\n    return x + 1",)
