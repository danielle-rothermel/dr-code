from __future__ import annotations

import pytest

from dr_code.preprocessing.failures import PreprocessingFailureCode
from dr_code.preprocessing.steps.base import StepFailedError
from dr_code.preprocessing.steps.filter_code_repr import FilterCodeRepr
from dr_code.preprocessing.steps.filter_compilable import FilterCompilable
from dr_code.preprocessing.steps.filter_plain_literal import FilterPlainLiteral
from dr_code.preprocessing.steps.filter_top_level_functions import (
    FilterTopLevelFunctions,
)
from dr_code.preprocessing.steps.inspect_candidates import (
    InspectCandidates,
    top_level_function_names,
)
from dr_code.preprocessing.steps.materialize_candidate_set import (
    MaterializeCandidateSet,
)
from dr_code.trace import (
    CandidateOrigin,
    CodeCandidate,
    CodeCandidateSetArtifact,
    ExtractionOperation,
    InspectedCodeCandidateSetArtifact,
)


def _candidate_set(*sources: str) -> CodeCandidateSetArtifact:
    return CodeCandidateSetArtifact(
        candidates=tuple(
            CodeCandidate(
                source=source,
                origins=(
                    CandidateOrigin(
                        operation=ExtractionOperation(
                            operation_name="text_segments"
                        ),
                        input_location=index,
                    ),
                ),
            )
            for index, source in enumerate(sources)
        )
    )


def _inspected(*sources: str) -> InspectedCodeCandidateSetArtifact:
    value = InspectCandidates().apply(_candidate_set(*sources)).value
    assert isinstance(value, InspectedCodeCandidateSetArtifact)
    return value


def _inspected_sources(
    value: InspectedCodeCandidateSetArtifact,
) -> tuple[str, ...]:
    return tuple(item.candidate.source for item in value.candidates)


def test_inspection_records_compilable_source_facts() -> None:
    out = _inspected("def f():\n    return 1\n")
    (item,) = out.candidates
    assert item.inspection.parses is True
    assert item.inspection.compiles is True
    assert item.inspection.parse_error is None
    assert item.inspection.compile_error is None
    assert item.inspection.top_level_function_names == ("f",)


def test_inspection_records_parse_failure() -> None:
    out = _inspected("def broken(:\n")
    (item,) = out.candidates
    assert item.inspection.parses is False
    assert item.inspection.compiles is False
    assert "SyntaxError" in (item.inspection.parse_error or "")
    assert item.inspection.top_level_function_names == ()


def test_inspection_carries_candidate_and_order_through() -> None:
    candidate_set = _candidate_set("def a():\n    return 1", "x = 1")
    out = InspectCandidates().apply(candidate_set)
    assert [item.candidate for item in out.value.candidates] == list(
        candidate_set.candidates
    )


def test_top_level_function_names_excludes_nested_definitions() -> None:
    import ast

    tree = ast.parse(
        "def outer():\n"
        "    def inner():\n"
        "        return 1\n"
        "    return inner\n"
        "class C:\n"
        "    def method(self):\n"
        "        return 2\n"
        "async def top_async():\n"
        "    return 3\n"
    )
    assert top_level_function_names(tree) == ("outer", "top_async")


def test_inspection_facts_report_counts() -> None:
    out = InspectCandidates().apply(
        _candidate_set("def f():\n    return 1", "def broken(:")
    )
    assert out.facts["inspected_count"] == 2
    assert out.facts["compiles_count"] == 1


def test_filter_compilable_uses_the_stored_inspection() -> None:
    out = FilterCompilable().apply(
        _inspected("def f():\n    return 1", "def broken(:")
    )
    assert _inspected_sources(out.value) == ("def f():\n    return 1",)
    assert "SyntaxError" in out.facts["rejected_1"]


def test_filter_compilable_reports_the_recorded_compile_error() -> None:
    inspected = _inspected("def broken(:")
    (item,) = inspected.candidates
    out = FilterCompilable().apply(inspected)
    assert out.facts["rejected_0"] == item.inspection.compile_error


def test_filter_top_level_functions_uses_the_stored_names() -> None:
    out = FilterTopLevelFunctions().apply(
        _inspected("def f():\n    return 1", "x = 1", "import os")
    )
    assert _inspected_sources(out.value) == ("def f():\n    return 1",)
    assert out.facts["rejected_1"] == "no top-level function definitions"
    assert out.facts["rejected_2"] == "no top-level function definitions"


def test_filter_plain_literal_drops_literal_modules() -> None:
    out = FilterPlainLiteral().apply(_inspected("[1, 2, 3]", "x = 1\n"))
    assert _inspected_sources(out.value) == ("x = 1\n",)
    assert out.facts["rejected_0"] == "plain literal module"


def test_filter_code_repr_drops_repr_assignments() -> None:
    out = FilterCodeRepr().apply(_inspected('code = "x = 1"', "x = 1\n"))
    assert _inspected_sources(out.value) == ("x = 1\n",)
    assert out.facts["rejected_0"] == "code representation assignment"


def test_filters_keep_survivors_and_their_inspections_identical() -> None:
    inspected = _inspected("def f():\n    return 1", "def broken(:")
    out = FilterCompilable().apply(inspected)
    assert out.value.candidates[0] == inspected.candidates[0]


def test_materialize_returns_the_complete_set_in_order() -> None:
    inspected = _inspected("def a():\n    return 1", "def b():\n    return 2")
    out = MaterializeCandidateSet().apply(inspected)
    assert out.value == inspected
    assert out.facts["candidate_count"] == 2


def test_materialize_empty_set_raises() -> None:
    with pytest.raises(StepFailedError) as excinfo:
        MaterializeCandidateSet().apply(_inspected())
    assert (
        excinfo.value.code
        is PreprocessingFailureCode.NO_CANDIDATE_SURVIVED_FILTERING
    )
