"""Per-step interface, composition, and determinism tests."""

from __future__ import annotations

import json
import unicodedata

import pytest

from dr_code.preprocessing.failures import PreprocessingFailureCode
from dr_code.preprocessing.import_inference import infer_necessary_imports
from dr_code.preprocessing.registry import REGISTRY
from dr_code.preprocessing.steps.add_last_return_salvage import (
    AddLastReturnSalvage,
)
from dr_code.preprocessing.steps.base import (
    Step,
    StepFailedError,
)
from dr_code.preprocessing.steps.collapse_blank_runs import (
    CollapseBlankRuns,
)
from dr_code.preprocessing.steps.dedent_candidates import DedentCandidates
from dr_code.preprocessing.steps.dedupe_candidates import DedupeCandidates
from dr_code.preprocessing.steps.dedupe_imports import DedupeImports
from dr_code.preprocessing.steps.drop_blank_candidates import (
    DropBlankCandidates,
)
from dr_code.preprocessing.steps.expand_tabs import ExpandTabs
from dr_code.preprocessing.steps.extract_all_representations import (
    ExtractAllRepresentations,
    Representation,
)
from dr_code.preprocessing.steps.filter_code_repr import FilterCodeRepr
from dr_code.preprocessing.steps.filter_compilable import (
    FilterCompilable,
)
from dr_code.preprocessing.steps.filter_plain_literal import (
    FilterPlainLiteral,
)
from dr_code.preprocessing.steps.filter_top_level_functions import (
    FilterTopLevelFunctions,
)
from dr_code.preprocessing.steps.infer_missing_imports import (
    InferMissingImports,
)
from dr_code.preprocessing.steps.inspect_candidates import (
    InspectCandidates,
    top_level_function_names,
)
from dr_code.preprocessing.steps.materialize_candidate_set import (
    MaterializeCandidateSet,
)
from dr_code.preprocessing.steps.normalize_line_endings import (
    NormalizeLineEndings,
)
from dr_code.preprocessing.steps.normalize_smart_quotes import (
    NormalizeSmartQuotes,
)
from dr_code.preprocessing.steps.normalize_unicode import (
    NormalizeUnicode,
)
from dr_code.preprocessing.steps.reject_blank_input import RejectBlankInput
from dr_code.preprocessing.steps.repair_import_lines import (
    RepairImportLines,
)
from dr_code.preprocessing.steps.split_on_name_guard import (
    SplitOnNameGuard,
)
from dr_code.preprocessing.steps.strip_fences import StripFences
from dr_code.preprocessing.steps.strip_trailing_whitespace import (
    StripTrailingWhitespace,
)
from dr_code.preprocessing.steps.trim_outer_blanks import TrimOuterBlanks
from dr_code.core.source.text_transforms import (
    collapse_blank_runs,
    drop_after_last_return,
    drop_if_name,
    normalize_line_endings,
    normalize_text,
    strip_code_fences,
    strip_trailing_whitespace,
)
from dr_code.trace import (
    ArtifactKind,
    CandidateOrigin,
    CodeCandidate,
    CodeCandidateSetArtifact,
    ExtractionOperation,
    InspectedCodeCandidateSetArtifact,
    TextArtifact,
)


def _candidate_set(*sources: str) -> CodeCandidateSetArtifact:
    """A candidate set whose records carry a plain synthetic origin."""
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
    """Run the inspection step to build a genuine inspected set."""
    value = InspectCandidates().apply(_candidate_set(*sources)).value
    assert isinstance(value, InspectedCodeCandidateSetArtifact)
    return value


def _sources(value: CodeCandidateSetArtifact) -> tuple[str, ...]:
    return tuple(candidate.source for candidate in value.candidates)


def _inspected_sources(
    value: InspectedCodeCandidateSetArtifact,
) -> tuple[str, ...]:
    return tuple(item.candidate.source for item in value.candidates)


def _operations(candidate: CodeCandidate) -> tuple[str, ...]:
    return tuple(
        origin.operation.operation_name for origin in candidate.origins
    )


# Garbage inputs reused from the wrapped modules' existing fixtures.
GARBAGE_TEXT = (
    "",
    "def broken(:\n",
    "```\nunterminated fence",
    "plain prose, no code at all",
    "smart ‘quotes’ and “doubles”\r\nCRLF\ttabs  \n\n\n\n",
)


# --- determinism battery (no rng) ------------------------------------


def _apply_twice(step_cls: type[Step], value) -> object:
    """Apply twice; return a comparable outcome (StepOutput or the
    StepFailedError marker) so determinism covers both success and the
    Absent path."""
    step = step_cls(step_cls.Settings())
    try:
        first = step.apply(value)
    except StepFailedError as exc:
        first = ("failed", exc.code, exc.cause)
    try:
        second = step.apply(value)
    except StepFailedError as exc:
        second = ("failed", exc.code, exc.cause)
    return (first, second)


def _sample_for(step_cls: type[Step]):
    """An input artifact of the step's INPUT kind, processable by the step."""
    if step_cls.INPUT is ArtifactKind.TEXT:
        return TextArtifact(text="```python\ndef f():\n    return 1\n```\n")
    if step_cls.INPUT is ArtifactKind.INSPECTED_CODE_CANDIDATE_SET:
        return _inspected(
            "def f():\n    return 1\n", "def g():\n    return 2\n"
        )
    return _candidate_set(
        "def f():\n    return 1\n", "def g():\n    return 2\n"
    )


@pytest.mark.parametrize("step_cls", REGISTRY.values())
def test_step_is_deterministic(step_cls: type[Step]) -> None:
    """apply twice with identical settings+input => equal (corruption-test
    pattern, minus rng). Covers both success and the Absent path."""
    first, second = _apply_twice(step_cls, _sample_for(step_cls))
    assert first == second


# --- atomic text steps wrap their functions --------------------------


def test_normalize_line_endings_wraps_function() -> None:
    raw = "a\r\nb\rc\n"
    out = NormalizeLineEndings().apply(TextArtifact(text=raw))
    assert out.value == TextArtifact(text=normalize_line_endings(raw))


def test_normalize_unicode_applies_nfkc() -> None:
    raw = "ｄｅｆ"
    out = NormalizeUnicode().apply(TextArtifact(text=raw))
    assert out.value == TextArtifact(text=unicodedata.normalize("NFKC", raw))


def test_expand_tabs_uses_tab_width_setting() -> None:
    raw = "a\tb"
    out = ExpandTabs(ExpandTabs.Settings(tab_width=2)).apply(
        TextArtifact(text=raw)
    )
    assert out.value == TextArtifact(text="a b")


def test_strip_trailing_whitespace_wraps_function() -> None:
    raw = "x = 1  \ny = 2\t\n"
    out = StripTrailingWhitespace().apply(TextArtifact(text=raw))
    assert out.value == TextArtifact(text=strip_trailing_whitespace(raw))


def test_collapse_blank_runs_wraps_function() -> None:
    raw = "a\n\n\n\nb"
    out = CollapseBlankRuns().apply(TextArtifact(text=raw))
    assert out.value == TextArtifact(text=collapse_blank_runs(raw))


def test_trim_outer_blanks_strips_newlines() -> None:
    out = TrimOuterBlanks().apply(TextArtifact(text="\n\nx\n\n"))
    assert out.value == TextArtifact(text="x")


@pytest.mark.parametrize("raw", GARBAGE_TEXT)
def test_atomic_text_sequence_equals_normalize_text(raw: str) -> None:
    """The six atomic steps, in order, reproduce normalize_text."""
    value = TextArtifact(text=raw)
    for step_cls in (
        NormalizeLineEndings,
        NormalizeUnicode,
        ExpandTabs,
        StripTrailingWhitespace,
        CollapseBlankRuns,
        TrimOuterBlanks,
    ):
        value = step_cls().apply(value).value
        assert isinstance(value, TextArtifact)
    assert value.text == normalize_text(raw)


# --- blank-input guard -----------------------------------------------


def test_reject_blank_input_passes_through_non_blank() -> None:
    value = TextArtifact(text="def f(): pass")
    assert RejectBlankInput().apply(value).value == value


@pytest.mark.parametrize("blank", ["", "   ", "\n\n", " \t\n "])
def test_reject_blank_input_fails_on_blank(blank: str) -> None:
    with pytest.raises(StepFailedError) as excinfo:
        RejectBlankInput().apply(TextArtifact(text=blank))
    assert excinfo.value.code is PreprocessingFailureCode.BLANK_INPUT
    assert excinfo.value.evidence == {"input_length": len(blank)}


# --- elementwise candidate steps -------------------------------------


def test_strip_fences_wraps_function() -> None:
    out = StripFences().apply(_candidate_set("```python\nx = 1\n```"))
    assert _sources(out.value) == (strip_code_fences("```python\nx = 1\n```"),)


def test_dedent_wraps_textwrap() -> None:
    out = DedentCandidates().apply(_candidate_set("    x = 1\n    y = 2\n"))
    assert _sources(out.value) == ("x = 1\ny = 2\n",)


def test_candidate_map_step_extends_lineage_with_its_operation() -> None:
    out = DedentCandidates().apply(_candidate_set("    x = 1\n"))
    (candidate,) = out.value.candidates
    assert _operations(candidate) == ("text_segments", "dedent_candidates")
    assert candidate.origins[-1].input_location == 0


def test_normalize_smart_quotes_converts_delimiters() -> None:
    out = NormalizeSmartQuotes().apply(_candidate_set("x = “a”\n"))
    assert _sources(out.value) == ('x = "a"\n',)


def test_normalize_smart_quotes_preserves_string_contents() -> None:
    src = 'x = "don’t “quote” me"\n'
    out = NormalizeSmartQuotes().apply(_candidate_set(src))
    assert _sources(out.value) == (src,)


def test_normalize_smart_quotes_comment_apostrophe_not_a_delimiter() -> None:
    # The apostrophe in the comment must not open string state; the real
    # literal's smart-quote contents stay preserved.
    src = "# don't\nx = 'a“b'\n"
    out = NormalizeSmartQuotes().apply(_candidate_set(src))
    assert _sources(out.value) == (src,)


def test_split_on_name_guard_flattens_in_place() -> None:
    src = "def f():\n    return 1\nif __name__ == '__main__':\n    pass"
    out = SplitOnNameGuard().apply(_candidate_set(src))
    assert _sources(out.value) == tuple(drop_if_name(src))


def test_split_on_name_guard_preserves_order_with_multiple() -> None:
    a = "def a():\n    return 1\nif __name__ == '__main__':\n    pass"
    b = "def b():\n    return 2\n"
    out = SplitOnNameGuard().apply(_candidate_set(a, b))
    assert _sources(out.value) == (*drop_if_name(a), *drop_if_name(b))
    # Every part records the ordinal of the candidate it was split out of.
    locations = [
        candidate.origins[-1].input_location
        for candidate in out.value.candidates
    ]
    assert locations == [0] * len(drop_if_name(a)) + [1] * len(drop_if_name(b))


# --- import-step sequence ≡ infer_necessary_imports -----------------


IMPORT_GARBAGE = (
    "",
    "def broken(:\n",
    "import numpy as np  // trailing junk\n\ndef f(x):\n    return np.array(x)\n",
    "from collections import (Counter, defaultdict\n\ndef f():\n    return Counter([1])\n",
    "def f():\n    return math.sqrt(2)\n",
)


@pytest.mark.parametrize("source", IMPORT_GARBAGE)
def test_import_step_sequence_equals_infer_necessary_imports(
    source: str,
) -> None:
    """repair -> infer -> dedupe, in order, reproduces
    infer_necessary_imports, on garbage-input cases."""
    value: CodeCandidateSetArtifact = _candidate_set(source)
    for step_cls in (RepairImportLines, InferMissingImports, DedupeImports):
        value = step_cls().apply(value).value
        assert isinstance(value, CodeCandidateSetArtifact)
    assert _sources(value) == (infer_necessary_imports(source),)


# --- last-return salvage is additive, never destructive --------------


def test_salvage_adds_a_candidate_and_keeps_the_original() -> None:
    src = "def f():\n    return 1\nprint('trailing')"
    out = AddLastReturnSalvage().apply(_candidate_set(src))
    assert _sources(out.value) == (src, "def f():\n    return 1\n")
    assert out.facts["salvaged_count"] == 1


def test_salvage_appends_immediately_after_its_source() -> None:
    a = "def a():\n    return 1\nprose"
    b = "def b():\n    return 2\nprose"
    out = AddLastReturnSalvage().apply(_candidate_set(a, b))
    assert _sources(out.value) == (
        a,
        "def a():\n    return 1\n",
        b,
        "def b():\n    return 2\n",
    )


def test_salvage_keeps_a_bracketed_return_whole() -> None:
    """The salvage of a multi-line return compiles, not cut mid-bracket."""
    src = "def f(x):\n    return (\n        x +\n        1\n    )\nProse.\n"
    out = AddLastReturnSalvage().apply(_candidate_set(src))
    _original, salvage = _sources(out.value)
    assert (
        salvage == "def f(x):\n    return (\n        x +\n        1\n    )\n"
    )
    compile(salvage, "<salvaged>", "exec")


def test_salvage_contributes_nothing_when_truncation_is_a_no_op() -> None:
    src = "def f():\n    return 1"
    out = AddLastReturnSalvage().apply(_candidate_set(src))
    assert _sources(out.value) == (src,)
    assert out.facts["salvaged_count"] == 0


def test_salvage_contributes_nothing_without_a_return_boundary() -> None:
    src = "def f():\n    pass\nProse."
    assert drop_after_last_return(src) is None
    out = AddLastReturnSalvage().apply(_candidate_set(src))
    assert _sources(out.value) == (src,)
    assert out.facts["salvaged_count"] == 0


def test_salvage_extends_the_lineage_of_the_candidate_it_came_from() -> None:
    src = "def f():\n    return 1\nprose"
    out = AddLastReturnSalvage().apply(_candidate_set(src))
    original, salvage = out.value.candidates
    assert _operations(original) == ("text_segments",)
    assert _operations(salvage) == (
        "text_segments",
        "add_last_return_salvage",
    )
    assert salvage.origins[-1].input_location == 0


# --- blank-candidate removal -----------------------------------------


def test_drop_blank_candidates_removes_whitespace_only() -> None:
    out = DropBlankCandidates().apply(_candidate_set("x = 1", "", "  \n\t"))
    assert _sources(out.value) == ("x = 1",)
    assert out.facts["dropped_count"] == 2


# --- deduplication merges origins in encounter order -----------------


def test_dedupe_keeps_the_first_occurrence_and_its_position() -> None:
    out = DedupeCandidates().apply(_candidate_set("a", "b", "a", "c"))
    assert _sources(out.value) == ("a", "b", "c")
    assert out.facts["duplicates_merged"] == 1


def test_dedupe_merges_origins_survivor_first_then_encounter_order() -> None:
    def _candidate(source: str, operation: str) -> CodeCandidate:
        return CodeCandidate(
            source=source,
            origins=(
                CandidateOrigin(
                    operation=ExtractionOperation(operation_name=operation),
                    input_location=0,
                ),
            ),
        )

    value = CodeCandidateSetArtifact(
        candidates=(
            _candidate("same", "raw_response"),
            _candidate("other", "text_segments"),
            _candidate("same", "markdown_segments"),
            _candidate("same", "escaped_python"),
        )
    )
    out = DedupeCandidates().apply(value)
    survivor, other = out.value.candidates
    # The survivor's own origins come first, then each absorbed
    # duplicate's origins in the order the duplicates were encountered.
    assert _operations(survivor) == (
        "raw_response",
        "markdown_segments",
        "escaped_python",
    )
    assert _operations(other) == ("text_segments",)


def test_dedupe_does_not_deduplicate_origins_themselves() -> None:
    # Two duplicates reaching the same source by the same route record
    # that route twice: the lineage is every route taken, not a set.
    out = DedupeCandidates().apply(_candidate_set("a", "a"))
    (survivor,) = out.value.candidates
    assert _operations(survivor) == ("text_segments", "text_segments")


# --- inspection: one parse, structural facts only --------------------


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


# --- filters read the stored inspection ------------------------------


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
    # Filters remove candidates; they never rewrite a survivor's source,
    # so a survivor's record and its inspection are carried through whole.
    inspected = _inspected("def f():\n    return 1", "def broken(:")
    out = FilterCompilable().apply(inspected)
    assert out.value.candidates[0] == inspected.candidates[0]


# --- materialization returns everything that survived ----------------


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


# --- additive extraction across every representation -----------------


def _extract(text: str):
    return ExtractAllRepresentations().apply(TextArtifact(text=text))


def _origin_operations(value: CodeCandidateSetArtifact) -> list[str]:
    return [
        candidate.origins[0].operation.operation_name
        for candidate in value.candidates
    ]


def test_extraction_reads_fenced_and_raw_representations_together() -> None:
    # A fenced response is read both as its whole raw text and as its
    # fenced segment: neither reading shadows the other.
    out = _extract("Intro\n```python\ndef f():\n    return 1\n```")
    operations = _origin_operations(out.value)
    assert Representation.RAW_RESPONSE.value in operations
    assert Representation.TEXT_SEGMENTS.value in operations
    assert "def f():\n    return 1" in _sources(out.value)


def test_extraction_reads_fenced_and_unfenced_code_additively() -> None:
    # Different code in each family: a fenced-or-else-unfenced reading can
    # only surface one of the two, so this pins the additive contract.
    out = _extract(
        "def outside():\n    return 1\n\n"
        "```python\ndef inside():\n    return 2\n```"
    )

    # Restricted to the segment reading, so the whole-text raw_response
    # candidate cannot stand in for the unfenced block.
    segment_sources = [
        candidate.source
        for candidate in out.value.candidates
        if candidate.origins[0].operation.operation_name
        == Representation.TEXT_SEGMENTS.value
    ]
    assert "def inside():\n    return 2" in segment_sources
    assert "def outside():\n    return 1\n" in segment_sources


def test_extraction_reads_unfenced_segments() -> None:
    out = _extract("Explanation first.\ndef f():\n    return 1")
    assert "def f():\n    return 1" in _sources(out.value)
    assert Representation.TEXT_SEGMENTS.value in _origin_operations(out.value)


def test_extraction_reads_markdown_wrapped_segments() -> None:
    out = _extract("> def f():\n>     return 1")
    assert "def f():\n    return 1" in _sources(out.value)
    assert Representation.MARKDOWN_SEGMENTS.value in _origin_operations(
        out.value
    )


def test_extraction_reads_a_whole_response_json_string() -> None:
    out = _extract(json.dumps("def f():\n    return 1"))
    assert "def f():\n    return 1" in _sources(out.value)
    assert Representation.JSON_STRING_RESPONSE.value in _origin_operations(
        out.value
    )


def test_extraction_reads_a_top_level_json_code_field() -> None:
    out = _extract(json.dumps({"code": "def f():\n    return 1"}))
    assert "def f():\n    return 1" in _sources(out.value)
    assert Representation.JSON_CODE_FIELD.value in _origin_operations(
        out.value
    )


@pytest.mark.parametrize("tag", ["json", ""])
def test_extraction_reads_a_json_code_field_inside_a_fence(tag: str) -> None:
    envelope = json.dumps({"code": "def f():\n    return 1"})
    out = _extract(f"Here it is:\n\n```{tag}\n{envelope}\n```\n\nDone.")
    assert "def f():\n    return 1" in _sources(out.value)
    assert Representation.JSON_CODE_FIELD.value in _origin_operations(
        out.value
    )


def test_extraction_reads_a_fenced_json_code_field_once() -> None:
    """A bare envelope inside its own fence is not read twice."""
    envelope = json.dumps({"code": "def f():\n    return 1"})
    out = _extract(f"```json\n{envelope}\n```")
    sources = [
        candidate.source
        for candidate in out.value.candidates
        if candidate.origins[0].operation.operation_name
        == Representation.JSON_CODE_FIELD.value
    ]
    assert len(sources) == len(set(sources))


def test_extraction_ignores_a_malformed_fenced_json_envelope() -> None:
    """Decoding stays strict: a truncated envelope is never repaired."""
    out = _extract('```json\n{"code": "def f():\\n    return 1"\n```')
    assert Representation.JSON_CODE_FIELD.value not in _origin_operations(
        out.value
    )


def test_extraction_ignores_a_fenced_envelope_outside_the_code_field() -> None:
    """A marked response's other fields are not read as its declaration.

    A ``[[ ## prompt ## ]]`` carrying a worked example is context the
    response was given, not an answer it wrote. Reading its envelope would
    put the example ahead of the marked answer under an acceptance policy
    that takes the lowest surviving ordinal.
    """
    envelope = json.dumps({"code": "def reference():\n    return 999"})
    out = _extract(
        f"[[ ## prompt ## ]]\nFor example:\n\n```json\n{envelope}\n```\n\n"
        "[[ ## code ## ]]\ndef f():\n    return 1\n"
    )
    assert "def reference():\n    return 999" not in _sources(out.value)
    assert Representation.JSON_CODE_FIELD.value not in _origin_operations(
        out.value
    )
    assert out.value.candidates[0].source == "def f():\n    return 1"


def test_extraction_reads_a_fenced_envelope_inside_the_code_field() -> None:
    """The marked answer's own fenced envelope is still the declaration."""
    envelope = json.dumps({"code": "def f():\n    return 1"})
    out = _extract(
        f"[[ ## prompt ## ]]\nWhat?\n[[ ## code ## ]]\n```json\n{envelope}\n```"
    )
    assert "def f():\n    return 1" in _sources(out.value)
    assert Representation.JSON_CODE_FIELD.value in _origin_operations(
        out.value
    )


def test_extraction_reads_a_field_marker_value() -> None:
    out = _extract(
        "[[ ## prompt ## ]]\nWhat?\n[[ ## code ## ]]\ndef f():\n    return 1\n"
    )
    assert "def f():\n    return 1" in _sources(out.value)
    assert Representation.FIELD_MARKER.value in _origin_operations(out.value)


def test_extraction_reads_escaped_python() -> None:
    out = _extract(r"Explanation:\ndef f():\n\treturn 1")
    assert Representation.ESCAPED_PYTHON.value in _origin_operations(out.value)


def test_extraction_reads_escaped_markdown() -> None:
    out = _extract(json.dumps("- def add(a, b):\n-     return a + b"))
    assert "def add(a, b):\n    return a + b" in _sources(out.value)
    assert Representation.ESCAPED_MARKDOWN.value in _origin_operations(
        out.value
    )


def test_extraction_contributes_in_declared_representation_order() -> None:
    out = _extract("Intro\n```python\ndef f():\n    return 1\n```")
    order = [Representation(name) for name in _origin_operations(out.value)]
    declared = list(Representation)
    positions = [declared.index(item) for item in order]
    assert positions == sorted(positions)


def test_extraction_records_per_representation_counts_as_facts() -> None:
    out = _extract("```python\ndef f():\n    return 1\n```")
    for representation in Representation:
        assert representation.value in out.facts
    assert out.facts["candidate_count"] == len(out.value.candidates)


def test_extraction_with_no_readable_representation_fails() -> None:
    with pytest.raises(StepFailedError) as excinfo:
        _extract("   ")
    assert (
        excinfo.value.code is PreprocessingFailureCode.NO_CANDIDATES_EXTRACTED
    )
    # The evidence names how much each representation contributed.
    assert set(excinfo.value.evidence) == {
        representation.value for representation in Representation
    }
    assert set(excinfo.value.evidence.values()) == {0}


def test_extraction_of_prose_yields_the_raw_response_only() -> None:
    # Prose is still a reading of the response; it is the filters, not
    # extraction, that decide prose is not code.
    out = _extract("This is an explanation with no code whatsoever.")
    assert _origin_operations(out.value) == [Representation.RAW_RESPONSE.value]
