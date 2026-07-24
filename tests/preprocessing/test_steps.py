"""Per-step interface, composition, and determinism tests."""

from __future__ import annotations

import ast
import json
import unicodedata

import pytest

from dr_code.preprocessing.registry import REGISTRY
from dr_code.preprocessing.identification import identify_candidates
from dr_code.preprocessing.steps.base import Step
from dr_code.preprocessing.steps.collapse_blank_runs import (
    CollapseBlankRuns,
)
from dr_code.preprocessing.steps.dedent import Dedent
from dr_code.preprocessing.steps.expand_tabs import ExpandTabs
from dr_code.preprocessing.steps.extract_candidates import (
    ExtractCandidates,
)
from dr_code.preprocessing.steps.filter_code_repr import FilterCodeRepr
from dr_code.preprocessing.steps.filter_compilable import (
    FilterCompilable,
)
from dr_code.preprocessing.steps.filter_has_top_level_function import (
    FilterHasTopLevelFunction,
)
from dr_code.preprocessing.steps.filter_plain_literal import (
    FilterPlainLiteral,
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
from dr_code.preprocessing.steps.return_all import ReturnAll
from dr_code.preprocessing.steps.split_on_name_guard import (
    SplitOnNameGuard,
)
from dr_code.preprocessing.steps.strip_fences import StripFences
from dr_code.preprocessing.steps.strip_trailing_whitespace import (
    StripTrailingWhitespace,
)
from dr_code.preprocessing.steps.trim_outer_blanks import TrimOuterBlanks
from dr_code.text_transforms import (
    collapse_blank_runs,
    drop_if_name,
    normalize_line_endings,
    normalize_text,
    strip_code_fences,
    strip_trailing_whitespace,
)
from dr_code.trace import (
    CandidateLineage,
    CandidateOrigin,
    CodeArtifact,
    CodeCandidateSetArtifact,
    ExtractionOperation,
    IdentifiedCandidateSetArtifact,
    TextArtifact,
)

# Garbage inputs reused from the wrapped modules' existing fixtures.
GARBAGE_TEXT = (
    "",
    "def broken(:\n",
    "```\nunterminated fence",
    "plain prose, no code at all",
    "smart ‘quotes’ and “doubles”\r\nCRLF\ttabs  \n\n\n\n",
)

TEXT_STEPS = (
    NormalizeLineEndings,
    NormalizeUnicode,
    ExpandTabs,
    StripTrailingWhitespace,
    CollapseBlankRuns,
    TrimOuterBlanks,
)


def _candidate_set(*sources: str) -> CodeCandidateSetArtifact:
    return CodeCandidateSetArtifact(
        candidates=sources,
        lineage=tuple(
            CandidateLineage(
                origins=(
                    CandidateOrigin(
                        path=(
                            ExtractionOperation(
                                kind="test_input",
                                details={"index": index},
                            ),
                        )
                    ),
                )
            )
            for index, _source in enumerate(sources)
        ),
    )


# --- determinism battery (no rng) ------------------------------------


def _apply_twice(step_cls: type[Step], value) -> object:
    """Apply twice; return a comparable outcome (StepOutput or the
    StepFailedError marker) so determinism covers both success and the
    Absent path."""
    from dr_code.preprocessing.steps.base import StepFailedError

    step = step_cls(step_cls.Settings())
    try:
        first = step.apply(value)
    except StepFailedError as exc:
        first = ("failed", exc.cause)
    try:
        second = step.apply(value)
    except StepFailedError as exc:
        second = ("failed", exc.cause)
    return (first, second)


@pytest.mark.parametrize("step_cls", REGISTRY.values())
def test_step_is_deterministic(step_cls: type[Step]) -> None:
    """apply twice with identical settings+input => equal (corruption-test
    pattern, minus rng). Covers both success and the Absent path."""
    value = _sample_for(step_cls)
    first, second = _apply_twice(step_cls, value)
    assert first == second


def _sample_for(step_cls: type[Step]):
    """An input artifact of the step's INPUT kind, processable by the step."""
    if step_cls.INPUT.value == "text":
        return TextArtifact(text="```python\ndef f():\n    return 1\n```\n")
    if step_cls.INPUT.value == "code":
        return CodeArtifact(source="def f():\n    return 1\n")
    if step_cls.INPUT.value == "identified_candidate_set":
        return _identified(
            _candidate_set(
                "def f():\n    return 1\n",
                "def g():\n    return 2\n",
            )
        )
    return _candidate_set(
        "def f():\n    return 1\n", "def g():\n    return 2\n"
    )


def _identified(
    value: CodeCandidateSetArtifact,
) -> IdentifiedCandidateSetArtifact:
    identified, _ = identify_candidates(value)
    return identified


# --- atomic text steps wrap their functions --------------------------


def test_normalize_line_endings_wraps_function() -> None:
    raw = "a\r\nb\rc\n"
    out = NormalizeLineEndings().apply(TextArtifact(text=raw))
    assert out.value == TextArtifact(text=normalize_line_endings(raw))


def test_normalize_unicode_applies_nfkc() -> None:
    raw = "ｄｅｆ"
    out = NormalizeUnicode().apply(TextArtifact(text=raw))
    assert out.value == TextArtifact(text=unicodedata.normalize("NFKC", raw))


def test_normalize_smart_quotes_converts_delimiters() -> None:
    cs = _candidate_set("x = “a”\n")
    out = NormalizeSmartQuotes().apply(cs)
    assert out.value == _candidate_set('x = "a"\n')


def test_normalize_smart_quotes_preserves_string_contents() -> None:
    src = 'x = "don’t “quote” me"\n'
    cs = _candidate_set(src)
    out = NormalizeSmartQuotes().apply(cs)
    assert out.value == _candidate_set(src)


def test_normalize_smart_quotes_comment_apostrophe_not_a_delimiter() -> None:
    # The apostrophe in the comment must not open string state; the real
    # literal's smart-quote contents stay preserved.
    src = "# don't\nx = 'a“b'\n"
    cs = _candidate_set(src)
    out = NormalizeSmartQuotes().apply(cs)
    assert out.value == _candidate_set(src)


def test_normalize_smart_quotes_converts_delimiters_after_comment() -> None:
    src = "# don't\ndef f():\n    return “x”"
    expected = '# don\'t\ndef f():\n    return "x"'
    cs = _candidate_set(src)
    out = NormalizeSmartQuotes().apply(cs)
    assert out.value == _candidate_set(expected)


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


# --- atomic text sequence ≡ normalize_text ---------------------------


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


# --- elementwise candidate steps -------------------------------------


def test_strip_fences_wraps_function() -> None:
    cs = _candidate_set("```python\nx = 1\n```")
    out = StripFences().apply(cs)
    assert out.value == _candidate_set(
        strip_code_fences("```python\nx = 1\n```")
    )


def test_dedent_wraps_textwrap() -> None:
    cs = _candidate_set("    x = 1\n    y = 2\n")
    out = Dedent().apply(cs)
    assert out.value == _candidate_set("x = 1\ny = 2\n")


def test_split_on_name_guard_flattens_in_place() -> None:
    src = "def f():\n    return 1\nif __name__ == '__main__':\n    pass"
    cs = _candidate_set(src)
    out = SplitOnNameGuard().apply(cs)
    assert out.value.candidates == tuple(drop_if_name(src))


def test_split_on_name_guard_preserves_order_with_multiple() -> None:
    a = "def a():\n    return 1\nif __name__ == '__main__':\n    pass"
    b = "def b():\n    return 2\n"
    cs = _candidate_set(a, b)
    out = SplitOnNameGuard().apply(cs)
    assert out.value.candidates == (*drop_if_name(a), *drop_if_name(b))


# --- filters record rejection facts ----------------------------------


def test_filter_compilable_keeps_compilable_with_parse_compile_facts() -> None:
    cs = _candidate_set("x = 1\n", "def broken(:\n")
    out = FilterCompilable().apply(_identified(cs))
    assert tuple(item.source for item in out.value.candidates) == ("x = 1\n",)
    assert out.facts["input_candidate_count"] == 2
    assert out.facts["survivor_candidate_count"] == 1
    survivor = out.facts["survivors"][0]
    assert survivor["input_index"] == 0
    assert survivor["parse_ok"] is True
    assert survivor["compile_ok"] is True
    assert survivor["compile_warnings"] == []
    assert survivor["candidate_id"]
    assert out.facts["rejections"][0]["input_index"] == 1
    assert out.facts["rejections"][0]["reason_code"] == "not_compilable"
    assert "SyntaxError" in out.facts["rejections"][0]["compile_error"]


def test_filter_plain_literal_drops_literals() -> None:
    cs = _candidate_set("[1, 2, 3]", "x = 1\n")
    out = FilterPlainLiteral().apply(_identified(cs))
    assert tuple(item.source for item in out.value.candidates) == ("x = 1\n",)
    assert out.facts["input_candidate_count"] == 2
    assert out.facts["survivor_candidate_count"] == 1
    assert out.facts["survivors"][0]["input_index"] == 1
    assert out.facts["rejections"][0]["reason_code"] == (
        "plain_literal_module"
    )


def test_filter_code_repr_drops_repr_assignments() -> None:
    cs = _candidate_set('code = "x = 1"', "x = 1\n")
    out = FilterCodeRepr().apply(_identified(cs))
    assert tuple(item.source for item in out.value.candidates) == ("x = 1\n",)
    assert out.facts["rejections"][0]["reason_code"] == "code_repr_assignment"


@pytest.mark.parametrize(
    ("step", "candidate", "failure_code", "reason_code"),
    (
        (
            FilterPlainLiteral(),
            "[1, 2, 3]",
            "plain_literal_only",
            "plain_literal_module",
        ),
        (
            FilterCodeRepr(),
            'code = "x = 1"',
            "code_repr_only",
            "code_repr_assignment",
        ),
        (
            FilterCompilable(),
            "def broken(:\n",
            "no_compilable_candidate",
            "not_compilable",
        ),
        (
            FilterHasTopLevelFunction(),
            "x = 1\n",
            "no_top_level_function_candidate",
            "no_top_level_function",
        ),
    ),
)
def test_filter_terminal_failures_include_structured_facts(
    step: Step,
    candidate: str,
    failure_code: str,
    reason_code: str,
) -> None:
    from dr_code.preprocessing.steps.base import StepFailedError

    with pytest.raises(StepFailedError) as raised:
        step.apply(_identified(_candidate_set(candidate)))

    error = raised.value
    assert error.failure_code == failure_code
    assert error.facts["input_candidate_count"] == 1
    assert error.facts["survivor_candidate_count"] == 0
    assert error.facts["rejections"][0]["input_index"] == 0
    assert error.facts["rejections"][0]["reason_code"] == reason_code


@pytest.mark.parametrize(
    ("step", "candidates"),
    (
        (FilterPlainLiteral(), ("[1]", "def retained():\n    return 1\n")),
        (
            FilterCodeRepr(),
            ('code = "x = 1"', "def retained():\n    return 1\n"),
        ),
        (
            FilterCompilable(),
            ("def broken(:\n", "def retained():\n    return 1\n"),
        ),
        (
            FilterHasTopLevelFunction(),
            ("x = 1\n", "def retained():\n    return 1\n"),
        ),
    ),
)
def test_filters_preserve_survivor_lineage(
    step: Step, candidates: tuple[str, str]
) -> None:
    initial = _candidate_set(*candidates)
    input_value = CodeCandidateSetArtifact(
        candidates=candidates,
        lineage=(
            initial.lineage[0].model_copy(update={"candidate_id": "rejected"}),
            initial.lineage[1].model_copy(update={"candidate_id": "retained"}),
        ),
    )

    identified = _identified(input_value)
    output = step.apply(identified).value

    assert isinstance(output, IdentifiedCandidateSetArtifact)
    assert tuple(item.source for item in output.candidates) == (candidates[1],)
    assert output.candidates[0].lineage.candidate_id
    assert step.apply(identified).facts["rejections"][0]["candidate_id"]


def test_filter_has_top_level_function_keeps_async_function_with_facts() -> (
    None
):
    candidate = "async def fetch():\n    return 1\n"
    out = FilterHasTopLevelFunction().apply(
        _identified(_candidate_set(candidate))
    )

    assert tuple(item.source for item in out.value.candidates) == (candidate,)
    survivor = out.facts["survivors"][0]
    assert survivor["input_index"] == 0
    assert survivor["parse_ok"] is True
    assert survivor["compile_ok"] is True
    assert survivor["top_level_function_count"] == 1
    assert survivor["top_level_function_names"] == ["fetch"]
    assert survivor["top_level_async_function_names"] == ["fetch"]
    assert survivor["has_async_top_level_function"] is True


def test_filter_has_top_level_function_rejects_nested_function() -> None:
    from dr_code.preprocessing.steps.base import StepFailedError

    nested_only = "if True:\n    def nested():\n        return 1\n"
    with pytest.raises(StepFailedError) as raised:
        FilterHasTopLevelFunction().apply(
            _identified(_candidate_set(nested_only))
        )

    assert raised.value.failure_code == "no_top_level_function_candidate"
    assert raised.value.facts["rejections"][0]["top_level_function_count"] == 0


@pytest.mark.parametrize(
    ("step", "failure_code"),
    (
        (FilterCompilable(), "no_compilable_candidate"),
        (FilterHasTopLevelFunction(), "no_top_level_function_candidate"),
    ),
)
def test_filters_reject_cpython_parser_stack_overflow(
    monkeypatch: pytest.MonkeyPatch, step: Step, failure_code: str
) -> None:
    from dr_code.preprocessing import identification
    from dr_code.preprocessing.steps.base import StepFailedError

    def parser_stack_overflow(source: str) -> ast.Module:
        raise MemoryError("Parser stack overflowed")

    monkeypatch.setattr(identification.ast, "parse", parser_stack_overflow)
    with pytest.raises(StepFailedError) as raised:
        step.apply(_identified(_candidate_set("x = 1\n")))

    assert raised.value.failure_code == failure_code
    rejection = raised.value.facts["rejections"][0]
    assert rejection["reason_code"] == "parser_stack_overflow"
    assert rejection["parse_error"] == "MemoryError: Parser stack overflowed"


@pytest.mark.parametrize("step", (FilterPlainLiteral(), FilterCodeRepr()))
def test_literal_and_repr_filters_preserve_parser_stack_candidates(
    monkeypatch: pytest.MonkeyPatch, step: Step
) -> None:
    from dr_code.preprocessing import identification

    def parser_stack_overflow(source: str) -> ast.Module:
        raise MemoryError("Parser stack overflowed")

    monkeypatch.setattr(identification.ast, "parse", parser_stack_overflow)
    candidate = "x = 1\n"
    output = step.apply(_identified(_candidate_set(candidate)))

    assert tuple(item.source for item in output.value.candidates) == (
        candidate,
    )
    assert output.facts["rejections"] == []


@pytest.mark.parametrize(
    "step",
    (
        FilterPlainLiteral(),
        FilterCodeRepr(),
        FilterCompilable(),
        FilterHasTopLevelFunction(),
    ),
)
def test_filters_reraise_unrelated_memory_error(
    monkeypatch: pytest.MonkeyPatch, step: Step
) -> None:
    from dr_code.preprocessing import identification

    def out_of_memory(source: str) -> ast.Module:
        raise MemoryError("allocation failed")

    monkeypatch.setattr(identification.ast, "parse", out_of_memory)
    with pytest.raises(MemoryError, match="allocation failed"):
        _identified(_candidate_set("x = 1\n"))


# --- cardinality knobs -----------------------------------------------


def test_return_all_passes_through_with_count() -> None:
    cs = _candidate_set("a", "b")
    out = ReturnAll().apply(cs)
    assert out.value == cs
    assert out.facts == {
        "outcome_code": "function_candidates_extracted",
        "candidate_count": 2,
    }


# --- extract_candidates: modular composition -------------------------


def test_extract_candidates_records_ordered_path() -> None:
    fenced = "```python\ndef f():\n    return 1\n```"
    out = ExtractCandidates().apply(TextArtifact(text=fenced))
    assert "def f():\n    return 1" in out.value.candidates
    assert len(out.value.lineage) == len(out.value.candidates)
    assert [
        operation.kind for operation in out.value.lineage[0].origins[0].path
    ] == [
        "response_representation",
        "fenced_block",
        "raw_fenced_block",
    ]


def test_extract_candidates_removes_markdown_wrapper() -> None:
    text = "> def f():\n>     return 1"
    out = ExtractCandidates().apply(TextArtifact(text=text))
    assert out.value.candidates == ("def f():\n    return 1",)
    assert any(
        operation.kind == "markdown_wrapper_removal"
        for operation in out.value.lineage[0].origins[0].path
    )


def test_extract_candidates_recovers_escaped_python() -> None:
    text = r"prose\ndef f():\n    return 1"
    out = ExtractCandidates().apply(TextArtifact(text=text))
    assert any("def f():" in candidate for candidate in out.value.candidates)
    assert any(
        operation.kind == "escaped_python_recovery"
        for lineage in out.value.lineage
        for origin in lineage.origins
        for operation in origin.path
    )


def test_extract_candidates_combines_response_and_wrapper_recovery() -> None:
    text = json.dumps("- def add(a, b):\n-     return a + b")
    out = ExtractCandidates().apply(TextArtifact(text=text))
    assert "def add(a, b):\n    return a + b" in out.value.candidates


def test_extract_candidates_all_fail_raises() -> None:
    from dr_code.preprocessing.steps.base import StepFailedError

    out_err = None
    try:
        ExtractCandidates().apply(TextArtifact(text="just prose, no code"))
    except StepFailedError:
        out_err = "raised"
    assert out_err == "raised"


def test_extract_candidates_empty_input_raises() -> None:
    from dr_code.preprocessing.steps.base import StepFailedError

    with pytest.raises(StepFailedError):
        ExtractCandidates().apply(TextArtifact(text=""))
