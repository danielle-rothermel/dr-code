"""Per-step interface, composition, and determinism tests."""

from __future__ import annotations

import ast
import json
import unicodedata

import pytest

from dr_code.preprocessing.import_inference import infer_necessary_imports
from dr_code.preprocessing.registry import REGISTRY
from dr_code.preprocessing.steps.base import Step
from dr_code.preprocessing.steps.collapse_blank_runs import (
    CollapseBlankRuns,
)
from dr_code.preprocessing.steps.dedupe_imports import DedupeImports
from dr_code.preprocessing.steps.dedent import Dedent
from dr_code.preprocessing.steps.drop_after_last_return import (
    DropAfterLastReturn,
)
from dr_code.preprocessing.steps.expand_tabs import ExpandTabs
from dr_code.preprocessing.steps.extract_candidates import (
    DEFAULT_STRATEGIES,
    ExtractCandidates,
    ExtractCandidatesSettings,
    ExtractionStrategy,
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
from dr_code.preprocessing.steps.infer_missing_imports import (
    InferMissingImports,
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
from dr_code.preprocessing.steps.repair_import_lines import (
    RepairImportLines,
)
from dr_code.preprocessing.steps.return_all import ReturnAll
from dr_code.preprocessing.steps.select_first import SelectFirst
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
    drop_after_last_return,
    drop_if_name,
    normalize_line_endings,
    normalize_text,
    strip_code_fences,
    strip_trailing_whitespace,
)
from dr_code.trace import (
    CandidateLineage,
    CodeArtifact,
    CodeCandidateSetArtifact,
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
        return TextArtifact(
            text="```python\ndef f():\n    return 1\n```\n"
        )
    if step_cls.INPUT.value == "code":
        return CodeArtifact(source="def f():\n    return 1\n")
    return CodeCandidateSetArtifact(
        candidates=("def f():\n    return 1\n", "def g():\n    return 2\n")
    )


# --- atomic text steps wrap their functions --------------------------


def test_normalize_line_endings_wraps_function() -> None:
    raw = "a\r\nb\rc\n"
    out = NormalizeLineEndings().apply(TextArtifact(text=raw))
    assert out.value == TextArtifact(text=normalize_line_endings(raw))


def test_normalize_unicode_applies_nfkc() -> None:
    raw = "ｄｅｆ"
    out = NormalizeUnicode().apply(TextArtifact(text=raw))
    assert out.value == TextArtifact(
        text=unicodedata.normalize("NFKC", raw)
    )


def test_normalize_smart_quotes_converts_delimiters() -> None:
    cs = CodeCandidateSetArtifact(candidates=("x = “a”\n",))
    out = NormalizeSmartQuotes().apply(cs)
    assert out.value == CodeCandidateSetArtifact(candidates=('x = "a"\n',))


def test_normalize_smart_quotes_preserves_string_contents() -> None:
    src = 'x = "don’t “quote” me"\n'
    cs = CodeCandidateSetArtifact(candidates=(src,))
    out = NormalizeSmartQuotes().apply(cs)
    assert out.value == CodeCandidateSetArtifact(candidates=(src,))


def test_normalize_smart_quotes_comment_apostrophe_not_a_delimiter() -> None:
    # The apostrophe in the comment must not open string state; the real
    # literal's smart-quote contents stay preserved.
    src = "# don't\nx = 'a“b'\n"
    cs = CodeCandidateSetArtifact(candidates=(src,))
    out = NormalizeSmartQuotes().apply(cs)
    assert out.value == CodeCandidateSetArtifact(candidates=(src,))


def test_normalize_smart_quotes_converts_delimiters_after_comment() -> None:
    src = "# don't\ndef f():\n    return “x”"
    expected = "# don't\ndef f():\n    return \"x\""
    cs = CodeCandidateSetArtifact(candidates=(src,))
    out = NormalizeSmartQuotes().apply(cs)
    assert out.value == CodeCandidateSetArtifact(candidates=(expected,))


def test_expand_tabs_uses_tab_width_setting() -> None:
    raw = "a\tb"
    out = ExpandTabs(ExpandTabs.Settings(tab_width=2)).apply(
        TextArtifact(text=raw)
    )
    assert out.value == TextArtifact(text="a b")


def test_strip_trailing_whitespace_wraps_function() -> None:
    raw = "x = 1  \ny = 2\t\n"
    out = StripTrailingWhitespace().apply(TextArtifact(text=raw))
    assert out.value == TextArtifact(
        text=strip_trailing_whitespace(raw)
    )


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
    cs = CodeCandidateSetArtifact(candidates=("```python\nx = 1\n```",))
    out = StripFences().apply(cs)
    assert out.value == CodeCandidateSetArtifact(
        candidates=(strip_code_fences("```python\nx = 1\n```"),)
    )


def test_dedent_wraps_textwrap() -> None:
    cs = CodeCandidateSetArtifact(candidates=("    x = 1\n    y = 2\n",))
    out = Dedent().apply(cs)
    assert out.value == CodeCandidateSetArtifact(
        candidates=("x = 1\ny = 2\n",)
    )


def test_split_on_name_guard_flattens_in_place() -> None:
    src = "def f():\n    return 1\nif __name__ == '__main__':\n    pass"
    cs = CodeCandidateSetArtifact(candidates=(src,))
    out = SplitOnNameGuard().apply(cs)
    assert out.value.candidates == tuple(drop_if_name(src))


def test_split_on_name_guard_preserves_order_with_multiple() -> None:
    a = "def a():\n    return 1\nif __name__ == '__main__':\n    pass"
    b = "def b():\n    return 2\n"
    cs = CodeCandidateSetArtifact(candidates=(a, b))
    out = SplitOnNameGuard().apply(cs)
    assert out.value.candidates == (*drop_if_name(a), *drop_if_name(b))


def test_drop_after_last_return_wraps_function() -> None:
    src = "def f():\n    return 1\nprint('x')"
    cs = CodeCandidateSetArtifact(candidates=(src,))
    out = DropAfterLastReturn().apply(cs)
    assert out.value == CodeCandidateSetArtifact(
        candidates=(drop_after_last_return(src),)
    )


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
    value: CodeCandidateSetArtifact = CodeCandidateSetArtifact(
        candidates=(source,)
    )
    for step_cls in (RepairImportLines, InferMissingImports, DedupeImports):
        value = step_cls().apply(value).value
        assert isinstance(value, CodeCandidateSetArtifact)
    assert value.candidates == (infer_necessary_imports(source),)


def test_import_inference_passes_parser_stack_candidate_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dr_code.preprocessing import import_inference

    def parser_stack_overflow(source: str) -> ast.Module:
        raise MemoryError("Parser stack overflowed")

    monkeypatch.setattr(import_inference.ast, "parse", parser_stack_overflow)
    source = "x = 1\n"
    output = InferMissingImports().apply(
        CodeCandidateSetArtifact(candidates=(source,))
    )

    assert output.value.candidates == (source,)


def test_import_inference_reraises_unrelated_memory_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dr_code.preprocessing import import_inference

    def out_of_memory(source: str) -> ast.Module:
        raise MemoryError("allocation failed")

    monkeypatch.setattr(import_inference.ast, "parse", out_of_memory)
    with pytest.raises(MemoryError, match="allocation failed"):
        InferMissingImports().apply(
            CodeCandidateSetArtifact(candidates=("x = 1\n",))
        )


# --- filters record rejection facts ----------------------------------


def test_filter_compilable_keeps_compilable_with_parse_compile_facts() -> None:
    cs = CodeCandidateSetArtifact(
        candidates=("x = 1\n", "def broken(:\n")
    )
    out = FilterCompilable().apply(cs)
    assert out.value.candidates == ("x = 1\n",)
    assert out.facts["input_candidate_count"] == 2
    assert out.facts["survivor_candidate_count"] == 1
    assert out.facts["survivors"] == [
        {
            "input_index": 0,
            "parse_ok": True,
            "parse_error": None,
            "compile_ok": True,
            "compile_error": None,
        }
    ]
    assert out.facts["rejections"][0]["input_index"] == 1
    assert out.facts["rejections"][0]["reason_code"] == "not_compilable"
    assert "SyntaxError" in out.facts["rejections"][0]["compile_error"]


def test_filter_plain_literal_drops_literals() -> None:
    cs = CodeCandidateSetArtifact(
        candidates=("[1, 2, 3]", "x = 1\n")
    )
    out = FilterPlainLiteral().apply(cs)
    assert out.value.candidates == ("x = 1\n",)
    assert out.facts == {
        "input_candidate_count": 2,
        "survivor_candidate_count": 1,
        "survivors": [{"input_index": 1}],
        "rejections": [
            {
                "input_index": 0,
                "reason_code": "plain_literal_module",
                "parse_ok": True,
                "parse_error": None,
                "compile_ok": True,
                "compile_error": None,
            }
        ],
    }


def test_filter_code_repr_drops_repr_assignments() -> None:
    cs = CodeCandidateSetArtifact(
        candidates=('code = "x = 1"', "x = 1\n")
    )
    out = FilterCodeRepr().apply(cs)
    assert out.value.candidates == ("x = 1\n",)
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
        step.apply(CodeCandidateSetArtifact(candidates=(candidate,)))

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
        (FilterCodeRepr(), ('code = "x = 1"', "def retained():\n    return 1\n")),
        (FilterCompilable(), ("def broken(:\n", "def retained():\n    return 1\n")),
        (FilterHasTopLevelFunction(), ("x = 1\n", "def retained():\n    return 1\n")),
    ),
)
def test_filters_preserve_survivor_lineage(
    step: Step, candidates: tuple[str, str]
) -> None:
    input_value = CodeCandidateSetArtifact(
        candidates=candidates,
        lineage=(
            CandidateLineage(candidate_id="rejected"),
            CandidateLineage(candidate_id="retained"),
        ),
    )

    output = step.apply(input_value).value

    assert isinstance(output, CodeCandidateSetArtifact)
    assert output.candidates == (candidates[1],)
    assert output.lineage_at(0).candidate_id == "retained"
    assert step.apply(input_value).facts["rejections"][0][
        "candidate_id"
    ] == "rejected"


def test_filter_has_top_level_function_keeps_async_function_with_facts() -> None:
    candidate = "async def fetch():\n    return 1\n"
    out = FilterHasTopLevelFunction().apply(
        CodeCandidateSetArtifact(candidates=(candidate,))
    )

    assert out.value.candidates == (candidate,)
    assert out.facts["survivors"] == [
        {
            "input_index": 0,
            "parse_ok": True,
            "parse_error": None,
            "compile_ok": True,
            "compile_error": None,
            "top_level_function_count": 1,
            "top_level_function_names": ["fetch"],
            "top_level_async_function_names": ["fetch"],
            "has_async_top_level_function": True,
        }
    ]


def test_filter_has_top_level_function_rejects_nested_function() -> None:
    from dr_code.preprocessing.steps.base import StepFailedError

    nested_only = "if True:\n    def nested():\n        return 1\n"
    with pytest.raises(StepFailedError) as raised:
        FilterHasTopLevelFunction().apply(
            CodeCandidateSetArtifact(candidates=(nested_only,))
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
    from dr_code import code_analysis
    from dr_code.preprocessing.steps.base import StepFailedError

    def parser_stack_overflow(source: str) -> ast.Module:
        raise MemoryError("Parser stack overflowed")

    monkeypatch.setattr(code_analysis.ast, "parse", parser_stack_overflow)
    with pytest.raises(StepFailedError) as raised:
        step.apply(CodeCandidateSetArtifact(candidates=("x = 1\n",)))

    assert raised.value.failure_code == failure_code
    rejection = raised.value.facts["rejections"][0]
    assert rejection["reason_code"] == "parser_stack_overflow"
    assert rejection["parse_error"] == "MemoryError: Parser stack overflowed"


@pytest.mark.parametrize("step", (FilterPlainLiteral(), FilterCodeRepr()))
def test_literal_and_repr_filters_preserve_parser_stack_candidates(
    monkeypatch: pytest.MonkeyPatch, step: Step
) -> None:
    from dr_code import code_analysis

    def parser_stack_overflow(source: str) -> ast.Module:
        raise MemoryError("Parser stack overflowed")

    monkeypatch.setattr(code_analysis.ast, "parse", parser_stack_overflow)
    candidate = "x = 1\n"
    output = step.apply(CodeCandidateSetArtifact(candidates=(candidate,)))

    assert output.value.candidates == (candidate,)
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
    from dr_code import code_analysis

    def out_of_memory(source: str) -> ast.Module:
        raise MemoryError("allocation failed")

    monkeypatch.setattr(code_analysis.ast, "parse", out_of_memory)
    with pytest.raises(MemoryError, match="allocation failed"):
        step.apply(CodeCandidateSetArtifact(candidates=("x = 1\n",)))


# --- cardinality knobs -----------------------------------------------


def test_select_first_picks_first_candidate() -> None:
    cs = CodeCandidateSetArtifact(candidates=("a", "b"))
    out = SelectFirst().apply(cs)
    assert out.value == CodeArtifact(source="a")


def test_select_first_empty_set_raises() -> None:
    from dr_code.preprocessing.steps.base import StepFailedError

    cs = CodeCandidateSetArtifact(candidates=())
    with pytest.raises(StepFailedError):
        SelectFirst().apply(cs)


def test_return_all_passes_through_with_count() -> None:
    cs = CodeCandidateSetArtifact(candidates=("a", "b"))
    out = ReturnAll().apply(cs)
    assert out.value == cs
    assert out.facts == {
        "outcome_code": "function_candidates_extracted",
        "candidate_count": 2,
    }


# --- extract_candidates: strategy ladder -----------------------------


def test_extract_candidates_records_chosen_alternative() -> None:
    fenced = "```python\ndef f():\n    return 1\n```"
    out = ExtractCandidates().apply(TextArtifact(text=fenced))
    assert out.facts["alternative"] == "fenced_blocks"
    assert "def f():\n    return 1" in out.value.candidates
    assert len(out.value.lineage) == len(out.value.candidates)


def test_extract_candidates_markdown_strategy_when_no_fence() -> None:
    # Prose with a blockquote-wrapped def: fenced strategy fails to find
    # a *code* candidate, markdown strategy strips the marker.
    text = "> def f():\n>     return 1"
    out = ExtractCandidates().apply(TextArtifact(text=text))
    assert out.facts["alternative"] == "markdown_wrapper"
    assert out.value.candidates == ("def f():\n    return 1",)


def test_extract_candidates_escaped_python_strategy() -> None:
    text = r"prose\ndef f():\n    return 1"
    out = ExtractCandidates().apply(TextArtifact(text=text))
    assert out.facts["alternative"] == "escaped_python"
    assert any("def f():" in candidate for candidate in out.value.candidates)


def test_extract_candidates_tuple_subset_setting() -> None:
    # A definition using only fenced_blocks + markdown_wrapper.
    settings = ExtractCandidatesSettings(
        alternatives=(
            ExtractionStrategy.FENCED_BLOCKS,
            ExtractionStrategy.MARKDOWN_WRAPPER,
        )
    )
    assert settings.alternatives == (
        ExtractionStrategy.FENCED_BLOCKS,
        ExtractionStrategy.MARKDOWN_WRAPPER,
    )
    # Prose-only text: fenced fails, markdown keeps the block.
    text = "> def f():\n>     return 1"
    out = ExtractCandidates(settings).apply(TextArtifact(text=text))
    assert out.facts["alternative"] == "markdown_wrapper"


def test_extract_candidates_default_strategies_order() -> None:
    assert DEFAULT_STRATEGIES == (
        ExtractionStrategy.FENCED_BLOCKS,
        ExtractionStrategy.MARKDOWN_WRAPPER,
        ExtractionStrategy.ESCAPED_PYTHON,
        ExtractionStrategy.ESCAPED_MARKDOWN_WRAPPER,
    )


def test_extract_candidates_escaped_markdown_wrapper_strategy() -> None:
    # JSON-wrapped, markdown-list-wrapped code: only the unescape + wrapper
    # rung recovers it.
    text = json.dumps("- def add(a, b):\n-     return a + b")
    out = ExtractCandidates().apply(TextArtifact(text=text))
    assert out.facts["alternative"] == "escaped_markdown_wrapper"
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
