"""Per-step interface, composition, and determinism tests."""

from __future__ import annotations

import json
import unicodedata

import pytest

from dr_code.code_analysis import validate_python_source
from dr_code.humaneval.import_inference import infer_necessary_imports
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.registry import REGISTRY
from dr_code.preprocessing.steps.base import Step, StepOutput
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
from dr_code.preprocessing.steps.field_marker import (
    FieldMarker,
    FieldMarkerSettings,
)
from dr_code.preprocessing.steps.filter_code_repr import FilterCodeRepr
from dr_code.preprocessing.steps.filter_compilable import (
    FilterCompilable,
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


# --- filters record rejection facts ----------------------------------


def test_filter_compilable_keeps_compilable() -> None:
    cs = CodeCandidateSetArtifact(
        candidates=("x = 1\n", "def broken(:\n")
    )
    out = FilterCompilable().apply(cs)
    assert out.value.candidates == ("x = 1\n",)
    assert "rejected_1" in out.facts
    assert "SyntaxError" in out.facts["rejected_1"]


def test_filter_plain_literal_drops_literals() -> None:
    cs = CodeCandidateSetArtifact(
        candidates=("[1, 2, 3]", "x = 1\n")
    )
    out = FilterPlainLiteral().apply(cs)
    assert out.value.candidates == ("x = 1\n",)
    assert out.facts["rejected_0"] == "plain literal module"


def test_filter_code_repr_drops_repr_assignments() -> None:
    cs = CodeCandidateSetArtifact(
        candidates=('code = "x = 1"', "x = 1\n")
    )
    out = FilterCodeRepr().apply(cs)
    assert out.value.candidates == ("x = 1\n",)
    assert out.facts["rejected_0"] == "code repr assignment"


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
    assert out.facts["candidate_count"] == "2"


# --- extract_candidates: strategy ladder -----------------------------


def test_extract_candidates_records_chosen_alternative() -> None:
    fenced = "```python\ndef f():\n    return 1\n```"
    out = ExtractCandidates().apply(TextArtifact(text=fenced))
    assert out.facts["alternative"] == "fenced_blocks"
    assert out.value.candidates == ("def f():\n    return 1",)


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
    assert "def f():" in out.value.candidates[0]


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
    assert out.value.candidates == ("def add(a, b):\n    return a + b",)


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


# --- field_marker step -----------------------------------------------


def test_field_marker_extracts_code_field() -> None:
    text = "[[ ## prompt ## ]]\nWhat?\n[[ ## code ## ]]\ndef f():\n    return 1\n"
    out = FieldMarker().apply(TextArtifact(text=text))
    assert out.value.candidates == ("def f():\n    return 1",)
    assert out.facts["field_name"] == "code"


def test_field_marker_missing_raises() -> None:
    from dr_code.preprocessing.steps.base import StepFailedError

    with pytest.raises(StepFailedError):
        FieldMarker().apply(TextArtifact(text="no markers here"))


def test_field_marker_custom_field_name() -> None:
    text = "[[ ## solution ## ]]\nx = 1\n"
    out = FieldMarker(
        FieldMarkerSettings(field_name="solution")
    ).apply(TextArtifact(text=text))
    assert out.value.candidates == ("x = 1",)
