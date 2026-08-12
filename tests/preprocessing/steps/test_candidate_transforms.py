from __future__ import annotations

import pytest

from dr_code.core.source.text_transforms import (
    drop_after_last_return,
    drop_if_name,
    normalize_text,
    strip_code_fences,
)
from dr_code.preprocessing.import_inference import infer_necessary_imports
from dr_code.preprocessing.steps.add_last_return_salvage import (
    AddLastReturnSalvage,
)
from dr_code.preprocessing.steps.dedent_candidates import DedentCandidates
from dr_code.preprocessing.steps.dedupe_candidates import DedupeCandidates
from dr_code.preprocessing.steps.dedupe_imports import DedupeImports
from dr_code.preprocessing.steps.drop_blank_candidates import (
    DropBlankCandidates,
)
from dr_code.preprocessing.steps.infer_missing_imports import (
    InferMissingImports,
)
from dr_code.preprocessing.steps.normalize_smart_quotes import (
    NormalizeSmartQuotes,
)
from dr_code.preprocessing.steps.normalize_text_preserving_semantics import (
    NormalizeTextPreservingSemantics,
)
from dr_code.preprocessing.steps.repair_import_lines import RepairImportLines
from dr_code.preprocessing.steps.split_on_name_guard import SplitOnNameGuard
from dr_code.preprocessing.steps.strip_fences import StripFences
from dr_code.trace import (
    CandidateOrigin,
    CodeCandidate,
    CodeCandidateSetArtifact,
    ExtractionOperation,
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


def _sources(value: CodeCandidateSetArtifact) -> tuple[str, ...]:
    return tuple(candidate.source for candidate in value.candidates)


def _operations(candidate: CodeCandidate) -> tuple[str, ...]:
    return tuple(
        origin.operation.operation_name for origin in candidate.origins
    )


def test_strip_fences_wraps_function() -> None:
    out = StripFences().apply(_candidate_set("```python\nx = 1\n```"))
    assert _sources(out.value) == (strip_code_fences("```python\nx = 1\n```"),)


def test_dedent_wraps_textwrap() -> None:
    out = DedentCandidates().apply(_candidate_set("    x = 1\n    y = 2\n"))
    assert _sources(out.value) == ("x = 1\ny = 2\n",)


def test_semantic_normalization_preserves_changed_python_contents() -> None:
    source = "def f():\n    return '''a\t \n\n\n\nb  \n'''\n"

    out = NormalizeTextPreservingSemantics().apply(_candidate_set(source))

    assert _sources(out.value) == (source.strip("\n"),)


def test_semantic_normalization_normalizes_equivalent_valid_python() -> None:
    source = "def f():  \n\n\n\n    return 1  \n"

    out = NormalizeTextPreservingSemantics().apply(_candidate_set(source))

    assert _sources(out.value) == (normalize_text(source),)


def test_semantic_normalization_repairs_non_python_text() -> None:
    source = "ｄｅｆ f():\n\treturn 1  \n\n\n"

    out = NormalizeTextPreservingSemantics().apply(_candidate_set(source))

    assert _sources(out.value) == (normalize_text(source),)


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

    locations = [
        candidate.origins[-1].input_location
        for candidate in out.value.candidates
    ]
    assert locations == [0] * len(drop_if_name(a)) + [1] * len(drop_if_name(b))


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
    value: CodeCandidateSetArtifact = _candidate_set(source)
    for step_cls in (RepairImportLines, InferMissingImports, DedupeImports):
        value = step_cls().apply(value).value
        assert isinstance(value, CodeCandidateSetArtifact)
    assert _sources(value) == (infer_necessary_imports(source),)


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


def test_drop_blank_candidates_removes_whitespace_only() -> None:
    out = DropBlankCandidates().apply(_candidate_set("x = 1", "", "  \n\t"))
    assert _sources(out.value) == ("x = 1",)
    assert out.facts["dropped_count"] == 2


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

    assert _operations(survivor) == (
        "raw_response",
        "markdown_segments",
        "escaped_python",
    )
    assert _operations(other) == ("text_segments",)


def test_dedupe_does_not_deduplicate_origins_themselves() -> None:
    out = DedupeCandidates().apply(_candidate_set("a", "a"))
    (survivor,) = out.value.candidates
    assert _operations(survivor) == ("text_segments", "text_segments")
