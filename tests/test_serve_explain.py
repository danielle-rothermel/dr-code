import pytest

from dr_code.humaneval.code_extraction import ExtractionTraceNode
from dr_code.humaneval.code_parsing import (
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
    PARSER_PROFILE_VERSION,
    STRICT_FIELD_MARKER_PARSER_PROFILE_ID,
    CandidateStatus,
    extract_code_with_profile,
    resolve_parser_profile,
)
from dr_code.serve.explain import explain_extraction
from dr_code.text_transforms import strip_code_fences

FENCED_TEXT = """Here is the solution:

```python
def add(a, b):
    return a + b
```

Hope that helps!
"""

FIELD_MARKER_TEXT = (
    "[[ ## code ## ]]\ndef add(a, b):\n    return a + b\n"
)

BROKEN_THEN_GOOD = """```python
def broken(:
```

```python
def works():
    return 1
```
"""


def node_names(nodes: list[ExtractionTraceNode]) -> set[str]:
    names: set[str] = set()
    for node in nodes:
        names.add(node.name)
        names.update(node_names(node.children))
    return names


def first_node(
    nodes: list[ExtractionTraceNode],
    *,
    name: str,
) -> ExtractionTraceNode:
    for node in nodes:
        if node.name == name:
            return node
        try:
            return first_node(node.children, name=name)
        except LookupError:
            pass
    raise LookupError(name)


def test_explain_returns_parser_emitted_trace() -> None:
    profile = resolve_parser_profile(
        parser_profile_id=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
        parser_version=PARSER_PROFILE_VERSION,
    )
    canonical = extract_code_with_profile(FENCED_TEXT, profile=profile)
    explanation = explain_extraction(FENCED_TEXT)

    assert explanation == canonical.trace


def test_trace_records_lineage_transforms_and_selection() -> None:
    trace = explain_extraction(FENCED_TEXT)

    assert trace.selected_candidate_index == 0
    assert trace.candidates[0].status is CandidateStatus.SELECTED
    assert "selected via" in trace.rationale
    names = node_names(trace.roots)
    assert {
        "normalize_text",
        "fence_split",
        "initial_pass",
        "function_pattern_fanout",
        "strip_code_fences",
        "dedent",
        "drop_after_last_return",
        "infer_necessary_imports",
    }.issubset(names)


def test_trace_records_pure_transform_output() -> None:
    text = "```python\ndef add(a, b):\n    return a + b\n```\n"
    trace = explain_extraction(text)
    strip_node = first_node(trace.roots, name="strip_code_fences")

    assert strip_node.before_text is not None
    assert strip_node.after_text == strip_code_fences(strip_node.before_text)


def test_trace_records_rejection_reasons_before_winner() -> None:
    trace = explain_extraction(BROKEN_THEN_GOOD)

    assert trace.selected_candidate_index == 1
    rejected = [
        candidate
        for candidate in trace.candidates
        if candidate.status is CandidateStatus.REJECTED
    ]
    assert rejected, "expected at least one rejected candidate"
    assert all(candidate.rejection_reason for candidate in rejected)


def test_trace_records_strict_field_marker_profile() -> None:
    trace = explain_extraction(
        FIELD_MARKER_TEXT,
        profile_id=STRICT_FIELD_MARKER_PARSER_PROFILE_ID,
    )

    assert trace.selected_candidate_index == 0
    assert trace.extraction_method == "field_marker"
    assert len(trace.candidates) == 1
    assert trace.candidates[0].status is CandidateStatus.SELECTED
    assert {"field_marker_present", "field_marker_extract"}.issubset(
        node_names(trace.roots)
    )


def test_trace_records_strict_profile_missing_marker_failure() -> None:
    trace = explain_extraction(
        "no marker here",
        profile_id=STRICT_FIELD_MARKER_PARSER_PROFILE_ID,
    )

    assert trace.extraction_error == "missing field marker for 'code'"
    assert "missing field marker" in trace.rationale
    assert trace.candidates == []


def test_explain_unknown_profile_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unsupported parser profile id"):
        explain_extraction(FENCED_TEXT, profile_id="nope")
