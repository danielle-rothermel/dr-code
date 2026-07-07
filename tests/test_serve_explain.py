from dr_code.humaneval.code_parsing import (
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
    STRICT_FIELD_MARKER_PARSER_PROFILE_ID,
    extract_code_with_profile,
    resolve_parser_profile,
)
from dr_code.serve.explain import (
    CandidateStatus,
    ExplainStage,
    explain_extraction,
)

import pytest

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


def test_explain_fenced_text_selects_winner_with_rationale() -> None:
    explanation = explain_extraction(FENCED_TEXT)

    assert explanation.result is not None
    assert explanation.result.succeeded
    assert explanation.candidates is not None
    selected = [
        candidate
        for candidate in explanation.candidates
        if candidate.status is CandidateStatus.SELECTED
    ]
    assert len(selected) == 1
    assert selected[0].index == explanation.result.selected_candidate_index
    assert explanation.selection is not None
    assert "selected via" in explanation.selection.rationale


def test_explain_matches_canonical_extraction_result() -> None:
    profile = resolve_parser_profile(
        parser_profile_id=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
        parser_version="v1",
    )
    canonical = extract_code_with_profile(FENCED_TEXT, profile=profile)
    explanation = explain_extraction(FENCED_TEXT)

    assert explanation.result == canonical


def test_explain_records_rejection_reasons_before_winner() -> None:
    explanation = explain_extraction(BROKEN_THEN_GOOD)

    assert explanation.result is not None
    assert explanation.result.succeeded
    assert explanation.candidates is not None
    rejected = [
        candidate
        for candidate in explanation.candidates
        if candidate.status is CandidateStatus.REJECTED
    ]
    assert rejected, "expected at least one rejected candidate"
    assert all(
        candidate.rejection_reason for candidate in rejected
    )


def test_explain_stage_filter_omits_unrequested_sections() -> None:
    explanation = explain_extraction(
        FENCED_TEXT,
        stages=frozenset({ExplainStage.RESULT}),
    )

    assert explanation.result is not None
    assert explanation.unwrap is None
    assert explanation.candidates is None
    assert explanation.selection is None


def test_explain_strict_field_marker_profile() -> None:
    explanation = explain_extraction(
        FIELD_MARKER_TEXT,
        profile_id=STRICT_FIELD_MARKER_PARSER_PROFILE_ID,
    )

    assert explanation.result is not None
    assert explanation.result.succeeded
    assert explanation.unwrap is not None
    assert explanation.unwrap.method == "field_marker"
    assert explanation.candidates is not None
    assert len(explanation.candidates) == 1
    assert explanation.candidates[0].status is CandidateStatus.SELECTED


def test_explain_strict_profile_missing_marker_fails_with_reason() -> None:
    explanation = explain_extraction(
        "no marker here",
        profile_id=STRICT_FIELD_MARKER_PARSER_PROFILE_ID,
    )

    assert explanation.result is not None
    assert not explanation.result.succeeded
    assert explanation.selection is not None
    assert "missing field marker" in explanation.selection.rationale


def test_explain_unknown_profile_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unsupported parser profile id"):
        explain_extraction(FENCED_TEXT, profile_id="nope")
