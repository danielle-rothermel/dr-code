"""Focused modular extraction interpretations and provenance."""

from __future__ import annotations

import json

import pytest

from dr_code.preprocessing.extraction import extract_candidate_drafts


def _sources(text: str) -> tuple[str, ...]:
    return tuple(draft.source for draft in extract_candidate_drafts(text))


def _path_kinds(text: str, source: str) -> tuple[str, ...]:
    draft = next(
        draft
        for draft in extract_candidate_drafts(text)
        if draft.source == source
        and any(
            operation.kind == "fenced_json_code"
            for operation in draft.origin.path
        )
    )
    return tuple(operation.kind for operation in draft.origin.path)


@pytest.mark.parametrize("tag", ("json", "JSON", ""))
def test_fenced_json_code_is_additive_and_strict(tag: str) -> None:
    code = "def recovered():\n    return 1"
    body = json.dumps({"code": code})
    text = f"```{tag}\n{body}\n```"

    sources = _sources(text)

    assert sources[0] == body
    assert code in sources
    assert _path_kinds(text, code) == (
        "response_representation",
        "fenced_block",
        "fenced_json_code",
        "unfenced_segment",
        "anchored_python_block",
    )


@pytest.mark.parametrize(
    "body",
    (
        "{not json}",
        json.dumps({"code": 7}),
        json.dumps({"nested": {"code": "def hidden(): pass"}}),
        json.dumps([{"code": "def hidden(): pass"}]),
    ),
)
def test_fenced_json_does_not_derive_incompatible_shapes(body: str) -> None:
    drafts = extract_candidate_drafts(f"```json\n{body}\n```")

    assert all(
        operation.kind != "fenced_json_code"
        for draft in drafts
        for operation in draft.origin.path
    )


def test_fenced_json_code_rediscovers_one_nested_python_fence() -> None:
    nested = "```python\ndef nested():\n    return 1\n```"
    text = f"```json\n{json.dumps({'code': nested})}\n```"

    assert "def nested():\n    return 1" in _sources(text)


def test_multiple_fenced_json_blocks_preserve_source_order() -> None:
    first = "def first():\n    return 1"
    second = "def second():\n    return 2"
    text = (
        f"```json\n{json.dumps({'code': first})}\n```\n"
        f"```json\n{json.dumps({'code': second})}\n```"
    )

    sources = _sources(text)

    assert sources.index(first) < sources.index(second)


def test_fenced_and_unfenced_segments_are_additive_in_source_order() -> None:
    text = (
        "def first():\n"
        "    return 1\n"
        "```python\n"
        "def second():\n"
        "    return 2\n"
        "```\n"
        "def third():\n"
        "    return 3"
    )

    sources = _sources(text)

    assert sources.index("def first():\n    return 1") < sources.index(
        "def second():\n    return 2"
    )
    assert sources.index("def second():\n    return 2") < sources.index(
        "def third():\n    return 3"
    )


@pytest.mark.parametrize("container", (list, tuple))
def test_singleton_string_container_is_bounded_interpretation(
    container,
) -> None:
    code = "def from_literal():\n    return 1"
    text = repr(container((code,)))

    assert code in _sources(text)


def test_multi_string_container_is_not_interpreted() -> None:
    drafts = extract_candidate_drafts(
        repr(["def first(): pass", "def second(): pass"])
    )

    assert all(
        operation.kind != "singleton_string_container"
        for draft in drafts
        for operation in draft.origin.path
    )


def test_whole_response_json_code_standalone_lambda_is_eligible() -> None:
    source = "lambda value: value + 1"
    text = json.dumps({"code": source})

    drafts = extract_candidate_drafts(text)

    assert source in tuple(draft.source for draft in drafts)
    assert any(
        operation.kind == "standalone_lambda"
        for draft in drafts
        if draft.source == source
        for operation in draft.origin.path
    )


@pytest.mark.parametrize(
    "source",
    (
        "values = list(filter(lambda value: value > 0, items))",
        "result = lambda value: value + 1",
        "prose mentioning lambda value: value + 1",
    ),
)
def test_nested_or_embedded_lambdas_are_not_standalone(source: str) -> None:
    assert all(
        operation.kind != "standalone_lambda"
        for draft in extract_candidate_drafts(source)
        for operation in draft.origin.path
    )


def test_eof_json_completion_supplies_only_the_envelope() -> None:
    text = '{"code": "def completed():\\n    return 1\\n'

    drafts = extract_candidate_drafts(text)

    recovered = next(
        draft for draft in drafts if draft.source.startswith("def completed")
    )
    assert recovered.source == "def completed():\n    return 1\n"
    assert recovered.origin.path[0].details["name"] == (
        "completed_top_level_json_code"
    )


@pytest.mark.parametrize(
    "text",
    (
        '{"other": "unfinished',
        '{"code": 3',
        '{"code": "def f(): pass", "other":',
        '{"code": "def f(): pass", "other": "unfinished',
        '{"other": 1, "code": "def f(): pass',
        '[{"code": "def f(): pass',
        '{"outer": {"code": "def f(): pass',
        '{"code": "def f(): pass\\',
        '{"code": "def f(): pass\\u12',
    ),
)
def test_eof_json_completion_rejects_other_truncation_shapes(
    text: str,
) -> None:
    assert all(
        operation.details.get("name") != "completed_top_level_json_code"
        for draft in extract_candidate_drafts(text)
        for operation in draft.origin.path
    )
