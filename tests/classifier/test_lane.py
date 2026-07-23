from __future__ import annotations

import pytest

from dr_code.classifier.classify import classify_one_repeat
from dr_code.classifier.lane import (
    LaneTransportError,
    PiLane,
    known_lanes,
    parse_label_response,
)


def test_parses_strict_json() -> None:
    response = parse_label_response('{"label": "prose-no-code", '
                                    '"rationale": "just prose"}')
    assert response.label == "prose-no-code"
    assert response.rationale == "just prose"


def test_parses_json_inside_a_code_fence() -> None:
    raw = '```json\n{"label": "empty-code", "rationale": "empty"}\n```'
    response = parse_label_response(raw)
    assert response.label == "empty-code"


def test_recovers_json_embedded_in_prose() -> None:
    raw = 'Here you go: {"label": "non-python", "rationale": "js"} thanks!'
    response = parse_label_response(raw)
    assert response.label == "non-python"


def test_rejects_missing_keys() -> None:
    with pytest.raises(ValueError):
        parse_label_response('{"label": "prose-no-code"}')


def test_rejects_extra_keys() -> None:
    with pytest.raises(ValueError):
        parse_label_response(
            '{"label": "x", "rationale": "y", "confidence": 0.9}'
        )


def test_rejects_non_json() -> None:
    with pytest.raises(ValueError):
        parse_label_response("I could not classify this.")


class _ScriptedLane:
    """A mock lane returning queued replies; records prompts seen."""

    name = "mock"
    model = "mock-model"

    def __init__(self, replies: list[str | Exception]) -> None:
        self._replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_reparse_retry_recovers_a_malformed_first_reply() -> None:
    lane = _ScriptedLane(
        ["not json at all", '{"label": "prose-no-code", "rationale": "ok"}']
    )
    outcome = classify_one_repeat(lane, "prompt")
    assert outcome.label == "prose-no-code"
    assert len(lane.prompts) == 2
    assert "Reply again" in lane.prompts[1]


def test_malformed_after_reparse_is_a_typed_failure() -> None:
    lane = _ScriptedLane(["nope", "still nope"])
    outcome = classify_one_repeat(lane, "prompt")
    assert outcome.label is None
    assert outcome.failure_reason is not None
    assert "malformed-response" in outcome.failure_reason


def test_transport_error_is_a_typed_failure() -> None:
    lane = _ScriptedLane([LaneTransportError("pi exited 1")])
    outcome = classify_one_repeat(lane, "prompt")
    assert outcome.label is None
    assert outcome.failure_reason is not None
    assert "transport" in outcome.failure_reason


def test_for_lane_maps_known_lanes() -> None:
    for name in known_lanes():
        lane = PiLane.for_lane(name)
        assert lane.name == name
        assert lane.model


def test_for_lane_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        PiLane.for_lane("gpt-lane")
