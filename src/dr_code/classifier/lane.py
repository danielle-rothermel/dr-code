"""Pluggable subscription LLM lanes and strict-JSON response parsing.

A *lane* turns a text prompt into a raw string reply. The pilot lane shells out
to ``pi`` headless against the ``glm-coding`` subscription provider; alternate
subscription providers (kimi/minimax/stepfun coding) are selectable by name so
the lane is pluggable via a ``--lane`` flag. No OpenRouter / OpenAI API is used.

The lane's raw reply is validated into a :class:`LabelResponse` (a Pydantic
model, used only at this subprocess/LLM JSON boundary). A malformed reply after
one reparse attempt becomes a typed :class:`LaneFailure`, never a fabricated
label.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError


DEFAULT_LANE: Final = "glm-coding"
DEFAULT_MODEL: Final = "glm-4.6"

# Subscription coding providers exposed by pi. Values are (provider, default
# model). The pilot uses glm-coding only; the rest are selectable alternates.
_LANE_PROVIDERS: Final[dict[str, tuple[str, str]]] = {
    "glm-coding": ("glm-coding", DEFAULT_MODEL),
    "kimi-coding": ("kimi-coding", "kimi-for-coding"),
    "minimax-coding": ("minimax-coding", "minimax-m2"),
    "stepfun-coding": ("stepfun-coding", "step-3"),
}


def known_lanes() -> tuple[str, ...]:
    """Return the selectable lane names."""
    return tuple(_LANE_PROVIDERS)


class LabelResponse(BaseModel):
    """Strict-JSON classifier reply parsed at the LLM boundary."""

    model_config = ConfigDict(extra="forbid")

    label: str
    rationale: str


@dataclass(frozen=True, slots=True)
class LaneFailure:
    """A typed lane failure for one attempt: no label was produced."""

    reason: str
    detail: str


class Lane(Protocol):
    """A subscription LLM lane: prompt in, raw string reply out."""

    name: str
    model: str

    def complete(self, prompt: str) -> str:
        """Return the raw model reply, or raise on transport failure."""
        ...


@dataclass(frozen=True, slots=True)
class PiLane:
    """A ``pi`` headless subscription lane.

    Keys resolve from the environment the process is launched with (callers run
    under ``eval "$(mise env)"`` so ``ZAI_API_KEY`` and friends are present).
    """

    name: str = DEFAULT_LANE
    model: str = DEFAULT_MODEL
    timeout_seconds: float = 120.0

    @classmethod
    def for_lane(cls, lane: str, *, model: str | None = None) -> PiLane:
        if lane not in _LANE_PROVIDERS:
            raise ValueError(
                f"unknown lane {lane!r}; choose one of "
                + ", ".join(known_lanes())
            )
        provider, default_model = _LANE_PROVIDERS[lane]
        return cls(name=provider, model=model or default_model)

    def complete(self, prompt: str) -> str:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [
                "pi",
                "-p",
                "--provider",
                self.name,
                "--model",
                self.model,
                "--no-tools",
                "--no-session",
                "--mode",
                "text",
                "--thinking",
                "off",
                prompt,
            ],
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
        if completed.returncode != 0:
            raise LaneTransportError(
                f"pi exited {completed.returncode}: "
                f"{completed.stderr.strip()[:500]}"
            )
        return completed.stdout


class LaneTransportError(RuntimeError):
    """The lane subprocess failed to produce any reply."""


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_label_response(raw: str) -> LabelResponse:
    """Parse a strict-JSON label reply, tolerating surrounding text.

    Raises :class:`ValueError` when no valid object can be recovered.
    """
    text = _strip_code_fence(raw.strip())
    candidates = [text]
    match = _JSON_OBJECT.search(text)
    if match is not None and match.group(0) != text:
        candidates.append(match.group(0))
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        try:
            return LabelResponse.model_validate(payload)
        except ValidationError as exc:
            last_error = exc
            continue
    raise ValueError(f"no valid label JSON in reply: {last_error}")


def _strip_code_fence(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


REPARSE_INSTRUCTION: Final = (
    "Your previous reply was not valid JSON matching "
    '{"label": "<one label>", "rationale": "<one line>"}. '
    "Reply again with ONLY that JSON object and nothing else."
)


__all__ = (
    "DEFAULT_LANE",
    "DEFAULT_MODEL",
    "REPARSE_INSTRUCTION",
    "LabelResponse",
    "Lane",
    "LaneFailure",
    "LaneTransportError",
    "PiLane",
    "known_lanes",
    "parse_label_response",
)
