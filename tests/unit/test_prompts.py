"""Unit tests for decoder prompt helpers."""

from __future__ import annotations

from dr_code.datasets.humaneval_loader import get_task
from dr_code.generation.prompts import (
    build_decoder_prompt,
    decoder_input_from_task,
)


def test_build_decoder_prompt_contains_docstring() -> None:
    task = get_task("HumanEval/0", prefer_snapshot=True)
    prompt = build_decoder_prompt(task)
    assert "Write functional code in Python" in prompt
    assert task.prompt in prompt


def test_decoder_input_from_task_matches_prompt() -> None:
    task = get_task("HumanEval/0", prefer_snapshot=True)
    assert decoder_input_from_task(task) == task.prompt
