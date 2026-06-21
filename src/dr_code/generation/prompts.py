"""Decoder prompt templates for stage 1b generation."""

from __future__ import annotations

from dr_code.models.humaneval import HumanEvalPlusTask

DECODER_TEMPLATE = '''Write functional code in Python according to the description.

"""
{description}
"""
'''


def build_decoder_prompt(task: HumanEvalPlusTask) -> str:
    """Format the decoder prompt for a HumanEval+ task."""
    return DECODER_TEMPLATE.format(description=task.prompt)


def decoder_input_from_task(task: HumanEvalPlusTask) -> str:
    """Return v1 stub-as-description decoder input."""
    return task.prompt
