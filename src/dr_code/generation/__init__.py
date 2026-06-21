"""Generation helpers for stage 1b."""

from dr_code.generation.prompts import (
    DECODER_TEMPLATE,
    build_decoder_prompt,
    decoder_input_from_task,
)

__all__ = [
    "DECODER_TEMPLATE",
    "build_decoder_prompt",
    "decoder_input_from_task",
]
