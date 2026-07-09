from __future__ import annotations

import textwrap

from dr_code.text_analysis import candidate_blocks, code_like_blocks
from dr_code.humaneval.import_inference import infer_necessary_imports
from dr_code.text_transforms import (
    drop_after_last_return,
    drop_if_name,
    normalize_text,
    strip_code_fences,
    strip_markdown_wrappers,
)


def apply_cleaning(
    gen_str: str,
    apply_dedent: bool = False,
) -> list[str]:
    normalized = normalize_text(gen_str)
    if not normalized:
        return []

    blocks = candidate_blocks(normalized)
    candidates = code_like_blocks(blocks)
    if not candidates:
        candidates = code_like_blocks(
            strip_markdown_wrappers(block) for block in blocks
        )

    cleaned: list[str] = []
    for candidate in candidates:
        candidate_text = strip_code_fences(candidate)
        if apply_dedent:
            candidate_text = textwrap.dedent(candidate_text)
        cleaned.extend(
            infer_necessary_imports(drop_after_last_return(split_candidate))
            for split_candidate in drop_if_name(candidate_text)
        )
    return cleaned
