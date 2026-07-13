from __future__ import annotations

import textwrap
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, StrictStr

from dr_code.text_analysis import (
    candidate_blocks,
    code_like_blocks,
    split_by_fences,
)
from dr_code.humaneval.import_inference import infer_necessary_imports
from dr_code.text_transforms import (
    drop_after_last_return,
    drop_if_name,
    normalize_text,
    strip_code_fences,
    strip_markdown_wrappers,
    unescape_literal_newlines,
)


class TraceNodeKind(StrEnum):
    FORK = "fork"
    TRANSFORM = "transform"
    CHECK = "check"


class TraceCheckVerdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"


class ExtractionTraceNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: TraceNodeKind
    name: StrictStr
    before_text: StrictStr | None = None
    after_text: StrictStr | None = None
    check_name: StrictStr | None = None
    verdict: TraceCheckVerdict | None = None
    reason: StrictStr | None = None
    children: list["ExtractionTraceNode"] = Field(default_factory=list)


class CleaningTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[StrictStr]
    roots: list[ExtractionTraceNode]


def apply_cleaning(
    gen_str: str,
    apply_dedent: bool = False,
    *,
    unescape_fallback: bool = True,
) -> list[str]:
    return apply_cleaning_with_trace(
        gen_str,
        apply_dedent=apply_dedent,
        unescape_fallback=unescape_fallback,
    ).candidates


def apply_cleaning_with_trace(
    gen_str: str,
    apply_dedent: bool = False,
    *,
    unescape_fallback: bool = True,
) -> CleaningTrace:
    normalized = normalize_text(gen_str)
    normalize_node = ExtractionTraceNode(
        kind=TraceNodeKind.TRANSFORM,
        name="normalize_text",
        before_text=gen_str,
        after_text=normalized,
    )
    if not normalized:
        return CleaningTrace(candidates=[], roots=[normalize_node])

    blocks, candidate_blocks_node = _candidate_blocks_with_trace(normalized)
    candidates, initial_node = _trace_candidate_pass(
        blocks,
        apply_dedent=apply_dedent,
        pass_name="initial_pass",
    )
    candidate_blocks_node.children = [initial_node]
    normalize_node.children.append(candidate_blocks_node)
    if not candidates:
        fallback_blocks = [strip_markdown_wrappers(block) for block in blocks]
        candidates, fallback_node = _trace_candidate_pass(
            fallback_blocks,
            apply_dedent=apply_dedent,
            pass_name="markdown_wrapper_fallback",
            original_blocks=blocks,
        )

        normalize_node.children.append(fallback_node)

    if not candidates and unescape_fallback:
        unescaped = unescape_literal_newlines(normalized)
        if unescaped is not None:
            unescape_node = ExtractionTraceNode(
                kind=TraceNodeKind.TRANSFORM,
                name="unescape_literal_newlines",
                before_text=normalized,
                after_text=unescaped,
            )
            fallback_blocks, candidate_blocks_node = (
                _candidate_blocks_with_trace(unescaped)
            )
            candidates, unescaped_initial_node = _trace_candidate_pass(
                fallback_blocks,
                apply_dedent=apply_dedent,
                pass_name="unescaped_initial_pass",
            )
            candidate_blocks_node.children = [unescaped_initial_node]
            unescape_node.children.append(candidate_blocks_node)

            if not candidates:
                stripped_blocks = [
                    strip_markdown_wrappers(block) for block in fallback_blocks
                ]
                candidates, unescaped_wrapper_node = _trace_candidate_pass(
                    stripped_blocks,
                    apply_dedent=apply_dedent,
                    pass_name="unescaped_markdown_wrapper_fallback",
                    original_blocks=fallback_blocks,
                )
                unescape_node.children.append(unescaped_wrapper_node)

            normalize_node.children.append(unescape_node)

    return CleaningTrace(candidates=candidates, roots=[normalize_node])


def _candidate_blocks_with_trace(
    normalized: str,
) -> tuple[list[str], ExtractionTraceNode]:
    unfenced, fenced = split_by_fences(normalized)
    blocks = candidate_blocks(normalized)
    selected = "fenced" if fenced else "first_unfenced"
    return (
        blocks,
        ExtractionTraceNode(
            kind=TraceNodeKind.FORK,
            name="fence_split",
            before_text=normalized,
            reason=(
                f"{len(fenced)} fenced block(s), "
                f"{len(unfenced)} unfenced block(s); using {selected}"
            ),
        ),
    )


def _trace_candidate_pass(
    blocks: Sequence[str],
    *,
    apply_dedent: bool,
    pass_name: str,
    original_blocks: Sequence[str] | None = None,
) -> tuple[list[str], ExtractionTraceNode]:
    root = ExtractionTraceNode(
        kind=TraceNodeKind.FORK,
        name=pass_name,
        reason=f"{len(blocks)} block(s)",
    )
    block_nodes: list[ExtractionTraceNode] = []
    candidates: list[str] = []

    for block_index, block in enumerate(blocks):
        before_block = (
            original_blocks[block_index]
            if original_blocks is not None
            else None
        )
        block_node = _trace_block(
            block,
            block_index=block_index,
            before_text=before_block,
            apply_dedent=apply_dedent,
            candidates=candidates,
        )
        block_nodes.append(block_node)

    root.children = block_nodes
    return candidates, root


def _trace_block(
    block: str,
    *,
    block_index: int,
    before_text: str | None,
    apply_dedent: bool,
    candidates: list[str],
) -> ExtractionTraceNode:
    if before_text is not None:
        parent = ExtractionTraceNode(
            kind=TraceNodeKind.TRANSFORM,
            name="strip_markdown_wrappers",
            before_text=before_text,
            after_text=block,
        )
    else:
        parent = ExtractionTraceNode(
            kind=TraceNodeKind.FORK,
            name=f"block_{block_index}",
            after_text=block,
        )

    candidate_texts = code_like_blocks([block])
    fanout = ExtractionTraceNode(
        kind=TraceNodeKind.FORK,
        name="function_pattern_fanout",
        before_text=block,
        reason=f"{len(candidate_texts)} candidate(s)",
        children=[
            _trace_candidate_transforms(
                candidate,
                apply_dedent=apply_dedent,
                candidates=candidates,
            )
            for candidate in candidate_texts
        ],
    )
    parent.children = [fanout]
    return parent


def _trace_candidate_transforms(
    candidate: str,
    *,
    apply_dedent: bool,
    candidates: list[str],
) -> ExtractionTraceNode:
    after_fence_strip = strip_code_fences(candidate)
    fence_node = ExtractionTraceNode(
        kind=TraceNodeKind.TRANSFORM,
        name="strip_code_fences",
        before_text=candidate,
        after_text=after_fence_strip,
    )

    current_node = fence_node
    current_text = after_fence_strip
    if apply_dedent:
        dedented = textwrap.dedent(current_text)
        dedent_node = ExtractionTraceNode(
            kind=TraceNodeKind.TRANSFORM,
            name="dedent",
            before_text=current_text,
            after_text=dedented,
        )
        current_node.children = [dedent_node]
        current_node = dedent_node
        current_text = dedented

    split_texts = drop_if_name(current_text)
    split_node = ExtractionTraceNode(
        kind=TraceNodeKind.FORK,
        name="drop_if_name",
        before_text=current_text,
        reason=f"{len(split_texts)} candidate(s)",
        children=[
            _trace_return_and_import(split_text, candidates)
            for split_text in split_texts
        ],
    )
    current_node.children = [split_node]
    return fence_node


def _trace_return_and_import(
    split_text: str,
    candidates: list[str],
) -> ExtractionTraceNode:
    after_drop_return = drop_after_last_return(split_text)
    drop_return_node = ExtractionTraceNode(
        kind=TraceNodeKind.TRANSFORM,
        name="drop_after_last_return",
        before_text=split_text,
        after_text=after_drop_return,
    )
    inferred = infer_necessary_imports(after_drop_return)
    drop_return_node.children = [
        ExtractionTraceNode(
            kind=TraceNodeKind.TRANSFORM,
            name="infer_necessary_imports",
            before_text=after_drop_return,
            after_text=inferred,
        )
    ]
    candidates.append(inferred)
    return drop_return_node
