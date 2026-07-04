"""Extraction-interface tests.

The Extraction module owns Text Normalization, extractor ordering,
raw/text-normalized fan-out, and extractor_path Attribution.
"""

from __future__ import annotations

import ast

from code_eval.config import DEFAULT_CONFIG
from code_eval.extraction import EXTRACTION_CATALOG, run_extraction, text_normalize
from code_eval.names import ExtractorName


def _assert_parses(source: str) -> None:
    ast.parse(source)


def _sources_for(raw: str, extractor: ExtractorName) -> list[str]:
    result = run_extraction(raw, DEFAULT_CONFIG)
    return [candidate.source for candidate in result.candidates if candidate.extractor is extractor]


def test_text_normalize_folds_fullwidth_crlf_tabs_and_blanks() -> None:
    raw = "\uff44\uff45\uff46 \uff46\uff4f\uff4f():\r\n\tpass\r\n\r\n\r\n"
    out = text_normalize(raw)
    assert out == "def foo():\n    pass"
    _assert_parses(out)


def test_extraction_order_is_catalog_order() -> None:
    result = run_extraction("def foo():\n    return 1\n", DEFAULT_CONFIG)
    assert [step.extractor for step in result.extraction_log[1:]] == [
        name for name, _ in EXTRACTION_CATALOG
    ]
    assert [extraction_pass.extractor for extraction_pass in result.passes] == [
        name for name, _ in EXTRACTION_CATALOG
    ]


def test_direct_parse_emits_raw_and_dedented_candidates() -> None:
    raw = "    def foo():\n        return 1\n"
    result = run_extraction(raw, DEFAULT_CONFIG)

    paths = [candidate.extractor_path for candidate in result.candidates]
    assert (ExtractorName.DIRECT_PARSE.value,) in paths
    assert (ExtractorName.DIRECT_PARSE_DEDENTED.value,) in paths
    assert any(
        "def foo" in source for source in _sources_for(raw, ExtractorName.DIRECT_PARSE_DEDENTED)
    )


def test_fences_extracts_multiple_code_candidates() -> None:
    raw = "Option 1:\n```python\ndef foo():\n    return 1\n```\nOption 2:\n```\ndef bar():\n    return 2\n```"
    sources = _sources_for(raw, ExtractorName.FENCES)
    assert len(sources) >= 2
    assert any("def foo" in source for source in sources)
    assert any("def bar" in source for source in sources)


def test_fences_extracts_unterminated_fence() -> None:
    raw = "```python\ndef foo():\n    return 1\n"
    sources = _sources_for(raw, ExtractorName.FENCES)
    assert sources
    _assert_parses(sources[0])


def test_keyword_anchor_locates_code_after_prose() -> None:
    sources = _sources_for("prose\ndef foo():\n    return 1\n", ExtractorName.KEYWORD_ANCHOR)
    assert "def foo():\n    return 1" in sources


def test_prose_patterns_trims_trailing_prose() -> None:
    raw = "Here is the solution:\n\ndef foo():\n    return 1\n\nHope this helps."
    sources = _sources_for(raw, ExtractorName.PROSE_PATTERNS)
    assert "def foo():\n    return 1" in [source.rstrip() for source in sources]


def test_indentation_block_picks_longest_code_run() -> None:
    raw = "prose paragraph\ndef foo():\n    return 1\nmore prose\n"
    sources = _sources_for(raw, ExtractorName.INDENTATION_BLOCK)
    assert "def foo():\n    return 1" in sources


def test_markdown_strip_blockquote() -> None:
    sources = _sources_for("> def foo():\n>     return 1\n", ExtractorName.MARKDOWN_STRIP)
    assert "def foo():\n    return 1" in sources


def test_inline_spans_full_wrap() -> None:
    sources = _sources_for("`def foo(): return 1`", ExtractorName.INLINE_SPANS)
    assert "def foo(): return 1" in sources


def test_text_normalized_candidates_are_attributed() -> None:
    raw = "\uff44\uff45\uff46 foo():\n\u3000\u3000pass\n"
    result = run_extraction(raw, DEFAULT_CONFIG)
    assert any(
        candidate.extractor_path
        == (
            ExtractorName.TEXT_NORMALIZE.value,
            ExtractorName.DIRECT_PARSE.value,
        )
        for candidate in result.candidates
    )
