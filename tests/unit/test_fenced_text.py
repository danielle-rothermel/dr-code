"""Structured fence document behavior."""

from __future__ import annotations

from dr_code.fenced_text import extract_fenced_document


def test_fenced_document_preserves_order_tag_marker_and_closure() -> None:
    document = extract_fenced_document(
        "before\n```JSON\n{\"code\": \"x\"}\n```\nmiddle\n~~~python\nx = 1"
    )

    assert [segment.content for segment in document.segments] == [
        "before",
        '{"code": "x"}',
        "middle",
        "x = 1",
    ]
    assert [segment.is_fenced for segment in document.segments] == [
        False,
        True,
        False,
        True,
    ]
    assert [
        (block.index, block.marker, block.tag, block.closed)
        for block in document.fenced_blocks
    ] == [
        (0, "```", "json", True),
        (1, "~~~", "python", False),
    ]


def test_fenced_document_requires_matching_closer() -> None:
    document = extract_fenced_document(
        "```python\nx = 1\n~~~\ny = 2\n```"
    )

    assert len(document.fenced_blocks) == 1
    assert document.fenced_blocks[0].content == "x = 1\n~~~\ny = 2"
    assert document.fenced_blocks[0].closed is True


def test_fenced_document_retains_empty_and_unclosed_blocks() -> None:
    closed = extract_fenced_document("```json\n```")
    unclosed = extract_fenced_document("~~~")

    assert closed.fenced_blocks[0].content == ""
    assert closed.fenced_blocks[0].closed is True
    assert unclosed.fenced_blocks[0].content == ""
    assert unclosed.fenced_blocks[0].closed is False


def test_fenced_document_is_total_for_arbitrary_text() -> None:
    for text in ("", "plain prose", "```\nunterminated", "```bad tag!"):
        extract_fenced_document(text)
