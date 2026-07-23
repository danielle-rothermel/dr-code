from __future__ import annotations

from dr_code.classifier.taxonomy import (
    OTHER_LABEL,
    TAXONOMY_VERSION,
    FailureKind,
    definitions_block,
    is_valid_label,
    label_names,
    labels_for,
)


def test_taxonomy_version_is_a_nonblank_string() -> None:
    assert isinstance(TAXONOMY_VERSION, str) and TAXONOMY_VERSION


def test_every_kind_includes_the_other_escape_label() -> None:
    for kind in FailureKind:
        assert OTHER_LABEL in label_names(kind)


def test_label_names_are_unique_per_kind() -> None:
    for kind in FailureKind:
        names = label_names(kind)
        assert len(names) == len(set(names))


def test_each_label_documents_a_definition_and_example() -> None:
    for kind in FailureKind:
        for label in labels_for(kind):
            assert label.definition.strip()
            assert label.example.strip()


def test_is_valid_label_matches_the_declared_names() -> None:
    assert is_valid_label(FailureKind.PARSE, "truncated-output")
    assert not is_valid_label(FailureKind.PARSE, "wrong-algorithm")
    assert is_valid_label(FailureKind.TEST, "wrong-algorithm")
    assert not is_valid_label(FailureKind.TEST, "not-a-label")


def test_definitions_block_lists_every_label() -> None:
    block = definitions_block(FailureKind.PARSE)
    for name in label_names(FailureKind.PARSE):
        assert name in block
