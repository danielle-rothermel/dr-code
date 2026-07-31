from __future__ import annotations

from dr_code.classifier.taxonomy import (
    OTHER_LABEL,
    TAXONOMY_VERSION,
    FailureFamily,
    label_names,
    labels_for,
)


def test_taxonomies_are_closed_versioned_and_include_other() -> None:
    assert TAXONOMY_VERSION
    for family in FailureFamily:
        labels = labels_for(family)
        names = label_names(family)
        assert names == tuple(label.name for label in labels)
        assert len(names) == len(set(names))
        assert OTHER_LABEL in names
        assert all(label.definition for label in labels)


def test_mixed_is_not_an_item_taxonomy_label() -> None:
    for family in FailureFamily:
        assert "mixed" not in label_names(family)
