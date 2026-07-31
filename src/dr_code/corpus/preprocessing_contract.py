"""Canonical persisted preprocessing corpus schema contracts."""

from __future__ import annotations

from typing import Final

PREPROCESSING_MANIFEST_SCHEMA_VERSION: Final = 3
PROJECTED_PART_SCHEMA_VERSION: Final = 2
PREPROCESSING_MANIFEST_FIELDS: Final = frozenset(
    {
        "schema_version",
        "run_id",
        "input",
        "preprocessing_definition_ref",
        "preprocessing_config",
        "preprocessing_definition_identity",
        "preprocessing_config_identity",
        "resolved_step_versions",
        "source",
        "installed_environment",
        "batch_size",
        "started_at",
        "updated_at",
        "completed_row_groups",
        "relation_totals",
        "outcome_totals",
        "complete",
        "relation_sha256",
        "completed_at",
    }
)
PREPROCESSING_INPUT_FIELDS: Final = frozenset(
    {
        "path",
        "sha256",
        "size",
        "schema_hex",
        "expected_rows",
        "expected_row_groups",
        "row_groups",
    }
)
PREPROCESSING_IDENTITY_FIELDS: Final = (
    "schema_version",
    "input",
    "preprocessing_definition_ref",
    "preprocessing_config",
    "preprocessing_definition_identity",
    "preprocessing_config_identity",
    "resolved_step_versions",
    "source",
    "installed_environment",
    "batch_size",
    "relation_totals",
    "outcome_totals",
    "relation_sha256",
)

__all__ = (
    "PREPROCESSING_IDENTITY_FIELDS",
    "PREPROCESSING_INPUT_FIELDS",
    "PREPROCESSING_MANIFEST_FIELDS",
    "PREPROCESSING_MANIFEST_SCHEMA_VERSION",
    "PROJECTED_PART_SCHEMA_VERSION",
)
