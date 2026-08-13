"""Load a generation corpus bundle for the task-difficulty workflow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl

from drc_generation_corpus.models import BuildManifest

_MANIFEST_NAME = "manifest.json"
_GENERATIONS_NAME = "generations.parquet"
_REQUESTS_NAME = "requests.parquet"


def manifest_sha256(bundle_dir: Path) -> str:
    manifest_path = bundle_dir / _MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def load_workflow_frame(
    bundle_dir: Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> pl.DataFrame:
    """Join generations and requests into the flat workflow frame."""

    bundle_dir = bundle_dir.expanduser().resolve()
    actual_sha256 = manifest_sha256(bundle_dir)
    if (
        expected_manifest_sha256 is not None
        and actual_sha256 != expected_manifest_sha256
    ):
        raise ValueError(
            "generation corpus manifest SHA-256 mismatch: "
            f"expected {expected_manifest_sha256}, got {actual_sha256}"
        )

    manifest = BuildManifest.model_validate_json(
        (bundle_dir / _MANIFEST_NAME).read_text(encoding="utf-8")
    )
    if manifest.format != "generation-corpus-v1":
        raise ValueError(
            f"unsupported generation corpus format: {manifest.format!r}"
        )

    generations_path = bundle_dir / _GENERATIONS_NAME
    requests_path = bundle_dir / _REQUESTS_NAME
    if not generations_path.is_file():
        raise FileNotFoundError(generations_path)
    if not requests_path.is_file():
        raise FileNotFoundError(requests_path)

    generations = pl.read_parquet(generations_path)
    requests = pl.read_parquet(requests_path).select(
        "generation_id",
        "generation_mode",
        "budget_mode",
        "max_characters",
    )
    joined = generations.join(requests, on="generation_id", how="inner")
    if joined.is_empty():
        raise ValueError(f"generation corpus bundle is empty: {bundle_dir}")

    return joined.select(
        pl.col("generation_id").alias("sample_id"),
        pl.col("task_id"),
        pl.col("model"),
        pl.col("decoder_model"),
        pl.col("encoder_model"),
        pl.col("encoder_output"),
        pl.col("encoder_user_prompt"),
        pl.col("decoder_output"),
        pl.col("generation_mode"),
        pl.col("budget_mode"),
        pl.col("max_characters"),
    )


def load_manifest_summary(bundle_dir: Path) -> dict[str, object]:
    manifest = BuildManifest.model_validate_json(
        (bundle_dir.expanduser().resolve() / _MANIFEST_NAME).read_text(
            encoding="utf-8"
        )
    )
    return {
        "format": manifest.format,
        "adapter_name": manifest.adapter_name,
        "adapter_version": manifest.adapter_version,
        "source_manifest_sha256": manifest.source_manifest_sha256,
        "generations": manifest.generations.rows,
        "requests": manifest.requests.rows,
        "manifest_sha256": manifest_sha256(bundle_dir),
    }


def format_manifest_summary(summary: dict[str, object]) -> str:
    return json.dumps(summary, sort_keys=True, separators=(",", ":"))
