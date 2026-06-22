"""Write stage 4 analysis artifacts to disk."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from dr_code.analysis.join import EnrichedRow


@dataclass(frozen=True)
class AnalysisArtifacts:
    """Paths written by export_analysis."""

    output_dir: Path
    enriched_path: Path
    summary_path: Path
    aggregate_paths: dict[str, Path]


def export_analysis(
    enriched: list[EnrichedRow],
    summary: dict[str, Any],
    aggregates: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> AnalysisArtifacts:
    """Write enriched table, summary JSON, and aggregate Parquet files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregates_dir = output_dir / "aggregates"
    aggregates_dir.mkdir(parents=True, exist_ok=True)

    enriched_path = write_enriched_rows(
        enriched, output_dir / "enriched.parquet"
    )
    summary_path = write_summary(summary, output_dir / "summary.json")

    aggregate_paths: dict[str, Path] = {}
    for name, table_rows in aggregates.items():
        path = aggregates_dir / f"{name}.parquet"
        write_aggregate_table(table_rows, path)
        aggregate_paths[name] = path

    return AnalysisArtifacts(
        output_dir=output_dir,
        enriched_path=enriched_path,
        summary_path=summary_path,
        aggregate_paths=aggregate_paths,
    )


def write_enriched_rows(rows: list[EnrichedRow], path: Path) -> Path:
    """Write enriched rows to Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    py_rows = [row.model_dump(mode="json") for row in rows]
    table = pa.Table.from_pylist(py_rows)
    pq.write_table(table, path)
    return path


def write_aggregate_table(rows: list[dict[str, Any]], path: Path) -> Path:
    """Write an aggregate slice table to Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)
    return path


def write_summary(summary: dict[str, Any], path: Path) -> Path:
    """Write summary JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path
