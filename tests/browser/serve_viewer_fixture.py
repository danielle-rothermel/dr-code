"""Serve a deterministic live viewer fixture for Playwright."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import uvicorn

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "tests"))

from viewer.helpers import write_bundle  # noqa: E402

from dr_code.corpus.preprocessing_artifacts import (  # noqa: E402
    PROJECTED_ARTIFACT_SCHEMAS,
)
from dr_code.corpus.run_descriptor import RunDescriptor, file_sha256  # noqa: E402
from dr_code.viewer.analytics import ViewerAnalytics  # noqa: E402
from dr_code.viewer.app import create_app  # noqa: E402
from dr_code.viewer.database import ViewerDatabase  # noqa: E402

HOST = "127.0.0.1"
PORT = 8011
SOURCE_SAMPLE_ID = "no-code-null"
REVIEW_SAMPLE_IDS = ("HumanEval/32",) + tuple(
    f"review/{index:02d}" for index in range(10)
)


def _expand_review_group(descriptor: RunDescriptor) -> None:
    corpus_table = pq.read_table(descriptor.corpus_path)
    corpus_rows = corpus_table.to_pylist()
    corpus_template = next(
        row for row in corpus_rows if row["sample_id"] == SOURCE_SAMPLE_ID
    )
    expanded_corpus = [
        row for row in corpus_rows if row["sample_id"] != SOURCE_SAMPLE_ID
    ]
    expanded_corpus.extend(
        {**corpus_template, "sample_id": sample_id}
        for sample_id in REVIEW_SAMPLE_IDS
    )
    pq.write_table(
        pa.Table.from_pylist(expanded_corpus, schema=corpus_table.schema),
        descriptor.corpus_path,
        row_group_size=4,
    )

    run = descriptor.preprocessing_manifest_path.parent
    for relation in ("results", "step_facts"):
        path = run / f"{relation}.parquet"
        table = pq.read_table(path)
        rows = table.to_pylist()
        templates = [
            row for row in rows if row["sample_id"] == SOURCE_SAMPLE_ID
        ]
        expanded = [
            row for row in rows if row["sample_id"] != SOURCE_SAMPLE_ID
        ]
        expanded.extend(
            {**template, "sample_id": sample_id}
            for sample_id in REVIEW_SAMPLE_IDS
            for template in templates
        )
        pq.write_table(
            pa.Table.from_pylist(
                expanded, schema=PROJECTED_ARTIFACT_SCHEMAS[relation]
            ),
            path,
            row_group_size=4,
        )

    manifest_path = descriptor.preprocessing_manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus = pq.ParquetFile(descriptor.corpus_path)
    manifest["input"].update(
        {
            "sha256": file_sha256(descriptor.corpus_path),
            "size": descriptor.corpus_path.stat().st_size,
            "schema_hex": corpus.schema_arrow.serialize().to_pybytes().hex(),
            "expected_rows": corpus.metadata.num_rows,
            "expected_row_groups": corpus.num_row_groups,
            "row_groups": [
                {
                    "index": index,
                    "rows": corpus.metadata.row_group(index).num_rows,
                    "total_byte_size": (
                        corpus.metadata.row_group(index).total_byte_size
                    ),
                }
                for index in range(corpus.num_row_groups)
            ],
        }
    )
    manifest["completed_row_groups"] = list(range(corpus.num_row_groups))
    result_rows = pq.read_table(run / "results.parquet").to_pylist()
    manifest["outcome_totals"] = {
        outcome: sum(row["outcome"] == outcome for row in result_rows)
        for outcome in sorted({row["outcome"] for row in result_rows})
    }
    for relation in PROJECTED_ARTIFACT_SCHEMAS:
        path = run / f"{relation}.parquet"
        manifest["relation_totals"][relation] = pq.ParquetFile(
            path
        ).metadata.num_rows
        manifest["relation_sha256"][relation] = file_sha256(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="dr-code-browser-") as temporary:
        root = Path(temporary)
        initial = write_bundle(
            root / "bundle",
            run_id="browser-smoke",
            with_evaluation=False,
        )
        _expand_review_group(initial)
        descriptor = RunDescriptor.from_paths(
            label="Browser smoke",
            dataset_id=initial.dataset_id,
            corpus_path=initial.corpus_path,
            preprocessing=initial.preprocessing_manifest_path.parent,
        )
        with ViewerDatabase(root / "viewer.duckdb") as database:
            service = ViewerAnalytics(database, [descriptor])
            uvicorn.run(
                create_app(service, allowed_host=HOST),
                host=HOST,
                port=PORT,
                workers=1,
            )


if __name__ == "__main__":
    main()
