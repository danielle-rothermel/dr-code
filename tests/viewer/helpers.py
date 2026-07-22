from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dr_code.corpus.candidate_evaluation import (
    MEMBERSHIP_SCHEMA,
    RESULTS_SCHEMA as EVALUATION_RESULTS_SCHEMA,
)
from dr_code.corpus.preprocessing_artifacts import (
    projected_artifact_schemas,
)
from dr_code.viewer.domain import RunDescriptor


CORPUS_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("task_id", pa.string(), nullable=False),
        pa.field("source_kind", pa.string(), nullable=False),
        pa.field("decoder_output", pa.string()),
    ]
)

CORPUS_ROWS = [
    ("missing", "Task/0", "test", None),
    ("blank", "Task/1", "test", "  \n"),
    ("no-code", "Task/2", "test", "This is prose."),
    ("no-code-alt", "Task/2", "test", "Alternate prose."),
    ("no-code-null", "Task/2", "test", "Unclassified prose."),
    ("compile-fail", "Task/3", "test", "def broken("),
    ("top-fail", "Task/4", "test", "answer = 42"),
    ("pass", "Task/5", "test", "def pass_me():\n    return 1"),
    ("fail", "Task/6", "test", "def fail_me():\n    return 0"),
]


def write_bundle(
    root: Path,
    *,
    run_id: str = "fixture-run",
    corpus_path: Path | None = None,
    definition_hash: str = "a" * 128,
    with_evaluation: bool = True,
    preprocessing_schema_version: int = 2,
    no_code_causes: tuple[str | None, str | None, str | None] = (
        "primary",
        "alternate",
        None,
    ),
) -> RunDescriptor:
    root.mkdir()
    corpus = corpus_path or root / "corpus.parquet"
    if corpus_path is None:
        _write(corpus, CORPUS_SCHEMA, CORPUS_ROWS)
    run = root / "run"
    run.mkdir()

    result_rows = [
        _result(
            "missing", "missing", "decoder_output_missing", None, None, None, 0
        ),
        _result(
            "blank",
            "present",
            "decoder_output_blank",
            "blank",
            "require_nonblank_text",
            "blank",
            0,
        ),
        _result(
            "no-code",
            "present",
            "no_code_candidates",
            "no_code_candidates",
            "extract_candidates",
            no_code_causes[0],
            0,
        ),
        _result(
            "no-code-alt",
            "present",
            "no_code_candidates",
            "no_code_candidates",
            "extract_candidates",
            no_code_causes[1],
            0,
        ),
        _result(
            "no-code-null",
            "present",
            "no_code_candidates",
            "no_code_candidates",
            "extract_candidates",
            no_code_causes[2],
            0,
        ),
        _result(
            "compile-fail",
            "present",
            "no_compilable_candidate",
            "no_compilable_candidate",
            "filter_compilable",
            "syntax",
            0,
        ),
        _result(
            "top-fail",
            "present",
            "no_top_level_function_candidate",
            "no_top_level_function_candidate",
            "filter_has_top_level_function",
            "no function",
            0,
        ),
        _result(
            "pass",
            "present",
            "function_candidates_extracted",
            None,
            None,
            None,
            1,
        ),
        _result(
            "fail",
            "present",
            "function_candidates_extracted",
            None,
            None,
            None,
            1,
        ),
    ]
    candidate_rows = [
        _candidate(
            "pass",
            "candidate-pass",
            "def pass_me():\n    return 1",
            schema_version=preprocessing_schema_version,
        ),
        _candidate(
            "fail",
            "candidate-fail",
            "def fail_me():\n    return 0",
            schema_version=preprocessing_schema_version,
        ),
    ]
    fact_rows = [
        _fact("blank", "require_nonblank_text", {"is_nonblank": False}),
        _fact("no-code", "require_nonblank_text", {"is_nonblank": True}),
        _fact("no-code", "extract_candidates", {"candidate_count": 0}),
        _fact("no-code-alt", "require_nonblank_text", {"is_nonblank": True}),
        _fact("no-code-alt", "extract_candidates", {"candidate_count": 0}),
        _fact("no-code-null", "require_nonblank_text", {"is_nonblank": True}),
        _fact("no-code-null", "extract_candidates", {"candidate_count": 0}),
        *(
            _survival_facts(
                "compile-fail", extracted=1, compilable=0, top=None
            )
        ),
        *(_survival_facts("top-fail", extracted=1, compilable=1, top=0)),
        *(_survival_facts("pass", extracted=1, compilable=1, top=1)),
        *(_survival_facts("fail", extracted=1, compilable=1, top=1)),
    ]
    rejection_rows = [
        {
            "sample_id": "compile-fail",
            "step_name": "filter_compilable",
            "candidate_id": "candidate-broken",
            "input_index": 0,
            "reason_code": "not_compilable",
            "details_json": '{"compile_error":"SyntaxError"}',
        },
        {
            "sample_id": "top-fail",
            "step_name": "filter_has_top_level_function",
            "candidate_id": "candidate-top",
            "input_index": 0,
            "reason_code": "no_top_level_function",
            "details_json": "{}",
        },
    ]
    relations = {
        "results": result_rows,
        "candidates": candidate_rows,
        "step_facts": fact_rows,
        "rejections": rejection_rows,
    }
    schemas = projected_artifact_schemas(preprocessing_schema_version)
    for name, rows in relations.items():
        _write(run / f"{name}.parquet", schemas[name], rows)
    corpus_file = pq.ParquetFile(corpus)
    manifest = {
        "schema_version": preprocessing_schema_version,
        "run_id": run_id,
        "input": {
            "sha256": sha256_file(corpus),
            "schema": corpus_file.schema_arrow.serialize().to_pybytes().hex(),
            "expected_rows": len(CORPUS_ROWS),
            "expected_row_groups": corpus_file.num_row_groups,
        },
        "definition": {
            "definition_id": "fixture-definition",
            "version": "v1",
            "steps": [
                {"instance_name": name, "step": name, "settings": {}}
                for name in (
                    "require_nonblank_text",
                    "extract_candidates",
                    "filter_compilable",
                    "filter_has_top_level_function",
                )
            ],
        },
        "definition_hash": definition_hash,
        "relation_totals": {
            name: len(rows) for name, rows in relations.items()
        },
        "complete": True,
    }
    _write_json(run / "manifest.json", manifest)

    evaluation: Path | None = None
    if with_evaluation:
        evaluation = root / "evaluation"
        evaluation.mkdir()
        memberships = [
            _membership(
                "pass",
                "candidate-pass",
                "Task/5",
                "key-pass",
                "def pass_me():\n    return 1",
            ),
            _membership(
                "fail",
                "candidate-fail",
                "Task/6",
                "key-fail",
                "def fail_me():\n    return 0",
            ),
        ]
        evaluation_results = [
            _evaluation_result(
                "key-pass", "Task/5", "def pass_me():\n    return 1", "passed"
            ),
            _evaluation_result(
                "key-fail", "Task/6", "def fail_me():\n    return 0", "failed"
            ),
        ]
        membership_path = evaluation / "candidate_membership.parquet"
        evaluation_results_path = evaluation / "candidate_results.parquet"
        _write(membership_path, MEMBERSHIP_SCHEMA, memberships)
        _write(
            evaluation_results_path,
            EVALUATION_RESULTS_SCHEMA,
            evaluation_results,
        )
        evaluation_manifest = {
            "schema_version": 1,
            "complete": True,
            "corpus_sha256": sha256_file(corpus),
            "preprocessing_manifest_sha256": sha256_file(
                run / "manifest.json"
            ),
            "preprocessing_candidates_sha256": sha256_file(
                run / "candidates.parquet"
            ),
            "preprocessing_results_sha256": sha256_file(
                run / "results.parquet"
            ),
            "membership_rows": len(memberships),
            "result_rows": len(evaluation_results),
            "metrics_profile": "fixture@v1",
            "operator": "code_test@1",
            "metrics_definition_hash": "b" * 128,
            "snapshot_sha256": "c" * 64,
            "runner_identity": "fixture",
            "execution_fingerprint": "d" * 64,
            "operator_settings": {"timeout_seconds": 1.0},
        }
        _write_json(
            evaluation / "candidate_evaluation_manifest.json",
            evaluation_manifest,
        )
    return RunDescriptor.from_paths(
        label=run_id,
        corpus_path=corpus,
        preprocessing=run,
        candidate_evaluation=evaluation,
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _result(
    sample_id: str,
    presence: str,
    outcome: str,
    failure_code: str | None,
    failed_step: str | None,
    cause: str | None,
    count: int,
) -> dict[str, object]:
    raw = next(row[3] for row in CORPUS_ROWS if row[0] == sample_id)
    return {
        "sample_id": sample_id,
        "decoder_output_presence": presence,
        "raw_output_sha256": _text_sha256(raw) if raw is not None else None,
        "outcome": outcome,
        "outcome_code": outcome if count else None,
        "failure_code": failure_code,
        "failed_step": failed_step,
        "cause": cause,
        "propagated_through": [] if failure_code else None,
        "final_candidate_count": count,
    }


def _candidate(
    sample_id: str,
    candidate_id: str,
    source: str,
    *,
    schema_version: int,
) -> dict[str, object]:
    origins = (
        [{"variant": "raw", "strategy": "fixture"}]
        if schema_version == 1
        else [
            {
                "path": [
                    {"kind": "raw", "details_json": "{}"},
                    {
                        "kind": "fixture",
                        "details_json": '{"fixture":true}',
                    },
                ]
            }
        ]
    )
    return {
        "sample_id": sample_id,
        "candidate_index": 0,
        "candidate_id": candidate_id,
        "cleaned_source": source,
        "source_sha256": _text_sha256(source),
        "origins": origins,
        "parse_ok": True,
        "parse_error": None,
        "compile_ok": True,
        "compile_error": None,
        "compile_warnings": [],
        "top_level_function_count": 1,
        "top_level_function_names": [candidate_id.removeprefix("candidate-")],
        "top_level_async_function_names": [],
    }


def _fact(
    sample_id: str, step: str, facts: dict[str, object]
) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "step_name": step,
        "facts_json": json.dumps(facts, sort_keys=True, separators=(",", ":")),
    }


def _survival_facts(
    sample_id: str, *, extracted: int, compilable: int, top: int | None
) -> list[dict[str, str]]:
    rows = [
        _fact(sample_id, "require_nonblank_text", {"is_nonblank": True}),
        _fact(sample_id, "extract_candidates", {"candidate_count": extracted}),
        _fact(
            sample_id,
            "filter_compilable",
            {"survivor_candidate_count": compilable},
        ),
    ]
    if top is not None:
        rows.append(
            _fact(
                sample_id,
                "filter_has_top_level_function",
                {"survivor_candidate_count": top},
            )
        )
    return rows


def _membership(
    sample_id: str, candidate_id: str, task_id: str, key: str, source: str
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "candidate_id": candidate_id,
        "candidate_index": 0,
        "task_id": task_id,
        "source_kind": "test",
        "source_sha256": _text_sha256(source),
        "task_fingerprint": _text_sha256(task_id),
        "evaluation_key": key,
        "metrics_profile": "fixture@v1",
        "operator": "code_test@1",
    }


def _evaluation_result(
    key: str, task_id: str, source: str, outcome: str
) -> dict[str, object]:
    passed = outcome == "passed"
    return {
        "evaluation_key": key,
        "task_id": task_id,
        "cleaned_source": source,
        "source_sha256": _text_sha256(source),
        "task_fingerprint": _text_sha256(task_id),
        "metrics_profile": "fixture@v1",
        "operator": "code_test@1",
        "record_status": "measured",
        "failure_type": None,
        "failure_message": None,
        "outcome": outcome,
        "function_count": 1,
        "best_function_name": "f",
        "total_cases": 1,
        "passed_count": int(passed),
        "failed_count": int(not passed),
        "error_count": 0,
        "timeout_count": 0,
        "coverage_complete": True,
    }


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write(path: Path, schema: pa.Schema, rows: list[object]) -> None:
    if rows and isinstance(rows[0], tuple):
        table = pa.Table.from_arrays(
            [
                pa.array(column, type=field.type)
                for column, field in zip(zip(*rows), schema)
            ],
            schema=schema,
        )
    else:
        table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
