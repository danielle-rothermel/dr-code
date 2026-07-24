from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dr_code.corpus.candidate_evaluation import (
    MEMBERSHIP_SCHEMA,
    RESULTS_SCHEMA as EVALUATION_RESULTS_SCHEMA,
    humaneval_metric_definition,
)
from dr_code.corpus.candidate_evaluation_contract import (
    CANDIDATE_EVALUATION_SCHEMA_VERSION,
    candidate_evaluation_identity,
    candidate_evaluation_key,
    preprocessing_run_identity,
)
from dr_code.corpus.preprocessing_artifacts import (
    PROJECTED_ARTIFACT_SCHEMAS,
    project_preprocessing_result,
)
from dr_code.corpus.evaluation_generation import (
    publish_generation_directory,
    staged_generation_directory,
    switch_current,
)
from dr_code.eval import EvaluationProcedureDefinition, OperatorCoordinates
from dr_code.preprocessing.definitions import (
    HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
)
from dr_code.preprocessing.candidate_identity import candidate_id_for_source
from dr_code.preprocessing.runner import bind_preprocessing
from dr_code.trace import TextArtifact, TraceProducer
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
PASS_SOURCE = "def pass_me():\n    return 1"
FAIL_SOURCE = "def fail_me():\n    return 0"
PASS_CANDIDATE_ID = candidate_id_for_source(PASS_SOURCE)
FAIL_CANDIDATE_ID = candidate_id_for_source(FAIL_SOURCE)
METRIC_DEFINITION = humaneval_metric_definition()
METRIC_CONFIG = METRIC_DEFINITION.materialize()
PREPROCESSING_CONFIG = (
    HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION.materialize()
)
PREPROCESSING_RUNNER = bind_preprocessing(PREPROCESSING_CONFIG)
PROCEDURE_CONFIG = EvaluationProcedureDefinition(
    definition_id="humaneval-candidate-evaluation",
    version="1",
).materialize(
    preprocessing=PREPROCESSING_CONFIG,
    metric_extraction=METRIC_CONFIG,
)
QUESTION = METRIC_CONFIG.questions[0]
QUESTION_IDENTITY = QUESTION.identity_hash()
(
    _,
    OPERATOR_NAME,
    OPERATOR_VERSION,
    _OPERATOR_IMPLEMENTATION,
) = METRIC_CONFIG.resolved_operator_versions[0]
TRACE_PRODUCER = TraceProducer(
    producer_id=PREPROCESSING_CONFIG.definition_ref.definition_id,
    version=PREPROCESSING_CONFIG.definition_ref.version,
    definition_hash=PREPROCESSING_CONFIG.definition_ref.identity_hash,
    preprocessing_config_hash=PREPROCESSING_CONFIG.config_identity_hash,
    implementation_hash=PREPROCESSING_CONFIG.implementation_hash,
)
OPERATOR_COORDINATES = OperatorCoordinates(
    name=OPERATOR_NAME,
    version=OPERATOR_VERSION,
    implementation_hash=_OPERATOR_IMPLEMENTATION,
    settings=tuple(QUESTION.settings_dict().items()),
)
INSTALLED_DISTRIBUTIONS = [{"name": "fixture", "version": "1"}]
INSTALLED_ENVIRONMENT = {
    "distributions": INSTALLED_DISTRIBUTIONS,
    "identity": hashlib.sha256(
        json.dumps(
            INSTALLED_DISTRIBUTIONS,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest(),
}
RUNNER_IDENTITY = "fixture"
HOST_RUNTIME = {
    "python_version": "fixture",
    "python_implementation": "CPython",
    "platform": {
        "system": "fixture",
        "release": "fixture",
        "machine": "fixture",
    },
    "byteorder": "little",
    "installed_distributions": [],
    "installed_distributions_sha256": "4" * 64,
}
TRUSTED_SOURCE_SHA256 = {"runner": "3" * 64}
RUNTIME_IDENTITY = hashlib.sha256(
    json.dumps(
        {
            "runner_identity": RUNNER_IDENTITY,
            "host_runtime": HOST_RUNTIME,
            "installed_environment": INSTALLED_ENVIRONMENT,
            "trusted_source_sha256": TRUSTED_SOURCE_SHA256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
TRACE_PRODUCER_JSON = json.dumps(
    TRACE_PRODUCER.model_dump(mode="json"),
    sort_keys=True,
    separators=(",", ":"),
)
OPERATOR_COORDINATES_JSON = json.dumps(
    OPERATOR_COORDINATES.model_dump(mode="json"),
    sort_keys=True,
    separators=(",", ":"),
)


def write_bundle(
    root: Path,
    *,
    run_id: str = "fixture-run",
    dataset_id: str = "fixture",
    task_namespace: str = "Task",
    corpus_path: Path | None = None,
    with_evaluation: bool = True,
    parse_failures_are_nonblank: bool = True,
    no_code_causes: tuple[str | None, str | None, str | None] = (
        "primary",
        "alternate",
        None,
    ),
) -> RunDescriptor:
    root.mkdir()
    corpus = corpus_path or root / "corpus.parquet"
    parse_failure_ids = {
        "no-code",
        "no-code-alt",
        "no-code-null",
        "compile-fail",
        "top-fail",
    }
    corpus_rows = [
        (
            sample_id,
            resolved_task_id,
            source_kind,
            (
                " \n"
                if (
                    not parse_failures_are_nonblank
                    and sample_id in parse_failure_ids
                )
                else decoder_output
            ),
        )
        for (
            sample_id,
            task_id,
            source_kind,
            decoder_output,
        ) in CORPUS_ROWS
        for resolved_task_id in (
            f"{task_namespace}/{task_id.rsplit('/', 1)[1]}",
        )
    ]
    if corpus_path is None:
        _write(corpus, CORPUS_SCHEMA, corpus_rows)
    run = root / "run"
    run.mkdir()

    relations = _projected_relations(corpus_rows, no_code_causes)
    result_rows = relations["results"]
    for name, rows in relations.items():
        _write(
            run / f"{name}.parquet",
            PROJECTED_ARTIFACT_SCHEMAS[name],
            rows,
        )
    corpus_file = pq.ParquetFile(corpus)
    row_groups = [
        {
            "index": index,
            "rows": corpus_file.metadata.row_group(index).num_rows,
            "total_byte_size": (
                corpus_file.metadata.row_group(index).total_byte_size
            ),
        }
        for index in range(corpus_file.num_row_groups)
    ]
    manifest = {
        "schema_version": 4,
        "run_id": run_id,
        "input": {
            "path": str(corpus.resolve()),
            "sha256": sha256_file(corpus),
            "size": corpus.stat().st_size,
            "schema_hex": (
                corpus_file.schema_arrow.serialize().to_pybytes().hex()
            ),
            "expected_rows": len(corpus_rows),
            "expected_row_groups": corpus_file.num_row_groups,
            "row_groups": row_groups,
        },
        "preprocessing_definition_ref": (
            PREPROCESSING_CONFIG.definition_ref.model_dump(mode="json")
        ),
        "preprocessing_config": PREPROCESSING_CONFIG.model_dump(mode="json"),
        "preprocessing_definition_identity": (
            PREPROCESSING_CONFIG.definition_ref.identity_hash
        ),
        "preprocessing_config_identity": (
            PREPROCESSING_CONFIG.config_identity_hash
        ),
        "resolved_step_versions": [
            {
                "instance_name": instance_name,
                "step": step,
                "version": version,
                "implementation_hash": implementation_hash,
            }
            for (
                instance_name,
                step,
                version,
                implementation_hash,
            ) in PREPROCESSING_CONFIG.resolved_step_versions
        ],
        "source": {
            "dr_code_python_package_sha256": "9" * 64,
            "python_implementation": "CPython",
            "python_version": "fixture",
        },
        "installed_environment": INSTALLED_ENVIRONMENT,
        "batch_size": 1_000,
        "started_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "completed_row_groups": list(range(corpus_file.num_row_groups)),
        "relation_totals": {
            name: len(rows) for name, rows in relations.items()
        },
        "outcome_totals": {
            outcome: sum(1 for row in result_rows if row["outcome"] == outcome)
            for outcome in {row["outcome"] for row in result_rows}
        },
        "relation_sha256": {
            name: sha256_file(run / f"{name}.parquet") for name in relations
        },
        "complete": True,
        "completed_at": "2026-01-01T00:00:00+00:00",
    }
    _write_json(run / "manifest.json", manifest)

    evaluation: Path | None = None
    if with_evaluation:
        evaluation = root / "evaluation"
        memberships = [
            _membership(
                "pass",
                PASS_CANDIDATE_ID,
                f"{task_namespace}/5",
                _evaluation_key(f"{task_namespace}/5", PASS_SOURCE),
                PASS_SOURCE,
            ),
            _membership(
                "fail",
                FAIL_CANDIDATE_ID,
                f"{task_namespace}/6",
                _evaluation_key(f"{task_namespace}/6", FAIL_SOURCE),
                FAIL_SOURCE,
            ),
        ]
        evaluation_results = [
            _evaluation_result(f"{task_namespace}/5", PASS_SOURCE, "passed"),
            _evaluation_result(
                f"{task_namespace}/6", FAIL_SOURCE, "tests_failed"
            ),
        ]
        preprocessing_coordinates = {
            "identity": preprocessing_run_identity(manifest),
            "relations": {
                name: {
                    "sha256": sha256_file(run / f"{name}.parquet"),
                    "rows": len(rows),
                }
                for name, rows in relations.items()
            },
        }
        evaluation_coordinates = {
            "schema_version": CANDIDATE_EVALUATION_SCHEMA_VERSION,
            "corpus_sha256": sha256_file(corpus),
            "preprocessing_run": preprocessing_coordinates,
            "metrics_profile": "fixture@v1",
            "snapshot_sha256": "c" * 64,
            "runner_identity": RUNNER_IDENTITY,
            "runtime_identity": RUNTIME_IDENTITY,
            "dataset": {
                "dataset_id": dataset_id,
                "split": "test",
                "hf_revision": "fixture",
            },
            "metric_extraction_definition_ref": (
                METRIC_CONFIG.definition_ref.model_dump(mode="json")
            ),
            "metric_extraction_config": METRIC_CONFIG.model_dump(mode="json"),
            "metric_extraction_definition_identity": (
                METRIC_CONFIG.definition_ref.identity_hash
            ),
            "metric_extraction_config_identity": (
                METRIC_CONFIG.config_identity_hash
            ),
            "evaluation_procedure_definition_ref": (
                PROCEDURE_CONFIG.definition_ref.model_dump(mode="json")
            ),
            "evaluation_procedure_config": PROCEDURE_CONFIG.model_dump(
                mode="json"
            ),
            "evaluation_procedure_definition_identity": (
                PROCEDURE_CONFIG.definition_ref.identity_hash
            ),
            "evaluation_procedure_config_identity": (
                PROCEDURE_CONFIG.config_identity_hash
            ),
            "trace_producer": TRACE_PRODUCER.model_dump(mode="json"),
            "operator_coordinates": OPERATOR_COORDINATES.model_dump(
                mode="json"
            ),
            "question_identity_hash": QUESTION_IDENTITY,
            "operator_name": OPERATOR_NAME,
            "operator_version": OPERATOR_VERSION,
            "operator_implementation": _OPERATOR_IMPLEMENTATION,
            "host_runtime": HOST_RUNTIME,
            "installed_environment": INSTALLED_ENVIRONMENT,
            "trusted_source_sha256": TRUSTED_SOURCE_SHA256,
            "max_infrastructure_retries": 2,
        }
        with staged_generation_directory(evaluation) as staged:
            membership_path = staged / "candidate_membership.parquet"
            evaluation_results_path = staged / "candidate_results.parquet"
            _write(membership_path, MEMBERSHIP_SCHEMA, memberships)
            _write(
                evaluation_results_path,
                EVALUATION_RESULTS_SCHEMA,
                evaluation_results,
            )
            evaluation_manifest = {
                **evaluation_coordinates,
                "evaluation_identity": candidate_evaluation_identity(
                    evaluation_coordinates
                ),
                "reuse_result_sources": [],
                "membership_rows": len(memberships),
                "result_rows": len(evaluation_results),
                "candidate_membership_sha256": sha256_file(membership_path),
                "candidate_results_sha256": sha256_file(
                    evaluation_results_path
                ),
                "record_status_totals": {"measured": len(evaluation_results)},
                "reused_result_rows": 0,
                "reused_result_rows_by_source": [],
                "complete": True,
            }
            _write_json(
                staged / "candidate_evaluation_manifest.json",
                evaluation_manifest,
            )
            generation = publish_generation_directory(evaluation, staged)
        switch_current(evaluation, generation)
    return RunDescriptor.from_paths(
        label=run_id,
        dataset_id=dataset_id,
        corpus_path=corpus,
        preprocessing=run,
        candidate_evaluation=evaluation,
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _projected_relations(
    corpus_rows: Sequence[tuple[str, str, str, str | None]],
    no_code_causes: tuple[str | None, str | None, str | None],
) -> dict[str, list[dict[str, object]]]:
    relations: dict[str, list[dict[str, object]]] = {
        name: [] for name in PROJECTED_ARTIFACT_SCHEMAS
    }
    failure_details = {
        "blank": ("decoder_output_blank", "blank"),
        "no-code": ("no_code_candidates", no_code_causes[0]),
        "no-code-alt": ("no_code_candidates", no_code_causes[1]),
        "no-code-null": ("no_code_candidates", no_code_causes[2]),
        "compile-fail": ("no_compilable_candidate", "syntax"),
        "top-fail": (
            "no_top_level_function_candidate",
            "no function",
        ),
    }
    for sample_id, _task_id, _source_kind, decoder_output in corpus_rows:
        trace = (
            None
            if decoder_output is None
            else PREPROCESSING_RUNNER.run(TextArtifact(text=decoder_output))
        )
        projected = project_preprocessing_result(
            sample_id,
            decoder_output,
            trace,
        )
        failure_detail = failure_details.get(sample_id)
        if (
            failure_detail is not None
            and projected.results[0]["outcome"] == failure_detail[0]
        ):
            projected.results[0]["cause"] = failure_detail[1]
        relations["results"].extend(projected.results)
        relations["candidates"].extend(projected.candidates)
        relations["step_facts"].extend(projected.step_facts)
        relations["rejections"].extend(projected.rejections)
    return relations


def _membership(
    sample_id: str, candidate_id: str, task_id: str, key: str, source: str
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "candidate_id": candidate_id,
        "candidate_index": 0,
        "task_id": task_id,
        "task_identity": _text_sha256(task_id),
        "source_kind": "test",
        "source_sha256": _text_sha256(source),
        "evaluation_key": key,
        "question_identity_hash": QUESTION_IDENTITY,
        "operator_name": OPERATOR_NAME,
        "operator_version": OPERATOR_VERSION,
        "trace_producer_json": TRACE_PRODUCER_JSON,
        "operator_coordinates_json": OPERATOR_COORDINATES_JSON,
        "metric_extraction_config_identity": METRIC_CONFIG.config_identity_hash,
        "evaluation_procedure_config_identity": (
            PROCEDURE_CONFIG.config_identity_hash
        ),
        "runtime_identity": RUNTIME_IDENTITY,
        "runner_identity": RUNNER_IDENTITY,
    }


def _evaluation_result(
    task_id: str, source: str, outcome: str
) -> dict[str, object]:
    passed = outcome == "passed"
    return {
        "evaluation_key": _evaluation_key(task_id, source),
        "task_id": task_id,
        "task_identity": _text_sha256(task_id),
        "cleaned_source": source,
        "source_sha256": _text_sha256(source),
        "question_identity_hash": QUESTION_IDENTITY,
        "operator_name": OPERATOR_NAME,
        "operator_version": OPERATOR_VERSION,
        "trace_producer_json": TRACE_PRODUCER_JSON,
        "operator_coordinates_json": OPERATOR_COORDINATES_JSON,
        "metric_extraction_config_identity": METRIC_CONFIG.config_identity_hash,
        "evaluation_procedure_config_identity": (
            PROCEDURE_CONFIG.config_identity_hash
        ),
        "runtime_identity": RUNTIME_IDENTITY,
        "runner_identity": RUNNER_IDENTITY,
        "metrics_profile": "fixture@v1",
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


def _evaluation_key(task_id: str, source: str) -> str:
    return candidate_evaluation_key(
        task_id=task_id,
        task_identity=_text_sha256(task_id),
        source_sha256=_text_sha256(source),
        question_identity_hash=QUESTION_IDENTITY,
        operator_name=OPERATOR_NAME,
        operator_version=OPERATOR_VERSION,
        metric_extraction_config_identity=METRIC_CONFIG.config_identity_hash,
        evaluation_procedure_config_identity=(
            PROCEDURE_CONFIG.config_identity_hash
        ),
        runtime_identity=RUNTIME_IDENTITY,
        runner_identity=RUNNER_IDENTITY,
    )


def _write(path: Path, schema: pa.Schema, rows: Sequence[object]) -> None:
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
