from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dr_code.corpus.candidate_evaluation import (
    CandidateEvaluationError,
    STATE_FILENAME,
    evaluate_preprocessing_candidates,
)
from dr_code.corpus.candidate_evaluation_contract import (
    CANDIDATE_RESULT_FACT_FIELDS,
    CandidateEvaluationContractError,
    canonical_candidate_result,
)
from dr_code.corpus.evaluation_generation import (
    CURRENT_FILENAME,
    GENERATIONS_DIRECTORY,
    MANIFEST_FILENAME,
    MEMBERSHIP_FILENAME,
    RESULTS_FILENAME,
    resolve_current_generation,
)
from dr_code.eval import OperatorCoordinates
from dr_code.corpus.preprocessing_run import run_preprocessing_corpus
from dr_code.execution.subprocess import (
    SubprocessInfrastructureError,
    run_python_subprocess,
)
from dr_code.humaneval.sampling import write_human_eval_snapshot_rows
from dr_code.preprocessing.definitions import (
    HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
)
from dr_code.preprocessing.candidate_identity import candidate_id_for_source
from dr_code.trace import TraceProducer

_CANONICAL_SNAPSHOT = Path(__file__).with_name("humanevalplus_snapshot.json")
_CANONICAL_TASK = json.loads(_CANONICAL_SNAPSHOT.read_text())["rows"][0]
_TASK_ID = _CANONICAL_TASK["task_id"]
_CANDIDATE = (
    "from typing import List\n\n"
    "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
    "    sorted_numbers = sorted(numbers)\n"
    "    for index in range(len(sorted_numbers) - 1):\n"
    "        if sorted_numbers[index + 1] - sorted_numbers[index] < threshold:\n"
    "            return True\n"
    "    return False\n"
)


def _unreadable_source_evidence() -> str:
    raise ValueError("unreadable")


def _snapshot(path: Path) -> Path:
    shutil.copyfile(_CANONICAL_SNAPSHOT, path)
    return path


def _corpus(
    path: Path,
    sample_ids: list[str],
    *,
    outputs: list[str | None] | None = None,
) -> Path:
    decoder_outputs = (
        [_CANDIDATE] * len(sample_ids) if outputs is None else outputs
    )
    schema = pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("decoder_output", pa.string(), nullable=True),
            pa.field("task_id", pa.string(), nullable=False),
            pa.field("source_kind", pa.string(), nullable=False),
        ]
    )
    pq.write_table(
        pa.Table.from_arrays(
            [
                pa.array(sample_ids),
                pa.array(decoder_outputs),
                pa.array([_TASK_ID] * len(sample_ids)),
                pa.array(["fixture"] * len(sample_ids)),
            ],
            schema=schema,
        ),
        path,
        row_group_size=1,
    )
    return path


def _inputs(tmp_path: Path, sample_ids: list[str]) -> tuple[Path, Path, Path]:
    corpus = _corpus(tmp_path / "corpus.parquet", sample_ids)
    run = run_preprocessing_corpus(
        input_path=corpus,
        output_root=tmp_path / "runs",
        run_id="candidates",
    )
    snapshot = _snapshot(tmp_path / "snapshot.json")
    return run, corpus, snapshot


class _CountingRunner:
    def __init__(self, *, failures: int = 0, forbid: bool = False) -> None:
        self.calls = 0
        self.failures = failures
        self.forbid = forbid

    def __call__(
        self, *, source: str, input_text: str, timeout_seconds: float
    ):
        self.calls += 1
        if self.forbid:
            raise AssertionError("reused work must not execute")
        if self.calls <= self.failures:
            raise SubprocessInfrastructureError("transient fixture failure")
        return run_python_subprocess(
            source=source,
            input_text=input_text,
            timeout_seconds=timeout_seconds,
        )


def _evaluate(
    *,
    run: Path,
    corpus: Path,
    snapshot: Path,
    output: Path,
    runner: _CountingRunner,
    reuse: tuple[Path, ...] = (),
) -> None:
    evaluate_preprocessing_candidates(
        preprocessing_run=run,
        corpus_path=corpus,
        output_dir=output,
        snapshot_path=snapshot,
        max_workers=2,
        run_in_subprocess=runner,
        runner_identity="test:python-isolated@v1",
        reuse_results_from=reuse,
    )


def _artifact_bytes(output: Path) -> dict[str, bytes]:
    generation = resolve_current_generation(output)
    return {
        "candidate_membership.parquet": generation.membership_path.read_bytes(),
        "candidate_results.parquet": generation.results_path.read_bytes(),
        "candidate_evaluation_manifest.json": (
            generation.manifest_path.read_bytes()
        ),
    }


def _artifact(output: Path, filename: str) -> Path:
    generation = resolve_current_generation(output)
    return {
        "candidate_membership.parquet": generation.membership_path,
        "candidate_results.parquet": generation.results_path,
        "candidate_evaluation_manifest.json": generation.manifest_path,
    }[filename]


def _artifact_hashes(artifacts: dict[str, bytes]) -> dict[str, str]:
    return {
        filename: hashlib.sha256(content).hexdigest()
        for filename, content in artifacts.items()
    }


def _tree_contents(root: Path) -> dict[Path, bytes | None]:
    return {
        path.relative_to(root): None if path.is_dir() else path.read_bytes()
        for path in root.rglob("*")
    }


def _rewrite_preprocessing_relation(
    run: Path,
    relation: str,
    rows: list[dict[str, object]],
) -> None:
    path = run / f"{relation}.parquet"
    schema = pq.read_schema(path)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["relation_sha256"][relation] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    manifest["relation_totals"][relation] = len(rows)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def _relocate_runtime_inputs(
    tmp_path: Path,
    run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    moved_run = tmp_path / "relocated" / run.name
    moved_run.parent.mkdir()
    shutil.move(run, moved_run)
    manifest_path = moved_run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["input"]["path"] = "/simulated/other-checkout/corpus.parquet"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    simulated_python = tmp_path / "other-venv" / "bin" / "python"
    simulated_python.parent.mkdir(parents=True)
    simulated_python.write_bytes(b"simulated interpreter")
    monkeypatch.setattr(sys, "executable", str(simulated_python))
    return moved_run


def test_membership_occurrences_are_separate_from_deduplicated_results(
    tmp_path: Path,
) -> None:
    run, corpus, snapshot = _inputs(tmp_path, ["one", "two"])
    runner = _CountingRunner()
    output = tmp_path / "evaluation"
    _evaluate(
        run=run,
        corpus=corpus,
        snapshot=snapshot,
        output=output,
        runner=runner,
    )

    memberships = pq.read_table(
        _artifact(output, "candidate_membership.parquet")
    ).to_pylist()
    results = pq.read_table(
        _artifact(output, "candidate_results.parquet")
    ).to_pylist()
    assert len(memberships) == 2
    assert len(results) == 1
    assert len({row["evaluation_key"] for row in memberships}) == 1
    assert results[0]["outcome"] == "passed"
    assert runner.calls == 1
    manifest = json.loads(
        _artifact(output, "candidate_evaluation_manifest.json").read_text()
    )
    assert manifest["runner_identity"] == "test:python-isolated@v1"
    assert len(manifest["runtime_identity"]) == 64
    assert len(manifest["evaluation_identity"]) == 64
    assert len(manifest["metric_extraction_config_identity"]) == 64
    assert manifest["schema_version"] == 6
    assert len(manifest["question_identity_hash"]) == 64
    assert manifest["operator_name"] == "code_test"
    assert manifest["operator_version"] == "1"
    assert (
        results[0]["question_identity_hash"]
        == (manifest["question_identity_hash"])
    )
    assert results[0]["operator_name"] == manifest["operator_name"]
    assert results[0]["operator_version"] == manifest["operator_version"]


def test_completed_resume_preserves_artifact_bytes_and_identity(
    tmp_path: Path,
) -> None:
    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    output = tmp_path / "evaluation"
    _evaluate(
        run=run,
        corpus=corpus,
        snapshot=snapshot,
        output=output,
        runner=_CountingRunner(),
    )
    before = _artifact_bytes(output)
    before_hashes = _artifact_hashes(before)

    resumed = _CountingRunner(forbid=True)
    _evaluate(
        run=run,
        corpus=corpus,
        snapshot=snapshot,
        output=output,
        runner=resumed,
    )

    after = _artifact_bytes(output)
    assert resumed.calls == 0
    assert after == before
    assert _artifact_hashes(after) == before_hashes
    before_manifest = json.loads(before["candidate_evaluation_manifest.json"])
    after_manifest = json.loads(after["candidate_evaluation_manifest.json"])
    assert len(before_manifest["evaluation_identity"]) == 64
    assert (
        after_manifest["evaluation_identity"]
        == before_manifest["evaluation_identity"]
    )


def test_completed_resume_rejects_valid_shape_tamper_without_publication(
    tmp_path: Path,
) -> None:
    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    output = tmp_path / "evaluation"
    _evaluate(
        run=run,
        corpus=corpus,
        snapshot=snapshot,
        output=output,
        runner=_CountingRunner(),
    )
    published = _artifact_bytes(output)
    with sqlite3.connect(output / STATE_FILENAME) as connection:
        values = json.loads(
            connection.execute("SELECT values_json FROM work").fetchone()[0]
        )
        values["passed_count"] -= 1
        values["failed_count"] += 1
        connection.execute(
            "UPDATE work SET values_json = ?",
            (json.dumps(values, sort_keys=True, separators=(",", ":")),),
        )

    resumed = _CountingRunner(forbid=True)
    with pytest.raises(
        CandidateEvaluationError,
        match="completed result evidence mismatch",
    ):
        _evaluate(
            run=run,
            corpus=corpus,
            snapshot=snapshot,
            output=output,
            runner=resumed,
        )

    assert resumed.calls == 0
    assert _artifact_bytes(output) == published


def test_result_evidence_digest_pins_canonical_payload() -> None:
    from dr_code.corpus import candidate_evaluation

    assert (
        candidate_evaluation._result_evidence_sha256(
            evaluation_key="a" * 64,
            task_id="HumanEval/0",
            task_identity="b" * 64,
            source_sha256="c" * 64,
            candidate_source="def f():\n    return 1\n",
            record_status="measured",
            failure_type=None,
            failure_message=None,
            values_json='{"passed_count":1}',
        )
        == "2eaea7f6a200250157222a9f541b2b1ba30fc976d01aef7f2678e08c3baa074e"
    )


def test_legacy_state_without_result_evidence_is_fail_closed(
    tmp_path: Path,
) -> None:
    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    output = tmp_path / "evaluation"
    _evaluate(
        run=run,
        corpus=corpus,
        snapshot=snapshot,
        output=output,
        runner=_CountingRunner(),
    )
    published = _artifact_bytes(output)
    with sqlite3.connect(output / STATE_FILENAME) as connection:
        connection.execute(
            "ALTER TABLE work DROP COLUMN result_evidence_sha256"
        )

    resumed = _CountingRunner(forbid=True)
    with pytest.raises(
        CandidateEvaluationError,
        match="predates canonical result evidence",
    ):
        _evaluate(
            run=run,
            corpus=corpus,
            snapshot=snapshot,
            output=output,
            runner=resumed,
        )

    assert resumed.calls == 0
    assert _artifact_bytes(output) == published


def test_no_candidate_corpus_task_must_exist_in_snapshot_before_state(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus.parquet"
    schema = pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("decoder_output", pa.string(), nullable=True),
            pa.field("task_id", pa.string(), nullable=False),
            pa.field("source_kind", pa.string(), nullable=False),
        ]
    )
    pq.write_table(
        pa.Table.from_arrays(
            [
                pa.array(["missing"]),
                pa.array([None], type=pa.string()),
                pa.array(["HumanEval/absent"]),
                pa.array(["fixture"]),
            ],
            schema=schema,
        ),
        corpus,
    )
    run = run_preprocessing_corpus(
        input_path=corpus,
        output_root=tmp_path / "runs",
        run_id="no-candidate",
    )
    runner = _CountingRunner(forbid=True)
    output = tmp_path / "evaluation"

    with pytest.raises(
        CandidateEvaluationError,
        match="corpus task_id is absent from snapshot",
    ):
        _evaluate(
            run=run,
            corpus=corpus,
            snapshot=_snapshot(tmp_path / "snapshot.json"),
            output=output,
            runner=runner,
        )

    assert runner.calls == 0
    assert not output.exists()


def test_forged_one_row_snapshot_cannot_claim_canonical_coordinates(
    tmp_path: Path,
) -> None:
    run, corpus, _snapshot_path = _inputs(tmp_path, ["one"])
    forged_snapshot = write_human_eval_snapshot_rows(
        [_CANONICAL_TASK],
        snapshot_path=tmp_path / "forged-snapshot.json",
    )
    runner = _CountingRunner(forbid=True)
    output = tmp_path / "evaluation"

    with pytest.raises(
        CandidateEvaluationError,
        match="snapshot does not match the pinned dataset coordinates",
    ):
        _evaluate(
            run=run,
            corpus=corpus,
            snapshot=forged_snapshot,
            output=output,
            runner=runner,
        )

    assert runner.calls == 0
    assert not output.exists()


def test_moved_preprocessing_and_python_path_preserve_resume_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    output = tmp_path / "evaluation"
    _evaluate(
        run=run,
        corpus=corpus,
        snapshot=snapshot,
        output=output,
        runner=_CountingRunner(),
    )
    before = _artifact_bytes(output)
    before_key = pq.read_table(
        _artifact(output, "candidate_results.parquet")
    ).to_pylist()[0]["evaluation_key"]

    moved_run = _relocate_runtime_inputs(tmp_path, run, monkeypatch)
    resumed = _CountingRunner(forbid=True)

    _evaluate(
        run=moved_run,
        corpus=corpus,
        snapshot=snapshot,
        output=output,
        runner=resumed,
    )

    assert resumed.calls == 0
    assert _artifact_bytes(output) == before
    assert (
        pq.read_table(
            _artifact(output, "candidate_results.parquet")
        ).to_pylist()[0]["evaluation_key"]
        == before_key
    )


def test_reuse_accepts_moved_inputs_and_different_python_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    source = tmp_path / "source"
    _evaluate(
        run=run,
        corpus=corpus,
        snapshot=snapshot,
        output=source,
        runner=_CountingRunner(),
    )
    source_manifest = json.loads(
        _artifact(source, "candidate_evaluation_manifest.json").read_text()
    )
    source_key = pq.read_table(
        _artifact(source, "candidate_results.parquet")
    ).to_pylist()[0]["evaluation_key"]

    moved_run = _relocate_runtime_inputs(tmp_path, run, monkeypatch)
    target = tmp_path / "target"
    reused = _CountingRunner(forbid=True)

    _evaluate(
        run=moved_run,
        corpus=corpus,
        snapshot=snapshot,
        output=target,
        runner=reused,
        reuse=(source,),
    )

    target_manifest = json.loads(
        _artifact(target, "candidate_evaluation_manifest.json").read_text()
    )
    assert reused.calls == 0
    assert (
        target_manifest["evaluation_identity"]
        == source_manifest["evaluation_identity"]
    )
    assert (
        pq.read_table(
            _artifact(target, "candidate_results.parquet")
        ).to_pylist()[0]["evaluation_key"]
        == source_key
    )


@pytest.mark.parametrize("replacement", [None, "def add_one(x):\n return 9\n"])
def test_corpus_decoder_output_must_match_preprocessing_result_coordinate(
    tmp_path: Path, replacement: str | None
) -> None:
    run, _corpus_path, snapshot = _inputs(tmp_path, ["one"])
    corpus = _corpus(
        tmp_path / "replacement.parquet",
        ["one"],
        outputs=[replacement],
    )
    output = tmp_path / "evaluation"

    with pytest.raises(
        CandidateEvaluationError,
        match="decoder_output does not match preprocessing results",
    ):
        _evaluate(
            run=run,
            corpus=corpus,
            snapshot=snapshot,
            output=output,
            runner=_CountingRunner(forbid=True),
        )

    assert not output.exists()


def test_corpus_decoder_output_column_is_required(tmp_path: Path) -> None:
    run, _corpus_path, snapshot = _inputs(tmp_path, ["one"])
    corpus = tmp_path / "missing-decoder-output.parquet"
    pq.write_table(
        pa.table(
            {
                "sample_id": ["one"],
                "task_id": [_TASK_ID],
                "source_kind": ["fixture"],
            }
        ),
        corpus,
    )

    with pytest.raises(
        CandidateEvaluationError,
        match=r"missing required column.*decoder_output",
    ):
        _evaluate(
            run=run,
            corpus=corpus,
            snapshot=snapshot,
            output=tmp_path / "evaluation",
            runner=_CountingRunner(forbid=True),
        )


@pytest.mark.parametrize(
    ("step_name", "field", "replacement", "message"),
    [
        (
            "extract_candidates",
            "candidate_count",
            "1",
            "typed schema",
        ),
        (
            "return_all",
            "candidate_count",
            2,
            "candidate count waterfall",
        ),
    ],
)
def test_hash_consistent_preprocessing_facts_must_reconcile(
    tmp_path: Path,
    step_name: str,
    field: str,
    replacement: object,
    message: str,
) -> None:
    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    rows = pq.read_table(run / "step_facts.parquet").to_pylist()
    row = next(item for item in rows if item["step_name"] == step_name)
    facts = json.loads(row["facts_json"])
    facts[field] = replacement
    row["facts_json"] = json.dumps(
        facts,
        sort_keys=True,
        separators=(",", ":"),
    )
    _rewrite_preprocessing_relation(run, "step_facts", rows)
    runner = _CountingRunner(forbid=True)

    with pytest.raises(CandidateEvaluationError, match=message):
        _evaluate(
            run=run,
            corpus=corpus,
            snapshot=snapshot,
            output=tmp_path / "evaluation",
            runner=runner,
        )

    assert runner.calls == 0


def test_hash_consistent_rejection_relation_must_match_step_facts(
    tmp_path: Path,
) -> None:
    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    rows = [
        {
            "sample_id": "one",
            "step_name": "filter_plain_literal",
            "candidate_id": None,
            "input_index": 0,
            "reason_code": "forged",
            "details_json": "{}",
        }
    ]
    _rewrite_preprocessing_relation(run, "rejections", rows)
    runner = _CountingRunner(forbid=True)

    with pytest.raises(
        CandidateEvaluationError,
        match="rejection relation does not match step facts",
    ):
        _evaluate(
            run=run,
            corpus=corpus,
            snapshot=snapshot,
            output=tmp_path / "evaluation",
            runner=runner,
        )

    assert runner.calls == 0


def test_hash_consistent_candidate_must_be_derived_from_decoder_output(
    tmp_path: Path,
) -> None:
    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    candidate_rows = pq.read_table(run / "candidates.parquet").to_pylist()
    assert len(candidate_rows) == 1
    original_id = candidate_rows[0]["candidate_id"]
    forged_source = candidate_rows[0]["cleaned_source"] + "\n# forged"
    forged_id = candidate_id_for_source(forged_source)
    candidate_rows[0].update(
        {
            "candidate_id": forged_id,
            "cleaned_source": forged_source,
            "source_sha256": hashlib.sha256(
                forged_source.encode()
            ).hexdigest(),
        }
    )
    _rewrite_preprocessing_relation(run, "candidates", candidate_rows)

    def replace_candidate_id(value: object) -> object:
        if value == original_id:
            return forged_id
        if isinstance(value, list):
            return [replace_candidate_id(item) for item in value]
        if isinstance(value, dict):
            return {
                key: replace_candidate_id(item) for key, item in value.items()
            }
        return value

    fact_rows = pq.read_table(run / "step_facts.parquet").to_pylist()
    for row in fact_rows:
        row["facts_json"] = json.dumps(
            replace_candidate_id(json.loads(row["facts_json"])),
            sort_keys=True,
            separators=(",", ":"),
        )
    _rewrite_preprocessing_relation(run, "step_facts", fact_rows)
    rejection_rows = pq.read_table(run / "rejections.parquet").to_pylist()
    for row in rejection_rows:
        if row["candidate_id"] == original_id:
            row["candidate_id"] = forged_id
    _rewrite_preprocessing_relation(run, "rejections", rejection_rows)

    runner = _CountingRunner(forbid=True)
    with pytest.raises(
        CandidateEvaluationError,
        match="candidates are not canonically derived",
    ):
        _evaluate(
            run=run,
            corpus=corpus,
            snapshot=snapshot,
            output=tmp_path / "evaluation",
            runner=runner,
        )

    assert runner.calls == 0
    assert not (tmp_path / "evaluation").exists()


def test_corpus_result_admission_streams_bounded_parquet_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dr_code.corpus import candidate_evaluation

    row_count = candidate_evaluation._ADMISSION_BATCH_SIZE * 2 + 1
    sample_ids = [f"sample-{index}" for index in range(row_count)]
    corpus = _corpus(tmp_path / "scaled-corpus.parquet", sample_ids)
    raw_output_sha256 = hashlib.sha256(_CANDIDATE.encode()).hexdigest()
    results = tmp_path / "scaled-results.parquet"
    pq.write_table(
        pa.table(
            {
                "sample_id": sample_ids,
                "raw_output_sha256": [raw_output_sha256] * row_count,
                "final_candidate_count": [1] * row_count,
            }
        ),
        results,
    )
    parquet_file = candidate_evaluation.pq.ParquetFile
    observed_batch_sizes: list[int] = []

    class TrackingParquetFile:
        def __init__(self, path: Path) -> None:
            self._delegate = parquet_file(path)

        @property
        def schema_arrow(self) -> pa.Schema:
            return self._delegate.schema_arrow

        def iter_batches(
            self,
            *,
            batch_size: int,
            columns: list[str],
        ):
            assert batch_size == candidate_evaluation._ADMISSION_BATCH_SIZE
            for batch in self._delegate.iter_batches(
                batch_size=batch_size,
                columns=columns,
            ):
                observed_batch_sizes.append(batch.num_rows)
                yield batch

    monkeypatch.setattr(
        candidate_evaluation.pq,
        "ParquetFile",
        TrackingParquetFile,
    )

    candidate_evaluation._validate_corpus_results_before_state(
        corpus,
        results,
    )

    assert sum(observed_batch_sizes) == row_count * 2
    assert (
        max(observed_batch_sizes) <= candidate_evaluation._ADMISSION_BATCH_SIZE
    )


def test_corpus_extra_column_rejects_before_evaluator_state_creation(
    tmp_path: Path,
) -> None:
    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    table = pq.read_table(corpus).append_column(
        "unprocessed_extra",
        pa.array(["drift"]),
    )
    altered = tmp_path / "altered.parquet"
    pq.write_table(table, altered, row_group_size=1)
    output = tmp_path / "evaluation"

    with pytest.raises(
        CandidateEvaluationError,
        match="does not match preprocessing input coordinate",
    ):
        _evaluate(
            run=run,
            corpus=altered,
            snapshot=snapshot,
            output=output,
            runner=_CountingRunner(forbid=True),
        )

    assert not output.exists()


def test_forged_preprocessing_config_identity_rejects_before_state(
    tmp_path: Path,
) -> None:
    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["preprocessing_config_identity"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "evaluation"

    with pytest.raises(CandidateEvaluationError, match="not canonical"):
        _evaluate(
            run=run,
            corpus=corpus,
            snapshot=snapshot,
            output=output,
            runner=_CountingRunner(forbid=True),
        )

    assert not output.exists()


@pytest.mark.parametrize("forgery", ["config_identity", "assignment"])
def test_forged_preprocessing_config_payload_rejects_before_state(
    tmp_path: Path,
    forgery: str,
) -> None:
    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    config = manifest["preprocessing_config"]
    if forgery == "config_identity":
        config["config_identity_hash"] = "0" * 64
    else:
        config["assignment"] = [["undeclared", True]]
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "evaluation"

    with pytest.raises(
        CandidateEvaluationError, match="coordinates are invalid"
    ):
        _evaluate(
            run=run,
            corpus=corpus,
            snapshot=snapshot,
            output=output,
            runner=_CountingRunner(forbid=True),
        )

    assert not output.exists()


def test_infrastructure_failures_retry_before_completion(
    tmp_path: Path,
) -> None:
    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    runner = _CountingRunner(failures=2)
    output = tmp_path / "retry"
    _evaluate(
        run=run,
        corpus=corpus,
        snapshot=snapshot,
        output=output,
        runner=runner,
    )

    assert runner.calls == 3
    result = pq.read_table(
        _artifact(output, "candidate_results.parquet")
    ).to_pylist()[0]
    assert result["record_status"] == "measured"


def test_terminal_infrastructure_failure_survives_resume(
    tmp_path: Path,
) -> None:
    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    output = tmp_path / "terminal"
    failed = _CountingRunner(failures=10)
    _evaluate(
        run=run,
        corpus=corpus,
        snapshot=snapshot,
        output=output,
        runner=failed,
    )
    assert failed.calls == 3
    before = pq.read_table(
        _artifact(output, "candidate_results.parquet")
    ).to_pylist()[0]
    assert before["record_status"] == "infrastructure_failure"

    resumed = _CountingRunner(forbid=True)
    _evaluate(
        run=run,
        corpus=corpus,
        snapshot=snapshot,
        output=output,
        runner=resumed,
    )
    after = pq.read_table(
        _artifact(output, "candidate_results.parquet")
    ).to_pylist()[0]
    assert resumed.calls == 0
    assert after == before


def test_blank_operator_failure_message_is_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dr_code.metrics.operators.code_test import CodeTest

    def fail_with_blank_message(*args: object, **kwargs: object) -> None:
        raise RuntimeError()

    monkeypatch.setattr(CodeTest, "compute", fail_with_blank_message)
    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    output = tmp_path / "blank-failure"
    _evaluate(
        run=run,
        corpus=corpus,
        snapshot=snapshot,
        output=output,
        runner=_CountingRunner(),
    )

    result = pq.read_table(
        _artifact(output, "candidate_results.parquet")
    ).to_pylist()[0]
    assert result["record_status"] == "operator_failure"
    assert result["failure_type"] == "RuntimeError"
    assert result["failure_message"] == ""


def test_candidate_result_requires_present_but_allows_blank_failure_message() -> (
    None
):
    source = "def f():\n    return 1\n"
    operator = OperatorCoordinates(
        name="code_test",
        version="1",
        implementation_hash="0" * 64,
        settings=(("task_key", "task"), ("timeout_seconds", 1.0)),
    )
    arguments = {
        "task_id": _TASK_ID,
        "task_identity": "1" * 64,
        "cleaned_source": source,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "question_identity_hash": operator.question_identity_hash(
            on_key="candidate"
        ),
        "operator_name": "code_test",
        "operator_version": "1",
        "trace_producer": TraceProducer(
            producer_id="preprocessing",
            version="1",
            definition_hash="2" * 64,
            preprocessing_config_hash="3" * 64,
            implementation_hash="4" * 64,
        ),
        "operator": operator,
        "metric_extraction_config_identity": "3" * 64,
        "evaluation_procedure_config_identity": "4" * 64,
        "runtime_identity": "5" * 64,
        "runner_identity": "test:runner",
        "metrics_profile": "test@1",
        "record_status": "operator_failure",
        "failure_type": "RuntimeError",
        "failure_message": "",
        "facts": {field: None for field in CANDIDATE_RESULT_FACT_FIELDS},
    }

    result = canonical_candidate_result(**arguments)
    assert result["failure_message"] == ""

    with pytest.raises(
        CandidateEvaluationContractError,
        match="failure type and message",
    ):
        canonical_candidate_result(**{**arguments, "failure_message": None})
    with pytest.raises(
        CandidateEvaluationContractError,
        match="failure type and message",
    ):
        canonical_candidate_result(**{**arguments, "failure_type": "  "})


def test_recovery_preserves_exhausted_abandoned_attempt_budget(
    tmp_path: Path,
) -> None:
    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    output = tmp_path / "abandoned"
    _evaluate(
        run=run,
        corpus=corpus,
        snapshot=snapshot,
        output=output,
        runner=_CountingRunner(),
    )
    with sqlite3.connect(output / STATE_FILENAME) as connection:
        connection.execute(
            """UPDATE work
                  SET status = 'running', attempt_count = 3,
                      record_status = NULL,
                      failure_type = 'PriorInfrastructureFailure',
                      failure_message = 'prior failure',
                      values_json = NULL, completed_at = NULL,
                      owner_lease_id = 'abandoned',
                      result_evidence_sha256 = NULL"""
        )

    resumed = _CountingRunner(forbid=True)
    _evaluate(
        run=run,
        corpus=corpus,
        snapshot=snapshot,
        output=output,
        runner=resumed,
    )
    row = pq.read_table(
        _artifact(output, "candidate_results.parquet")
    ).to_pylist()[0]
    assert resumed.calls == 0
    assert row["record_status"] == "infrastructure_failure"
    with sqlite3.connect(output / STATE_FILENAME) as connection:
        assert connection.execute(
            "SELECT attempt_count FROM work"
        ).fetchone() == (3,)


def test_hash_authenticated_reuse_executes_no_candidate_work(
    tmp_path: Path,
) -> None:
    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    source = tmp_path / "source"
    _evaluate(
        run=run,
        corpus=corpus,
        snapshot=snapshot,
        output=source,
        runner=_CountingRunner(),
    )
    target_runner = _CountingRunner(forbid=True)
    target = tmp_path / "target"
    _evaluate(
        run=run,
        corpus=corpus,
        snapshot=snapshot,
        output=target,
        runner=target_runner,
        reuse=(source,),
    )

    assert target_runner.calls == 0
    manifest = json.loads(
        _artifact(target, "candidate_evaluation_manifest.json").read_text()
    )
    assert manifest["reused_result_rows"] == 1
    provenance = manifest["reused_result_rows_by_source"][0]
    assert provenance["reused_result_rows"] == 1
    assert len(provenance["manifest_sha256"]) == 64
    assert len(provenance["candidate_results_sha256"]) == 64


def test_reuse_imports_the_exact_captured_bytes_after_source_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dr_code.corpus import candidate_evaluation

    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    source = tmp_path / "source"
    _evaluate(
        run=run,
        corpus=corpus,
        snapshot=snapshot,
        output=source,
        runner=_CountingRunner(),
    )
    original = candidate_evaluation._load_reuse_sources

    def mutate_after_validation(*args: object, **kwargs: object):
        result = original(*args, **kwargs)
        live_results = _artifact(source, "candidate_results.parquet")
        live_results.write_bytes(live_results.read_bytes() + b"mutated")
        return result

    monkeypatch.setattr(
        candidate_evaluation,
        "_load_reuse_sources",
        mutate_after_validation,
    )
    runner = _CountingRunner(forbid=True)
    target = tmp_path / "target"
    _evaluate(
        run=run,
        corpus=corpus,
        snapshot=snapshot,
        output=target,
        runner=runner,
        reuse=(source,),
    )

    assert runner.calls == 0
    assert (
        pq.read_table(_artifact(target, "candidate_results.parquet")).num_rows
        == 1
    )


def test_reuse_rejects_generation_mutated_between_resolution_and_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dr_code.corpus import candidate_evaluation

    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    source = tmp_path / "source"
    _evaluate(
        run=run,
        corpus=corpus,
        snapshot=snapshot,
        output=source,
        runner=_CountingRunner(),
    )
    original_stable_files = candidate_evaluation.stable_files
    reuse_loader_called = False

    @contextmanager
    def mutate_before_capture(paths):
        results = _artifact(source, "candidate_results.parquet")
        results.write_bytes(results.read_bytes() + b"mutated-before-capture")
        with original_stable_files(paths) as captured:
            yield captured

    def forbidden_reuse_loader(*args: object, **kwargs: object):
        nonlocal reuse_loader_called
        reuse_loader_called = True
        raise AssertionError("reuse loading must not begin")

    monkeypatch.setattr(
        candidate_evaluation, "stable_files", mutate_before_capture
    )
    monkeypatch.setattr(
        candidate_evaluation, "_load_reuse_sources", forbidden_reuse_loader
    )
    runner = _CountingRunner(forbid=True)
    target = tmp_path / "target"

    with pytest.raises(
        CandidateEvaluationError,
        match="captured evaluation generation hash",
    ):
        _evaluate(
            run=run,
            corpus=corpus,
            snapshot=snapshot,
            output=target,
            runner=runner,
            reuse=(source,),
        )

    assert runner.calls == 0
    assert reuse_loader_called is False
    assert not target.exists()


def test_corpus_and_snapshot_mutation_after_capture_do_not_change_consumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dr_code.corpus import candidate_evaluation

    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    original = candidate_evaluation._validate_preprocessing_run

    def mutate_live_inputs(*args: object, **kwargs: object):
        corpus.write_bytes(b"corrupt")
        snapshot.write_bytes(b"corrupt")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        candidate_evaluation,
        "_validate_preprocessing_run",
        mutate_live_inputs,
    )
    output = tmp_path / "captured"
    _evaluate(
        run=run,
        corpus=corpus,
        snapshot=snapshot,
        output=output,
        runner=_CountingRunner(),
    )

    assert (
        pq.read_table(_artifact(output, "candidate_results.parquet")).num_rows
        == 1
    )


def test_different_question_settings_change_identity_and_evaluation_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dr_code.corpus import candidate_evaluation

    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    first = tmp_path / "first"
    _evaluate(
        run=run,
        corpus=corpus,
        snapshot=snapshot,
        output=first,
        runner=_CountingRunner(),
    )
    first_manifest = json.loads(
        _artifact(first, "candidate_evaluation_manifest.json").read_text()
    )
    first_key = pq.read_table(
        _artifact(first, "candidate_results.parquet")
    ).to_pylist()[0]["evaluation_key"]
    original = candidate_evaluation.humaneval_metric_definition

    def alternate_definition():
        definition = original()
        question = definition.questions[0].model_copy(
            update={
                "settings": (
                    ("task_key", "task"),
                    ("timeout_seconds", 3.0),
                )
            }
        )
        return definition.model_copy(update={"questions": (question,)})

    monkeypatch.setattr(
        candidate_evaluation,
        "humaneval_metric_definition",
        alternate_definition,
    )
    second = tmp_path / "second"
    _evaluate(
        run=run,
        corpus=corpus,
        snapshot=snapshot,
        output=second,
        runner=_CountingRunner(),
    )
    second_manifest = json.loads(
        _artifact(second, "candidate_evaluation_manifest.json").read_text()
    )
    second_key = pq.read_table(
        _artifact(second, "candidate_results.parquet")
    ).to_pylist()[0]["evaluation_key"]

    assert (
        second_manifest["question_identity_hash"]
        != (first_manifest["question_identity_hash"])
    )
    assert (
        second_manifest["evaluation_identity"]
        != (first_manifest["evaluation_identity"])
    )
    assert second_key != first_key


def test_forged_question_coordinates_reject_before_candidate_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dr_code.corpus import candidate_evaluation

    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    preprocessing = HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION.materialize()
    forged = replace(
        candidate_evaluation._evaluation_config(preprocessing),
        question_identity_hash="0" * 64,
    )
    monkeypatch.setattr(
        candidate_evaluation,
        "_evaluation_config",
        lambda _preprocessing: forged,
    )
    output = tmp_path / "forged"
    runner = _CountingRunner(forbid=True)

    with pytest.raises(CandidateEvaluationError, match="forged"):
        _evaluate(
            run=run,
            corpus=corpus,
            snapshot=snapshot,
            output=output,
            runner=runner,
        )

    assert runner.calls == 0
    assert not output.exists()


@pytest.mark.parametrize("relationship", ["equal", "child", "ancestor"])
def test_output_overlap_with_preprocessing_run_rejects_without_mutation(
    tmp_path: Path,
    relationship: str,
) -> None:
    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    output = {
        "equal": run,
        "child": run / "evaluation",
        "ancestor": run.parent,
    }[relationship]
    before = _tree_contents(run)
    runner = _CountingRunner(forbid=True)

    with pytest.raises(CandidateEvaluationError, match="overlaps.*run"):
        _evaluate(
            run=run,
            corpus=corpus,
            snapshot=snapshot,
            output=output,
            runner=runner,
        )

    assert runner.calls == 0
    assert _tree_contents(run) == before


@pytest.mark.parametrize(
    "relationship",
    [
        "equal_root",
        "child_root",
        "ancestor_of_multiple_roots",
        "equal_generation",
        "child_generation",
    ],
)
def test_output_overlap_with_reuse_roots_rejects_without_mutation(
    tmp_path: Path,
    relationship: str,
) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    run, corpus, snapshot = _inputs(inputs, ["one"])
    reuse_parent = tmp_path / "reuse"
    first = reuse_parent / "first"
    second = reuse_parent / "second"
    _evaluate(
        run=run,
        corpus=corpus,
        snapshot=snapshot,
        output=first,
        runner=_CountingRunner(),
    )
    _evaluate(
        run=run,
        corpus=corpus,
        snapshot=snapshot,
        output=second,
        runner=_CountingRunner(),
    )
    generation = resolve_current_generation(first).generation_dir
    output = {
        "equal_root": first,
        "child_root": first / "nested",
        "ancestor_of_multiple_roots": reuse_parent,
        "equal_generation": generation,
        "child_generation": generation / "nested",
    }[relationship]
    before = _tree_contents(reuse_parent)
    runner = _CountingRunner(forbid=True)

    with pytest.raises(CandidateEvaluationError, match="overlaps.*reuse"):
        _evaluate(
            run=run,
            corpus=corpus,
            snapshot=snapshot,
            output=output,
            runner=runner,
            reuse=(first, second),
        )

    assert runner.calls == 0
    assert _tree_contents(reuse_parent) == before


def test_flat_only_evaluation_root_rejects_before_state_mutation(
    tmp_path: Path,
) -> None:
    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    output = tmp_path / "flat"
    output.mkdir()
    sentinel = output / "candidate_results.parquet"
    sentinel.write_bytes(b"legacy-flat")

    with pytest.raises(CandidateEvaluationError, match="flat"):
        _evaluate(
            run=run,
            corpus=corpus,
            snapshot=snapshot,
            output=output,
            runner=_CountingRunner(forbid=True),
        )

    assert sentinel.read_bytes() == b"legacy-flat"
    assert not (output / STATE_FILENAME).exists()


def test_dangling_evaluation_root_symlink_rejects_before_state_mutation(
    tmp_path: Path,
) -> None:
    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    target = tmp_path / "missing-evaluation"
    output = tmp_path / "evaluation"
    output.symlink_to(target, target_is_directory=True)
    runner = _CountingRunner(forbid=True)

    with pytest.raises(CandidateEvaluationError, match="symlink"):
        _evaluate(
            run=run,
            corpus=corpus,
            snapshot=snapshot,
            output=output,
            runner=runner,
        )

    assert runner.calls == 0
    assert output.is_symlink()
    assert not target.exists()


@pytest.mark.parametrize(
    ("reserved_name", "directory_target"),
    [
        (CURRENT_FILENAME, False),
        (GENERATIONS_DIRECTORY, True),
        (STATE_FILENAME, False),
        (MANIFEST_FILENAME, False),
        (MEMBERSHIP_FILENAME, False),
        (RESULTS_FILENAME, False),
    ],
)
def test_reserved_evaluation_child_symlink_rejects_before_external_mutation(
    tmp_path: Path,
    reserved_name: str,
    directory_target: bool,
) -> None:
    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    output = tmp_path / "evaluation"
    output.mkdir()
    target = tmp_path / f"external-{reserved_name}"
    if directory_target:
        target.mkdir()
        marker = target / "marker"
    else:
        marker = target
    marker.write_bytes(b"unchanged")
    (output / reserved_name).symlink_to(
        target,
        target_is_directory=directory_target,
    )

    with pytest.raises(CandidateEvaluationError, match="symlink"):
        _evaluate(
            run=run,
            corpus=corpus,
            snapshot=snapshot,
            output=output,
            runner=_CountingRunner(forbid=True),
        )

    assert marker.read_bytes() == b"unchanged"
    assert (output / reserved_name).is_symlink()


def test_evaluation_output_parent_symlink_alias_rejects_before_mutation(
    tmp_path: Path,
) -> None:
    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    external = tmp_path / "external"
    external.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(external, target_is_directory=True)

    with pytest.raises(CandidateEvaluationError, match="symlink"):
        _evaluate(
            run=run,
            corpus=corpus,
            snapshot=snapshot,
            output=alias / "evaluation",
            runner=_CountingRunner(forbid=True),
        )

    assert not (external / "evaluation").exists()


def test_installed_package_source_digest_drift_rejects_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dr_code.corpus import candidate_evaluation

    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    output = tmp_path / "evaluation"
    _evaluate(
        run=run,
        corpus=corpus,
        snapshot=snapshot,
        output=output,
        runner=_CountingRunner(),
    )
    before = json.loads(
        _artifact(output, "candidate_evaluation_manifest.json").read_text()
    )
    monkeypatch.setattr(
        candidate_evaluation,
        "package_source_digest",
        lambda: "0" * 64,
    )
    resumed = _CountingRunner(forbid=True)

    with pytest.raises(CandidateEvaluationError, match="incompatible"):
        _evaluate(
            run=run,
            corpus=corpus,
            snapshot=snapshot,
            output=output,
            runner=resumed,
        )

    assert resumed.calls == 0
    assert before["trusted_source_sha256"] != {
        "dr_code_python_package": "0" * 64
    }


@pytest.mark.parametrize(
    ("source_evidence", "message"),
    [
        (lambda: "", "invalid"),
        (_unreadable_source_evidence, "unavailable"),
    ],
)
def test_installed_source_evidence_failure_rejects_before_output_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_evidence: Callable[[], str],
    message: str,
) -> None:
    from dr_code.corpus import candidate_evaluation

    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    output = tmp_path / "evaluation"
    runner = _CountingRunner(forbid=True)
    monkeypatch.setattr(
        candidate_evaluation,
        "package_source_digest",
        source_evidence,
    )

    with pytest.raises(CandidateEvaluationError, match=message):
        _evaluate(
            run=run,
            corpus=corpus,
            snapshot=snapshot,
            output=output,
            runner=runner,
        )

    assert runner.calls == 0
    assert not output.exists()


def test_reuse_rejects_result_bytes_that_do_not_match_manifest(
    tmp_path: Path,
) -> None:
    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    source = tmp_path / "source"
    _evaluate(
        run=run,
        corpus=corpus,
        snapshot=snapshot,
        output=source,
        runner=_CountingRunner(),
    )
    results_path = _artifact(source, "candidate_results.parquet")
    results_path.write_bytes(results_path.read_bytes() + b"corrupt")

    with pytest.raises(CandidateEvaluationError, match="generation hash"):
        _evaluate(
            run=run,
            corpus=corpus,
            snapshot=snapshot,
            output=tmp_path / "target",
            runner=_CountingRunner(forbid=True),
            reuse=(source,),
        )


def test_reuse_rejects_forged_evaluation_identity(tmp_path: Path) -> None:
    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    source = tmp_path / "source"
    _evaluate(
        run=run,
        corpus=corpus,
        snapshot=snapshot,
        output=source,
        runner=_CountingRunner(),
    )
    manifest_path = _artifact(source, "candidate_evaluation_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["evaluation_identity"] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    with pytest.raises(CandidateEvaluationError, match="generation hash"):
        _evaluate(
            run=run,
            corpus=corpus,
            snapshot=snapshot,
            output=tmp_path / "target",
            runner=_CountingRunner(forbid=True),
            reuse=(source,),
        )


def test_stale_running_work_is_recovered_on_resume(tmp_path: Path) -> None:
    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    output = tmp_path / "stale"
    _evaluate(
        run=run,
        corpus=corpus,
        snapshot=snapshot,
        output=output,
        runner=_CountingRunner(),
    )
    connection = sqlite3.connect(output / STATE_FILENAME)
    connection.execute(
        """UPDATE work
              SET status = 'running', record_status = NULL,
                  failure_type = NULL, failure_message = NULL,
                  values_json = NULL, completed_at = NULL,
                  result_evidence_sha256 = NULL"""
    )
    connection.commit()
    connection.close()

    resumed = _CountingRunner()
    _evaluate(
        run=run,
        corpus=corpus,
        snapshot=snapshot,
        output=output,
        runner=resumed,
    )

    assert resumed.calls == 1
    row = pq.read_table(
        _artifact(output, "candidate_results.parquet")
    ).to_pylist()[0]
    assert row["record_status"] == "measured"


def test_live_owner_lease_blocks_concurrent_resume(tmp_path: Path) -> None:
    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    output = tmp_path / "owned"
    _evaluate(
        run=run,
        corpus=corpus,
        snapshot=snapshot,
        output=output,
        runner=_CountingRunner(),
    )
    connection = sqlite3.connect(output / STATE_FILENAME)
    connection.execute(
        """INSERT INTO evaluator_lease(singleton, lease_id, heartbeat_at)
           VALUES (1, 'other', ?)""",
        (time.time(),),
    )
    connection.commit()
    connection.close()

    with pytest.raises(CandidateEvaluationError, match="live evaluator"):
        _evaluate(
            run=run,
            corpus=corpus,
            snapshot=snapshot,
            output=output,
            runner=_CountingRunner(),
        )


def test_production_preflight_holds_no_expiring_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dr_code.corpus import candidate_evaluation

    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    output = tmp_path / "production"
    first_preflight_entered = threading.Event()
    release_first_preflight = threading.Event()
    candidate_started = threading.Event()
    release_candidate = threading.Event()
    preflight_calls = 0
    preflight_guard = threading.Lock()
    clock = [100.0]

    def coordinated_preflight(**_kwargs: object) -> None:
        nonlocal preflight_calls
        with preflight_guard:
            preflight_calls += 1
            call = preflight_calls
        if call == 1:
            clock[0] += candidate_evaluation._LEASE_SECONDS + 1
            first_preflight_entered.set()
            assert release_first_preflight.wait(timeout=10)

    def blocking_runner(
        *, source: str, input_text: str, timeout_seconds: float
    ):
        candidate_started.set()
        assert release_candidate.wait(timeout=10)
        return run_python_subprocess(
            source=source,
            input_text=input_text,
            timeout_seconds=timeout_seconds,
        )

    monkeypatch.setattr(
        candidate_evaluation, "_preflight_production", coordinated_preflight
    )
    monkeypatch.setattr(
        candidate_evaluation, "run_python_subprocess", blocking_runner
    )
    monkeypatch.setattr(candidate_evaluation.time, "time", lambda: clock[0])

    def evaluate() -> None:
        evaluate_preprocessing_candidates(
            preprocessing_run=run,
            corpus_path=corpus,
            output_dir=output,
            snapshot_path=snapshot,
            max_workers=1,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(evaluate)
        assert first_preflight_entered.wait(timeout=10)
        assert not output.exists()
        second = executor.submit(evaluate)
        assert candidate_started.wait(timeout=10)
        release_first_preflight.set()
        with pytest.raises(CandidateEvaluationError, match="live evaluator"):
            first.result(timeout=10)
        release_candidate.set()
        second.result(timeout=10)

    assert (
        pq.read_table(_artifact(output, "candidate_results.parquet")).num_rows
        == 1
    )


def test_displaced_lease_owner_cannot_complete_running_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dr_code.corpus import candidate_evaluation

    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    output = tmp_path / "displaced"
    _evaluate(
        run=run,
        corpus=corpus,
        snapshot=snapshot,
        output=output,
        runner=_CountingRunner(),
    )
    state = output / STATE_FILENAME
    with sqlite3.connect(state) as connection:
        connection.execute(
            """UPDATE work
                  SET status = 'pending', record_status = NULL,
                      values_json = NULL, completed_at = NULL,
                      result_evidence_sha256 = NULL"""
        )

    clock = [100.0]
    monkeypatch.setattr(candidate_evaluation.time, "time", lambda: clock[0])
    first = candidate_evaluation._open_state(state)
    second = candidate_evaluation._open_state(state)
    try:
        candidate_evaluation._acquire_lease(first, "first")
        claimed_first = candidate_evaluation._claim_next_work(first, "first")
        assert claimed_first is not None

        clock[0] += candidate_evaluation._LEASE_SECONDS + 1
        candidate_evaluation._acquire_lease(second, "second")
        candidate_evaluation._recover_work(
            second,
            lease_id="second",
            max_infrastructure_retries=2,
        )
        claimed_second = candidate_evaluation._claim_next_work(
            second, "second"
        )
        assert claimed_second == claimed_first

        with pytest.raises(CandidateEvaluationError, match="lease was lost"):
            candidate_evaluation._complete_work(
                first,
                claimed_first.evaluation_key,
                None,
                SubprocessInfrastructureError("late result"),
                lease_id="first",
                max_infrastructure_retries=0,
            )
        owner = second.execute(
            """SELECT owner_lease_id FROM work
                WHERE evaluation_key = ?""",
            (claimed_second.evaluation_key,),
        ).fetchone()
        assert owner == ("second",)
    finally:
        candidate_evaluation._release_lease(first, "first")
        candidate_evaluation._release_lease(second, "second")
        first.close()
        second.close()


def test_displaced_exporter_cannot_publish_terminal_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dr_code.corpus import candidate_evaluation

    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    output = tmp_path / "export-race"
    export_blocked = threading.Event()
    release_export = threading.Event()
    maintenance_waiting = threading.Event()
    allow_maintenance = threading.Event()
    write_guard = threading.Lock()
    wait_guard = threading.Lock()
    write_calls = 0
    maintenance_thread: list[int] = []
    evaluator_threads: dict[str, int] = {}
    staging_paths: list[tuple[int, Path]] = []
    first_staging_path: list[Path] = []
    clock = [100.0]
    original_write = candidate_evaluation._write_query_parquet
    original_wait = candidate_evaluation._wait_for_lease_heartbeat

    def blocking_first_export(*args: object, **kwargs: object) -> None:
        nonlocal write_calls
        path = Path(str(args[0]))
        staging_paths.append((threading.get_ident(), path))
        with write_guard:
            write_calls += 1
            call = write_calls
        if call == 1:
            path.write_bytes(b"live first-exporter temp")
            first_staging_path.append(path)
            export_blocked.set()
            assert release_export.wait(timeout=10)
            path.unlink()
        original_write(*args, **kwargs)

    def controlled_heartbeat_wait(stopped: threading.Event) -> bool:
        thread = threading.get_ident()
        with wait_guard:
            if not maintenance_thread:
                maintenance_thread.append(thread)
            is_first_exporter = thread == maintenance_thread[0]
        if is_first_exporter:
            maintenance_waiting.set()
            assert allow_maintenance.wait(timeout=10)
            return stopped.is_set()
        return original_wait(stopped)

    monkeypatch.setattr(
        candidate_evaluation, "_write_query_parquet", blocking_first_export
    )
    monkeypatch.setattr(
        candidate_evaluation,
        "_wait_for_lease_heartbeat",
        controlled_heartbeat_wait,
    )
    monkeypatch.setattr(candidate_evaluation.time, "time", lambda: clock[0])

    def evaluate(label: str) -> None:
        evaluator_threads[label] = threading.get_ident()
        _evaluate(
            run=run,
            corpus=corpus,
            snapshot=snapshot,
            output=output,
            runner=_CountingRunner(),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(evaluate, "first")
        assert export_blocked.wait(timeout=10)
        assert maintenance_waiting.wait(timeout=10)
        clock[0] += candidate_evaluation._LEASE_SECONDS + 1
        allow_maintenance.set()
        second = executor.submit(evaluate, "second")
        second.result(timeout=10)
        assert first_staging_path[0].read_bytes() == (
            b"live first-exporter temp"
        )
        release_export.set()
        with pytest.raises(CandidateEvaluationError, match="lease was lost"):
            first.result(timeout=10)

    first_temps = {
        path
        for thread, path in staging_paths
        if thread == evaluator_threads["first"]
    }
    second_temps = {
        path
        for thread, path in staging_paths
        if thread == evaluator_threads["second"]
    }
    assert first_temps.isdisjoint(second_temps)
    assert not list(output.glob(".CURRENT.json.*.tmp"))
    assert (
        pq.read_table(
            _artifact(output, "candidate_membership.parquet")
        ).num_rows
        == 1
    )
    assert (
        pq.read_table(_artifact(output, "candidate_results.parquet")).num_rows
        == 1
    )
    manifest = json.loads(
        _artifact(output, "candidate_evaluation_manifest.json").read_text()
    )
    assert manifest["complete"] is True


def test_durable_current_switch_is_success_even_if_clock_expires_afterward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dr_code.corpus import candidate_evaluation

    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    output = tmp_path / "terminal-success"
    clock = [100.0]
    original_publish = candidate_evaluation.publish_staged_current_switch

    def publish_then_expire(staged):
        generation = original_publish(staged)
        clock[0] += candidate_evaluation._LEASE_SECONDS + 1
        return generation

    monkeypatch.setattr(candidate_evaluation.time, "time", lambda: clock[0])
    monkeypatch.setattr(
        candidate_evaluation,
        "publish_staged_current_switch",
        publish_then_expire,
    )

    _evaluate(
        run=run,
        corpus=corpus,
        snapshot=snapshot,
        output=output,
        runner=_CountingRunner(),
    )

    generation = resolve_current_generation(output)
    assert generation.manifest_path.is_file()


def test_generation_hash_and_fsync_publication_keeps_heartbeat_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dr_code.corpus import candidate_evaluation

    run, corpus, snapshot = _inputs(tmp_path, ["one"])
    output = tmp_path / "heartbeat-during-publish"
    publication_entered = threading.Event()
    heartbeat_seen = threading.Event()
    heartbeat_wait_calls = 0
    original_publish = candidate_evaluation.publish_generation_directory
    original_heartbeat = candidate_evaluation._heartbeat

    def observe_heartbeat(connection, lease_id: str) -> None:
        original_heartbeat(connection, lease_id)
        if threading.current_thread().name.startswith(
            "candidate-evaluation-heartbeat-"
        ):
            heartbeat_seen.set()

    def publish_after_heartbeat(root: Path, staging: Path):
        publication_entered.set()
        assert heartbeat_seen.wait(timeout=10)
        return original_publish(root, staging)

    def controlled_heartbeat_wait(stopped: threading.Event) -> bool:
        nonlocal heartbeat_wait_calls
        heartbeat_wait_calls += 1
        if heartbeat_wait_calls == 1:
            assert publication_entered.wait(timeout=10)
            return False
        assert heartbeat_wait_calls == 2
        return stopped.wait()

    monkeypatch.setattr(candidate_evaluation, "_heartbeat", observe_heartbeat)
    monkeypatch.setattr(
        candidate_evaluation,
        "_wait_for_lease_heartbeat",
        controlled_heartbeat_wait,
    )
    monkeypatch.setattr(
        candidate_evaluation,
        "publish_generation_directory",
        publish_after_heartbeat,
    )

    _evaluate(
        run=run,
        corpus=corpus,
        snapshot=snapshot,
        output=output,
        runner=_CountingRunner(),
    )

    assert heartbeat_seen.is_set()
    assert heartbeat_wait_calls == 2
