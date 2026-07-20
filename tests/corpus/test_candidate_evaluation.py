"""Contract tests for resumable candidate evaluation state and exports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dr_code.corpus.candidate_evaluation import (
    CandidateEvaluationError,
    evaluate_preprocessing_candidates,
)
from dr_code.corpus.preprocessing_artifacts import (
    CANDIDATES_SCHEMA,
    RESULTS_SCHEMA,
)
from dr_code.humaneval.sandbox import SandboxCompletedProcess
from dr_code.humaneval.sandbox import SandboxError
from dr_code.humaneval.sampling import write_human_eval_snapshot_rows
from dr_code.preprocessing.steps.dedupe_candidates import (
    candidate_id_for_source,
)


def _snapshot(path: Path) -> Path:
    return write_human_eval_snapshot_rows(
        [
            {
                "task_id": "HumanEval/fixture",
                "prompt": "def add_one(x):\n",
                "canonical_solution": "    return x + 1\n",
                "entry_point": "add_one",
                "test": (
                    "def check(candidate):\n"
                    "    inputs = [(1,)]\n"
                    "    results = [2]\n"
                    "    for inp, expected in zip(inputs, results):\n"
                    "        assertion(candidate(*inp), expected)\n"
                ),
            }
        ],
        snapshot_path=path,
    )


def _corpus(path: Path, sample_ids: list[str]) -> Path:
    pq.write_table(
        pa.table(
            {
                "sample_id": sample_ids,
                "task_id": ["HumanEval/fixture"] * len(sample_ids),
                "source_kind": ["test-fixture"] * len(sample_ids),
            }
        ),
        path,
    )
    return path


def _run(path: Path, sample_ids: list[str], source: str) -> Path:
    path.mkdir()
    (path / "manifest.json").write_text(
        '{"complete":true}\n', encoding="utf-8"
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "sample_id": sample_id,
                    "decoder_output_presence": "present",
                    "raw_output_sha256": None,
                    "outcome": "function_candidates_extracted",
                    "outcome_code": "function_candidates_extracted",
                    "failure_code": None,
                    "failed_step": None,
                    "cause": None,
                    "propagated_through": None,
                    "final_candidate_count": 1,
                }
                for sample_id in sample_ids
            ],
            schema=RESULTS_SCHEMA,
        ),
        path / "results.parquet",
    )
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "sample_id": sample_id,
                    "candidate_index": 0,
                    "candidate_id": candidate_id_for_source(source),
                    "cleaned_source": source,
                    "source_sha256": digest,
                    "origins": [],
                    "parse_ok": True,
                    "parse_error": None,
                    "compile_ok": True,
                    "compile_error": None,
                    "compile_warnings": [],
                    "top_level_function_count": 1,
                    "top_level_function_names": ["add_one"],
                    "top_level_async_function_names": [],
                }
                for index, sample_id in enumerate(sample_ids)
            ],
            schema=CANDIDATES_SCHEMA,
        ),
        path / "candidates.parquet",
    )
    return path


class _PassingRunner:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self, *, source: str, input_json: str, timeout_seconds: float
    ) -> SandboxCompletedProcess:
        del source, timeout_seconds
        self.calls += 1
        payload = json.loads(input_json)
        return SandboxCompletedProcess(
            returncode=0,
            stdout=json.dumps(
                [
                    {"case_id": check["case_id"], "status": "passed"}
                    for check in payload["checks"]
                ]
            ),
            stderr="",
        )


def test_deduplicates_work_persists_membership_and_resume_is_zero_reruns(
    tmp_path: Path,
) -> None:
    source = "def add_one(x):\n    return x + 1\n"
    sample_ids = ["sample-a", "sample-b"]
    runner = _PassingRunner()
    kwargs = {
        "preprocessing_run": _run(tmp_path / "run", sample_ids, source),
        "corpus_path": _corpus(tmp_path / "corpus.parquet", sample_ids),
        "output_dir": tmp_path / "evaluation",
        "snapshot_path": _snapshot(tmp_path / "snapshot.json"),
        "max_workers": 2,
        "run_in_sandbox": runner,
        "runner_identity": "test-passing-runner@v1",
    }

    artifacts = evaluate_preprocessing_candidates(**kwargs)

    assert runner.calls == 1
    membership = pq.read_table(artifacts.membership_path).to_pylist()
    results = pq.read_table(artifacts.results_path).to_pylist()
    assert len(membership) == 2
    assert "source_kind" in pq.read_schema(artifacts.membership_path).names
    assert "source_kind" not in pq.read_schema(artifacts.results_path).names
    assert len({row["evaluation_key"] for row in membership}) == 1
    assert results[0]["outcome"] == "passed"
    assert results[0]["total_cases"] == 1
    first_membership = artifacts.membership_path.read_bytes()
    first_results = artifacts.results_path.read_bytes()

    evaluate_preprocessing_candidates(**kwargs)

    assert runner.calls == 1
    assert artifacts.membership_path.read_bytes() == first_membership
    assert artifacts.results_path.read_bytes() == first_results


def test_rejects_candidate_source_hash_corruption(tmp_path: Path) -> None:
    source = "def add_one(x):\n    return x + 1\n"
    run = _run(tmp_path / "run", ["sample-a"], source)
    table = pq.read_table(run / "candidates.parquet")
    rows = table.to_pylist()
    rows[0]["source_sha256"] = "not-a-digest"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=CANDIDATES_SCHEMA),
        run / "candidates.parquet",
    )

    with pytest.raises(CandidateEvaluationError, match="source_sha256"):
        evaluate_preprocessing_candidates(
            preprocessing_run=run,
            corpus_path=_corpus(tmp_path / "corpus.parquet", ["sample-a"]),
            output_dir=tmp_path / "evaluation",
            snapshot_path=_snapshot(tmp_path / "snapshot.json"),
            run_in_sandbox=_PassingRunner(),
            runner_identity="test-passing-runner@v1",
        )


def test_rejects_preprocessing_result_membership_mismatch(
    tmp_path: Path,
) -> None:
    source = "def add_one(x):\n    return x + 1\n"
    run = _run(tmp_path / "run", ["sample-a", "sample-b"], source)

    with pytest.raises(CandidateEvaluationError, match="exactly match"):
        evaluate_preprocessing_candidates(
            preprocessing_run=run,
            corpus_path=_corpus(tmp_path / "corpus.parquet", ["sample-a"]),
            output_dir=tmp_path / "evaluation",
            snapshot_path=_snapshot(tmp_path / "snapshot.json"),
            run_in_sandbox=_PassingRunner(),
            runner_identity="test-passing-runner@v1",
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("candidate_index", 1, "final_candidate_count"),
        ("candidate_id", "candidate-not-content-derived", "content-derived"),
    ],
)
def test_rejects_candidate_index_and_identity_corruption(
    tmp_path: Path, field: str, value: object, error: str
) -> None:
    source = "def add_one(x):\n    return x + 1\n"
    run = _run(tmp_path / "run", ["sample-a"], source)
    table = pq.read_table(run / "candidates.parquet")
    rows = table.to_pylist()
    rows[0][field] = value
    pq.write_table(
        pa.Table.from_pylist(rows, schema=CANDIDATES_SCHEMA),
        run / "candidates.parquet",
    )

    with pytest.raises(CandidateEvaluationError, match=error):
        evaluate_preprocessing_candidates(
            preprocessing_run=run,
            corpus_path=_corpus(tmp_path / "corpus.parquet", ["sample-a"]),
            output_dir=tmp_path / "evaluation",
            snapshot_path=_snapshot(tmp_path / "snapshot.json"),
            run_in_sandbox=_PassingRunner(),
            runner_identity="test-passing-runner@v1",
        )


def test_rejects_partial_candidate_export_count(tmp_path: Path) -> None:
    source = "def add_one(x):\n    return x + 1\n"
    run = _run(tmp_path / "run", ["sample-a"], source)
    results = pq.read_table(run / "results.parquet")
    rows = results.to_pylist()
    rows[0]["final_candidate_count"] = 2
    pq.write_table(
        pa.Table.from_pylist(rows, schema=RESULTS_SCHEMA),
        run / "results.parquet",
    )

    with pytest.raises(
        CandidateEvaluationError, match="final_candidate_count"
    ):
        evaluate_preprocessing_candidates(
            preprocessing_run=run,
            corpus_path=_corpus(tmp_path / "corpus.parquet", ["sample-a"]),
            output_dir=tmp_path / "evaluation",
            snapshot_path=_snapshot(tmp_path / "snapshot.json"),
            run_in_sandbox=_PassingRunner(),
            runner_identity="test-passing-runner@v1",
        )


def test_injected_runner_requires_stable_identity(tmp_path: Path) -> None:
    with pytest.raises(CandidateEvaluationError, match="runner_identity"):
        evaluate_preprocessing_candidates(
            preprocessing_run=_run(
                tmp_path / "run",
                ["sample-a"],
                "def add_one(x):\n    return x + 1\n",
            ),
            corpus_path=_corpus(tmp_path / "corpus.parquet", ["sample-a"]),
            output_dir=tmp_path / "evaluation",
            snapshot_path=_snapshot(tmp_path / "snapshot.json"),
            run_in_sandbox=_PassingRunner(),
        )


def test_production_requires_explicit_sandbox_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DR_CODE_SANDBOX_IMAGE", raising=False)

    with pytest.raises(
        CandidateEvaluationError, match="DR_CODE_SANDBOX_IMAGE"
    ):
        evaluate_preprocessing_candidates(
            preprocessing_run=tmp_path / "missing-run",
            corpus_path=tmp_path / "missing-corpus.parquet",
            output_dir=tmp_path / "evaluation",
            snapshot_path=tmp_path / "missing-snapshot.json",
        )


def test_production_requires_image_and_preflights_before_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dr_code.corpus.candidate_evaluation as evaluation

    source = "def add_one(x):\n    return x + 1\n"
    called: list[int] = []
    monkeypatch.setenv("DR_CODE_SANDBOX_IMAGE", "test@sha256:" + "0" * 64)
    monkeypatch.setattr(
        evaluation,
        "_preflight_production",
        lambda tasks: called.append(len(tasks)),
    )
    monkeypatch.setattr(evaluation, "run_python_in_sandbox", _PassingRunner())

    evaluate_preprocessing_candidates(
        preprocessing_run=_run(tmp_path / "run", ["sample-a"], source),
        corpus_path=_corpus(tmp_path / "corpus.parquet", ["sample-a"]),
        output_dir=tmp_path / "evaluation",
        snapshot_path=_snapshot(tmp_path / "snapshot.json"),
    )

    assert called == [1]


def test_production_preflight_uses_a_separate_generous_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dr_code.corpus.candidate_evaluation as evaluation

    tasks = evaluation._load_tasks(_snapshot(tmp_path / "snapshot.json"))
    observed_timeouts: list[float] = []
    runner = _PassingRunner()

    def observe_timeout(
        *, source: str, input_json: str, timeout_seconds: float
    ) -> SandboxCompletedProcess:
        observed_timeouts.append(timeout_seconds)
        if input_json == "{}":
            return SandboxCompletedProcess(
                returncode=0, stdout="2.2.6\n", stderr=""
            )
        return runner(
            source=source,
            input_json=input_json,
            timeout_seconds=timeout_seconds,
        )

    monkeypatch.setattr(evaluation, "run_python_in_sandbox", observe_timeout)

    evaluation._preflight_production(tasks)

    assert observed_timeouts
    assert set(observed_timeouts) == {
        evaluation._PRODUCTION_PREFLIGHT_TIMEOUT_SECONDS
    }
    assert evaluation._PRODUCTION_PREFLIGHT_TIMEOUT_SECONDS > (
        evaluation.DEFAULT_HUMANEVAL_TIMEOUT_SECONDS
    )


def test_infrastructure_sandbox_error_is_not_reported_as_tests_failed(
    tmp_path: Path,
) -> None:
    source = "def add_one(x):\n    return x + 1\n"

    def unavailable(**_: object) -> SandboxCompletedProcess:
        raise SandboxError("runtime unavailable")

    artifacts = evaluate_preprocessing_candidates(
        preprocessing_run=_run(tmp_path / "run", ["sample-a"], source),
        corpus_path=_corpus(tmp_path / "corpus.parquet", ["sample-a"]),
        output_dir=tmp_path / "evaluation",
        snapshot_path=_snapshot(tmp_path / "snapshot.json"),
        run_in_sandbox=unavailable,
        runner_identity="test-unavailable-runner@v1",
    )

    result = pq.read_table(artifacts.results_path).to_pylist()[0]
    assert result["record_status"] == "infrastructure_failure"
    assert result["failure_type"] == "SandboxError"
    assert result["outcome"] is None

    evaluate_preprocessing_candidates(
        preprocessing_run=tmp_path / "run",
        corpus_path=tmp_path / "corpus.parquet",
        output_dir=tmp_path / "evaluation",
        snapshot_path=tmp_path / "snapshot.json",
        run_in_sandbox=_PassingRunner(),
        runner_identity="test-unavailable-runner@v1",
    )

    retried = pq.read_table(artifacts.results_path).to_pylist()[0]
    assert retried["record_status"] == "measured"
    assert retried["outcome"] == "passed"


def test_timeout_uses_official_timed_out_outcome(tmp_path: Path) -> None:
    source = "def add_one(x):\n    return x + 1\n"

    def timeout(
        *, source: str, input_json: str, timeout_seconds: float
    ) -> SandboxCompletedProcess:
        del source, input_json, timeout_seconds
        from dr_code.humaneval.sandbox import SandboxTimeoutError

        raise SandboxTimeoutError("candidate timed out")

    artifacts = evaluate_preprocessing_candidates(
        preprocessing_run=_run(tmp_path / "run", ["sample-a"], source),
        corpus_path=_corpus(tmp_path / "corpus.parquet", ["sample-a"]),
        output_dir=tmp_path / "evaluation",
        snapshot_path=_snapshot(tmp_path / "snapshot.json"),
        run_in_sandbox=timeout,
        runner_identity="test-timeout-runner@v1",
    )

    result = pq.read_table(artifacts.results_path).to_pylist()[0]
    assert result["record_status"] == "measured"
    assert result["outcome"] == "timed_out"
