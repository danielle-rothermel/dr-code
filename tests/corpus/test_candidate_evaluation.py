"""Contract tests for resumable candidate evaluation state and exports."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dr_code.corpus.candidate_evaluation import (
    CandidateEvaluationError,
    STATE_FILENAME,
    evaluate_preprocessing_candidates,
)
from dr_code.corpus.preprocessing_artifacts import (
    CANDIDATES_SCHEMA,
    RESULTS_SCHEMA,
)
from dr_code.humaneval.sampling import write_human_eval_snapshot_rows
from dr_code.humaneval.subprocess_runner import (
    SubprocessCompletedProcess,
    SubprocessError,
)
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
    ) -> SubprocessCompletedProcess:
        del source, timeout_seconds
        self.calls += 1
        payload = json.loads(input_json)
        return SubprocessCompletedProcess(
            returncode=0,
            stdout=json.dumps(
                [
                    {"case_id": check["case_id"], "status": "passed"}
                    for check in payload["checks"]
                ]
            ),
            stderr="",
        )


def _runtime_coordinates(
    *, release: str = "test-release", numpy_version: str = "2.5.0"
) -> dict[str, object]:
    distributions = [
        {"name": "numpy", "version": numpy_version},
        {"name": "pyarrow", "version": "23.0.0"},
    ]
    distributions_json = json.dumps(
        distributions, sort_keys=True, separators=(",", ":")
    )
    return {
        "platform": {
            "system": "TestOS",
            "release": release,
            "machine": "test-machine",
        },
        "python_executable_sha256": "a" * 64,
        "installed_distributions": distributions,
        "installed_distributions_sha256": hashlib.sha256(
            distributions_json.encode("utf-8")
        ).hexdigest(),
    }


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
        "run_in_subprocess": runner,
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
            run_in_subprocess=_PassingRunner(),
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
            run_in_subprocess=_PassingRunner(),
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
            run_in_subprocess=_PassingRunner(),
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
            run_in_subprocess=_PassingRunner(),
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
            run_in_subprocess=_PassingRunner(),
        )


def test_production_defaults_to_subprocess_and_preflights_before_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dr_code.corpus.candidate_evaluation as evaluation

    source = "def add_one(x):\n    return x + 1\n"
    called: list[tuple[int, object]] = []
    runner = _PassingRunner()
    monkeypatch.setenv("DR_CODE_SANDBOX_RUNTIME", "unavailable-runtime")
    monkeypatch.setenv("DR_CODE_SANDBOX_IMAGE", "unavailable-image")

    def record_preflight(
        tasks: dict[str, object], *, run_in_subprocess: object
    ) -> None:
        called.append((len(tasks), run_in_subprocess))

    monkeypatch.setattr(
        evaluation,
        "_preflight_production",
        record_preflight,
    )
    monkeypatch.setattr(evaluation, "run_python_subprocess", runner)

    evaluate_preprocessing_candidates(
        preprocessing_run=_run(tmp_path / "run", ["sample-a"], source),
        corpus_path=_corpus(tmp_path / "corpus.parquet", ["sample-a"]),
        output_dir=tmp_path / "evaluation",
        snapshot_path=_snapshot(tmp_path / "snapshot.json"),
    )

    assert called == [(1, runner)]
    manifest = json.loads(
        (
            tmp_path / "evaluation" / "candidate_evaluation_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["runner_identity"] == "subprocess:python-isolated@v1"
    assert manifest["sandbox_image"] is None
    assert (
        "dr_code.humaneval.subprocess_runner"
        in manifest["trusted_source_sha256"]
    )
    assert "dr_code.humaneval.sandbox" not in manifest["trusted_source_sha256"]
    host_runtime = manifest["host_runtime"]
    assert set(host_runtime["platform"]) == {"system", "release", "machine"}
    assert len(host_runtime["python_executable_sha256"]) == 64
    assert host_runtime["installed_distributions"] == sorted(
        host_runtime["installed_distributions"],
        key=lambda item: (item["name"], item["version"]),
    )
    distributions = {
        item["name"]: item["version"]
        for item in host_runtime["installed_distributions"]
    }
    assert "numpy" in distributions
    assert len(host_runtime["installed_distributions_sha256"]) == 64


@pytest.mark.parametrize(
    "changed_runtime",
    [
        _runtime_coordinates(release="changed-release"),
        _runtime_coordinates(numpy_version="changed-numpy"),
    ],
    ids=["platform", "dependency"],
)
def test_host_runtime_changes_execution_fingerprint_and_evaluation_key(
    changed_runtime: dict[str, object],
) -> None:
    import dr_code.corpus.candidate_evaluation as evaluation

    definition = evaluation.humaneval_metrics_definition()
    original_fingerprint = evaluation._execution_fingerprint(
        "subprocess:python-isolated@v1",
        host_runtime=_runtime_coordinates(),
    )
    changed_fingerprint = evaluation._execution_fingerprint(
        "subprocess:python-isolated@v1",
        host_runtime=changed_runtime,
    )

    original_key = evaluation._evaluation_key(
        task_id="HumanEval/fixture",
        task_fingerprint="task-fingerprint",
        candidate_source="def add_one(x):\n    return x + 1\n",
        definition=definition,
        execution_fingerprint=original_fingerprint,
    )
    changed_key = evaluation._evaluation_key(
        task_id="HumanEval/fixture",
        task_fingerprint="task-fingerprint",
        candidate_source="def add_one(x):\n    return x + 1\n",
        definition=definition,
        execution_fingerprint=changed_fingerprint,
    )

    assert changed_fingerprint != original_fingerprint
    assert changed_key != original_key


@pytest.mark.parametrize(
    "changed_runtime",
    [
        _runtime_coordinates(release="changed-release"),
        _runtime_coordinates(numpy_version="changed-numpy"),
    ],
    ids=["platform", "dependency"],
)
def test_partial_resume_rejects_host_runtime_coordinate_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_runtime: dict[str, object],
) -> None:
    import dr_code.corpus.candidate_evaluation as evaluation

    source = "def add_one(x):\n    return x + 1\n"
    runner = _PassingRunner()
    output = tmp_path / "evaluation"
    runtime = _runtime_coordinates()
    monkeypatch.setattr(
        evaluation, "_host_runtime_coordinates", lambda: runtime
    )
    run = _run(tmp_path / "run", ["sample-a"], source)
    corpus = _corpus(tmp_path / "corpus.parquet", ["sample-a"])
    snapshot = _snapshot(tmp_path / "snapshot.json")

    def evaluate() -> None:
        evaluate_preprocessing_candidates(
            preprocessing_run=run,
            corpus_path=corpus,
            output_dir=output,
            snapshot_path=snapshot,
            run_in_subprocess=runner,
            runner_identity="test-passing-runner@v1",
        )

    evaluate()
    with sqlite3.connect(output / STATE_FILENAME) as connection:
        connection.execute("UPDATE work SET status = 'running'")

    monkeypatch.setattr(
        evaluation,
        "_host_runtime_coordinates",
        lambda: changed_runtime,
    )

    with pytest.raises(CandidateEvaluationError, match="incompatible"):
        evaluate()
    assert runner.calls == 1


def test_rejects_reusing_partial_state_with_legacy_oci_coordinates(
    tmp_path: Path,
) -> None:
    output = tmp_path / "evaluation"
    output.mkdir()
    with sqlite3.connect(output / STATE_FILENAME) as connection:
        connection.execute(
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value_json TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO metadata(key, value_json) VALUES (?, ?)",
            (
                (
                    "runner_identity",
                    json.dumps("oci:docker:python@sha256:legacy"),
                ),
                (
                    "sandbox_image",
                    json.dumps("python@sha256:legacy"),
                ),
            ),
        )

    with pytest.raises(CandidateEvaluationError, match="incompatible"):
        evaluate_preprocessing_candidates(
            preprocessing_run=_run(
                tmp_path / "run",
                ["sample-a"],
                "def add_one(x):\n    return x + 1\n",
            ),
            corpus_path=_corpus(tmp_path / "corpus.parquet", ["sample-a"]),
            output_dir=output,
            snapshot_path=_snapshot(tmp_path / "snapshot.json"),
        )


def test_production_preflight_uses_a_separate_generous_timeout(
    tmp_path: Path,
) -> None:
    import dr_code.corpus.candidate_evaluation as evaluation

    tasks = evaluation._load_tasks(_snapshot(tmp_path / "snapshot.json"))
    observed_timeouts: list[float] = []
    observed_inputs: list[str] = []
    runner = _PassingRunner()

    def observe_timeout(
        *, source: str, input_json: str, timeout_seconds: float
    ) -> SubprocessCompletedProcess:
        observed_timeouts.append(timeout_seconds)
        observed_inputs.append(input_json)
        if input_json == "{}":
            return SubprocessCompletedProcess(
                returncode=0, stdout="2.2.6\n", stderr=""
            )
        return runner(
            source=source,
            input_json=input_json,
            timeout_seconds=timeout_seconds,
        )

    evaluation._preflight_production(tasks, run_in_subprocess=observe_timeout)

    assert observed_timeouts
    assert observed_inputs[0] == "{}"
    assert any("return x + 1" in payload for payload in observed_inputs[1:])
    assert set(observed_timeouts) == {
        evaluation._PRODUCTION_PREFLIGHT_TIMEOUT_SECONDS
    }
    assert evaluation._PRODUCTION_PREFLIGHT_TIMEOUT_SECONDS > (
        evaluation.DEFAULT_HUMANEVAL_TIMEOUT_SECONDS
    )


def test_production_runs_with_host_subprocess_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DR_CODE_SANDBOX_RUNTIME", "unavailable-runtime")
    monkeypatch.setenv("DR_CODE_SANDBOX_IMAGE", "unavailable-image")

    artifacts = evaluate_preprocessing_candidates(
        preprocessing_run=_run(
            tmp_path / "run",
            ["sample-a"],
            "def add_one(x):\n    return x + 1\n",
        ),
        corpus_path=_corpus(tmp_path / "corpus.parquet", ["sample-a"]),
        output_dir=tmp_path / "evaluation",
        snapshot_path=_snapshot(tmp_path / "snapshot.json"),
    )

    result = pq.read_table(artifacts.results_path).to_pylist()[0]
    assert result["record_status"] == "measured"
    assert result["outcome"] == "passed"


def test_infrastructure_subprocess_error_is_not_reported_as_tests_failed(
    tmp_path: Path,
) -> None:
    source = "def add_one(x):\n    return x + 1\n"

    def unavailable(**_: object) -> SubprocessCompletedProcess:
        raise SubprocessError("runtime unavailable")

    artifacts = evaluate_preprocessing_candidates(
        preprocessing_run=_run(tmp_path / "run", ["sample-a"], source),
        corpus_path=_corpus(tmp_path / "corpus.parquet", ["sample-a"]),
        output_dir=tmp_path / "evaluation",
        snapshot_path=_snapshot(tmp_path / "snapshot.json"),
        run_in_subprocess=unavailable,
        runner_identity="test-unavailable-runner@v1",
    )

    result = pq.read_table(artifacts.results_path).to_pylist()[0]
    assert result["record_status"] == "infrastructure_failure"
    assert result["failure_type"] == "SubprocessError"
    assert result["outcome"] is None

    evaluate_preprocessing_candidates(
        preprocessing_run=tmp_path / "run",
        corpus_path=tmp_path / "corpus.parquet",
        output_dir=tmp_path / "evaluation",
        snapshot_path=tmp_path / "snapshot.json",
        run_in_subprocess=_PassingRunner(),
        runner_identity="test-unavailable-runner@v1",
    )

    retried = pq.read_table(artifacts.results_path).to_pylist()[0]
    assert retried["record_status"] == "measured"
    assert retried["outcome"] == "passed"


def test_timeout_uses_official_timed_out_outcome(tmp_path: Path) -> None:
    source = "def add_one(x):\n    return x + 1\n"

    def timeout(
        *, source: str, input_json: str, timeout_seconds: float
    ) -> SubprocessCompletedProcess:
        del source, input_json, timeout_seconds
        from dr_code.humaneval.subprocess_runner import SubprocessTimeoutError

        raise SubprocessTimeoutError("candidate timed out")

    artifacts = evaluate_preprocessing_candidates(
        preprocessing_run=_run(tmp_path / "run", ["sample-a"], source),
        corpus_path=_corpus(tmp_path / "corpus.parquet", ["sample-a"]),
        output_dir=tmp_path / "evaluation",
        snapshot_path=_snapshot(tmp_path / "snapshot.json"),
        run_in_subprocess=timeout,
        runner_identity="test-timeout-runner@v1",
    )

    result = pq.read_table(artifacts.results_path).to_pylist()[0]
    assert result["record_status"] == "measured"
    assert result["outcome"] == "timed_out"
