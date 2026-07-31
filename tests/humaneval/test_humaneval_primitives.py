from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest
from dr_exec import (
    Attribution,
    BudgetAxis,
    ExitVerdict,
    FakeExecutor,
    ItemResult,
    Measurements,
    Outcome,
    Records,
    RunResult,
    ScriptedBatch,
    TruncationMark,
)
from pydantic import ValidationError

import dr_code.humaneval as humaneval
from dr_code.code_analysis import validate_python_source
from dr_code.eval import (
    RepeatPlan,
    SamplingDefinition,
    TaskSet,
    humaneval_source_content_hash,
    humaneval_task_identity,
)
from dr_code.humaneval import (
    EvaluationCaseStatus,
    HumanEvalTask,
    parse_human_eval_dataset,
)
from dr_code.humaneval.batch_runner import (
    PRODUCTION_EXECUTOR,
    CANDIDATE_KILL_RETURNCODES,
    driver_body_template,
    evaluate_human_eval_code,
    require_parsed_tests,
    run_function_batch,
)
from dr_code.humaneval.parsed_code import ParsedCode, parse_code
from dr_code.humaneval.parsed_tests import (
    HumanEvalTestCaseKind,
    UnsupportedTestFormatError,
    parse_human_eval_tests,
)
from dr_code.humaneval.sampling import (
    DEFAULT_HUMAN_EVAL_DATASET_NAME,
    DEFAULT_HUMAN_EVAL_DATASET_SPLIT,
    DEFAULT_HUMAN_EVAL_HF_REVISION,
    DEFAULT_HUMAN_EVAL_SNAPSHOT_SHA256,
    HumanEvalRawRowsSnapshot,
    SampledHumanEvalTask,
    load_human_eval_rows,
    run_human_eval_sampling,
    sample_human_eval_tasks_from_rows,
)
from dr_code.humaneval.scoring import (
    CandidateHarnessFailure,
    CompletedCandidateScore,
    CompletedScore,
    HarnessFailure,
    SubmissionOutcome,
    evaluation_outcome,
    score_humaneval_submission,
    submission_outcome,
)
from dr_code.humaneval.task import (
    EvaluationCaseResult,
    EvaluationHarnessError,
    EvaluationTaskResult,
    HumanEvalOverride,
    apply_human_eval_override,
)
from dr_code.preprocessing import (
    HUMANEVAL_FUNCTION_CANDIDATES_DEFINITION_ID,
    HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
    resolve_preprocessing_definition,
)

EXPECTED_HUMANEVAL_PUBLIC_API = {
    "CandidateHarnessFailure",
    "CompletedCandidateScore",
    "CompletedScore",
    "DEFAULT_HUMANEVAL_SCORING_PROFILE",
    "DEFAULT_HUMANEVAL_TIMEOUT_SECONDS",
    "DEFAULT_HUMAN_EVAL_SNAPSHOT_SHA256",
    "EvaluationCaseStatus",
    "EvaluationCaseSummary",
    "EvaluationTaskSummary",
    "HUMANEVAL_SCORING_PROFILE_ID",
    "HUMANEVAL_SCORING_PROFILE_VERSION",
    "HarnessFailure",
    "HarnessFailureCause",
    "HumanEvalCandidateScore",
    "HumanEvalScoringProfile",
    "HumanEvalSubmissionScore",
    "HumanEvalTask",
    "HumanEvalTestCaseKind",
    "SampledHumanEvalTask",
    "SubmissionOutcome",
    "evaluation_aggregate_metrics",
    "load_human_eval_rows",
    "parse_human_eval_dataset",
    "resolve_humaneval_scoring_profile",
    "run_human_eval_sampling",
    "sample_human_eval_tasks",
    "sample_human_eval_tasks_from_rows",
    "score_humaneval_submission",
}


def _task(*, test: str | None = None) -> HumanEvalTask:
    return HumanEvalTask(
        task_id="HumanEval/fixture",
        prompt="def add_one(x):\n",
        canonical_solution="    return x + 1\n",
        entry_point="add_one",
        test=test or _input_result_test(),
    )


def _row(task_id: str, offset: int) -> dict[str, str]:
    return {
        "task_id": task_id,
        "prompt": f"def f_{offset}(x):\n",
        "canonical_solution": f"    return x + {offset}\n",
        "entry_point": f"f_{offset}",
        "test": (
            "def check(candidate):\n"
            "    inputs = [(1,)]\n"
            f"    results = [{1 + offset}]\n"
            "    for inp, expected in zip(inputs, results):\n"
            "        assertion(candidate(*inp), expected)\n"
        ),
    }


def _check_payload_bytes(task: HumanEvalTask) -> list[bytes]:
    parsed_tests = require_parsed_tests(task)
    return [
        case.as_check(
            candidate_name="candidate",
            assertion_name=parsed_tests.assertion_name,
        )
        .model_dump_json()
        .encode("utf-8")
        for case in parsed_tests.cases
    ]


def _input_result_test() -> str:
    return (
        "def check(candidate):\n"
        "    inputs = [(1,), (2,)]\n"
        "    results = [2, 3]\n"
        "    for inp, expected in zip(inputs, results):\n"
        "        assertion(candidate(*inp), expected)\n"
    )


def test_humaneval_public_api_is_curated() -> None:
    assert set(humaneval.__all__) == EXPECTED_HUMANEVAL_PUBLIC_API


# ---------------------------------------------------------------------------
# dr-exec executor doubles (logic tests that need a specific batch outcome).
# ---------------------------------------------------------------------------


@pytest.fixture
def real_executor() -> object:
    """The real dr-exec batch executor; records disabled per call site."""
    return PRODUCTION_EXECUTOR


def _no_records() -> Records:
    return Records.none()


def _measurements(*, stdout: str = "", stderr: str = "") -> Measurements:
    return Measurements(
        duration_seconds=0.0,
        teardown_seconds=0.0,
        stdout_bytes_produced=len(stdout.encode("utf-8")),
        stderr_bytes_produced=len(stderr.encode("utf-8")),
        input_bytes=0,
    )


def _payload_run(returncode: int, *, stderr: str = "") -> RunResult:
    return RunResult(
        returncode=returncode,
        stdout="",
        stderr=stderr,
        truncation=TruncationMark(),
        measurements=_measurements(stderr=stderr),
        outcome=Outcome(
            attribution=Attribution.PAYLOAD,
            exit_verdict=ExitVerdict.REPORT_ONLY,
        ),
    )


def _budget_run(axis: BudgetAxis) -> RunResult:
    dropped = 1024 if axis is BudgetAxis.OUTPUT else 0
    return RunResult(
        returncode=-9,
        stdout="",
        stderr="",
        truncation=TruncationMark(stderr_bytes_dropped=dropped),
        measurements=Measurements(
            duration_seconds=0.0,
            teardown_seconds=0.0,
            stdout_bytes_produced=0,
            stderr_bytes_produced=dropped,
            input_bytes=0,
        ),
        outcome=Outcome(attribution=Attribution.BUDGET, violated_axis=axis),
    )


def _scripted_executor(
    *,
    case_payloads: dict[str, object] | None = None,
    run: RunResult | None = None,
) -> FakeExecutor:
    """A FakeExecutor answering every batch with the given payloads and run."""
    payloads = case_payloads or {}

    def batch_for(call):
        results = tuple(
            ItemResult(item_id=item_id, payload=payload)
            for item_id, payload in payloads.items()
        )
        return ScriptedBatch(
            run=run if run is not None else _payload_run(0),
            results=results,
            completion_seen=True,
        )

    fake = FakeExecutor()
    fake.script_batches_with(batch_for)
    return fake


def _passed_case(case_id: str) -> dict[str, object]:
    return {
        "status": "passed",
        "message": "",
        "input_repr": "[1]",
        "expected_output_repr": "2",
        "actual_output_repr": "",
        "elapsed_seconds": 0.0,
    }


def test_humaneval_facade_supports_explicit_star_and_dir_introspection() -> (
    None
):
    assert humaneval.CompletedScore is CompletedScore
    namespace: dict[str, object] = {}
    exec("from dr_code.humaneval import *", namespace)
    assert EXPECTED_HUMANEVAL_PUBLIC_API <= namespace.keys()
    assert EXPECTED_HUMANEVAL_PUBLIC_API <= set(dir(humaneval))


def test_sampling_from_rows_is_deterministic_and_indexed() -> None:
    rows = [_row(f"HumanEval/{index}", index) for index in range(5)]

    first = sample_human_eval_tasks_from_rows(
        rows,
        seed=17,
        sample_count=3,
        dataset_name=DEFAULT_HUMAN_EVAL_DATASET_NAME,
        dataset_split=DEFAULT_HUMAN_EVAL_DATASET_SPLIT,
        hf_revision=DEFAULT_HUMAN_EVAL_HF_REVISION,
    )
    second = sample_human_eval_tasks_from_rows(
        rows,
        seed=17,
        sample_count=3,
        dataset_name=DEFAULT_HUMAN_EVAL_DATASET_NAME,
        dataset_split=DEFAULT_HUMAN_EVAL_DATASET_SPLIT,
        hf_revision=DEFAULT_HUMAN_EVAL_HF_REVISION,
    )

    assert [sample.sample_index for sample in first] == [0, 1, 2]
    assert [sample.repeat_id.rng_seed for sample in first] == [17, 17, 17]
    assert [sample.task.task_id for sample in first] == [
        sample.task.task_id for sample in second
    ]


@pytest.mark.parametrize("sample_count", [True, 1.0, "1"])
def test_sampling_rejects_non_integer_counts(sample_count: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        sample_human_eval_tasks_from_rows(
            [_row("HumanEval/0", 0)],
            seed=17,
            sample_count=sample_count,  # type: ignore[arg-type]
            dataset_name=DEFAULT_HUMAN_EVAL_DATASET_NAME,
            dataset_split=DEFAULT_HUMAN_EVAL_DATASET_SPLIT,
            hf_revision=DEFAULT_HUMAN_EVAL_HF_REVISION,
        )


def test_sampling_count_bounds_and_zero_policy() -> None:
    rows = [_row("HumanEval/0", 0)]
    common = {
        "seed": 17,
        "dataset_name": DEFAULT_HUMAN_EVAL_DATASET_NAME,
        "dataset_split": DEFAULT_HUMAN_EVAL_DATASET_SPLIT,
        "hf_revision": DEFAULT_HUMAN_EVAL_HF_REVISION,
    }

    assert (
        sample_human_eval_tasks_from_rows(
            rows,
            sample_count=0,
            **common,
        )
        == []
    )
    with pytest.raises(ValueError, match="non-negative"):
        sample_human_eval_tasks_from_rows(
            rows,
            sample_count=-1,
            **common,
        )
    with pytest.raises(ValueError, match="exceeds"):
        sample_human_eval_tasks_from_rows(
            rows,
            sample_count=2,
            **common,
        )


def test_canonical_sampling_runner_honors_repeats_and_seed_provenance() -> (
    None
):
    tasks = parse_human_eval_dataset(
        [_row("HumanEval/0", 0), _row("HumanEval/1", 1)]
    )
    identities = tuple(humaneval_task_identity(task) for task in tasks)
    task_set = TaskSet(
        manifest_id="tasks",
        version="1",
        dataset_id="fixture",
        dataset_split="test",
        dataset_revision="fixture",
        source_content_hash=humaneval_source_content_hash(tuple(tasks)),
        source_task_identities=identities,
        task_identities=identities,
    )
    repeat_plan = RepeatPlan(
        plan_id="repeats",
        version="1",
        task_identities=identities,
        repeat_count=2,
        seeds=(
            (f"{identities[0]}#0", 10),
            (f"{identities[0]}#1", 11),
            (f"{identities[1]}#0", 20),
            (f"{identities[1]}#1", 21),
        ),
    )
    sampling = SamplingDefinition(
        definition_id="sampling",
        version="1",
    ).materialize(task_set=task_set, repeat_plan=repeat_plan)
    sampled = run_human_eval_sampling(
        tasks,
        sampling=sampling,
    )
    assert [sample.task.task_id for sample in sampled] == [
        "HumanEval/0",
        "HumanEval/0",
        "HumanEval/1",
        "HumanEval/1",
    ]
    assert [sample.repeat_id.rng_seed for sample in sampled] == [
        10,
        11,
        20,
        21,
    ]
    assert {sample.sampling_config_identity for sample in sampled} == {
        sampling.config_identity_hash
    }
    assert (
        type(sampled[0]).model_validate_json(
            sampled[0].model_dump_json(exclude_computed_fields=True)
        )
        == sampled[0]
    )
    assert len({sample.repeat_id.identity_hash() for sample in sampled}) == 4


def test_sampling_authenticates_nondefault_dataset_coordinates() -> None:
    rows = [_row("HumanEval/0", 0)]
    first = sample_human_eval_tasks_from_rows(
        rows,
        seed=17,
        sample_count=1,
        dataset_name="dataset-a",
        dataset_split="validation",
        hf_revision="revision-a",
    )[0]
    second = sample_human_eval_tasks_from_rows(
        rows,
        seed=17,
        sample_count=1,
        dataset_name="dataset-b",
        dataset_split="test",
        hf_revision="revision-b",
    )[0]
    assert first.sampling_config_identity != second.sampling_config_identity


def test_sampling_authenticates_the_complete_preselection_population() -> None:
    rows = [_row(f"HumanEval/{index}", index) for index in range(3)]
    first = sample_human_eval_tasks_from_rows(
        rows,
        seed=17,
        sample_count=1,
        dataset_name="dataset",
        dataset_split="test",
        hf_revision="revision",
    )[0]
    selected_id = first.task.task_id
    changed_rows = [dict(row) for row in rows]
    changed_index = next(
        index
        for index, row in enumerate(changed_rows)
        if row["task_id"] != selected_id
    )
    changed_rows[changed_index]["prompt"] += "# source changed\n"
    second = sample_human_eval_tasks_from_rows(
        changed_rows,
        seed=17,
        sample_count=1,
        dataset_name="dataset",
        dataset_split="test",
        hf_revision="revision",
    )[0]

    assert second.task == first.task
    assert (
        second.sampling_config.task_set.source_content_hash
        != first.sampling_config.task_set.source_content_hash
    )
    assert second.sampling_config_identity != first.sampling_config_identity
    with pytest.raises(ValueError, match="source population"):
        run_human_eval_sampling(
            parse_human_eval_dataset(changed_rows),
            sampling=first.sampling_config,
        )


def test_sampled_task_rejects_negative_and_unrelated_sampling_identity() -> (
    None
):
    sampled = sample_human_eval_tasks_from_rows(
        [_row("HumanEval/0", 0)],
        seed=17,
        sample_count=1,
        dataset_name="dataset",
        dataset_split="test",
        hf_revision="revision",
    )[0]
    negative = sampled.model_dump(mode="json")
    negative["sample_index"] = -1
    with pytest.raises(ValidationError, match="non-negative"):
        SampledHumanEvalTask.model_validate(negative)

    unrelated = sampled.model_dump(
        mode="json",
        exclude_computed_fields=True,
    )
    unrelated["sampling_config_identity"] = "f" * 64
    with pytest.raises(ValidationError, match="embedded SamplingConfig"):
        SampledHumanEvalTask.model_validate(unrelated)


def test_sampled_task_serializer_has_explicit_authenticated_shape() -> None:
    sampled = sample_human_eval_tasks_from_rows(
        [_row("HumanEval/0", 0)],
        seed=17,
        sample_count=1,
        dataset_name="dataset",
        dataset_split="test",
        hf_revision="revision",
    )[0]

    assert set(sampled.model_dump(mode="json")) == {
        "sample_index",
        "sampling_config_identity",
        "sampling_config",
        "task",
        "repeat_id",
        "sample_identity",
    }
    assert set(sampled.sample_identity.model_dump(mode="json")) == {
        "sampling_config_identity",
        "repeat_identity",
        "ordinal",
        "task_identity",
        "identity_hash",
    }


def test_sampled_task_and_embedded_task_are_frozen() -> None:
    sampled = sample_human_eval_tasks_from_rows(
        [_row("HumanEval/0", 0)],
        seed=17,
        sample_count=1,
        dataset_name="dataset",
        dataset_split="test",
        hf_revision="revision",
    )[0]

    with pytest.raises(ValidationError, match="frozen"):
        sampled.sample_index = 2
    with pytest.raises(ValidationError, match="frozen"):
        sampled.task.task_id = "forged"


def test_sampled_task_rejects_embedded_task_identity_mismatch() -> None:
    sampled = sample_human_eval_tasks_from_rows(
        [_row("HumanEval/0", 0)],
        seed=17,
        sample_count=1,
        dataset_name="dataset",
        dataset_split="test",
        hf_revision="revision",
    )[0]
    forged = sampled.model_dump(mode="json")
    forged["task"] = _task().model_dump(
        mode="json",
        exclude_computed_fields=True,
    )
    with pytest.raises(
        ValidationError,
        match="embedded HumanEval task identity",
    ):
        SampledHumanEvalTask.model_validate(forged)


def test_raw_row_snapshot_rehydrates_byte_equal_checks() -> None:
    snapshot_path = Path("tests/corpus/humanevalplus_snapshot.json")
    raw_snapshot = HumanEvalRawRowsSnapshot.model_validate_json(
        snapshot_path.read_text(encoding="utf-8")
    )
    fresh_tasks = parse_human_eval_dataset(
        [row.model_dump(mode="json") for row in raw_snapshot.rows]
    )
    snapshot_tasks = parse_human_eval_dataset(
        load_human_eval_rows(snapshot_path=snapshot_path)
    )

    assert [task.task_id for task in snapshot_tasks] == [
        task.task_id for task in fresh_tasks
    ]
    for fresh_task, snapshot_task in zip(
        fresh_tasks,
        snapshot_tasks,
        strict=True,
    ):
        assert _check_payload_bytes(snapshot_task) == _check_payload_bytes(
            fresh_task
        )


def test_default_snapshot_digest_is_pinned_before_parsing(
    tmp_path: Path,
) -> None:
    snapshot_path = Path("tests/corpus/humanevalplus_snapshot.json")
    assert (
        hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
        == DEFAULT_HUMAN_EVAL_SNAPSHOT_SHA256
    )
    malformed = tmp_path / "malformed.json"
    malformed.write_bytes(b"not-json")

    with pytest.raises(ValueError, match="content digest mismatch"):
        load_human_eval_rows(snapshot_path=malformed)


def test_custom_snapshot_coordinates_require_and_verify_digest(
    tmp_path: Path,
) -> None:
    source = Path("tests/corpus/humanevalplus_snapshot.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["header"]["dataset_id"] = "custom/humaneval"
    custom_path = tmp_path / "custom.json"
    custom_path.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(custom_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="require an explicit"):
        load_human_eval_rows(
            dataset_name="custom/humaneval",
            snapshot_path=custom_path,
        )
    rows = load_human_eval_rows(
        dataset_name="custom/humaneval",
        snapshot_path=custom_path,
        expected_snapshot_sha256=digest,
    )
    assert rows


def test_raw_row_snapshot_rejects_forged_dataset_split(
    tmp_path: Path,
) -> None:
    snapshot_path = Path("tests/corpus/humanevalplus_snapshot.json")
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    payload["header"]["dataset_id"] = "custom/humaneval"
    payload["header"]["dataset_split"] = "validation"
    forged_path = tmp_path / "forged-split.json"
    forged_path.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(forged_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="dataset split mismatch"):
        load_human_eval_rows(
            dataset_name="custom/humaneval",
            dataset_split=DEFAULT_HUMAN_EVAL_DATASET_SPLIT,
            snapshot_path=forged_path,
            expected_snapshot_sha256=digest,
        )


def test_parse_input_result_tests_have_stable_case_ids() -> None:
    parsed = parse_human_eval_tests(_input_result_test())

    assert parsed.test_type is HumanEvalTestCaseKind.INPUT_RESULT
    assert [case.case_id for case in parsed.cases] == ["case_0", "case_1"]
    assert [case.kind for case in parsed.cases] == [
        HumanEvalTestCaseKind.INPUT_RESULT,
        HumanEvalTestCaseKind.INPUT_RESULT,
    ]
    checks = list(parsed.iter_checks(candidate_name="candidate"))
    assert checks[0].input_repr == "[1]"
    assert "candidate(*[1])" in checks[0].code

    summary = parsed.to_summary()
    assert summary.test_type is HumanEvalTestCaseKind.INPUT_RESULT
    assert [case.case_id for case in summary.cases] == ["case_0", "case_1"]
    assert summary.cases[0].input_repr == "[1]"
    assert "code" not in summary.cases[0].model_dump(mode="json")


def test_parse_oracle_tests_have_expected_expression_metadata() -> None:
    parsed = parse_human_eval_tests(
        "def ref(x):\n"
        "    return x + 1\n"
        "\n"
        "def check(candidate):\n"
        "    inputs = [(1,), (2,)]\n"
        "    for inp in inputs:\n"
        "        assertion(candidate(*inp), ref(*inp))\n"
    )

    assert parsed.test_type is HumanEvalTestCaseKind.INPUT_ORACLE
    assert [case.case_id for case in parsed.cases] == ["case_0", "case_1"]
    checks = list(parsed.iter_checks(candidate_name="candidate"))
    assert checks[0].expected_output_expr == "ref(*[1])"


def test_parse_expression_tests_preserve_indexed_assertion() -> None:
    parsed = parse_human_eval_tests(
        "def check(candidate):\n"
        "    inputs = [(1,), (2,)]\n"
        "    results = [2, 3]\n"
        "    for i, (inp, expected) in enumerate(zip(inputs, results)):\n"
        "        assert candidate(*inp) == expected\n"
    )

    assert parsed.test_type is HumanEvalTestCaseKind.INPUT_EXPRESSION
    checks = list(parsed.iter_checks(candidate_name="candidate"))
    assert checks[1].case_id == "case_1"
    assert "i = 1" in checks[1].code
    assert "assert candidate(*inp) == expected" in checks[1].code


def test_parsed_code_summary_excludes_runtime_ast() -> None:
    parsed = parse_code(
        display_title="fixture",
        code_str=(
            'def add_one(x: int) -> int:\n    """doc"""\n    return x + 1\n'
        ),
    )

    assert isinstance(parsed, ParsedCode)
    assert parsed.display_title == "fixture"
    assert parsed.signatures[0].function_name == "add_one"
    assert parsed.signatures[0].function_args[0].name == "x"
    dumped = parsed.model_dump(mode="json")
    assert "tree" not in dumped
    assert "doc" in dumped["comments"]


def test_evaluation_passes_when_best_function_passes(
    real_executor: object,
) -> None:
    result = evaluate_human_eval_code(
        task=_task(),
        candidate_code=(
            "def broken_helper(x):\n"
            "    return x\n"
            "\n"
            "def add_one(x):\n"
            "    return x + 1\n"
        ),
        timeout_seconds=2.0,
        executor=real_executor,
        records=_no_records(),
    )

    assert result.best_function_name == "add_one"
    assert result.passed is True
    assert result.status_counts == {"passed": 2}
    assert result.failures == []
    summary = result.to_summary()
    assert summary.passed is True
    assert summary.best_function_name == "add_one"
    assert summary.failure_count == 0


def test_evaluation_prefers_entry_point_when_pass_counts_tie(
    real_executor: object,
) -> None:
    result = evaluate_human_eval_code(
        task=_task(),
        candidate_code=(
            "def add_one(x):\n"
            "    return x + 1\n"
            "\n"
            "def also_add_one(x):\n"
            "    return x + 1\n"
        ),
        timeout_seconds=2.0,
        executor=real_executor,
        records=_no_records(),
    )

    assert result.best_function_name == "add_one"
    assert result.passed is True


def test_evaluation_fails_when_best_function_does_not_pass_all_cases(
    real_executor: object,
) -> None:
    result = evaluate_human_eval_code(
        task=_task(),
        candidate_code=(
            "def broken_helper(x):\n"
            "    return x\n"
            "\n"
            "def add_one(x):\n"
            "    return x + 1 if x == 1 else x\n"
        ),
        timeout_seconds=2.0,
        executor=real_executor,
        records=_no_records(),
    )

    assert result.best_function_name == "add_one"
    assert result.passed is False
    assert result.status_counts == {"passed": 1, "failed": 1}


def test_evaluation_uses_highest_pass_count(
    real_executor: object,
) -> None:
    result = evaluate_human_eval_code(
        task=_task(),
        candidate_code=(
            "def add_one(x):\n"
            "    return x\n"
            "\n"
            "def helper(x):\n"
            "    return x + 1\n"
        ),
        timeout_seconds=2.0,
        executor=real_executor,
        records=_no_records(),
    )

    assert result.best_function_name == "helper"
    assert result.passed is True
    assert result.status_counts == {"passed": 2}


def test_evaluate_humaneval_code_reports_timeout_per_case(
    real_executor: object,
) -> None:
    result = evaluate_human_eval_code(
        task=_task(),
        candidate_code=("def add_one(x):\n    while True:\n        pass\n"),
        timeout_seconds=0.2,
        executor=real_executor,
        records=_no_records(),
    )

    assert result.passed is False
    assert result.status_counts == {"timeout": 2}
    assert {case.case_id for case in result.results} == {"case_0", "case_1"}
    assert {case.timeout_seconds for case in result.results} == {0.2}
    assert evaluation_outcome(result) is SubmissionOutcome.TIMED_OUT


@pytest.mark.parametrize(
    ("candidate_code", "test_source", "expected_stderr"),
    [
        (
            "print('candidate top-level output')\n"
            "def add_one(x):\n"
            "    return x + 1\n",
            None,
            "candidate top-level output",
        ),
        (
            "def add_one(x):\n"
            "    print('candidate function output')\n"
            "    return x + 1\n",
            None,
            "candidate function output",
        ),
        (
            "def add_one(x):\n    return x + 1\n",
            "print('support top-level output')\n" + _input_result_test(),
            "support top-level output",
        ),
        (
            "import sys\n"
            "print('dunder stdout output', file=sys.__stdout__)\n"
            "def add_one(x):\n"
            "    return x + 1\n",
            None,
            "dunder stdout output",
        ),
    ],
)
def test_python_print_output_does_not_corrupt_runner_protocol(
    candidate_code: str,
    test_source: str | None,
    expected_stderr: str,
) -> None:
    """Candidate/support prints land on the payload stream, never the NDJSON
    protocol channel: the kit captures the protocol handle before reassigning
    stdout, so a noisy payload cannot interleave with result lines."""
    from dr_code.humaneval.batch_runner import (
        HUMANEVAL_ENVIRONMENT,
        HUMANEVAL_PROFILE,
        HUMANEVAL_RUNTIME,
        build_human_eval_batch_plan,
    )

    task = _task(test=test_source)
    plan = build_human_eval_batch_plan(
        task=task,
        candidate_code=candidate_code,
        function_name="add_one",
        timeout_seconds=2.0,
    )
    batch = PRODUCTION_EXECUTOR.run_batch(
        plan.request,
        profile=HUMANEVAL_PROFILE,
        budgets=plan.budgets,
        records=_no_records(),
        runtime=HUMANEVAL_RUNTIME,
        environment=HUMANEVAL_ENVIRONMENT,
    )

    result = evaluate_human_eval_code(
        task=task,
        candidate_code=candidate_code,
        timeout_seconds=2.0,
        executor=PRODUCTION_EXECUTOR,
        records=_no_records(),
    )

    assert result.passed is True
    assert batch.complete
    assert expected_stderr in batch.run.stderr
    assert expected_stderr not in batch.run.stdout


def test_scoring_uses_canonical_preprocessing_and_persists_its_trace() -> None:
    result = score_humaneval_submission(
        raw_submission="def add_one(x):\n    return “ok”\n",
        task=_task(),
        timeout_seconds=2.0,
        executor=_scripted_executor(
            case_payloads={"case_0": _passed_case("case_0")}
        ),
        records=_no_records(),
    )

    assert isinstance(result, CompletedScore)
    assert result.candidates[0].candidate_code == (
        'def add_one(x):\n    return "ok"'
    )
    assert result.preprocessing.producer.producer_id == (
        HUMANEVAL_FUNCTION_CANDIDATES_DEFINITION_ID
    )
    assert (
        result.preprocessing.producer.preprocessing_config_hash
        == resolve_preprocessing_definition(
            definition_id=HUMANEVAL_FUNCTION_CANDIDATES_DEFINITION_ID,
            version=HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION.version,
        )
        .materialize()
        .config_identity_hash
    )
    assert (
        CompletedScore.model_validate_json(result.model_dump_json()) == result
    )


def _partial_evaluation_result(task: HumanEvalTask) -> EvaluationTaskResult:
    return EvaluationTaskResult(
        task_id=task.task_id,
        entry_point=task.entry_point,
        function_names=[task.entry_point],
        total_cases=2,
        results=[
            EvaluationCaseResult(
                task_id=task.task_id,
                case_id="case_0",
                function_name=task.entry_point,
                status=EvaluationCaseStatus.PASSED,
                test_type=HumanEvalTestCaseKind.INPUT_RESULT,
            ),
        ],
    )


def test_evaluation_outcome_reports_incomplete_for_partial_coverage() -> None:
    evaluation = _partial_evaluation_result(_task())

    assert evaluation.coverage_complete is False
    assert evaluation.failures == []
    assert evaluation_outcome(evaluation) is (
        SubmissionOutcome.EVALUATION_INCOMPLETE
    )


def test_evaluation_outcome_reports_tests_failed_when_case_fails() -> None:
    task = _task()
    evaluation = EvaluationTaskResult(
        task_id=task.task_id,
        entry_point=task.entry_point,
        function_names=[task.entry_point],
        total_cases=2,
        results=[
            EvaluationCaseResult(
                task_id=task.task_id,
                case_id="case_0",
                function_name=task.entry_point,
                status=EvaluationCaseStatus.FAILED,
                message="bad",
                test_type=HumanEvalTestCaseKind.INPUT_RESULT,
            ),
            EvaluationCaseResult(
                task_id=task.task_id,
                case_id="case_1",
                function_name=task.entry_point,
                status=EvaluationCaseStatus.PASSED,
                test_type=HumanEvalTestCaseKind.INPUT_RESULT,
            ),
        ],
    )

    assert evaluation_outcome(evaluation) is SubmissionOutcome.TESTS_FAILED


def test_submission_outcome_reports_no_candidates_for_empty_input() -> None:
    assert submission_outcome(()) is SubmissionOutcome.NO_CANDIDATES


def test_score_humaneval_submission_reports_incomplete_runner_output() -> None:
    result = score_humaneval_submission(
        raw_submission="def add_one(x):\n    return x + 1\n",
        task=_task(),
        timeout_seconds=2.0,
        executor=_scripted_executor(
            case_payloads={"case_0": _passed_case("case_0")}
        ),
        records=_no_records(),
    )

    assert isinstance(result, CompletedScore)
    assert result.outcome is SubmissionOutcome.EVALUATION_INCOMPLETE
    assert result.score == 0.0
    candidate = result.candidates[0]
    assert isinstance(candidate, CompletedCandidateScore)
    assert candidate.evaluation.failures == []
    assert candidate.evaluation.coverage_complete is False


def test_score_humaneval_submission_returns_harness_failure() -> None:
    """A delivered item whose payload the case schema rejects surfaces as a
    candidate harness failure carrying the validation cause."""
    result = score_humaneval_submission(
        raw_submission="def add_one(x):\n    return x + 1\n",
        task=_task(),
        timeout_seconds=2.0,
        executor=_scripted_executor(
            case_payloads={"case_0": {"status": "not-a-status"}}
        ),
        records=_no_records(),
    )

    assert isinstance(result, HarnessFailure)
    assert result.kind == "harness_failure"
    assert result.outcome is SubmissionOutcome.HARNESS_FAILURE
    assert not hasattr(result, "score")
    candidate = result.candidates[0]
    assert isinstance(candidate, CandidateHarnessFailure)
    assert candidate.failure_class == "unknown"
    assert candidate.cause.exception_type == "ValidationError"


def test_score_humaneval_submission_reports_execution_breakage() -> None:
    """Broken execution is a harness failure, never a scored result.

    A candidate must not benefit from infrastructure breakage: an
    ``ExecutorFailure`` — a failure with no result to attribute — surfaces as a
    ``CandidateHarnessFailure`` rather than a ``CompletedScore`` with a zero
    score. The dr-exec attribution vocabulary is what the cause records.
    """
    from dr_exec import ExecutorFailure

    def broken(call):
        raise ExecutorFailure("executor is unavailable")

    executor = FakeExecutor()
    executor.script_batches_with(broken)

    result = score_humaneval_submission(
        raw_submission="def add_one(x):\n    return x + 1\n",
        task=_task(),
        timeout_seconds=2.0,
        executor=executor,
        records=_no_records(),
    )

    assert isinstance(result, HarnessFailure)
    assert result.kind == "harness_failure"
    assert not hasattr(result, "score")
    candidate = result.candidates[0]
    assert isinstance(candidate, CandidateHarnessFailure)
    assert candidate.cause.exception_type == "ExecutorFailure"


def test_score_humaneval_submission_reports_empty_submission() -> None:
    result = score_humaneval_submission(
        raw_submission=" \n\t ",
        task=_task(),
        timeout_seconds=2.0,
    )

    assert isinstance(result, CompletedScore)
    assert result.kind == "completed"
    assert result.raw_submission == " \n\t "
    assert result.outcome is SubmissionOutcome.PREPROCESSING_FAILED
    assert result.preprocessing_failure_code is not None
    assert result.candidates == ()


@pytest.mark.parametrize(
    ("raw_submission", "safe_raw_submission"),
    (
        (
            "def add_one(x):\n    return x + 1\n\x00",
            "def add_one(x):\n    return x + 1\n\\x00",
        ),
        (
            "def add_one(x):\n    return '\ud800'\n",
            "def add_one(x):\n    return '\\ud800'\n",
        ),
    ),
)
def test_score_humaneval_submission_returns_one_result_for_invalid_decoder_text(
    raw_submission: str,
    safe_raw_submission: str,
) -> None:
    result = score_humaneval_submission(
        raw_submission=raw_submission,
        task=_task(),
        timeout_seconds=2.0,
    )

    assert isinstance(result, CompletedScore)
    assert result.outcome is SubmissionOutcome.PREPROCESSING_FAILED
    assert result.preprocessing_failure_code == "decoder_output_invalid"
    assert result.candidates == ()
    assert result.raw_submission == safe_raw_submission
    assert (
        CompletedScore.model_validate_json(result.model_dump_json()) == result
    )


def test_score_humaneval_submission_evaluates_every_candidate() -> None:
    executor = _scripted_executor(
        case_payloads={
            "case_0": _passed_case("case_0"),
            "case_1": _passed_case("case_1"),
        }
    )

    result = score_humaneval_submission(
        raw_submission=(
            "```python\n"
            "def add_one(x):\n"
            "    return x\n"
            "```\n"
            "```python\n"
            "def add_one(x):\n"
            "    return x + 1\n"
            "```\n"
        ),
        task=_task(),
        timeout_seconds=2.0,
        executor=executor,
        records=_no_records(),
    )

    assert isinstance(result, CompletedScore)
    assert [candidate.candidate_index for candidate in result.candidates] == [
        0,
        1,
    ]
    assert (
        len({candidate.candidate_id for candidate in result.candidates}) == 2
    )
    # One dr-exec batch run spawned per candidate (the per-candidate fan-out).
    assert len(executor.batch_calls) == 2


@pytest.mark.parametrize(
    "raw_submission",
    (
        "def first():\n"
        "    return 1\n"
        "\n"
        "The preceding function is complete.\n"
        "\n"
        "def second():\n"
        "    return 2\n",
        r"def first():\n"
        r"    return 1\n"
        r"\n"
        r"The preceding function is complete.\n"
        r"\n"
        r"def second():\n"
        r"    return 2",
    ),
    ids=("unfenced", "escaped-unfenced"),
)
def test_score_humaneval_submission_evaluates_unfenced_functions_separated_by_prose(
    raw_submission: str,
) -> None:
    executor = _scripted_executor(
        case_payloads={
            "case_0": _passed_case("case_0"),
            "case_1": _passed_case("case_1"),
        }
    )

    result = score_humaneval_submission(
        raw_submission=raw_submission,
        task=_task(),
        timeout_seconds=2.0,
        executor=executor,
        records=_no_records(),
    )

    assert isinstance(result, CompletedScore)
    assert [candidate.candidate_code for candidate in result.candidates] == [
        "def first():\n    return 1",
        "def second():\n    return 2",
    ]
    assert len(executor.batch_calls) == 2


def test_scoring_does_not_filter_candidates_by_task_entry_point() -> None:
    result = score_humaneval_submission(
        raw_submission=(
            "def deliberately_different_name(x):\n    return x + 1\n"
        ),
        task=_task(),
        timeout_seconds=2.0,
        executor=_scripted_executor(
            case_payloads={
                "case_0": _passed_case("case_0"),
                "case_1": _passed_case("case_1"),
            }
        ),
        records=_no_records(),
    )

    assert isinstance(result, CompletedScore)
    assert result.outcome is SubmissionOutcome.PASSED
    assert result.candidates[0].candidate_code.startswith(
        "def deliberately_different_name"
    )


def test_evaluation_incomplete_when_runner_returns_partial_results() -> None:
    result = evaluate_human_eval_code(
        task=_task(),
        candidate_code="def add_one(x):\n    return x + 1\n",
        timeout_seconds=2.0,
        executor=_scripted_executor(
            case_payloads={"case_0": _passed_case("case_0")}
        ),
        records=_no_records(),
    )

    assert result.passed is False
    assert result.coverage_complete is False
    assert result.failures == []
    assert result.status_counts == {"passed": 1}


def test_validate_python_source_reports_syntax_errors() -> None:
    validation = validate_python_source("def bad(x)\n  pass")

    assert validation.parse_ok is False
    assert validation.compile_ok is False
    assert validation.parse_error is not None
    assert validation.compile_error is not None


def test_score_humaneval_submission_rejects_non_string_input() -> None:
    with pytest.raises(TypeError, match="raw_submission must be str"):
        score_humaneval_submission(
            raw_submission={"code": "def add_one(x):\n    return x + 1\n"},  # type: ignore[arg-type]
            task=_task(),
            timeout_seconds=2.0,
        )


@pytest.mark.parametrize(
    ("test_source", "match"),
    [
        ("def helper():\n    pass\n", "Could not find check"),
        (
            "def check(a, b):\n    pass\n",
            "one positional argument",
        ),
        (
            "def check(candidate):\n"
            "    inputs = [(1,)]\n"
            "    results = [1, 2]\n"
            "    for inp, expected in zip(inputs, results):\n"
            "        assertion(candidate(*inp), expected)\n",
            "does not match",
        ),
        (
            "def check(candidate):\n"
            "    inputs = range(3)\n"
            "    results = [0, 1, 2]\n"
            "    for inp, expected in zip(inputs, results):\n"
            "        assertion(candidate(*inp), expected)\n",
            "not a literal",
        ),
    ],
)
def test_parse_human_eval_tests_rejects_invalid_formats(
    test_source: str,
    match: str,
) -> None:
    with pytest.raises(UnsupportedTestFormatError, match=match):
        parse_human_eval_tests(test_source)


def test_candidate_module_level_sys_exit_is_scored(
    real_executor: object,
) -> None:
    result = evaluate_human_eval_code(
        task=_task(),
        candidate_code=(
            "import sys\nsys.exit(5)\ndef add_one(x):\n    return x + 1\n"
        ),
        timeout_seconds=2.0,
        executor=real_executor,
        records=_no_records(),
    )

    assert result.passed is False
    assert result.status_counts == {"error": 2}


def test_apply_human_eval_override_passthrough() -> None:
    row = _row("HumanEval/99", 1)
    assert apply_human_eval_override(row, {}) == dict(row)

    updated = apply_human_eval_override(
        row,
        {
            "HumanEval/99": HumanEvalOverride(
                canonical_solution="    return x + 99\n",
            ),
        },
    )
    assert updated["canonical_solution"] == "    return x + 99\n"

    with pytest.raises(ValueError, match="replacement text not found"):
        apply_human_eval_override(
            row,
            {
                "HumanEval/99": HumanEvalOverride(
                    test_replacements={"missing": "text"},
                ),
            },
        )


def test_parse_human_eval_dataset_builds_tasks() -> None:
    tasks = parse_human_eval_dataset([_row("HumanEval/0", 0)])

    assert len(tasks) == 1
    assert tasks[0].task_id == "HumanEval/0"
    assert tasks[0].parsed_tests is not None


def test_humaneval_task_rejects_derived_fields_from_other_raw_fields() -> None:
    original = _task()
    other = HumanEvalTask(
        task_id="HumanEval/other",
        prompt="def other(x):\n",
        canonical_solution="    return x - 1\n",
        entry_point="other",
        test=(
            "def check(candidate):\n"
            "    inputs = [(1,)]\n"
            "    results = [0]\n"
            "    for inp, expected in zip(inputs, results):\n"
            "        assertion(candidate(*inp), expected)\n"
        ),
    )
    assert other.parsed is not None
    assert other.parsed_tests is not None

    with pytest.raises(ValidationError, match="parsed code must match"):
        HumanEvalTask(
            task_id=original.task_id,
            prompt=original.prompt,
            canonical_solution=original.canonical_solution,
            entry_point=original.entry_point,
            test=original.test,
            parsed=other.parsed,
        )
    with pytest.raises(ValidationError, match="parsed tests must match"):
        HumanEvalTask(
            task_id=original.task_id,
            prompt=original.prompt,
            canonical_solution=original.canonical_solution,
            entry_point=original.entry_point,
            test=original.test,
            parsed_tests=other.parsed_tests,
        )


def test_require_parsed_tests_raises_when_missing() -> None:
    task = HumanEvalTask.model_construct(
        task_id="HumanEval/fixture",
        prompt="def add_one(x):\n",
        canonical_solution="    return x + 1\n",
        entry_point="add_one",
        test=_input_result_test(),
        parsed_tests=None,
    )

    with pytest.raises(
        ValueError,
        match=r"HumanEvalTask\.parsed_tests is required",
    ):
        require_parsed_tests(task)


def test_run_function_batch_raises_for_malformed_item_payload() -> None:
    """A delivered item whose payload the case schema rejects is a harness
    failure carrying the validated cases up to and including the bad one."""
    executor = _scripted_executor(
        case_payloads={
            "case_0": _passed_case("case_0"),
            "case_1": {"status": "nonsense"},
        }
    )

    with pytest.raises(EvaluationHarnessError) as exc_info:
        run_function_batch(
            task=_task(),
            candidate_code="def add_one(x):\n    return x + 1\n",
            function_name="add_one",
            timeout_seconds=2.0,
            executor=executor,
            records=_no_records(),
        )

    results = exc_info.value.case_results
    assert results[-1].case_id == "case_1"
    assert results[-1].status is EvaluationCaseStatus.ERROR
    assert "Invalid runner output" in results[-1].message


def test_batch_candidate_kill_returncode_scored_as_error() -> None:
    """A candidate-process death (SIGKILL) reports no items; every case is
    synthesized as an error scored against the candidate."""
    results = run_function_batch(
        task=_task(),
        candidate_code="def add_one(x):\n    return x + 1\n",
        function_name="add_one",
        timeout_seconds=2.0,
        executor=_scripted_executor(
            run=_payload_run(next(iter(CANDIDATE_KILL_RETURNCODES)))
        ),
        records=_no_records(),
    )

    assert len(results) == 2
    assert all(
        result.status is EvaluationCaseStatus.ERROR for result in results
    )
    assert "subprocess killed candidate execution" in results[0].message


def test_batch_output_budget_scored_as_error() -> None:
    """An output-budget death reports no items; every case is synthesized as an
    error scored against the candidate."""
    results = run_function_batch(
        task=_task(),
        candidate_code="def add_one(x):\n    return x + 1\n",
        function_name="add_one",
        timeout_seconds=2.0,
        executor=_scripted_executor(run=_budget_run(BudgetAxis.OUTPUT)),
        records=_no_records(),
    )

    assert len(results) == 2
    assert all(
        result.status is EvaluationCaseStatus.ERROR for result in results
    )
    assert "output budget exceeded" in results[0].message


def test_batch_executor_attribution_is_a_harness_failure() -> None:
    """A machine/executor/channel/absence attribution never scores against the
    candidate: it is a harness failure the raising lane surfaces. A no-spawn
    outcome carries no transcript, so this is exercised at the interpretation
    boundary directly rather than through the fake's spawn-consistent batch."""
    from dr_exec import BatchItem, BatchRequest, BatchResult

    from dr_code.humaneval.batch_runner import interpret_batch_result

    task = _task()
    request = BatchRequest(
        items=(
            BatchItem(item_id="case_0", payload={}),
            BatchItem(item_id="case_1", payload={}),
        ),
        body_source="def run_item(item_id, payload):\n    return {}\n",
        item_schema="humaneval-case@v1",
        config={"id": "x"},
    )
    run = RunResult(
        returncode=None,
        stdout="",
        stderr="",
        truncation=TruncationMark(),
        measurements=_measurements(),
        outcome=Outcome(attribution=Attribution.MACHINE, spawn_errno=13),
    )
    result = BatchResult(
        request=request,
        run=run,
        results=(),
        completion_seen=False,
        results_emitted_claim=None,
    )
    with pytest.raises(EvaluationHarnessError, match="machine attribution"):
        interpret_batch_result(
            task=task,
            function_name="add_one",
            result=result,
            timeout_seconds=2.0,
        )


def test_composed_driver_body_compiles() -> None:
    from dr_code.humaneval.batch_runner import compose_body

    composed = compose_body(
        candidate_code="def add_one(x):\n    return x + 1\n",
        support_code="",
        function_name="add_one",
    )
    compile(composed, "<driver-body>", "exec")


def test_driver_body_template_is_dependency_free() -> None:
    tree = ast.parse(driver_body_template())
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.append(node.module or "")

    assert not any(
        module == "dr_code" or module.startswith("dr_code.")
        for module in imported_modules
    )
