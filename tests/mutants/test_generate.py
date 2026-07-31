"""Search, cache, duplicate, and real-snapshot generation contracts."""

from __future__ import annotations

import ast
import json
from collections.abc import Callable

import pytest
from dr_exec import (
    Attribution,
    Budgets,
    ContainmentProfile,
    EnvironmentGrant,
    ExitVerdict,
    Measurements,
    Outcome,
    PythonRuntime,
    Records,
    RunResult,
    TruncationMark,
)

from dr_code.humaneval.task import parse_human_eval_dataset
from dr_code.mutants import provenance as provenance_module
from dr_code.mutants.generate import generate_mutants
from dr_code.mutants.operators import OperatorFamily
from dr_code.mutants.oracle import run_program_on_inputs
from dr_code.synthetic import humaneval_loader as humaneval_loader_module
from dr_code.synthetic.humaneval_loader import (
    HumanEvalPlusTask,
    HumanEvalSource,
)

_BEGIN = "<<<DR_CODE_MUTANTS_V2_BEGIN>>>"
_END = "<<<DR_CODE_MUTANTS_V2_END>>>"
_TEST = """def check(candidate):
    inputs = [[0], [1]]
    results = [True, False]
    for i, (inp, exp) in enumerate(zip(inputs, results)):
        assertion(candidate(*inp), exp, 0)
"""


def _task(
    *,
    source: str | None = None,
    test: str = _TEST,
) -> HumanEvalPlusTask:
    prompt = "def f(x):\n"
    solution = "    return x < 1\n"
    if source is not None:
        prompt = source
        solution = ""
    return HumanEvalPlusTask(
        task_id="HumanEval/fixture",
        prompt=prompt,
        canonical_solution=solution,
        entry_point="f",
        test=test,
    )


def _response(input_text: str, *values: str) -> RunResult:
    envelope = {
        "invocation_id": json.loads(input_text)["invocation_id"],
        "protocol_version": 2,
        "outcomes": [
            {"kind": "value", "value_repr": value} for value in values
        ],
    }
    stdout = f"{_BEGIN}{json.dumps(envelope)}{_END}"
    return RunResult(
        returncode=0,
        stdout=stdout,
        stderr="",
        truncation=TruncationMark(),
        measurements=Measurements(
            duration_seconds=0.0,
            teardown_seconds=0.0,
            stdout_bytes_produced=len(stdout.encode("utf-8")),
            stderr_bytes_produced=0,
            input_bytes=len(input_text.encode("utf-8")),
        ),
        outcome=Outcome(
            attribution=Attribution.PAYLOAD,
            exit_verdict=ExitVerdict.SUCCESS,
        ),
    )


class _ProgramRunner:
    """A ``PythonRunner`` that answers each run from the fed input payload.

    Injected-runner logic tests script the oracle's response from the JSON
    request the batch feeds as ``input_text``; the executor's spawn path is
    dr-exec's, tested there, so these tests never spawn.
    """

    def __init__(self, respond: Callable[[str], RunResult]) -> None:
        self._respond = respond

    def run_untrusted_python(
        self,
        source: str,
        *,
        profile: ContainmentProfile,
        budgets: Budgets,
        records: Records,
        runtime: PythonRuntime,
        input_text: str,
        environment: EnvironmentGrant,
    ) -> RunResult:
        _ = source, profile, budgets, records, runtime, environment
        return self._respond(input_text)


def test_canonical_is_cached_and_duplicate_mutants_are_skipped(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        provenance_module,
        "load_humaneval_plus",
        lambda source: [_task()],
    )
    canonical_calls = 0
    mutant_calls = 0

    def respond(input_text: str) -> RunResult:
        nonlocal canonical_calls, mutant_calls
        program = json.loads(input_text)["program"]
        if "<=" in program:
            mutant_calls += 1
            return _response(input_text, "True", "True")
        canonical_calls += 1
        return _response(input_text, "True", "False")

    generated = generate_mutants(
        families=(OperatorFamily.COMPARISON_FLIP,),
        seeds=3,
        max_inputs_per_mutant=2,
        timeout_seconds=1.0,
        task_ids=("HumanEval/fixture",),
        runner=_ProgramRunner(respond),
        runner_identity="fixture-runner@v1",
        runtime_identity="fixture-runtime@v1",
    )

    assert canonical_calls == 2
    assert mutant_calls == 2
    assert len(generated.records) == 1
    assert generated.records[0].distinct_input_indices == (1,)
    assert [(skip.seed, skip.reason) for skip in generated.skipped] == [
        (1, "duplicate of an earlier accepted mutant"),
        (2, "duplicate of an earlier accepted mutant"),
    ]


def test_generation_captures_humaneval_32_evaluator_canonical(
    monkeypatch,
) -> None:
    raw_task = next(
        task
        for task in humaneval_loader_module.load_humaneval_plus(
            source=HumanEvalSource.SNAPSHOT
        )
        if task.task_id == "HumanEval/32"
    )
    evaluator_task = parse_human_eval_dataset(
        [raw_task.model_dump(mode="json")]
    )[0]
    monkeypatch.setattr(
        provenance_module,
        "load_humaneval_plus",
        lambda source: [raw_task],
    )

    generated = generate_mutants(
        families=(OperatorFamily.COMPARISON_FLIP,),
        seeds=1,
        max_inputs_per_mutant=1,
        timeout_seconds=1.0,
        task_ids=(raw_task.task_id,),
        runner=_ProgramRunner(lambda input_text: _response(input_text, "0")),
        runner_identity="fixture-runner@v1",
        runtime_identity="fixture-runtime@v1",
    )

    canonical_task = generated.canonical_suite[0]
    assert canonical_task.canonical_full_source == ast.unparse(
        ast.parse(evaluator_task.ground_truth_code)
    )
    assert canonical_task.canonical_test == evaluator_task.test


def test_two_independent_mutant_runs_must_agree(monkeypatch) -> None:
    monkeypatch.setattr(
        provenance_module,
        "load_humaneval_plus",
        lambda source: [_task()],
    )
    mutant_calls = 0

    def respond(input_text: str) -> RunResult:
        nonlocal mutant_calls
        program = json.loads(input_text)["program"]
        if "<=" not in program:
            return _response(input_text, "True", "False")
        mutant_calls += 1
        if mutant_calls == 1:
            return _response(input_text, "True", "True")
        return _response(input_text, "True", "False")

    generated = generate_mutants(
        families=(OperatorFamily.COMPARISON_FLIP,),
        seeds=1,
        max_inputs_per_mutant=2,
        timeout_seconds=1.0,
        task_ids=("HumanEval/fixture",),
        runner=_ProgramRunner(respond),
        runner_identity="fixture-runner@v1",
        runtime_identity="fixture-runtime@v1",
    )

    assert generated.records == ()
    assert generated.skipped[0].reason == (
        "mutant is non-deterministic across two runs"
    )


def test_two_independent_canonical_runs_must_agree(monkeypatch) -> None:
    monkeypatch.setattr(
        provenance_module,
        "load_humaneval_plus",
        lambda source: [_task()],
    )
    canonical_calls = 0

    def respond(input_text: str) -> RunResult:
        nonlocal canonical_calls
        canonical_calls += 1
        if canonical_calls == 1:
            return _response(input_text, "True", "False")
        return _response(input_text, "False", "False")

    generated = generate_mutants(
        families=(OperatorFamily.COMPARISON_FLIP,),
        seeds=1,
        max_inputs_per_mutant=2,
        timeout_seconds=1.0,
        task_ids=("HumanEval/fixture",),
        runner=_ProgramRunner(respond),
        runner_identity="fixture-runner@v1",
        runtime_identity="fixture-runtime@v1",
    )

    assert canonical_calls == 2
    assert generated.records == ()
    assert generated.skipped[0].reason == (
        "canonical is non-deterministic across two runs"
    )


def test_malformed_canonical_source_is_an_honest_task_skip(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        provenance_module,
        "load_humaneval_plus",
        lambda source: [_task(source="def f(:\n")],
    )

    def respond(input_text: str) -> RunResult:
        raise AssertionError(input_text)

    generated = generate_mutants(
        families=(OperatorFamily.COMPARISON_FLIP,),
        seeds=1,
        max_inputs_per_mutant=2,
        timeout_seconds=1.0,
        runner=_ProgramRunner(respond),
        runner_identity="fixture-runner@v1",
        runtime_identity="fixture-runtime@v1",
    )

    assert generated.records == ()
    assert generated.skipped[0].reason == "canonical source is malformed"


def test_malformed_canonical_test_is_an_honest_task_skip(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        provenance_module,
        "load_humaneval_plus",
        lambda source: [_task(test="def check(:\n")],
    )

    generated = generate_mutants(
        families=(OperatorFamily.COMPARISON_FLIP,),
        seeds=2,
        max_inputs_per_mutant=2,
        timeout_seconds=1.0,
        task_ids=("HumanEval/fixture",),
        runner=_ProgramRunner(
            lambda input_text: (_ for _ in ()).throw(
                AssertionError(input_text)
            )
        ),
        runner_identity="fixture-runner@v1",
        runtime_identity="fixture-runtime@v1",
    )

    assert generated.records == ()
    assert generated.skipped[0].operator_family == "*"
    assert generated.skipped[0].seed is None
    assert generated.skipped[0].reason == "canonical test is malformed"


def test_injected_runner_requires_explicit_nonproduction_provenance(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        provenance_module,
        "load_humaneval_plus",
        lambda source: [_task()],
    )

    def respond(input_text: str) -> RunResult:
        raise AssertionError(input_text)

    with pytest.raises(ValueError, match="explicit runner and runtime"):
        generate_mutants(
            families=(OperatorFamily.COMPARISON_FLIP,),
            task_ids=("HumanEval/fixture",),
            runner=_ProgramRunner(respond),
        )


def test_selected_suite_content_changes_config_identity(monkeypatch) -> None:
    selected = _task()
    monkeypatch.setattr(
        provenance_module,
        "load_humaneval_plus",
        lambda source: [selected],
    )

    runner = _ProgramRunner(
        lambda input_text: _response(input_text, "True", "False")
    )

    first = generate_mutants(
        families=(OperatorFamily.AGGREGATION_SWAP,),
        task_ids=("HumanEval/fixture",),
        runner=runner,
        runner_identity="fixture-runner@v1",
        runtime_identity="fixture-runtime@v1",
    )
    changed = selected.model_copy(
        update={"prompt": "def f(x):\n    # changed\n"}
    )
    monkeypatch.setattr(
        provenance_module,
        "load_humaneval_plus",
        lambda source: [changed],
    )
    second = generate_mutants(
        families=(OperatorFamily.AGGREGATION_SWAP,),
        task_ids=("HumanEval/fixture",),
        runner=runner,
        runner_identity="fixture-runner@v1",
        runtime_identity="fixture-runtime@v1",
    )

    assert first.config.task_ids == ("HumanEval/fixture",)
    assert first.config.canonical_suite_digest != (
        second.config.canonical_suite_digest
    )
    assert first.config.identity_hash() != second.config.identity_hash()


def test_unknown_requested_task_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        provenance_module,
        "load_humaneval_plus",
        lambda source: [_task()],
    )

    with pytest.raises(ValueError, match="unknown HumanEval"):
        generate_mutants(task_ids=("HumanEval/missing",))


def test_small_pinned_snapshot_slice_produces_replayable_divergence() -> None:
    generated = generate_mutants(
        families=(OperatorFamily.COMPARISON_FLIP,),
        seeds=1,
        max_inputs_per_mutant=50,
        timeout_seconds=5.0,
        task_ids=("HumanEval/0",),
    )

    assert len(generated.records) == 1
    record = generated.records[0]
    assert record.task_id == "HumanEval/0"
    assert record.distinct_input_indices == (18,)
    assert all(
        isinstance(ast.literal_eval(value), tuple)
        for value in record.input_reprs
    )

    replay = run_program_on_inputs(
        program=record.mutated_full_source,
        entry_point=record.entry_point,
        input_reprs=record.input_reprs,
        timeout_seconds=5.0,
    )
    assert tuple(
        outcome.model_dump(mode="json") for outcome in replay.outcomes
    ) == tuple(
        outcome.model_dump(mode="json") for outcome in record.mutant_expected
    )
