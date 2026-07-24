"""Search, cache, duplicate, and real-snapshot generation contracts."""

from __future__ import annotations

import ast
import json

import pytest

from dr_code.mutants import generate as generate_module
from dr_code.mutants import oracle as oracle_module
from dr_code.execution.subprocess import SubprocessCompletedProcess
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


def _response(input_text: str, *values: str) -> SubprocessCompletedProcess:
    envelope = {
        "invocation_id": json.loads(input_text)["invocation_id"],
        "protocol_version": 2,
        "outcomes": [
            {"kind": "value", "value_repr": value} for value in values
        ],
    }
    return SubprocessCompletedProcess(
        returncode=0,
        stdout=f"{_BEGIN}{json.dumps(envelope)}{_END}",
        stderr="",
    )


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

    def runner(
        *,
        source: str,
        input_text: str,
        timeout_seconds: float,
    ) -> SubprocessCompletedProcess:
        nonlocal canonical_calls, mutant_calls
        _ = source, timeout_seconds
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
        runner=runner,
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
        runner=lambda **kwargs: _response(kwargs["input_text"], "0"),
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

    def runner(
        *,
        source: str,
        input_text: str,
        timeout_seconds: float,
    ) -> SubprocessCompletedProcess:
        nonlocal mutant_calls
        _ = source, timeout_seconds
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
        runner=runner,
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

    def runner(
        *,
        source: str,
        input_text: str,
        timeout_seconds: float,
    ) -> SubprocessCompletedProcess:
        nonlocal canonical_calls
        _ = source, timeout_seconds
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
        runner=runner,
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

    def runner(
        *,
        source: str,
        input_text: str,
        timeout_seconds: float,
    ) -> SubprocessCompletedProcess:
        raise AssertionError((source, input_text, timeout_seconds))

    generated = generate_mutants(
        families=(OperatorFamily.COMPARISON_FLIP,),
        seeds=1,
        max_inputs_per_mutant=2,
        timeout_seconds=1.0,
        runner=runner,
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
        runner=lambda **kwargs: (_ for _ in ()).throw(AssertionError(kwargs)),
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

    def runner(**kwargs):
        raise AssertionError(kwargs)

    with pytest.raises(ValueError, match="explicit runner and runtime"):
        generate_mutants(
            families=(OperatorFamily.COMPARISON_FLIP,),
            task_ids=("HumanEval/fixture",),
            runner=runner,
        )


def test_injected_runner_cannot_claim_production_identity_namespace(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        provenance_module,
        "load_humaneval_plus",
        lambda source: [_task()],
    )

    def runner(**kwargs):
        raise AssertionError(kwargs)

    with pytest.raises(ValueError, match="must be non-production"):
        generate_mutants(
            families=(OperatorFamily.COMPARISON_FLIP,),
            task_ids=("HumanEval/fixture",),
            runner=runner,
            runner_identity="python-subprocess-oracle@unrecognized",
            runtime_identity="fixture-runtime@v1",
        )


def test_production_identity_and_execution_share_captured_runner_source(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        provenance_module,
        "load_humaneval_plus",
        lambda source: [_task()],
    )
    captured_source = "# captured runner source\n"
    later_source = "# later runner source!\n"
    monkeypatch.setattr(oracle_module, "_RUNNER_SOURCE", captured_source)

    def package_digest() -> str:
        monkeypatch.setattr(oracle_module, "_RUNNER_SOURCE", later_source)
        return "d" * 64

    monkeypatch.setattr(
        provenance_module,
        "package_source_digest",
        package_digest,
    )
    executed_sources: list[str] = []

    def runner(
        *,
        source: str,
        input_text: str,
        timeout_seconds: float,
    ) -> SubprocessCompletedProcess:
        _ = timeout_seconds
        executed_sources.append(source)
        program = json.loads(input_text)["program"]
        values = ("True", "True") if "<=" in program else ("True", "False")
        return _response(input_text, *values)

    monkeypatch.setattr(generate_module, "run_python_subprocess", runner)

    generated = generate_module.generate_mutants(
        families=(OperatorFamily.COMPARISON_FLIP,),
        seeds=1,
        max_inputs_per_mutant=2,
        timeout_seconds=1.0,
        task_ids=("HumanEval/fixture",),
        runner=runner,
    )

    payload = provenance_module._production_runner_identity_payload(
        runner_source_utf8=captured_source.encode("utf-8"),
        dr_code_python_package_sha256="d" * 64,
    )
    assert generated.config.runner_identity == (
        provenance_module._production_runner_identity(payload)
    )
    assert executed_sources
    assert set(executed_sources) == {captured_source}
    assert oracle_module._RUNNER_SOURCE == later_source


def test_selected_suite_content_changes_config_identity(monkeypatch) -> None:
    selected = _task()
    monkeypatch.setattr(
        provenance_module,
        "load_humaneval_plus",
        lambda source: [selected],
    )

    def runner(
        *,
        source: str,
        input_text: str,
        timeout_seconds: float,
    ) -> SubprocessCompletedProcess:
        _ = source, timeout_seconds
        return _response(input_text, "True", "False")

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
