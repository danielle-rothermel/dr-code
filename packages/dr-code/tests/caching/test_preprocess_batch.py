from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest
from dr_exec import (
    BudgetAxis,
    BudgetExceededOutcome,
    ExecutionSubmission,
    ExitedOutcome,
    ImportableEntryPoint,
    JobId,
    WorkerPoolImportableJsonExecutor,
    build_in_process_importable_json_job,
)

from dr_code.caching.preprocess_batch import (
    candidate_sources_batch,
    preprocess_batch,
)
from dr_code.preprocessing import EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION
from dr_code.preprocessing.execution import preprocess_job_budgets
from dr_code.trace import OUTPUT_KEY, InspectedCodeCandidateSetArtifact, Trace

pytestmark = pytest.mark.asyncio

_FENCED = "Here is the code:\n```python\ndef f(x):\n    return x + 1\n```\n"

_BLOCKING_ENTRY_POINT = ImportableEntryPoint(
    module_name="_blocking_entry_point",
    attribute_name="blocking_job",
)
_TESTS_ROOT = Path(__file__).resolve().parents[1]


async def test_preprocess_batch_returns_one_trace_per_distinct_text() -> None:
    duplicated = [_FENCED, _FENCED]

    results = await preprocess_batch(
        duplicated,
        definition=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
        worker_count=2,
    )

    assert set(results) == {_FENCED}
    output = results[_FENCED].value(OUTPUT_KEY)
    assert isinstance(output, InspectedCodeCandidateSetArtifact)


async def test_preprocess_batch_runs_every_distinct_text() -> None:
    texts = [
        f"```python\ndef f_{index}(x):\n    return x + {index}\n```"
        for index in range(8)
    ]

    results = await preprocess_batch(
        texts,
        definition=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
        worker_count=2,
    )

    assert set(results) == set(texts)


async def test_preprocess_batch_streams_to_the_observer_without_retaining() -> (
    None
):
    texts = [
        f"```python\ndef f_{index}(x):\n    return x + {index}\n```"
        for index in range(4)
    ]
    observed: dict[str, Trace | None] = {}

    results = await preprocess_batch(
        texts,
        definition=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
        worker_count=2,
        on_trace=observed.__setitem__,
    )

    assert results == {}
    assert set(observed) == set(texts)
    assert all(trace is not None for trace in observed.values())


async def test_preprocess_batch_returns_empty_for_no_texts() -> None:
    assert (
        await preprocess_batch(
            [],
            definition=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
        )
        == {}
    )


async def test_candidate_sources_batch_returns_sources_per_text() -> None:
    texts = [_FENCED, "Just an explanation, no code at all.\n"]

    results = await candidate_sources_batch(
        texts,
        definition=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
        worker_count=2,
    )

    assert results[_FENCED] == ("def f(x):\n    return x + 1",)
    assert results[texts[1]] == ()


async def test_candidate_sources_batch_streams_without_retaining() -> None:
    texts = [
        f"```python\ndef f_{index}(x):\n    return x + {index}\n```"
        for index in range(4)
    ]
    observed: dict[str, tuple[str, ...] | None] = {}

    results = await candidate_sources_batch(
        texts,
        definition=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
        worker_count=2,
        on_sources=observed.__setitem__,
    )

    assert results == {}
    assert set(observed) == set(texts)
    assert all(sources for sources in observed.values())


@pytest.fixture
def blocking_entry_point_on_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Let spawned workers import the blocking entry point module."""

    existing = os.environ.get("PYTHONPATH", "")
    joined = (
        f"{_TESTS_ROOT}{os.pathsep}{existing}"
        if existing
        else str(_TESTS_ROOT)
    )
    monkeypatch.setenv("PYTHONPATH", joined)


async def test_wall_time_budget_kills_only_the_wedged_item(
    blocking_entry_point_on_path: None,
) -> None:
    """A wedged item fails alone; its siblings still complete.

    The wedged job blocks on a pipe read that never receives a byte, so the
    budget is the only thing that can end it. Assertions are on terminal
    outcomes, never on how long anything took.
    """

    def _submission(*, block: bool) -> ExecutionSubmission[str]:
        label = "wedged" if block else f"healthy-{uuid4()}"
        return ExecutionSubmission(
            job=build_in_process_importable_json_job(
                JobId(uuid4()),
                _BLOCKING_ENTRY_POINT,
                {"block": block, "label": label},
                budgets=preprocess_job_budgets(0.5),
            ),
            context=label,
        )

    async def _submissions():
        yield _submission(block=True)
        for _ in range(3):
            yield _submission(block=False)

    outcomes: dict[str, object] = {}
    with WorkerPoolImportableJsonExecutor(
        entry_point=_BLOCKING_ENTRY_POINT,
        worker_count=2,
    ) as executor:
        async with executor.open_pool() as pool:
            async for completion in pool.map_stream(_submissions()):
                outcomes[completion.context] = (
                    completion.completed_execution.result.outcome
                )

    assert len(outcomes) == 4, "every submission yields one completion"

    wedged = outcomes.pop("wedged")
    assert isinstance(wedged, BudgetExceededOutcome)
    assert wedged.axis is BudgetAxis.WALL_TIME

    assert outcomes, "sibling items were submitted"
    for label, outcome in outcomes.items():
        assert isinstance(outcome, ExitedOutcome), label
        assert outcome.exit_code == 0, label


async def test_pool_serves_further_jobs_after_a_budget_kill(
    blocking_entry_point_on_path: None,
) -> None:
    """The killed worker is respawned, so the pool keeps serving."""

    def _job(*, block: bool, budget_seconds: float):
        return build_in_process_importable_json_job(
            JobId(uuid4()),
            _BLOCKING_ENTRY_POINT,
            {"block": block},
            budgets=preprocess_job_budgets(budget_seconds),
        )

    async def _one(submission: ExecutionSubmission[str]):
        async for completion in pool.map_stream(_only(submission)):
            return completion.completed_execution
        raise AssertionError("the pool yielded no completion")

    async def _only(submission: ExecutionSubmission[str]):
        yield submission

    with WorkerPoolImportableJsonExecutor(
        entry_point=_BLOCKING_ENTRY_POINT,
        worker_count=1,
    ) as executor:
        async with executor.open_pool() as pool:
            killed = await _one(
                ExecutionSubmission(
                    job=_job(block=True, budget_seconds=0.5),
                    context="wedged",
                )
            )
            # Only a respawned worker can serve this, since the pool has
            # width one and the previous worker was killed.
            after = await _one(
                ExecutionSubmission(
                    job=_job(block=False, budget_seconds=60.0),
                    context="after",
                )
            )

    assert isinstance(killed.result.outcome, BudgetExceededOutcome)
    assert killed.result.outcome.axis is BudgetAxis.WALL_TIME

    assert isinstance(after.result.outcome, ExitedOutcome)
    assert after.result.outcome.exit_code == 0
