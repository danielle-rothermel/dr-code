"""Execution-cache contracts (plan section: ``engine/execution.py``).

Covers ``ExecutionRequest.cache_key`` (deterministic, derived from
``computation_id`` plus dr-exec's declared invocation identity),
``ExecutionOutcome`` (structured attribution fields), the ``ExecutionCache``
protocol + ``InMemoryExecutionCache`` get/put, ``run_requests`` dedupe +
at-most-once execution, and a GOLDEN pin on the cache-key derivation so drift
is loud.

Execution is driven through a ``FakeExecutor`` scripted with dr-exec batch
outcomes; the real executor is exercised in the parity/oracle suites.
"""

from __future__ import annotations

import pytest
from dr_exec import (
    Attribution,
    BatchItem,
    BatchRequest,
    BudgetAxis,
    Budgets,
    EnvironmentGrant,
    OutputBudget,
    OverflowPolicy,
    PROCESS_BOUNDARY_ONLY,
    HERMETIC,
)

from dr_code.metrics.engine.execution import (
    ExecutionOutcome,
    ExecutionRequest,
    InMemoryExecutionCache,
    InvocationIdentity,
    is_output_limit_outcome,
    is_timeout_outcome,
    run_requests,
)

from metrics.helpers import (
    clean_run,
    full_pass_batch,
    output_budget_run,
    scripted_batch,
    wall_clock_run,
)
from dr_exec import Records


_BODY = "def run_item(item_id, payload):\n    return {}\n"


def _batch_request(*, case_ids=("case_0", "case_1"), config="cfg") -> BatchRequest:
    return BatchRequest(
        items=tuple(
            BatchItem(item_id=cid, payload={"code": "pass"}) for cid in case_ids
        ),
        body_source=_BODY,
        item_schema="humaneval-case@v1",
        config={"identity": config},
    )


def _budgets() -> Budgets:
    return Budgets(
        wall_clock=2.0,
        output=OutputBudget(limit_bytes=1024, overflow_policy=OverflowPolicy.FAIL),
        input=4096,
    )


def _request(
    *,
    case_ids=("case_0", "case_1"),
    config="cfg",
    computation_id: str = "humaneval-runner@v1",
    executor_identity: str = "dr-exec@1.2.3",
) -> ExecutionRequest:
    batch = _batch_request(case_ids=case_ids, config=config)
    budgets = _budgets()
    identity = InvocationIdentity.of(
        executor_identity=executor_identity,
        source=batch.driver_source(),
        input_text="",
        budgets=budgets,
        environment=EnvironmentGrant.fixed({"OPENBLAS_NUM_THREADS": "1"}),
        profile=PROCESS_BOUNDARY_ONLY,
        runtime=HERMETIC,
    )
    return ExecutionRequest(
        batch_request=batch,
        budgets=budgets,
        environment=EnvironmentGrant.fixed({"OPENBLAS_NUM_THREADS": "1"}),
        profile=PROCESS_BOUNDARY_ONLY,
        runtime=HERMETIC,
        computation_id=computation_id,
        identity=identity,
    )


def _run(requests, *, executor, cache=None):
    return run_requests(
        requests,
        executor=executor,
        records=Records.none(),
        cache=cache or InMemoryExecutionCache(),
    )


# ===========================================================================
# ExecutionRequest.cache_key — content-addressed + deterministic.
# ===========================================================================

def test_cache_key_is_a_deterministic_string() -> None:
    assert isinstance(_request().cache_key, str)
    assert len(_request().cache_key) > 0
    assert _request().cache_key == _request().cache_key


@pytest.mark.parametrize(
    ("kwargs_a", "kwargs_b"),
    [
        ({"case_ids": ("case_0",)}, {"case_ids": ("case_1",)}),
        ({"config": "a"}, {"config": "b"}),
        ({"computation_id": "a@v1"}, {"computation_id": "a@v2"}),
        ({"executor_identity": "dr-exec@1"}, {"executor_identity": "dr-exec@2"}),
    ],
)
def test_cache_key_depends_on_each_identity_component(kwargs_a, kwargs_b) -> None:
    assert _request(**kwargs_a).cache_key != _request(**kwargs_b).cache_key


def test_cache_key_folds_in_executor_identity() -> None:
    """A dr-exec version bump moves EXECUTOR_IDENTITY and so the key."""
    a = _request(executor_identity="dr-exec@1.0.0")
    b = _request(executor_identity="dr-exec@1.0.1")
    assert a.cache_key != b.cache_key


# ===========================================================================
# GOLDEN: the exact identity components and their order, pinned.
# ===========================================================================

def test_invocation_identity_key_components_are_pinned() -> None:
    """The cache-key derivation is a persisted contract: pin the exact ordered
    components so any silent change to what the cache distinguishes fails
    loudly here."""
    identity = InvocationIdentity(
        executor_identity="dr-exec@9.9.9",
        source_digest="src-digest",
        input_digest="in-digest",
        budget_wall_clock="2.0",
        budget_output_bytes="1024",
        budget_output_overflow_policy="fail",
        budget_input_bytes="4096",
        unbudgeted_axes=(),
        grant_kind="fixed",
        grant_names=("OPENBLAS_NUM_THREADS",),
        grant_exclusions=(),
        grant_contents_digest="grant-digest",
        profile_name="process_boundary_only",
        runtime_name="hermetic",
    )
    assert identity.key_components() == (
        "executor=dr-exec@9.9.9",
        "source_digest=src-digest",
        "input_digest=in-digest",
        "budget_wall_clock=2.0",
        "budget_output_bytes=1024",
        "budget_output_overflow_policy=fail",
        "budget_input_bytes=4096",
        "unbudgeted_axes=",
        "grant_kind=fixed",
        "grant_names=OPENBLAS_NUM_THREADS",
        "grant_exclusions=",
        "grant_contents_digest=grant-digest",
        "profile=process_boundary_only",
        "runtime=hermetic",
    )


# ===========================================================================
# ExecutionOutcome — structured attribution fields.
# ===========================================================================

def test_execution_outcome_carries_attribution_fields() -> None:
    outcome = ExecutionOutcome(
        attribution=Attribution.PAYLOAD,
        violated_axis=None,
        returncode=0,
        stdout="{}",
        stderr="warn",
        stdout_bytes_dropped=0,
        stderr_bytes_dropped=0,
    )
    assert outcome.attribution is Attribution.PAYLOAD
    assert outcome.returncode == 0
    assert outcome.stderr == "warn"


def test_execution_outcome_is_frozen() -> None:
    outcome = ExecutionOutcome(
        attribution=Attribution.PAYLOAD,
        violated_axis=None,
        returncode=0,
        stdout="",
        stderr="",
        stdout_bytes_dropped=0,
        stderr_bytes_dropped=0,
    )
    with pytest.raises(Exception):  # noqa: PT011
        outcome.returncode = 1  # type: ignore[misc]


def test_timeout_and_output_limit_classification() -> None:
    timeout = ExecutionOutcome(
        attribution=Attribution.BUDGET,
        violated_axis=BudgetAxis.WALL_CLOCK,
        returncode=-9,
        stdout="",
        stderr="",
        stdout_bytes_dropped=0,
        stderr_bytes_dropped=0,
    )
    overflow = ExecutionOutcome(
        attribution=Attribution.BUDGET,
        violated_axis=BudgetAxis.OUTPUT,
        returncode=-9,
        stdout="",
        stderr="",
        stdout_bytes_dropped=0,
        stderr_bytes_dropped=0,
    )
    assert is_timeout_outcome(timeout)
    assert not is_output_limit_outcome(timeout)
    assert is_output_limit_outcome(overflow)
    assert not is_timeout_outcome(overflow)


# ===========================================================================
# InMemoryExecutionCache get/put.
# ===========================================================================

def _outcome() -> ExecutionOutcome:
    return ExecutionOutcome(
        attribution=Attribution.PAYLOAD,
        violated_axis=None,
        returncode=0,
        stdout="[]",
        stderr="",
        stdout_bytes_dropped=0,
        stderr_bytes_dropped=0,
    )


def test_in_memory_cache_miss_returns_none() -> None:
    assert InMemoryExecutionCache().get("nope") is None


def test_in_memory_cache_get_put_round_trip() -> None:
    request = _request()
    outcome = _outcome()
    cache = InMemoryExecutionCache()
    assert cache.get(request.cache_key) is None
    cache.put(request.cache_key, outcome)
    assert cache.get(request.cache_key) == outcome


def test_in_memory_cache_put_overwrites() -> None:
    cache = InMemoryExecutionCache()
    first = _outcome()
    second = ExecutionOutcome(
        attribution=Attribution.BUDGET,
        violated_axis=BudgetAxis.WALL_CLOCK,
        returncode=-9,
        stdout="",
        stderr="",
        stdout_bytes_dropped=0,
        stderr_bytes_dropped=0,
    )
    cache.put("k", first)
    cache.put("k", second)
    assert cache.get("k") == second


# ===========================================================================
# run_requests — dedupe + at-most-once execution.
# ===========================================================================

def test_run_requests_dedupes_identical_requests() -> None:
    from metrics.helpers import fake_executor_scripted

    executor = fake_executor_scripted(full_pass_batch())
    requests = [_request(), _request(), _request()]
    outcomes = _run(requests, executor=executor)
    assert len(executor.batch_calls) == 1
    assert len(outcomes) == 1
    assert set(outcomes) == {requests[0].cache_key}


def test_run_requests_executes_distinct_requests() -> None:
    from metrics.helpers import fake_executor_scripted

    executor = fake_executor_scripted(full_pass_batch(), full_pass_batch())
    _run([_request(config="a"), _request(config="b")], executor=executor)
    assert len(executor.batch_calls) == 2


def test_run_requests_serves_cache_hits_without_running() -> None:
    from metrics.helpers import fake_executor_scripted

    request = _request()
    cache = InMemoryExecutionCache()
    cache.put(request.cache_key, _outcome())

    executor = fake_executor_scripted()  # nothing enqueued: must not be called
    outcomes = _run([request], executor=executor, cache=cache)
    assert len(executor.batch_calls) == 0
    assert outcomes[request.cache_key] == _outcome()


def test_run_requests_populates_cache_for_misses() -> None:
    from metrics.helpers import fake_executor_scripted

    cache = InMemoryExecutionCache()
    request = _request()
    _run([request], executor=fake_executor_scripted(full_pass_batch()), cache=cache)
    assert cache.get(request.cache_key) is not None


def test_run_requests_empty_input_returns_empty_dict() -> None:
    from metrics.helpers import fake_executor_scripted

    assert _run([], executor=fake_executor_scripted()) == {}


def test_run_requests_wall_clock_budget_is_a_cacheable_outcome() -> None:
    from metrics.helpers import fake_executor_scripted

    executor = fake_executor_scripted(
        scripted_batch(case_payloads={}, run=wall_clock_run(_budgets().output))
    )
    cache = InMemoryExecutionCache()
    request = _request()
    outcomes = _run([request], executor=executor, cache=cache)
    outcome = outcomes[request.cache_key]
    assert is_timeout_outcome(outcome)
    assert cache.get(request.cache_key) == outcome


def test_run_requests_output_budget_is_a_cacheable_outcome() -> None:
    from metrics.helpers import fake_executor_scripted

    executor = fake_executor_scripted(
        scripted_batch(case_payloads={}, run=output_budget_run())
    )
    cache = InMemoryExecutionCache()
    request = _request()
    outcomes = _run([request], executor=executor, cache=cache)
    outcome = outcomes[request.cache_key]
    assert is_output_limit_outcome(outcome)
    assert cache.get(request.cache_key) == outcome


def test_run_requests_clean_batch_outcome_is_payload_attributed() -> None:
    from metrics.helpers import fake_executor_scripted

    executor = fake_executor_scripted(full_pass_batch())
    request = _request()
    outcomes = _run([request], executor=executor)
    assert outcomes[request.cache_key].attribution is Attribution.PAYLOAD
