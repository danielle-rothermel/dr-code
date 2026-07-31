"""Content-addressed execution requests and cache orchestration.

The execution boundary runs one dr-exec batch per request and caches its
outcome. A request's identity is ``computation_id`` (dr-code's deliberate
domain namespace) plus dr-exec's declared invocation identity: which executor
would run it, the source and input it would carry, the budgets in force, the
environment grant, and the containment profile and runtime. Two requests that
would spawn the identical run share a cache entry; anything that changes what
would spawn — including a dr-exec version bump moving ``EXECUTOR_IDENTITY`` —
is a distinct key.

Outcomes are data: a batch that timed out, overflowed its output budget, or
failed to start is a value carrying dr-exec's attribution, never an exception.
Only a genuine executor failure with no result (``ExecutorFailure``) or a
pre-spawn declaration error (``DeclarationError``) escapes as an exception.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Protocol

from dr_exec import (
    Attribution,
    BatchRequest,
    BatchResult,
    BudgetAxis,
    Budgets,
    ContainmentProfile,
    EnvironmentGrant,
    ExecutorFailure,
    Measurements,
    Outcome,
    PythonRuntime,
    Records,
    RunResult,
    TruncationMark,
    contents_digest_of,
    serialize_budgets,
    serialize_grant,
)
from dr_exec.batch import account_transcript

# The single hash family for execution-cache keys: BLAKE2b hex over the
# newline-joined identity components. Kept separate from dr-exec's SHA-256
# content digests, which are themselves *inputs* to this key.
_KEY_HASH = hashlib.blake2b


@dataclass(frozen=True, slots=True)
class InvocationIdentity:
    """dr-exec's declared identity for the run a request would spawn.

    Every field is a string or digest that answers "what would spawn": the
    executor that would run it, the content-addressed source and input, the
    declared budgets and grant, and the profile and runtime. Values never
    enter the identity — the grant contributes its introspectable names and a
    value-sensitive digest, never its secrets.
    """

    executor_identity: str
    source_digest: str
    input_digest: str
    budget_wall_clock: str
    budget_output_bytes: str
    budget_output_overflow_policy: str
    budget_input_bytes: str
    unbudgeted_axes: tuple[str, ...]
    grant_kind: str
    grant_names: tuple[str, ...]
    grant_exclusions: tuple[str, ...]
    grant_contents_digest: str
    profile_name: str
    runtime_name: str

    @classmethod
    def of(
        cls,
        *,
        executor_identity: str,
        source: str,
        input_text: str,
        budgets: Budgets,
        environment: EnvironmentGrant,
        profile: ContainmentProfile,
        runtime: PythonRuntime,
    ) -> InvocationIdentity:
        serialized_budgets = serialize_budgets(budgets)
        grant = serialize_grant(environment)
        return cls(
            executor_identity=executor_identity,
            source_digest=contents_digest_of(source),
            input_digest=contents_digest_of(input_text),
            budget_wall_clock=str(serialized_budgets.wall_clock_seconds),
            budget_output_bytes=str(serialized_budgets.output_bytes),
            budget_output_overflow_policy=(
                serialized_budgets.output_overflow_policy
            ),
            budget_input_bytes=str(serialized_budgets.input_bytes),
            unbudgeted_axes=serialized_budgets.unbudgeted_axes,
            grant_kind=str(grant["grant_kind"]),
            grant_names=tuple(grant["grant_names"]),
            grant_exclusions=tuple(grant["grant_exclusions"]),
            grant_contents_digest=str(grant["grant_contents_digest"]),
            profile_name=profile.name,
            runtime_name=runtime.name,
        )

    def key_components(self) -> tuple[str, ...]:
        """The ordered identity components folded into the cache key.

        Order and membership are pinned by a golden test so a silent drift of
        what the cache distinguishes is a loud failure.
        """
        return (
            f"executor={self.executor_identity}",
            f"source_digest={self.source_digest}",
            f"input_digest={self.input_digest}",
            f"budget_wall_clock={self.budget_wall_clock}",
            f"budget_output_bytes={self.budget_output_bytes}",
            f"budget_output_overflow_policy="
            f"{self.budget_output_overflow_policy}",
            f"budget_input_bytes={self.budget_input_bytes}",
            f"unbudgeted_axes={','.join(self.unbudgeted_axes)}",
            f"grant_kind={self.grant_kind}",
            f"grant_names={','.join(self.grant_names)}",
            f"grant_exclusions={','.join(self.grant_exclusions)}",
            f"grant_contents_digest={self.grant_contents_digest}",
            f"profile={self.profile_name}",
            f"runtime={self.runtime_name}",
        )


@dataclass(frozen=True)
class ExecutionRequest:
    """One dr-exec batch to run, plus the identity that caches its outcome.

    ``batch_request`` is the dr-exec request the executor runs; ``budgets``,
    ``environment``, ``profile``, and ``runtime`` are the declared execution
    parameters the batch call carries. ``computation_id`` is dr-code's domain
    namespace, retained from the prior key shape; the rest of the key is
    dr-exec's declared invocation identity.
    """

    batch_request: BatchRequest
    budgets: Budgets
    environment: EnvironmentGrant
    profile: ContainmentProfile
    runtime: PythonRuntime
    computation_id: str
    identity: InvocationIdentity

    @cached_property
    def cache_key(self) -> str:
        components = (
            f"computation_id={self.computation_id}",
            *self.identity.key_components(),
        )
        blob = "\n".join(components).encode("utf-8")
        return _KEY_HASH(blob).hexdigest()


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    """The cacheable execution facts of one batch run.

    Attribution is dr-exec's: exactly one of the ``Attribution`` literals, with
    the violated axis named only for a budget outcome. The captured protocol
    transcript (``stdout``) and payload stream (``stderr``) are byte-exact for
    what was retained; ``*_bytes_dropped`` names what a budget bound cut. The
    domain layer reconstructs the batch's per-item results from these facts, so
    execution stores what happened and the operator decides what it means.
    """

    attribution: Attribution
    violated_axis: BudgetAxis | None
    returncode: int | None
    stdout: str
    stderr: str
    stdout_bytes_dropped: int
    stderr_bytes_dropped: int

    @classmethod
    def from_batch_result(cls, result: BatchResult) -> ExecutionOutcome:
        run = result.run
        return cls(
            attribution=run.outcome.attribution,
            violated_axis=run.outcome.violated_axis,
            returncode=run.returncode,
            stdout=run.stdout,
            stderr=run.stderr,
            stdout_bytes_dropped=run.truncation.stdout_bytes_dropped,
            stderr_bytes_dropped=run.truncation.stderr_bytes_dropped,
        )

    def as_run_result(self) -> RunResult:
        """Rebuild the dr-exec ``RunResult`` the transcript accounting reads.

        Only the fields the batch accounting and the domain mapping consult
        are reconstructed; timing measurements are execution-varying and stay
        out of the cache, so they are reported as zero on the rehydrated run.
        """
        return RunResult(
            returncode=self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
            truncation=TruncationMark(
                stdout_bytes_dropped=self.stdout_bytes_dropped,
                stderr_bytes_dropped=self.stderr_bytes_dropped,
            ),
            measurements=Measurements(
                duration_seconds=0.0,
                teardown_seconds=0.0,
                stdout_bytes_produced=len(self.stdout.encode("utf-8"))
                + self.stdout_bytes_dropped,
                stderr_bytes_produced=len(self.stderr.encode("utf-8"))
                + self.stderr_bytes_dropped,
                input_bytes=0,
            ),
            outcome=Outcome(
                attribution=self.attribution,
                violated_axis=self.violated_axis,
            ),
        )

    def batch_result_for(self, request: BatchRequest) -> BatchResult:
        """Rehydrate the batch result: dr-exec's accounting over the transcript."""
        return account_transcript(request=request, run=self.as_run_result())


class Executor(Protocol):
    """The batch-running executor the engine drives.

    dr-exec's real entry point and its ``FakeExecutor`` both satisfy this: a
    single ``run_batch`` that spawns one warm child per request and returns
    the accounted batch result.
    """

    def run_batch(
        self,
        request: BatchRequest,
        *,
        profile: ContainmentProfile,
        budgets: Budgets,
        records: Records,
        runtime: PythonRuntime,
        environment: EnvironmentGrant,
    ) -> BatchResult: ...


class ExecutionCache(Protocol):
    """Outcome cache keyed by ``ExecutionRequest.cache_key``.

    The key folds in dr-exec's ``EXECUTOR_IDENTITY``, so a cache instance is
    already scoped to the executor that produced its entries: an outcome from
    one executor can never be served for a request another would run.
    """

    def get(self, key: str) -> ExecutionOutcome | None: ...

    def put(self, key: str, outcome: ExecutionOutcome) -> None: ...


class InMemoryExecutionCache:
    """A process-local execution outcome cache."""

    def __init__(self) -> None:
        self._outcomes: dict[str, ExecutionOutcome] = {}

    def get(self, key: str) -> ExecutionOutcome | None:
        return self._outcomes.get(key)

    def put(self, key: str, outcome: ExecutionOutcome) -> None:
        self._outcomes[key] = outcome


def run_requests(
    requests: Sequence[ExecutionRequest],
    *,
    executor: Executor,
    records: Records,
    cache: ExecutionCache,
) -> dict[str, ExecutionOutcome]:
    """Run each distinct cache miss at most once.

    Budget, channel, machine, and absence outcomes are all data: the batch
    result carries dr-exec's attribution and the outcome is cached. A genuine
    ``ExecutorFailure`` — a failure with no result to attribute — propagates,
    and its (non-existent) outcome is never cached.
    """
    outcomes: dict[str, ExecutionOutcome] = {}
    unique_requests: dict[str, ExecutionRequest] = {}
    for request in requests:
        unique_requests.setdefault(request.cache_key, request)

    for key, request in unique_requests.items():
        cached = cache.get(key)
        if cached is not None:
            outcomes[key] = cached
            continue
        result = executor.run_batch(
            request.batch_request,
            profile=request.profile,
            budgets=request.budgets,
            records=records,
            runtime=request.runtime,
            environment=request.environment,
        )
        outcome = ExecutionOutcome.from_batch_result(result)
        cache.put(key, outcome)
        outcomes[key] = outcome
    return outcomes


def is_timeout_outcome(outcome: ExecutionOutcome) -> bool:
    """Whether a batch died on its wall-clock budget (scored against candidate)."""
    return (
        outcome.attribution is Attribution.BUDGET
        and outcome.violated_axis is BudgetAxis.WALL_CLOCK
    )


def is_output_limit_outcome(outcome: ExecutionOutcome) -> bool:
    """Whether a batch died on its output budget (scored against candidate)."""
    return (
        outcome.attribution is Attribution.BUDGET
        and outcome.violated_axis is BudgetAxis.OUTPUT
    )


__all__ = [
    "ExecutionCache",
    "ExecutionOutcome",
    "ExecutionRequest",
    "Executor",
    "ExecutorFailure",
    "InMemoryExecutionCache",
    "InvocationIdentity",
    "is_output_limit_outcome",
    "is_timeout_outcome",
    "run_requests",
]
