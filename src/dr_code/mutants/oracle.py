"""Typed execution oracle for behavioral mutants.

The oracle runs one untrusted program per call through dr-exec's
:func:`run_untrusted_python` and reads back a sentinel-framed JSON envelope
the child writes to a duplicated trusted descriptor. dr-exec's byte-exact
capture, real dup-able descriptors, and deterministic ``0/1/2`` child fd
table are what make the anti-spoofing protocol sound: a payload that reopens
stdout cannot forge the envelope, because the trusted channel is a dup taken
before the payload runs.

Outcomes are data. Before any envelope parsing, ``_parse_outcomes`` branches
on the run's ``outcome.attribution``: a budget, output, channel, executor,
machine, or absence outcome is an execution failure — never a protocol
violation. Only a run dr-exec attributes to the payload reaches the
sentinel-envelope check.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from typing import Final, Literal, Protocol

from dr_exec import (
    HERMETIC,
    PROCESS_BOUNDARY_ONLY,
    Attribution,
    Budgets,
    ContainmentProfile,
    EnvironmentGrant,
    OutputBudget,
    OverflowPolicy,
    PythonRuntime,
    Records,
    RunResult,
    run_untrusted_python,
)
from pydantic import BaseModel, ConfigDict, ValidationError

from dr_code.mutants.outcomes import ExecutionOutcome

_RESULT_BEGIN: Final = "<<<DR_CODE_MUTANTS_V2_BEGIN>>>"
_RESULT_END: Final = "<<<DR_CODE_MUTANTS_V2_END>>>"
_PROTOCOL_VERSION: Final = 2

# The oracle's declared execution parameters, at the site where the protocol
# knowledge lives. Output overflow is FAIL: a flooding payload is killed and
# budget-attributed, and _parse_outcomes rejects it before reading any
# transcript, so a truncated envelope is never mistaken for a valid one.
_MAX_ORACLE_OUTPUT_BYTES: Final[int] = 1024 * 1024
_MAX_ORACLE_INPUT_BYTES: Final[int] = 4 * 1024 * 1024

# The containment profile and runtime the oracle declares for every program.
_ORACLE_PROFILE: Final[ContainmentProfile] = PROCESS_BOUNDARY_ONLY
_ORACLE_RUNTIME: Final[PythonRuntime] = HERMETIC

# OPENBLAS_NUM_THREADS=1 is a determinism and thread-oversubscription
# control: BLAS thread count changes float reduction order, which the
# double-run determinism gate would see as spurious nondeterminism.
_ORACLE_ENVIRONMENT: Final[EnvironmentGrant] = EnvironmentGrant.fixed(
    {"OPENBLAS_NUM_THREADS": "1"}
)


class PythonRunner(Protocol):
    """The single-program executor the oracle drives (real or fake).

    dr-exec's :func:`run_untrusted_python` and its ``FakeExecutor`` both
    satisfy this: one untrusted-Python run with the oracle's declared
    budgets, grant, profile, and runtime, returning a full ``RunResult``.
    """

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
    ) -> RunResult: ...


class _ProductionRunner:
    """Adapts dr-exec's ``run_untrusted_python`` to the runner protocol.

    The real entry point is a module function, not an object; this thin
    object lets the same call site drive either it or a ``FakeExecutor``
    without a branch. It claims no identity of its own — the run it produces
    carries ``EXECUTOR_IDENTITY``.
    """

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
        return run_untrusted_python(
            source,
            profile=profile,
            budgets=budgets,
            records=records,
            runtime=runtime,
            input_text=input_text,
            environment=environment,
        )


PRODUCTION_RUNNER: Final[PythonRunner] = _ProductionRunner()
"""The default oracle runner: dr-exec's real ``run_untrusted_python``."""


@dataclass(frozen=True, slots=True)
class ProgramOutcomes:
    """Ordered outcomes aligned to one input sequence."""

    outcomes: tuple[ExecutionOutcome, ...]


class OracleError(RuntimeError):
    """The oracle could not produce a complete trustworthy result."""


class OracleInputError(OracleError):
    """Oracle input could not cross the subprocess protocol."""


class OracleExecutionError(OracleError):
    """The subprocess did not complete normally."""


class OracleProtocolError(OracleError):
    """The subprocess response violated the oracle protocol."""


class _WireEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    protocol_version: Literal[2]
    invocation_id: str
    outcomes: list[ExecutionOutcome]


@dataclass(frozen=True, slots=True)
class GateReport:
    """Determinism and divergence gate results."""

    canonical_deterministic: bool
    mutant_deterministic: bool
    distinct_input_count: int

    @property
    def accepted(self) -> bool:
        return (
            self.canonical_deterministic
            and self.mutant_deterministic
            and self.distinct_input_count >= 1
        )

    def rejection_reason(self) -> str | None:
        if not self.canonical_deterministic:
            return "canonical is non-deterministic across two runs"
        if not self.mutant_deterministic:
            return "mutant is non-deterministic across two runs"
        if self.distinct_input_count < 1:
            return "behaviorally identical to canonical on every input"
        return None


_RUNNER_SOURCE: Final = f"""
import ast
import json
import os
import sys

def _main():
    payload = json.loads(sys.stdin.read())
    trusted_fd = os.dup(1)
    try:
        with open(os.devnull, "w") as discarded:
            os.dup2(discarded.fileno(), 1)
            namespace = {{"__name__": "__dr_code_mutant__"}}
            exec(payload["program"], namespace)
            function = namespace[payload["entry_point"]]
            outcomes = []
            for input_repr in payload["input_reprs"]:
                args = ast.literal_eval(input_repr)
                try:
                    value = function(*args)
                except Exception as exc:
                    outcomes.append(
                        {{
                            "kind": "error",
                            "exception_type": (
                                type(exc).__module__
                                + "."
                                + type(exc).__qualname__
                            ),
                            "exception_args_repr": repr(exc.args),
                        }}
                    )
                else:
                    outcomes.append(
                        {{"kind": "value", "value_repr": repr(value)}}
                    )
        envelope = {{
            "invocation_id": payload["invocation_id"],
            "protocol_version": {_PROTOCOL_VERSION},
            "outcomes": outcomes,
        }}
        encoded = (
            {_RESULT_BEGIN!r}
            + json.dumps(envelope, sort_keys=True)
            + {_RESULT_END!r}
        ).encode()
        os.write(trusted_fd, encoded)
    finally:
        os.close(trusted_fd)

_main()
"""


def run_program_on_inputs(
    *,
    program: str,
    entry_point: str,
    input_reprs: tuple[str, ...],
    timeout_seconds: float,
    runner: PythonRunner = PRODUCTION_RUNNER,
) -> ProgramOutcomes:
    """Run one program on literal argument tuples in a fresh process."""

    invocation_id = secrets.token_hex(32)
    try:
        input_text = json.dumps(
            {
                "entry_point": entry_point,
                "input_reprs": input_reprs,
                "invocation_id": invocation_id,
                "program": program,
            },
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise OracleInputError(
            "oracle input is not JSON serializable"
        ) from exc
    result = runner.run_untrusted_python(
        _RUNNER_SOURCE,
        profile=_ORACLE_PROFILE,
        budgets=Budgets(
            wall_clock=timeout_seconds,
            output=OutputBudget(
                limit_bytes=_MAX_ORACLE_OUTPUT_BYTES,
                overflow_policy=OverflowPolicy.FAIL,
            ),
            input=_MAX_ORACLE_INPUT_BYTES,
        ),
        records=Records.none(),
        runtime=_ORACLE_RUNTIME,
        input_text=input_text,
        environment=_ORACLE_ENVIRONMENT,
    )
    return _parse_outcomes(
        result,
        expected_count=len(input_reprs),
        expected_invocation_id=invocation_id,
    )


def _parse_outcomes(
    result: RunResult,
    *,
    expected_count: int,
    expected_invocation_id: str,
) -> ProgramOutcomes:
    _reject_execution_failure(result)
    stdout = result.stdout
    if not stdout.startswith(_RESULT_BEGIN) or not stdout.endswith(
        _RESULT_END
    ):
        raise OracleProtocolError(
            "oracle did not emit one complete final envelope"
        )
    payload_start = len(_RESULT_BEGIN)
    end = len(stdout) - len(_RESULT_END)
    try:
        envelope = _WireEnvelope.model_validate_json(stdout[payload_start:end])
    except ValidationError as exc:
        raise OracleProtocolError(
            "oracle result did not match protocol version 2"
        ) from exc
    if not secrets.compare_digest(
        envelope.invocation_id,
        expected_invocation_id,
    ):
        raise OracleProtocolError("oracle invocation binding mismatch")
    if len(envelope.outcomes) != expected_count:
        raise OracleProtocolError(
            f"oracle returned {len(envelope.outcomes)} outcomes; "
            f"expected {expected_count}"
        )
    return ProgramOutcomes(outcomes=tuple(envelope.outcomes))


def _reject_execution_failure(result: RunResult) -> None:
    """Turn any non-payload attribution into an execution failure.

    Mandatory pre-branch before the envelope check: a budget deadline, an
    output overflow, a protocol-channel budget, an executor or machine
    failure, or an absent interpreter is an execution failure that never
    produced a trustworthy envelope. Only a run dr-exec attributes to the
    payload — the child ran and exited on its own terms — is read as a
    protocol response.
    """
    attribution = result.outcome.attribution
    if attribution is Attribution.PAYLOAD:
        if result.returncode == 0:
            return
        detail = result.stderr.strip().replace("\n", " ")[:200]
        raise OracleExecutionError(
            f"oracle child exited {result.returncode}"
            + (f": {detail}" if detail else "")
        )
    if (
        attribution is Attribution.BUDGET
        and result.outcome.violated_axis is not None
    ):
        raise OracleExecutionError(
            f"oracle run exceeded its {result.outcome.violated_axis.value} "
            "budget"
        )
    raise OracleExecutionError(
        f"oracle run did not complete: {attribution.value} failure"
    )


def distinct_input_indices(
    canonical: ProgramOutcomes,
    mutant: ProgramOutcomes,
) -> tuple[int, ...]:
    """Return aligned input indices whose outcomes differ."""

    if len(canonical.outcomes) != len(mutant.outcomes):
        raise OracleProtocolError("outcome sequences are not aligned")
    return tuple(
        index
        for index, (expected, observed) in enumerate(
            zip(canonical.outcomes, mutant.outcomes, strict=True)
        )
        if expected != observed
    )


def evaluate_gates(
    *,
    canonical_first: ProgramOutcomes,
    canonical_second: ProgramOutcomes,
    mutant_first: ProgramOutcomes,
    mutant_second: ProgramOutcomes,
) -> GateReport:
    """Accept only deterministic canonical and mutant divergence."""

    if len(canonical_first.outcomes) != len(canonical_second.outcomes):
        raise OracleProtocolError(
            "canonical outcome sequences are not aligned"
        )
    if len(mutant_first.outcomes) != len(mutant_second.outcomes):
        raise OracleProtocolError("mutant outcome sequences are not aligned")
    distinct = distinct_input_indices(canonical_first, mutant_first)
    return GateReport(
        canonical_deterministic=canonical_first == canonical_second,
        mutant_deterministic=mutant_first == mutant_second,
        distinct_input_count=len(distinct),
    )


__all__ = (
    "PRODUCTION_RUNNER",
    "GateReport",
    "OracleError",
    "OracleExecutionError",
    "OracleInputError",
    "OracleProtocolError",
    "ProgramOutcomes",
    "PythonRunner",
    "distinct_input_indices",
    "evaluate_gates",
    "run_program_on_inputs",
)
