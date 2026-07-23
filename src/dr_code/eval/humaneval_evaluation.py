"""Crosswalk seam: HumanEval candidate evaluation onto the eval kernel.

This is the single canonical mapping from the operational HumanEval
execution result (``dr_code.metrics.operators.code_test.CodeTestResult``)
onto the kernel's canonical identities and Metric Facts/Records. It exists
so new persisted evaluation artifacts can be written under one identity and
model family (the eval kernel) instead of the corpus stack's private hash
helpers and outcome records.

Design decisions honored here:

- **Canonical candidate-execution identity.** Built from the four
  coordinates the integration guide names: HumanEval task identity,
  candidate content identity, evaluation-procedure (metric + procedure)
  configuration identity, and the explicit execution/runtime fingerprint.
  RNG seeds never enter this identity (they are Repeat slot data).
- **Lossless facts.** Every field of ``CodeTestResult`` becomes an explicit
  :class:`MetricFact` with a unit and resolved operator lineage. Nothing is
  dropped.
- **Empty vs. failure.** A processed input that yields zero candidates is a
  valid empty result (``AbsenceMode.EMPTY_CANDIDATE_SET``), never a
  Preprocessing Failure. A causal preprocessing failure keeps its native
  ``Absent`` role and terminal cause upstream; this module only maps the
  execution-stage record.

This module is additive: it introduces no change to any existing
``dr_code.eval`` symbol's signature or behavior.
"""

from __future__ import annotations

from dr_code.eval.code import CodeArtifact
from dr_code.eval.facts import (
    AbsenceMode,
    Applicability,
    FactScalar,
    MetricFact,
    MetricRecord,
    OperatorLineage,
    RecordStatus,
)
from dr_code.eval.identity import (
    SCHEMA_CANDIDATE_EXECUTION,
    identity_hash_for,
)
from dr_code.eval.resolved_versions import resolved_operator_version
from dr_code.metrics.names import MetricName
from dr_code.metrics.operators.code_test import CodeTestResult

# The canonical operator coordinate for HumanEval case execution.
CODE_TEST_OPERATOR = str(MetricName.CODE_TEST)
CODE_TEST_QUESTION = "humaneval_case_execution"
CODE_TEST_ON_KEY = "code"


def candidate_content_identity(source: str) -> str:
    """Identity of one candidate's exact source content."""

    return identity_hash_for(
        schema="dr_code.candidate_content",
        payload={"source": source},
    )


def candidate_execution_identity(
    *,
    task_identity: str,
    candidate_source: str,
    evaluation_procedure_config_hash: str,
    execution_fingerprint: str,
) -> str:
    """Canonical identity of one candidate execution on the kernel.

    Composes the four coordinates the integration guide requires; the
    execution is reproducible from exactly these. RNG seeds are excluded by
    construction (identity invariance is proved in the eval task tests).
    """

    return identity_hash_for(
        schema=SCHEMA_CANDIDATE_EXECUTION,
        payload={
            "task_identity": task_identity,
            "candidate_content_identity": candidate_content_identity(
                candidate_source
            ),
            "evaluation_procedure_config_hash": (
                evaluation_procedure_config_hash
            ),
            "execution_fingerprint": execution_fingerprint,
        },
    )


def code_test_lineage(
    evaluation_procedure_config_hash: str,
) -> OperatorLineage:
    """Resolved operator lineage for the code-test operator."""

    return OperatorLineage(
        evaluation_procedure_config_hash=evaluation_procedure_config_hash,
        operator=CODE_TEST_OPERATOR,
        operator_version=resolved_operator_version(CODE_TEST_OPERATOR),
    )


def code_test_facts(
    result: CodeTestResult,
    *,
    evaluation_procedure_config_hash: str,
) -> tuple[MetricFact, ...]:
    """Represent every ``CodeTestResult`` field as a Metric Fact.

    Each fact carries an explicit unit and the resolved operator lineage.
    The mapping is lossless: one fact per descriptive field, so downstream
    records/scores/aggregation derive from neutral facts rather than a
    pre-reduced binary outcome.
    """

    lineage = code_test_lineage(evaluation_procedure_config_hash)

    def _fact(name: str, value: FactScalar, unit: str) -> MetricFact:
        return MetricFact(
            name=name,
            value=value,
            unit=unit,
            applicability=Applicability.APPLICABLE,
            lineage=lineage,
        )

    facts: list[MetricFact] = [
        _fact("total_cases", result.total_cases, "case"),
        _fact("passed_count", result.passed_count, "case"),
        _fact("failed_count", result.failed_count, "case"),
        _fact("error_count", result.error_count, "case"),
        _fact("timeout_count", result.timeout_count, "case"),
        _fact(
            "coverage_complete",
            result.coverage_complete,
            "boolean",
        ),
        _fact("function_count", result.function_count, "function"),
    ]
    if result.best_function_name is not None:
        facts.append(
            _fact("best_function_name", result.best_function_name, "name")
        )
    return tuple(facts)


def code_test_record(
    result: CodeTestResult,
    *,
    evaluation_procedure_config_hash: str,
) -> MetricRecord:
    """Derive a measured Metric Record from a code-test result."""

    return MetricRecord(
        question=CODE_TEST_QUESTION,
        on_key=CODE_TEST_ON_KEY,
        evaluation_procedure_config_hash=evaluation_procedure_config_hash,
        status=RecordStatus.MEASURED,
        facts=code_test_facts(
            result,
            evaluation_procedure_config_hash=(
                evaluation_procedure_config_hash
            ),
        ),
    )


def empty_candidate_set_record(
    *,
    evaluation_procedure_config_hash: str,
    cause: str = "preprocessing produced zero candidates",
) -> MetricRecord:
    """A valid empty-candidate-set record (not a Preprocessing Failure).

    Distinguished explicitly from a causal ``Absent``: the input processed
    successfully and simply yielded no candidate to execute.
    """

    return MetricRecord(
        question=CODE_TEST_QUESTION,
        on_key=CODE_TEST_ON_KEY,
        evaluation_procedure_config_hash=evaluation_procedure_config_hash,
        status=RecordStatus.NOT_APPLICABLE,
        absence_mode=AbsenceMode.EMPTY_CANDIDATE_SET,
        absence_cause=cause,
    )


def compile_facts_for_candidate(source: str) -> bool:
    """Whether a candidate source compiles (kernel Code Artifact gate).

    A thin bridge to the kernel's compile-validating artifact so callers
    can classify a compile failure without duplicating the compile gate.
    """

    try:
        CodeArtifact(source=source)
    except Exception:  # noqa: BLE001 - compile failure is the signal
        return False
    return True


__all__ = [
    "CODE_TEST_ON_KEY",
    "CODE_TEST_OPERATOR",
    "CODE_TEST_QUESTION",
    "candidate_content_identity",
    "candidate_execution_identity",
    "code_test_facts",
    "code_test_lineage",
    "code_test_record",
    "compile_facts_for_candidate",
    "empty_candidate_set_record",
]
