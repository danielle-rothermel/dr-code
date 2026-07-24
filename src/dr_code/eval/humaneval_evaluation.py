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

from collections.abc import Mapping

from dr_code.eval.aggregation import (
    AggregationInput,
    AggregationOutput,
    aggregate,
)
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
from dr_code.eval.lifecycle import (
    AggregationConfig,
    AggregationDefinition,
    EvaluationProcedureConfig,
    EvaluationProcedureDefinition,
    MetricExtractionDefinition,
    MetricQuestionBinding,
    PreprocessingDefinition,
    PreprocessingStepBinding,
)
from dr_code.eval.resolved_versions import resolved_operator_version
from dr_code.metrics.definition import MetricsDefinition
from dr_code.metrics.names import MetricName
from dr_code.metrics.operators.code_test import CodeTestResult
from dr_code.preprocessing.definition import (
    PreprocessingDefinition as OperationalPreprocessingDefinition,
)

# The canonical operator coordinate for HumanEval case execution.
CODE_TEST_OPERATOR = str(MetricName.CODE_TEST)
CODE_TEST_QUESTION = "humaneval_case_execution"
CODE_TEST_ON_KEY = "code"


# The canonical identity scheme label for corpus artifacts written under
# the eval kernel. Distinct from the legacy BLAKE2 stable_hash scheme; the
# two identity families must never be mixed under one label (guide step 9).
IDENTITY_SCHEME = "eval_kernel_v1"


def kernel_preprocessing_definition(
    operational: OperationalPreprocessingDefinition,
) -> PreprocessingDefinition:
    """Losslessly map an operational definition to the kernel definition.

    The operational ``StepSpec(instance_name, step, settings)`` maps directly
    onto the kernel ``PreprocessingStepBinding``. Ordered settings preserve
    every key. The result carries the canonical eval-kernel identity so new
    corpus artifacts use one identity family instead of the legacy
    ``preprocessing_definition_hash`` (BLAKE2) helper.
    """

    return PreprocessingDefinition(
        definition_id=operational.definition_id,
        version=operational.version,
        steps=tuple(
            PreprocessingStepBinding(
                instance_name=spec.instance_name,
                step=str(spec.step),
                settings=tuple(sorted(spec.settings.items())),
            )
            for spec in operational.steps
        ),
    )


def kernel_metric_extraction_definition(
    operational: MetricsDefinition,
) -> MetricExtractionDefinition:
    """Losslessly map an operational MetricsDefinition to the kernel form.

    Operational ``MetricQuestion(metric, on, settings)`` maps directly onto
    the kernel ``MetricQuestionBinding``; ordered settings preserve every key.
    """

    return MetricExtractionDefinition(
        definition_id=operational.definition_id,
        version=operational.version,
        questions=tuple(
            MetricQuestionBinding(
                metric=str(question.metric),
                on=question.on,
                settings=tuple(sorted(question.settings.items())),
            )
            for question in operational.questions
        ),
    )


def evaluation_procedure_config(
    *,
    preprocessing: OperationalPreprocessingDefinition,
    metrics: MetricsDefinition,
    procedure_definition_id: str = "humaneval-evaluation-procedure",
    procedure_version: str = "v1",
) -> EvaluationProcedureConfig:
    """Compose the canonical Evaluation Procedure Config for HumanEval.

    Folds the resolved preprocessing + metric-extraction Config identities
    (which include resolved step/operator versions) into one canonical
    Procedure Config identity. This is the ``evaluation_procedure_config_hash``
    every candidate execution and Metric Fact is lineage-stamped with.
    """

    preprocessing_config = kernel_preprocessing_definition(
        preprocessing
    ).materialize()
    metric_extraction_config = kernel_metric_extraction_definition(
        metrics
    ).materialize()
    return EvaluationProcedureDefinition(
        definition_id=procedure_definition_id,
        version=procedure_version,
    ).materialize(
        preprocessing=preprocessing_config,
        metric_extraction=metric_extraction_config,
        # Explicit zero-denominator policy; the procedure definition requires
        # it. "not_applicable" means an empty denominator yields an absent
        # record rather than an operator error.
        assignment={"zero_denominator": "not_applicable"},
    )


def evaluation_procedure_config_identity(
    *,
    preprocessing: OperationalPreprocessingDefinition,
    metrics: MetricsDefinition,
) -> str:
    """The canonical Evaluation Procedure Config identity hash."""

    return evaluation_procedure_config(
        preprocessing=preprocessing, metrics=metrics
    ).config_identity_hash


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


def pass_rate_aggregation_config() -> AggregationConfig:
    """The explicit reduction policy for a candidate pass *rate*.

    A pass rate is a ``mean`` over per-candidate pass indicators. The policy
    is explicit end to end: missing indicators propagate (a missing outcome
    is not silently treated as a failure), and an empty denominator is
    ``not_applicable`` rather than an error. This is the canonical reduction
    the descriptive analysis rate must reconcile with (guide step 7).
    """

    definition = AggregationDefinition(
        definition_id="humaneval-pass-rate", version="v1"
    )
    return definition.materialize(
        assignment={
            "reduction": "mean",
            "missing_data": "propagate",
            "zero_denominator": "not_applicable",
        }
    )


def pass_rate(
    pass_indicators: tuple[bool | None, ...],
    *,
    config: AggregationConfig | None = None,
) -> AggregationOutput:
    """Reduce per-candidate pass indicators into a pass rate via the kernel.

    ``None`` marks a candidate whose outcome is missing (not a failure). The
    reduction is the kernel's pure :func:`aggregate`, so the rate carries an
    explicit status and the exact contributing counts.
    """

    resolved = config or pass_rate_aggregation_config()
    inputs = tuple(
        AggregationInput(value=None if indicator is None else float(indicator))
        for indicator in pass_indicators
    )
    return aggregate(resolved, inputs)


def record_from_result_row(
    row: Mapping[str, object],
    *,
    evaluation_procedure_config_hash: str,
) -> MetricRecord:
    """Derive an eval-kernel Metric Record from one persisted result row.

    The candidate-evaluation parquet already carries the raw CodeTestResult
    fields losslessly; this reconstructs the neutral kernel record from a row
    so analysis derives records/scores from facts rather than a pre-reduced
    outcome. A row whose ``record_status`` is not ``measured`` becomes the
    matching non-measured record (operator failure or empty/absent), never a
    silent success.
    """

    status = str(row.get("record_status"))
    if status == str(RecordStatus.MEASURED):
        result = CodeTestResult(
            total_cases=_row_int(row, "total_cases"),
            passed_count=_row_int(row, "passed_count"),
            failed_count=_row_int(row, "failed_count"),
            error_count=_row_int(row, "error_count"),
            timeout_count=_row_int(row, "timeout_count"),
            coverage_complete=bool(row.get("coverage_complete")),
            function_count=_row_int(row, "function_count"),
            best_function_name=(
                str(row["best_function_name"])
                if row.get("best_function_name") is not None
                else None
            ),
        )
        return code_test_record(
            result,
            evaluation_procedure_config_hash=(
                evaluation_procedure_config_hash
            ),
        )

    failure_type = row.get("failure_type")
    return MetricRecord(
        question=CODE_TEST_QUESTION,
        on_key=CODE_TEST_ON_KEY,
        evaluation_procedure_config_hash=evaluation_procedure_config_hash,
        status=RecordStatus.OPERATOR_FAILURE,
        failure_type=(
            str(failure_type)
            if failure_type is not None
            else "evaluation_incomplete"
        ),
        failure_message=(
            str(row["failure_message"])
            if row.get("failure_message") is not None
            else "candidate evaluation did not complete"
        ),
    )


def _row_int(row: Mapping[str, object], key: str) -> int:
    value = row.get(key)
    if not isinstance(value, (int, bool)):
        raise ValueError(
            f"measured result row {key!r} must be an integer, got {value!r}"
        )
    return int(value)


__all__ = [
    "CODE_TEST_ON_KEY",
    "CODE_TEST_OPERATOR",
    "CODE_TEST_QUESTION",
    "IDENTITY_SCHEME",
    "candidate_content_identity",
    "candidate_execution_identity",
    "code_test_facts",
    "code_test_lineage",
    "code_test_record",
    "compile_facts_for_candidate",
    "empty_candidate_set_record",
    "evaluation_procedure_config",
    "evaluation_procedure_config_identity",
    "kernel_metric_extraction_definition",
    "kernel_preprocessing_definition",
    "pass_rate",
    "pass_rate_aggregation_config",
    "record_from_result_row",
]
