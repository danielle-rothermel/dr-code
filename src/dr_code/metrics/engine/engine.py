"""Bind, plan, execute, and compute declared metric questions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from dr_code.eval.facts import (
    AbsenceMode,
    Applicability,
    EvaluationProcedureConfigHash,
    MetricFact,
    MetricRecord,
    OperatorCoordinates,
    OperatorLineage,
    validate_evaluation_procedure_config_hash,
)
from dr_code.eval.lifecycle import (
    EvaluationProcedureConfig,
    MetricExtractionConfig,
    MetricQuestionBinding,
)
from dr_code.execution.subprocess import (
    SubprocessError,
    PythonSubprocessRunner,
    run_python_subprocess,
)
from dr_code.metrics.engine.execution import (
    ExecutionCache,
    ExecutionOutcome,
    ExecutionRequest,
    InMemoryExecutionCache,
    run_requests,
)
from dr_code.metrics.engine.views import ViewCache
from dr_code.metrics.operators.base import MetricOperator
from dr_code.metrics.validation import validated_metric_operator
from dr_code.trace import Absent, Artifact, Trace, TraceProducer, WiringError
from dr_code.eval.resolved_versions import implementation_identity
from dr_code.trace.observation import SampleIdentity


@dataclass(frozen=True, slots=True)
class _QuestionBinding:
    question: MetricQuestionBinding
    operator: MetricOperator


@dataclass(slots=True)
class _TraceBinding:
    trace: Trace
    question_binding: _QuestionBinding  # pairing validated in _bind_questions
    value: Artifact | None
    auxiliary: dict[str, Artifact]
    absence: Absent | None
    planning_failure: Exception | None = None


@dataclass(frozen=True, slots=True)
class _RecordIdentity:
    question: str
    question_identity_hash: str
    on_key: str
    evaluation_procedure_config_hash: EvaluationProcedureConfigHash
    trace_producer: TraceProducer
    sample_identity: SampleIdentity | None
    operator: OperatorCoordinates
    lineage: OperatorLineage

    def __post_init__(self) -> None:
        validate_evaluation_procedure_config_hash(
            self.evaluation_procedure_config_hash
        )

    @classmethod
    def from_binding(
        cls,
        procedure: EvaluationProcedureConfig,
        binding: _TraceBinding,
    ) -> _RecordIdentity:
        question_binding = binding.question_binding
        question = question_binding.question
        return cls(
            question=question.metric,
            question_identity_hash=question.identity_hash(),
            on_key=question.on,
            evaluation_procedure_config_hash=(procedure.config_identity_hash),
            trace_producer=binding.trace.producer,
            sample_identity=binding.trace.sample_identity,
            operator=OperatorCoordinates(
                name=question.metric,
                version=question_binding.operator.VERSION,
                implementation_hash=implementation_identity(
                    type(question_binding.operator)
                ),
                settings=tuple(
                    sorted(
                        question_binding.operator.settings.model_dump(
                            mode="json"
                        ).items()
                    )
                ),
            ),
            lineage=OperatorLineage(
                evaluation_procedure_config_hash=(
                    procedure.config_identity_hash
                ),
                question_identity_hash=question.identity_hash(),
                operator=question.metric,
                operator_version=question_binding.operator.VERSION,
                operator_implementation=implementation_identity(
                    type(question_binding.operator)
                ),
            ),
        )


class EngineInvariantError(Exception):
    """Raised when the engine's own bind/plan/execute invariants break.

    For example: an operator's ``compute`` rebuilds an ``ExecutionRequest``
    that diverges from what ``execution_requests`` planned, so no outcome
    was ever computed for it. This is an engine bug, not a metric bug, and
    must never be attributed to the operator as an ``operator_failure``.
    """


@dataclass(frozen=True, slots=True)
class _EngineContext:
    views: ViewCache
    outcomes: Mapping[str, ExecutionOutcome]

    def outcome_for(self, request: ExecutionRequest) -> ExecutionOutcome:
        try:
            return self.outcomes[request.cache_key]
        except KeyError as exc:
            raise EngineInvariantError(
                f"no execution outcome planned for request "
                f"{request.cache_key!r}"
            ) from exc


def extract_metrics(
    trace: Trace,
    *,
    metric_extraction: MetricExtractionConfig,
    evaluation_procedure: EvaluationProcedureConfig,
    run_in_subprocess: PythonSubprocessRunner = run_python_subprocess,
    execution_cache: ExecutionCache | None = None,
) -> tuple[MetricRecord, ...]:
    """Extract one record per question from one trace."""

    return extract_metrics_batch(
        (trace,),
        metric_extraction=metric_extraction,
        evaluation_procedure=evaluation_procedure,
        run_in_subprocess=run_in_subprocess,
        execution_cache=execution_cache,
    )[0]


def extract_metrics_batch(
    traces: Sequence[Trace],
    *,
    metric_extraction: MetricExtractionConfig,
    evaluation_procedure: EvaluationProcedureConfig,
    run_in_subprocess: PythonSubprocessRunner = run_python_subprocess,
    execution_cache: ExecutionCache | None = None,
) -> tuple[tuple[MetricRecord, ...], ...]:
    """Extract records after collecting work across every supplied trace."""

    metric_extraction = MetricExtractionConfig.model_validate(
        metric_extraction.model_dump(mode="python")
    )
    evaluation_procedure = EvaluationProcedureConfig.model_validate(
        evaluation_procedure.model_dump(mode="python")
    )
    if (
        evaluation_procedure.metric_extraction_config_hash
        != metric_extraction.config_identity_hash
    ):
        raise WiringError(
            "evaluation procedure does not reference this metric "
            "extraction definition"
        )
    for trace in traces:
        evaluation_procedure.validate_trace_producer(trace.producer)
    question_bindings = _bind_questions(metric_extraction)
    trace_bindings = tuple(
        tuple(
            _bind_trace_question(trace, question_binding)
            for question_binding in question_bindings
        )
        for trace in traces
    )

    requests: list[ExecutionRequest] = []
    for per_trace in trace_bindings:
        for binding in per_trace:
            if binding.absence is not None:
                continue
            assert binding.value is not None
            operator = binding.question_binding.operator
            try:
                binding_requests = operator.execution_requests(
                    binding.value,
                    binding.auxiliary,
                )
            except (SubprocessError, EngineInvariantError):
                raise
            except Exception as exc:
                binding.planning_failure = exc
                continue
            requests.extend(binding_requests)

    cache = (
        execution_cache
        if execution_cache is not None
        else InMemoryExecutionCache()
    )
    outcomes = run_requests(
        requests,
        run_in_subprocess=run_in_subprocess,
        cache=cache,
    )
    context = _EngineContext(views=ViewCache(), outcomes=outcomes)
    return tuple(
        tuple(
            _compute_record(evaluation_procedure, binding, context)
            for binding in per_trace
        )
        for per_trace in trace_bindings
    )


def _bind_questions(
    metric_extraction: MetricExtractionConfig,
) -> tuple[_QuestionBinding, ...]:
    if len(metric_extraction.questions) != len(
        metric_extraction.resolved_operator_versions
    ):
        raise WiringError(
            "metric extraction config has a stale operator resolution count: "
            f"configured {len(metric_extraction.resolved_operator_versions)}, "
            f"expected {len(metric_extraction.questions)}"
        )
    bindings: list[_QuestionBinding] = []
    for question, resolved_operator in zip(
        metric_extraction.questions,
        metric_extraction.resolved_operator_versions,
    ):
        try:
            operator = validated_metric_operator(
                name=str(question.metric),
                settings=question.settings_dict(),
                expected_version=resolved_operator[2],
                expected_implementation=resolved_operator[3],
            )
        except Exception as exc:
            raise WiringError(
                f"invalid executable metric {question.metric}: {exc}"
            ) from exc
        live_resolution = (
            question.identity_hash(),
            question.metric,
            str(type(operator).VERSION),
            implementation_identity(type(operator)),
        )
        if resolved_operator != live_resolution:
            raise WiringError(
                "metric extraction config has a stale operator resolution "
                f"for metric {question.metric!r}: "
                f"configured {resolved_operator!r}, live {live_resolution!r}"
            )
        bindings.append(_QuestionBinding(question=question, operator=operator))
    return tuple(bindings)


def _bind_trace_question(
    trace: Trace,
    question_binding: _QuestionBinding,
) -> _TraceBinding:
    question = question_binding.question
    operator = question_binding.operator

    raw_value = trace.value(question.on)
    value: Artifact | None
    absence: Absent | None
    if isinstance(raw_value, Absent):
        value = None
        absence = raw_value
    else:
        value = raw_value
        absence = None
        if value.kind not in operator.accepted_input_kinds():
            accepted = ", ".join(
                sorted(str(kind) for kind in operator.accepted_input_kinds())
            )
            raise WiringError(
                f"metric {question.metric} requires {question.on!r} to have "
                f"kind in {{{accepted}}}, got {value.kind}"
            )

    auxiliary: dict[str, Artifact] = {}
    auxiliary_absence: Absent | None = None
    for key in operator.auxiliary_keys():
        raw_auxiliary = trace.value(key)
        if isinstance(raw_auxiliary, Absent):
            if auxiliary_absence is None:
                auxiliary_absence = raw_auxiliary
            continue
        accepted = operator.accepted_auxiliary_kinds(key)
        if raw_auxiliary.kind not in accepted:
            expected = ", ".join(sorted(str(kind) for kind in accepted))
            raise WiringError(
                f"metric {question.metric} requires auxiliary key {key!r} "
                f"to have kind in {{{expected}}}, got {raw_auxiliary.kind}"
            )
        auxiliary[key] = raw_auxiliary

    if auxiliary_absence is None:
        try:
            operator.validate_auxiliary(auxiliary)
        except Exception as exc:
            raise WiringError(
                f"invalid auxiliary input for metric {question.metric}: {exc}"
            ) from exc

    return _TraceBinding(
        trace=trace,
        question_binding=question_binding,
        value=value,
        auxiliary=auxiliary,
        absence=absence or auxiliary_absence,
    )


def _compute_record(
    procedure: EvaluationProcedureConfig,
    binding: _TraceBinding,
    context: _EngineContext,
) -> MetricRecord:
    identity = _RecordIdentity.from_binding(procedure, binding)
    if binding.absence is not None:
        return _build_record(
            identity,
            absence_mode=AbsenceMode.PREPROCESSING_FAILURE,
            absence_cause=(
                f"{binding.absence.failed_step}: {binding.absence.cause}"
            ),
            failure_code=binding.absence.failure_code,
        )
    if binding.planning_failure is not None:
        return _failure_record(identity, binding.planning_failure)

    assert binding.value is not None
    try:
        result = binding.question_binding.operator.compute(
            binding.value,
            binding.auxiliary,
            context,
        )
        values = result.to_values()
        unknown_facts = set(values) - set(
            binding.question_binding.operator.FACT_UNITS
        )
        if unknown_facts:
            raise ValueError(
                "operator returned undeclared facts: "
                + ", ".join(sorted(unknown_facts))
            )
        facts = tuple(
            MetricFact(
                name=name,
                value=values.get(name),
                unit=unit,
                applicability=(
                    Applicability.APPLICABLE
                    if values.get(name) is not None
                    else Applicability.NOT_APPLICABLE
                ),
                reason=(
                    None
                    if values.get(name) is not None
                    else binding.question_binding.operator.undefined_fact_reason(
                        name
                    )
                ),
                lineage=identity.lineage,
            )
            for name, unit in binding.question_binding.operator.FACT_UNITS.items()
        )
        return MetricRecord.measured(
            question=identity.question,
            question_identity_hash=identity.question_identity_hash,
            on_key=identity.on_key,
            evaluation_procedure_config_hash=(
                identity.evaluation_procedure_config_hash
            ),
            trace_producer=binding.trace.producer,
            sample_identity=binding.trace.sample_identity,
            operator=identity.operator,
            facts=facts,
        )
    except (SubprocessError, EngineInvariantError):
        raise
    except Exception as exc:
        return _failure_record(identity, exc)


def _failure_record(
    identity: _RecordIdentity,
    failure: Exception,
) -> MetricRecord:
    return MetricRecord.operator_failure(
        question=identity.question,
        question_identity_hash=identity.question_identity_hash,
        on_key=identity.on_key,
        evaluation_procedure_config_hash=(
            identity.evaluation_procedure_config_hash
        ),
        trace_producer=identity.trace_producer,
        sample_identity=identity.sample_identity,
        operator=identity.operator,
        failure_type=type(failure).__name__,
        failure_message=str(failure),
    )


def _build_record(
    identity: _RecordIdentity,
    *,
    absence_mode: AbsenceMode,
    absence_cause: str,
    failure_code: str,
) -> MetricRecord:
    return MetricRecord.not_applicable(
        question=identity.question,
        question_identity_hash=identity.question_identity_hash,
        on_key=identity.on_key,
        evaluation_procedure_config_hash=(
            identity.evaluation_procedure_config_hash
        ),
        trace_producer=identity.trace_producer,
        sample_identity=identity.sample_identity,
        operator=identity.operator,
        absence_mode=absence_mode,
        cause=absence_cause,
        failure_code=failure_code,
    )
