"""Bind, plan, execute, and compute declared metric questions.

``extract_metrics`` answers a ``MetricsDefinition`` against one trace and
returns one ``MetricRecord`` per declared question, in declaration order.
``extract_metrics_batch`` does the same across several traces, collecting all
subprocess work before running any of it so equivalent requests execute once.

An evaluation procedure may be supplied to bind the run into the eval kernel.
It contributes the trace-source contract (which producer kind the procedure
accepts) and the operator-resolution check against the live registry; the
answers themselves are identical either way.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from typing import TYPE_CHECKING

from dr_code.execution.subprocess import (
    SubprocessError,
    PythonSubprocessRunner,
    run_python_subprocess,
)
from dr_code.metrics.definition import MetricQuestion, MetricsDefinition
from dr_code.metrics.engine.execution import (
    ExecutionCache,
    ExecutionOutcome,
    ExecutionRequest,
    InMemoryExecutionCache,
    run_requests,
)
from dr_code.metrics.engine.views import ViewCache
from dr_code.metrics.names import MetricName
from dr_code.metrics.operators.base import MetricOperator
from dr_code.metrics.records import MetricRecord, MetricScalar, RecordStatus
from dr_code.metrics.settings import OperatorSettings
from dr_code.metrics.validation import validated_metric_operator
from dr_code.trace import Absent, Artifact, Trace, TraceProducer, WiringError

if TYPE_CHECKING:
    from dr_code.eval.facts import MetricFact
    from dr_code.eval.lifecycle import (
        EvaluationProcedureConfig,
        MetricExtractionConfig,
    )


@dataclass(frozen=True, slots=True)
class _QuestionBinding:
    question: MetricQuestion
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
    metric: MetricName
    metric_version: str
    settings: OperatorSettings
    on_key: str
    producer: TraceProducer
    metrics_definition: MetricsDefinition

    @classmethod
    def from_binding(
        cls, definition: MetricsDefinition, binding: _TraceBinding
    ) -> _RecordIdentity:
        question_binding = binding.question_binding
        question = question_binding.question
        return cls(
            metric=question.metric,
            metric_version=str(type(question_binding.operator).VERSION),
            settings=question_binding.operator.settings,
            on_key=question.on,
            producer=binding.trace.producer,
            metrics_definition=definition,
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
    outcomes: Mapping[ExecutionRequest, ExecutionOutcome]

    def outcome_for(self, request: ExecutionRequest) -> ExecutionOutcome:
        try:
            return self.outcomes[request]
        except KeyError as exc:
            raise EngineInvariantError(
                "no execution outcome planned for request "
                f"{request.computation_id!r}"
            ) from exc


def extract_metrics(
    definition: MetricsDefinition,
    trace: Trace,
    *,
    evaluation_procedure: EvaluationProcedureConfig | None = None,
    metric_extraction: MetricExtractionConfig | None = None,
    run_in_subprocess: PythonSubprocessRunner = run_python_subprocess,
    execution_cache: ExecutionCache | None = None,
) -> tuple[MetricRecord, ...]:
    """Extract one record per question from one trace."""

    return extract_metrics_batch(
        definition,
        (trace,),
        evaluation_procedure=evaluation_procedure,
        metric_extraction=metric_extraction,
        run_in_subprocess=run_in_subprocess,
        execution_cache=execution_cache,
    )[0]


def extract_metrics_batch(
    definition: MetricsDefinition,
    traces: Sequence[Trace],
    *,
    evaluation_procedure: EvaluationProcedureConfig | None = None,
    metric_extraction: MetricExtractionConfig | None = None,
    run_in_subprocess: PythonSubprocessRunner = run_python_subprocess,
    execution_cache: ExecutionCache | None = None,
) -> tuple[tuple[MetricRecord, ...], ...]:
    """Extract records after collecting work across every supplied trace."""

    if evaluation_procedure is not None:
        definition = _kernel_bound_definition(
            definition,
            evaluation_procedure=evaluation_procedure,
            metric_extraction=metric_extraction,
        )
        for trace in traces:
            evaluation_procedure.validate_trace_producer(trace.producer)

    question_bindings = _bind_questions(definition)
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
            _compute_record(definition, binding, context)
            for binding in per_trace
        )
        for per_trace in trace_bindings
    )


def _kernel_bound_definition(
    definition: MetricsDefinition,
    *,
    evaluation_procedure: EvaluationProcedureConfig,
    metric_extraction: MetricExtractionConfig | None,
) -> MetricsDefinition:
    """Check the kernel wiring and return the definition the procedure owns."""

    if metric_extraction is None:
        raise WiringError(
            "an evaluation procedure requires the metric extraction config "
            "it references"
        )
    if (
        evaluation_procedure.metric_extraction_config
        != metric_extraction.coordinate()
    ):
        raise WiringError(
            "evaluation procedure does not reference this metric "
            "extraction definition"
        )
    if metric_extraction.definition != definition:
        raise WiringError(
            "metric extraction config does not carry this metrics definition"
        )
    return metric_extraction.definition


def _bind_questions(
    definition: MetricsDefinition,
) -> tuple[_QuestionBinding, ...]:
    bindings: list[_QuestionBinding] = []
    for question in definition.questions:
        try:
            operator = validated_metric_operator(
                name=question.metric.value,
                settings=question.settings.model_dump(mode="json"),
            )
        except Exception as exc:
            raise WiringError(
                f"invalid settings for metric {question.metric}: {exc}"
            ) from exc
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
    definition: MetricsDefinition,
    binding: _TraceBinding,
    context: _EngineContext,
) -> MetricRecord:
    identity = _RecordIdentity.from_binding(definition, binding)
    if binding.absence is not None:
        return _build_record(
            identity,
            status=RecordStatus.NOT_APPLICABLE,
            absence_failed_step=binding.absence.failed_step,
            absence_cause=binding.absence.cause,
        )
    if binding.planning_failure is not None:
        return _failure_record(identity, binding.planning_failure)

    assert binding.value is not None
    operator = binding.question_binding.operator
    try:
        result = operator.compute(
            binding.value,
            binding.auxiliary,
            context,
        )
        values = result.to_values()
        undeclared = set(values) - set(operator.FACT_UNITS)
        if undeclared:
            raise ValueError(
                "operator returned undeclared facts: "
                + ", ".join(sorted(undeclared))
            )
        return _build_record(
            identity,
            status=RecordStatus.MEASURED,
            values={name: values.get(name) for name in operator.FACT_UNITS},
        )
    except (SubprocessError, EngineInvariantError):
        raise
    except Exception as exc:
        return _failure_record(identity, exc)


def record_facts(
    record: MetricRecord,
    *,
    evaluation_procedure: EvaluationProcedureConfig,
) -> tuple[MetricFact, ...]:
    """Project one measured record onto unit-carrying, lineage-stamped facts.

    Each value the record carries becomes a ``MetricFact`` stamped with the
    unit its operator declares in ``FACT_UNITS``. A declared fact with no value
    for this observation is reported as not-applicable with the operator's
    reason, so a declared fact never silently disappears.
    """

    from dr_code.eval.facts import (
        Applicability,
        MetricFact,
        OperatorLineage,
    )

    if record.status is not RecordStatus.MEASURED:
        return ()
    operator = validated_metric_operator(
        name=record.metric.value,
        settings=record.settings.model_dump(mode="json"),
    )
    lineage = OperatorLineage(
        evaluation_procedure_config=evaluation_procedure.coordinate(),
        operator=record.metric,
        operator_version=record.metric_version,
        on_key=record.on_key,
    )
    return tuple(
        MetricFact(
            name=name,
            value=record.values.get(name),
            unit=unit,
            applicability=(
                Applicability.APPLICABLE
                if record.values.get(name) is not None
                else Applicability.NOT_APPLICABLE
            ),
            reason=(
                None
                if record.values.get(name) is not None
                else operator.undefined_fact_reason(name)
            ),
            lineage=lineage,
        )
        for name, unit in operator.FACT_UNITS.items()
    )


def _failure_record(
    identity: _RecordIdentity,
    failure: Exception,
) -> MetricRecord:
    return _build_record(
        identity,
        status=RecordStatus.OPERATOR_FAILURE,
        failure_type=type(failure).__name__,
        failure_message=str(failure),
    )


def _build_record(
    identity: _RecordIdentity,
    *,
    status: RecordStatus,
    values: dict[str, MetricScalar] | None = None,
    absence_failed_step: str | None = None,
    absence_cause: str | None = None,
    failure_type: str | None = None,
    failure_message: str | None = None,
) -> MetricRecord:
    return MetricRecord(
        metric=identity.metric,
        metric_version=identity.metric_version,
        settings=identity.settings,
        on_key=identity.on_key,
        producer=identity.producer,
        metrics_definition=identity.metrics_definition,
        status=status,
        values=values or {},
        absence_failed_step=absence_failed_step,
        absence_cause=absence_cause,
        failure_type=failure_type,
        failure_message=failure_message,
    )
