"""Bind, plan, execute, and compute declared metric questions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from dr_exec import ExecutorFailure, Records
from pydantic import JsonValue

from dr_code.metrics.definition import MetricsDefinition, MetricQuestion
from dr_code.metrics.engine.execution import (
    ExecutionCache,
    ExecutionOutcome,
    ExecutionRequest,
    Executor,
    InMemoryExecutionCache,
    run_requests,
)
from dr_code.metrics.engine.views import ViewCache
from dr_code.metrics.names import MetricName
from dr_code.metrics.operators.base import MetricOperator
from dr_code.metrics.records import (
    MetricRecord,
    MetricScalar,
    RecordStatus,
)
from dr_code.metrics.registry import REGISTRY
from dr_code.trace import Absent, Artifact, Trace, WiringError

_NO_RECORDS: Records = Records.none()
"""The default record sink: hot metric sweeps persist no run records."""


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
    settings: dict[str, JsonValue]
    on_key: str
    producer_id: str
    producer_version: str | None
    producer_definition_hash: str | None
    metrics_definition_id: str
    metrics_definition_version: str

    @classmethod
    def from_binding(
        cls, definition: MetricsDefinition, binding: _TraceBinding
    ) -> _RecordIdentity:
        question_binding = binding.question_binding
        question = question_binding.question
        producer = binding.trace.producer
        return cls(
            metric=question.metric,
            metric_version=question_binding.operator.VERSION,
            settings=dict(question.settings),
            on_key=question.on,
            producer_id=producer.producer_id,
            producer_version=producer.version,
            producer_definition_hash=producer.definition_hash,
            metrics_definition_id=definition.definition_id,
            metrics_definition_version=definition.version,
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
    definition: MetricsDefinition,
    trace: Trace,
    *,
    executor: Executor,
    records: Records = _NO_RECORDS,
    execution_cache: ExecutionCache | None = None,
) -> tuple[MetricRecord, ...]:
    """Extract one record per question from one trace."""

    return extract_metrics_batch(
        definition,
        (trace,),
        executor=executor,
        records=records,
        execution_cache=execution_cache,
    )[0]


def extract_metrics_batch(
    definition: MetricsDefinition,
    traces: Sequence[Trace],
    *,
    executor: Executor,
    records: Records = _NO_RECORDS,
    execution_cache: ExecutionCache | None = None,
) -> tuple[tuple[MetricRecord, ...], ...]:
    """Extract records after collecting work across every supplied trace."""

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
            except (ExecutorFailure, EngineInvariantError):
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
        executor=executor,
        records=records,
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


def _bind_questions(
    definition: MetricsDefinition,
) -> tuple[_QuestionBinding, ...]:
    bindings: list[_QuestionBinding] = []
    for question in definition.questions:
        operator_class = REGISTRY.get(str(question.metric))
        if operator_class is None:
            raise WiringError(
                f"no metric operator registered for {question.metric!r}"
            )
        try:
            settings = operator_class.Settings.model_validate(
                question.settings
            )
            operator = operator_class(settings)
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
    try:
        result = binding.question_binding.operator.compute(
            binding.value,
            binding.auxiliary,
            context,
        )
        return _build_record(
            identity,
            status=RecordStatus.MEASURED,
            values=result.to_values(),
        )
    except (ExecutorFailure, EngineInvariantError):
        raise
    except Exception as exc:
        return _failure_record(identity, exc)


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
        producer_id=identity.producer_id,
        producer_version=identity.producer_version,
        producer_definition_hash=identity.producer_definition_hash,
        metrics_definition_id=identity.metrics_definition_id,
        metrics_definition_version=identity.metrics_definition_version,
        status=status,
        values=values or {},
        absence_failed_step=absence_failed_step,
        absence_cause=absence_cause,
        failure_type=failure_type,
        failure_message=failure_message,
    )
