"""Bind, plan, execute, and compute declared metric questions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from dr_code.humaneval.sandbox import (
    SandboxError,
    SandboxRunner,
    run_python_in_sandbox,
)
from dr_code.metrics.coordinates import (
    MetricQuestionCoordinate,
    MetricsDefinitionCoordinate,
)
from dr_code.metrics.definition import MetricsDefinition, MetricQuestion
from dr_code.metrics.engine.execution import (
    ExecutionCache,
    ExecutionOutcome,
    ExecutionRequest,
    InMemoryExecutionCache,
    run_requests,
)
from dr_code.metrics.engine.views import ViewCache
from dr_code.metrics.operators.base import MetricOperator
from dr_code.metrics.records import (
    MeasuredRecord,
    MetricFact,
    MetricRecord,
    MetricRecordIdentity,
    NotApplicableRecord,
    OperatorFailure,
    OperatorFailureRecord,
)
from dr_code.metrics.registry import REGISTRY
from dr_code.trace import Absent, Artifact, Trace, WiringError


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


def _record_identity(
    definition: MetricsDefinition,
    binding: _TraceBinding,
) -> MetricRecordIdentity:
    """Project a bound question into the record's persisted identity."""

    question_binding = binding.question_binding
    return MetricRecordIdentity(
        question=MetricQuestionCoordinate.of(question_binding.question),
        metric_version=question_binding.operator.VERSION,
        producer=binding.trace.producer,
        metrics_definition=MetricsDefinitionCoordinate.of(definition),
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
    run_in_sandbox: SandboxRunner = run_python_in_sandbox,
    execution_cache: ExecutionCache | None = None,
) -> tuple[MetricRecord, ...]:
    """Extract one record per question from one trace."""

    return extract_metrics_batch(
        definition,
        (trace,),
        run_in_sandbox=run_in_sandbox,
        execution_cache=execution_cache,
    )[0]


def extract_metrics_batch(
    definition: MetricsDefinition,
    traces: Sequence[Trace],
    *,
    run_in_sandbox: SandboxRunner = run_python_in_sandbox,
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
            except (SandboxError, EngineInvariantError):
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
        run_in_sandbox=run_in_sandbox,
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
    identity = _record_identity(definition, binding)
    if binding.absence is not None:
        return NotApplicableRecord(identity=identity, absence=binding.absence)
    if binding.planning_failure is not None:
        return _failure_record(identity, binding.planning_failure)

    assert binding.value is not None
    try:
        result = binding.question_binding.operator.compute(
            binding.value,
            binding.auxiliary,
            context,
        )
        facts: tuple[MetricFact, ...] = result.to_facts()
    except (SandboxError, EngineInvariantError):
        raise
    except Exception as exc:
        return _failure_record(identity, exc)
    return MeasuredRecord(identity=identity, facts=facts)


def _failure_record(
    identity: MetricRecordIdentity,
    failure: Exception,
) -> OperatorFailureRecord:
    return OperatorFailureRecord(
        identity=identity,
        failure=OperatorFailure(
            failure_type=type(failure).__name__,
            failure_message=str(failure),
        ),
    )
