from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dr_code.metrics.coordinates import (
    MetricQuestionCoordinate,
    MetricsDefinitionCoordinate,
)
from dr_code.metrics.definition import MetricsDefinition, MetricQuestion
from dr_code.evaluation.records import CandidateExecutionOutcome
from dr_code.humaneval.job import HumanEvalEvaluatorSuite
from dr_code.humaneval.metric_operator import CodeTest
from dr_code.metrics.engine.views import ViewCache
from dr_code.metrics.operators.base import MetricOperator
from dr_code.metrics.records import (
    MeasuredRecord,
    MetricValue,
    MetricRecord,
    MetricRecordId,
    NotApplicableRecord,
    OperatorFailure,
    OperatorFailureRecord,
)
from dr_code.metrics.registry import REGISTRY
from dr_code.trace import Absent, Artifact, Trace, WiringError

if TYPE_CHECKING:
    from dr_code.evaluation.id import MaterializedEvalCandidate


@dataclass(frozen=True, slots=True)
class _QuestionBinding:
    question: MetricQuestion
    operator: MetricOperator


@dataclass(slots=True)
class _TraceBinding:
    trace: Trace
    question_binding: _QuestionBinding
    value: Artifact | None
    auxiliary: dict[str, Artifact]
    absence: Absent | None
    planning_failure: Exception | None = None


def _record_id(
    definition: MetricsDefinition,
    binding: _TraceBinding,
) -> MetricRecordId:
    question_binding = binding.question_binding
    return MetricRecordId(
        question=MetricQuestionCoordinate.of(question_binding.question),
        metric_version=question_binding.operator.VERSION,
        producer=binding.trace.producer,
        metrics_definition=MetricsDefinitionCoordinate.of(definition),
    )


class EngineInvariantError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class _EngineContext:
    views: ViewCache
    question: MetricQuestionCoordinate
    candidate_execution_outcome: object | None


@dataclass(frozen=True, slots=True)
class _CandidateMetricPlan:
    definition: MetricsDefinition
    bindings: tuple[_TraceBinding, ...]
    suites: tuple[HumanEvalEvaluatorSuite, ...]

    def records(
        self,
        outcome: CandidateExecutionOutcome,
    ) -> tuple[MetricRecord, ...]:
        views = ViewCache()
        return tuple(
            _compute_record(
                self.definition,
                binding,
                _EngineContext(
                    views=views,
                    question=MetricQuestionCoordinate.of(
                        binding.question_binding.question
                    ),
                    candidate_execution_outcome=outcome,
                ),
            )
            for binding in self.bindings
        )


def _plan_candidate_metrics(
    definition: MetricsDefinition,
    trace: Trace,
    candidate: MaterializedEvalCandidate,
    /,
) -> _CandidateMetricPlan:
    """Bind one candidate to the plan's questions without executing it."""

    question_bindings = _bind_questions(definition)
    values = dict(trace.values)
    for question_binding in question_bindings:
        if isinstance(question_binding.operator, CodeTest):
            values[question_binding.question.on] = candidate.source
    candidate_trace = Trace(
        values=values,
        producer=trace.producer,
        step_facts=trace.step_facts,
    )
    bindings = tuple(
        _bind_trace_question(candidate_trace, question_binding)
        for question_binding in question_bindings
    )
    suites: list[HumanEvalEvaluatorSuite] = []
    for binding in bindings:
        if binding.absence is not None:
            continue
        operator = binding.question_binding.operator
        if not isinstance(operator, CodeTest):
            continue
        assert binding.value is not None
        try:
            suites.append(
                operator.evaluator_suite(
                    binding.value,
                    binding.auxiliary,
                    MetricQuestionCoordinate.of(
                        binding.question_binding.question
                    ),
                )
            )
        except Exception as error:
            binding.planning_failure = error
    return _CandidateMetricPlan(
        definition=definition,
        bindings=bindings,
        suites=tuple(suites),
    )


async def extract_metrics(
    definition: MetricsDefinition,
    trace: Trace,
) -> tuple[MetricRecord, ...]:
    record_sets = await extract_metrics_batch(definition, (trace,))
    return record_sets[0]


async def extract_metrics_batch(
    definition: MetricsDefinition,
    traces: Sequence[Trace],
) -> tuple[tuple[MetricRecord, ...], ...]:
    question_bindings = _bind_questions(definition)
    trace_bindings = tuple(
        tuple(
            _bind_trace_question(trace, question_binding)
            for question_binding in question_bindings
        )
        for trace in traces
    )

    views = ViewCache()
    return tuple(
        tuple(
            _compute_record(
                definition,
                binding,
                _EngineContext(
                    views=views,
                    question=MetricQuestionCoordinate.of(
                        binding.question_binding.question
                    ),
                    candidate_execution_outcome=None,
                ),
            )
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
    identity = _record_id(definition, binding)
    if binding.absence is not None:
        return NotApplicableRecord(identity=identity, absence=binding.absence)
    if binding.planning_failure is not None:
        return _failure_record(identity, binding.planning_failure)

    assert binding.value is not None
    # Invalid operator result models or values become operator failures.
    try:
        result = binding.question_binding.operator.compute(
            binding.value,
            binding.auxiliary,
            context,
        )
        values: tuple[MetricValue, ...] = result.to_values()
        return MeasuredRecord(identity=identity, values=values)
    except EngineInvariantError:
        raise
    except Exception as exc:
        return _failure_record(identity, exc)


def _failure_record(
    identity: MetricRecordId,
    failure: Exception,
) -> OperatorFailureRecord:
    return OperatorFailureRecord(
        identity=identity,
        failure=OperatorFailure(
            failure_type=type(failure).__name__,
            failure_message=str(failure),
        ),
    )
