from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from dr_exec import ExecutionPoolConfig, Executor, resolve_pool_capacity
from dr_store import ArtifactBundlePublication, ObjectStore

from dr_code.caching.preprocess_batch import preprocess_batch
from dr_code.evaluation._batch import trace_candidate_sources
from dr_code.evaluation.batch import (
    EvaluationBatchRequest,
    EvaluationBatchResult,
    SampleEvaluationInput,
    evaluate_batch,
)
from dr_code.evaluation.comparison import (
    EvaluationEvidenceResolver,
    StructuralEvaluationComparison,
    compare_evaluation_attempts,
)
from dr_code.evaluation.records import EvaluationAttemptRecord

if TYPE_CHECKING:
    from dr_code.caching import WindowedExecutionCache


@dataclass(frozen=True, slots=True)
class PreprocessingCoverage:
    """How one preprocessing definition treated a corpus of distinct texts.

    The three counters partition the corpus: every distinct text is counted
    exactly once, and a text whose preprocessing job failed is its own count
    rather than an omission.
    """

    texts_with_candidates: int
    texts_without_candidates: int
    texts_failed: int

    @property
    def corpus_size(self) -> int:
        return (
            self.texts_with_candidates
            + self.texts_without_candidates
            + self.texts_failed
        )


@dataclass(frozen=True, slots=True)
class PreprocessingValidation:
    """One preprocessing-change validation over a corpus and a plan."""

    coverage: PreprocessingCoverage
    result: EvaluationBatchResult
    comparison: StructuralEvaluationComparison | None


@dataclass(frozen=True, slots=True)
class TestingValidation:
    """One testing-change validation over a plan."""

    result: EvaluationBatchResult
    comparison: StructuralEvaluationComparison | None


async def validate_preprocessing(
    request: EvaluationBatchRequest,
    /,
    *,
    executor: Executor,
    execution_cache: WindowedExecutionCache,
    object_store: ObjectStore | None,
    publication: ArtifactBundlePublication | None,
    pool_config: ExecutionPoolConfig,
    reference: EvaluationAttemptRecord | None = None,
    evidence_resolver: EvaluationEvidenceResolver | None = None,
) -> PreprocessingValidation:
    """Validate a preprocessing change over the pooled preprocessing leg.

    The request's sample inputs are the corpus. Their distinct raw texts run
    once through `preprocess_batch` under `request.plan.procedure.preprocessing`
    — the definition the attempt is evaluated under — and those traces both
    report the coverage and feed `evaluate_batch`, so the corpus is
    preprocessed once and the coverage describes the evaluated definition. A
    reference attempt makes the flow structural: the new attempt is compared
    against it, resolving both sides' evidence through the caller's resolver.
    """

    _validate_comparison_pair(reference, evidence_resolver)
    corpus = _corpus_texts(request)
    traces_by_text = await preprocess_batch(
        corpus,
        definition=request.plan.procedure.preprocessing,
        worker_count=resolve_pool_capacity(
            pool_config.capacity
        ).max_active_jobs,
    )
    failed_texts = frozenset(
        text for text in corpus if text not in traces_by_text
    )
    with_candidates = sum(
        1
        for text in corpus
        if (trace := traces_by_text.get(text)) is not None
        and trace_candidate_sources(trace)
    )
    result = await evaluate_batch(
        _request_without_texts(request, failed_texts),
        executor=executor,
        execution_cache=execution_cache,
        object_store=object_store,
        publication=publication,
        pool_config=pool_config,
        preprocessed_traces=traces_by_text,
    )
    return PreprocessingValidation(
        coverage=PreprocessingCoverage(
            texts_with_candidates=with_candidates,
            texts_without_candidates=len(traces_by_text) - with_candidates,
            texts_failed=len(corpus) - len(traces_by_text),
        ),
        result=result,
        comparison=await _compare(
            result.attempt, reference, evidence_resolver
        ),
    )


async def validate_testing(
    request: EvaluationBatchRequest,
    /,
    *,
    executor: Executor,
    execution_cache: WindowedExecutionCache,
    object_store: ObjectStore | None,
    publication: ArtifactBundlePublication | None,
    pool_config: ExecutionPoolConfig,
    reference: EvaluationAttemptRecord | None = None,
    evidence_resolver: EvaluationEvidenceResolver | None = None,
) -> TestingValidation:
    """Validate a testing change over the pooled evaluation leg.

    The request runs through `evaluate_batch` for its attempt verdicts. A
    reference attempt makes the flow structural: the new attempt is compared
    against it, resolving both sides' evidence through the caller's resolver.
    """

    _validate_comparison_pair(reference, evidence_resolver)
    result = await evaluate_batch(
        request,
        executor=executor,
        execution_cache=execution_cache,
        object_store=object_store,
        publication=publication,
        pool_config=pool_config,
    )
    return TestingValidation(
        result=result,
        comparison=await _compare(
            result.attempt, reference, evidence_resolver
        ),
    )


def _corpus_texts(request: EvaluationBatchRequest) -> tuple[str, ...]:
    """Give the distinct sample texts the preprocessing leg runs over."""

    return tuple(
        dict.fromkeys(
            item.sample.raw_input.text
            for item in request.inputs
            if isinstance(item, SampleEvaluationInput)
        )
    )


def _request_without_texts(
    request: EvaluationBatchRequest,
    excluded_texts: frozenset[str],
) -> EvaluationBatchRequest:
    """Drop sample inputs whose raw text failed pooled preprocessing."""

    if not excluded_texts:
        return request
    filtered = tuple(
        item
        for item in request.inputs
        if not (
            isinstance(item, SampleEvaluationInput)
            and item.sample.raw_input.text in excluded_texts
        )
    )
    if not filtered:
        raise ValueError(
            "every distinct preprocessing text failed; nothing to evaluate"
        )
    return request.model_copy(update={"inputs": filtered})


def _validate_comparison_pair(
    reference: EvaluationAttemptRecord | None,
    evidence_resolver: EvaluationEvidenceResolver | None,
) -> None:
    if (reference is None) != (evidence_resolver is None):
        raise ValueError(
            "structural comparison requires both a reference attempt and an "
            "evidence resolver"
        )


async def _compare(
    attempt: EvaluationAttemptRecord,
    reference: EvaluationAttemptRecord | None,
    evidence_resolver: EvaluationEvidenceResolver | None,
) -> StructuralEvaluationComparison | None:
    if reference is None or evidence_resolver is None:
        return None
    return await compare_evaluation_attempts(
        reference,
        attempt,
        resolver=evidence_resolver,
    )


__all__ = [
    "PreprocessingCoverage",
    "PreprocessingValidation",
    "TestingValidation",
    "validate_preprocessing",
    "validate_testing",
]
