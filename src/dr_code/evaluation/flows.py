from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dr_exec import ExecutionPoolConfig, Executor
from dr_store import ArtifactBundlePublication, ObjectStore

from dr_code.caching.preprocess_batch import candidate_sources_batch
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
from dr_code.evaluation.records import (
    AttemptCompleteness,
    AttemptLimitExhaustion,
    AttemptValidity,
    EvaluationAttemptRecord,
)
from dr_code.preprocessing.definition import PreprocessingDefinition

if TYPE_CHECKING:
    from dr_code.caching import WindowedExecutionCache


@dataclass(frozen=True, slots=True)
class AttemptVerdict:
    """The attempt-level verdicts one validation flow observed."""

    completeness: AttemptCompleteness
    validity: AttemptValidity
    limit_exhaustion: AttemptLimitExhaustion | None

    @classmethod
    def of(cls, attempt: EvaluationAttemptRecord, /) -> AttemptVerdict:
        return cls(
            completeness=attempt.completeness,
            validity=attempt.validity,
            limit_exhaustion=attempt.limit_exhaustion,
        )


@dataclass(frozen=True, slots=True)
class PreprocessingValidation:
    """One preprocessing-change validation over a corpus and a plan."""

    texts_with_candidates: int
    texts_without_candidates: int
    result: EvaluationBatchResult
    verdict: AttemptVerdict
    comparison: StructuralEvaluationComparison | None


@dataclass(frozen=True, slots=True)
class TestingValidation:
    """One testing-change validation over a plan."""

    result: EvaluationBatchResult
    verdict: AttemptVerdict
    comparison: StructuralEvaluationComparison | None


async def validate_preprocessing(
    request: EvaluationBatchRequest,
    /,
    *,
    definition: PreprocessingDefinition,
    executor: Executor,
    execution_cache: WindowedExecutionCache,
    object_store: ObjectStore | None,
    publication: ArtifactBundlePublication | None,
    pool_config: ExecutionPoolConfig,
    worker_count: int | None = None,
    reference: EvaluationAttemptRecord | None = None,
    evidence_resolver: EvaluationEvidenceResolver | None = None,
) -> PreprocessingValidation:
    """Validate a preprocessing change over the pooled preprocessing leg.

    The request's sample inputs are the corpus: their raw text runs through
    `candidate_sources_batch` under `definition` to report which texts the
    change leaves with candidates, then the same request runs through
    `evaluate_batch` for its attempt verdicts. A reference attempt makes the
    flow structural: the new attempt is compared against it, resolving both
    sides' evidence through the caller's resolver.
    """

    _validate_comparison_pair(reference, evidence_resolver)
    sources_by_text = await candidate_sources_batch(
        _corpus_texts(request),
        definition=definition,
        worker_count=worker_count,
    )
    with_candidates = sum(1 for sources in sources_by_text.values() if sources)
    result = await evaluate_batch(
        request,
        executor=executor,
        execution_cache=execution_cache,
        object_store=object_store,
        publication=publication,
        pool_config=pool_config,
    )
    return PreprocessingValidation(
        texts_with_candidates=with_candidates,
        texts_without_candidates=len(sources_by_text) - with_candidates,
        result=result,
        verdict=AttemptVerdict.of(result.attempt),
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
        verdict=AttemptVerdict.of(result.attempt),
        comparison=await _compare(
            result.attempt, reference, evidence_resolver
        ),
    )


def _corpus_texts(request: EvaluationBatchRequest) -> Iterable[str]:
    return tuple(
        item.sample.raw_input.text
        for item in request.inputs
        if isinstance(item, SampleEvaluationInput)
    )


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
    "AttemptVerdict",
    "PreprocessingValidation",
    "TestingValidation",
    "validate_preprocessing",
    "validate_testing",
]
