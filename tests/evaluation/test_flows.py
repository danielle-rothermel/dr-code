from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest
from dr_exec import ExecutionPoolConfig
from dr_store import ArtifactBundlePublication, MemoryBackend, ObjectStore

from _executor_stubs import importable_json_executor
from dr_code.evaluation import (
    AttemptCompleteness,
    AttemptValidity,
    ComparisonStatus,
    EvaluationAttemptIdentity,
    EvidenceReference,
    SampleEvaluationRecord,
    restore_evaluation_attempt,
    validate_preprocessing,
    validate_testing,
)
from dr_code.preprocessing import EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION

from ._batch_builders import BatchStore, cache, request
from ._bundle_builders import read_limits, stored_source_request

pytestmark = pytest.mark.asyncio


class RestoredEvidence:
    """Resolve published attempts' evidence from their bundles on demand."""

    def __init__(self, *bundle_roots: Path, object_store: ObjectStore) -> None:
        self._roots = bundle_roots
        self._object_store = object_store
        self._records: dict[EvidenceReference, SampleEvaluationRecord] = {}
        self._restored: set[Path] = set()

    async def resolve(
        self, reference: EvidenceReference, /
    ) -> SampleEvaluationRecord:
        if reference not in self._records:
            await self._restore_published_bundles()
        return self._records[reference]

    async def _restore_published_bundles(self) -> None:
        for root in self._roots:
            for bundle_path in sorted(root.iterdir()):
                if bundle_path in self._restored:
                    continue
                self._restored.add(bundle_path)
                restored = await restore_evaluation_attempt(
                    bundle_path,
                    object_store=self._object_store,
                    limits=read_limits(),
                )
                references = tuple(
                    member.record
                    for member in restored.attempt.members
                    if member.record is not None
                )
                self._records.update(
                    zip(references, restored.samples, strict=True)
                )


async def test_validate_testing_reports_attempt_verdicts(
    tmp_path: Path,
) -> None:
    publication = ArtifactBundlePublication.allocate(
        tmp_path, prefix="testing"
    )
    execution_cache = cache(BatchStore())
    try:
        validation = await validate_testing(
            request(2),
            executor=importable_json_executor(),
            execution_cache=execution_cache,
            object_store=ObjectStore(MemoryBackend()),
            publication=publication,
            pool_config=ExecutionPoolConfig(),
        )
    finally:
        await execution_cache.close()

    assert validation.verdict.completeness is AttemptCompleteness.COMPLETE
    assert validation.verdict.validity is AttemptValidity.VALID
    assert validation.verdict.limit_exhaustion is None
    assert validation.comparison is None
    assert len(validation.result.attempt.members) == 2


async def test_validate_preprocessing_reports_corpus_coverage(
    tmp_path: Path,
) -> None:
    publication = ArtifactBundlePublication.allocate(
        tmp_path, prefix="preprocessing"
    )
    execution_cache = cache(BatchStore())
    try:
        validation = await validate_preprocessing(
            request(2),
            definition=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
            executor=importable_json_executor(),
            execution_cache=execution_cache,
            object_store=ObjectStore(MemoryBackend()),
            publication=publication,
            pool_config=ExecutionPoolConfig(),
            worker_count=1,
        )
    finally:
        await execution_cache.close()

    # Every sample in the fixture corpus carries the same one candidate text.
    assert validation.texts_with_candidates == 1
    assert validation.texts_without_candidates == 0
    assert validation.verdict.completeness is AttemptCompleteness.COMPLETE
    assert validation.verdict.validity is AttemptValidity.VALID
    assert validation.comparison is None


async def test_structural_comparison_needs_a_reference_and_a_resolver(
    tmp_path: Path,
) -> None:
    object_store = ObjectStore(MemoryBackend())
    execution_cache = cache(BatchStore())
    try:
        with pytest.raises(ValueError, match="structural comparison"):
            await validate_testing(
                request(1),
                executor=importable_json_executor(),
                execution_cache=execution_cache,
                object_store=object_store,
                publication=ArtifactBundlePublication.allocate(
                    tmp_path, prefix="testing"
                ),
                pool_config=ExecutionPoolConfig(),
                evidence_resolver=RestoredEvidence(object_store=object_store),
            )
    finally:
        await execution_cache.close()


async def test_validate_testing_compares_against_a_reference_attempt(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    reference_publication = ArtifactBundlePublication.allocate(
        reference_root, prefix="testing"
    )
    object_store = ObjectStore(MemoryBackend())
    execution_cache = cache(BatchStore())
    reference_request = await stored_source_request(
        request(1), object_store=object_store
    )
    candidate_request = reference_request.model_copy(
        update={
            "attempt": EvaluationAttemptIdentity(attempt_id=UUID(int=2)),
        }
    )
    try:
        reference_run = await validate_testing(
            reference_request,
            executor=importable_json_executor(),
            execution_cache=execution_cache,
            object_store=object_store,
            publication=reference_publication,
            pool_config=ExecutionPoolConfig(),
        )
        assert reference_run.result.bundle_path is not None
        reference = await restore_evaluation_attempt(
            reference_run.result.bundle_path,
            object_store=object_store,
            limits=read_limits(),
        )
        candidate_publication = ArtifactBundlePublication.allocate(
            candidate_root, prefix="testing"
        )
        validation = await validate_testing(
            candidate_request,
            executor=importable_json_executor(),
            execution_cache=execution_cache,
            object_store=object_store,
            publication=candidate_publication,
            pool_config=ExecutionPoolConfig(),
            reference=reference.attempt,
            evidence_resolver=RestoredEvidence(
                reference_root,
                candidate_root,
                object_store=object_store,
            ),
        )
    finally:
        await execution_cache.close()

    comparison = validation.comparison
    assert comparison is not None
    assert comparison.left == reference.attempt.identity
    assert comparison.right == validation.result.attempt.identity
    assert comparison.added == ()
    assert comparison.removed == ()
    assert not comparison.ordering_changed
    assert [entry.metrics for entry in comparison.matched] == [
        ComparisonStatus.UNCHANGED
    ]
