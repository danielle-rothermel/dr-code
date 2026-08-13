from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest
from dr_exec import AutoPoolCapacity, ExecutionPoolConfig
from dr_store import ArtifactBundlePublication, MemoryBackend, ObjectStore

from _executor_stubs import importable_json_executor
from dr_code.evaluation import (
    AttemptCompleteness,
    AttemptValidity,
    ComparisonStatus,
    EvalAttemptId,
    EvidenceReference,
    PreprocessMode,
    SampleEvalRecord,
    restore_eval_attempt,
    validate_preprocessing,
    validate_testing,
)
from dr_code.evaluation import flows

from ._batch_builders import BatchStore, cache, request
from ._bundle_builders import read_limits, stored_source_request

pytestmark = pytest.mark.asyncio


class RestoredEvidence:
    """Resolve published attempts' evidence from their bundles on demand."""

    def __init__(self, *bundle_roots: Path, object_store: ObjectStore) -> None:
        self._roots = bundle_roots
        self._object_store = object_store
        self._records: dict[EvidenceReference, SampleEvalRecord] = {}
        self._restored: set[Path] = set()

    async def resolve(
        self, reference: EvidenceReference, /
    ) -> SampleEvalRecord:
        if reference not in self._records:
            await self._restore_published_bundles()
        return self._records[reference]

    async def _restore_published_bundles(self) -> None:
        for root in self._roots:
            for bundle_path in sorted(root.iterdir()):
                if bundle_path in self._restored:
                    continue
                self._restored.add(bundle_path)
                restored = await restore_eval_attempt(
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
            request(2, preprocess_mode=PreprocessMode.IN_PROCESS),
            executor=importable_json_executor(),
            execution_cache=execution_cache,
            object_store=ObjectStore(MemoryBackend()),
            publication=publication,
            pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
        )
    finally:
        await execution_cache.close()

    attempt = validation.result.attempt
    assert attempt.completeness is AttemptCompleteness.COMPLETE
    assert attempt.validity is AttemptValidity.VALID
    assert attempt.limit_exhaustion is None
    assert validation.comparison is None
    assert len(attempt.members) == 2


async def test_validate_preprocessing_reports_corpus_coverage(
    tmp_path: Path,
) -> None:
    publication = ArtifactBundlePublication.allocate(
        tmp_path, prefix="preprocessing"
    )
    execution_cache = cache(BatchStore())
    try:
        validation = await validate_preprocessing(
            request(2, preprocess_mode=PreprocessMode.PROCESS_POOL),
            executor=importable_json_executor(),
            execution_cache=execution_cache,
            object_store=ObjectStore(MemoryBackend()),
            publication=publication,
            pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
        )
    finally:
        await execution_cache.close()

    # Every sample in the fixture corpus carries the same one candidate text.
    coverage = validation.coverage
    assert coverage.texts_with_candidates == 1
    assert coverage.texts_without_candidates == 0
    assert coverage.texts_failed == 0
    assert coverage.corpus_size == 1
    attempt = validation.result.attempt
    assert attempt.completeness is AttemptCompleteness.COMPLETE
    assert attempt.validity is AttemptValidity.VALID
    assert validation.comparison is None


async def test_preprocessing_coverage_counters_partition_the_corpus(
    tmp_path: Path,
) -> None:
    """Every distinct corpus text lands in exactly one coverage counter."""

    corpus = (
        "def observed_load_count(_x):\n    return 1\n",
        "this is not python source\n",
        "x = 1\n",
    )
    publication = ArtifactBundlePublication.allocate(
        tmp_path, prefix="preprocessing"
    )
    execution_cache = cache(BatchStore())
    try:
        validation = await validate_preprocessing(
            request(
                len(corpus),
                texts=corpus,
                preprocess_mode=PreprocessMode.PROCESS_POOL,
            ),
            executor=importable_json_executor(),
            execution_cache=execution_cache,
            object_store=ObjectStore(MemoryBackend()),
            publication=publication,
            pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
        )
    finally:
        await execution_cache.close()

    coverage = validation.coverage
    assert coverage.texts_with_candidates == 1
    assert coverage.texts_without_candidates == 2
    assert coverage.texts_failed == 0
    assert coverage.corpus_size == len(corpus)


async def test_preprocessing_coverage_counts_a_failed_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A text whose preprocessing job failed is counted, never dropped."""

    corpus = (
        "def observed_load_count(_x):\n    return 1\n",
        "def wedged(_x):\n    return 2\n",
    )
    real_preprocess_batch = flows.preprocess_batch

    async def drop_the_second_text(texts, **keywords):  # type: ignore[no-untyped-def]
        traces = await real_preprocess_batch(texts, **keywords)
        return {
            text: trace for text, trace in traces.items() if text != corpus[1]
        }

    monkeypatch.setattr(flows, "preprocess_batch", drop_the_second_text)
    publication = ArtifactBundlePublication.allocate(
        tmp_path, prefix="preprocessing"
    )
    execution_cache = cache(BatchStore())
    try:
        validation = await validate_preprocessing(
            request(
                len(corpus),
                texts=corpus,
                preprocess_mode=PreprocessMode.PROCESS_POOL,
            ),
            executor=importable_json_executor(),
            execution_cache=execution_cache,
            object_store=ObjectStore(MemoryBackend()),
            publication=publication,
            pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
        )
    finally:
        await execution_cache.close()

    coverage = validation.coverage
    assert coverage.texts_with_candidates == 1
    assert coverage.texts_without_candidates == 0
    assert coverage.texts_failed == 1
    assert coverage.corpus_size == len(corpus)
    assert len(validation.result.attempt.members) == 1
    assert validation.result.attempt.members[0].sample.sample_id == "sample-0"


async def test_structural_comparison_needs_a_reference_and_a_resolver(
    tmp_path: Path,
) -> None:
    object_store = ObjectStore(MemoryBackend())
    execution_cache = cache(BatchStore())
    try:
        with pytest.raises(ValueError, match="structural comparison"):
            await validate_testing(
                request(1, preprocess_mode=PreprocessMode.IN_PROCESS),
                executor=importable_json_executor(),
                execution_cache=execution_cache,
                object_store=object_store,
                publication=ArtifactBundlePublication.allocate(
                    tmp_path, prefix="testing"
                ),
                pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
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
        request(1, preprocess_mode=PreprocessMode.IN_PROCESS),
        object_store=object_store,
    )
    candidate_request = reference_request.model_copy(
        update={
            "attempt": EvalAttemptId(attempt_id=UUID(int=2)),
        }
    )
    try:
        reference_run = await validate_testing(
            reference_request,
            executor=importable_json_executor(),
            execution_cache=execution_cache,
            object_store=object_store,
            publication=reference_publication,
            pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
        )
        assert reference_run.result.bundle_path is not None
        reference = await restore_eval_attempt(
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
            pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
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
