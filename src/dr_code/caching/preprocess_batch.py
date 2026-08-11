from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TypeVar
from uuid import uuid4

from dr_exec import (
    CompletedExecution,
    ExecutionJob,
    ExecutionSubmission,
    ImportableEntryPoint,
    JobId,
    WorkerPoolImportableJsonExecutor,
)

from dr_code.preprocessing.definition import PreprocessingDefinition
from dr_code.preprocessing.execution import (
    PreprocessTextExecutionError,
    build_candidate_sources_job,
    build_preprocess_text_job,
    parse_candidate_sources_result,
    parse_preprocess_text_result,
    preprocess_text_request,
)
from dr_code.preprocessing.job import (
    CANDIDATE_SOURCES_ENTRY_POINT,
    PREPROCESS_TEXT_ENTRY_POINT,
    PreprocessTextJobRequest,
)
from dr_code.trace import Trace

_ResultT = TypeVar("_ResultT")

TraceObserver = Callable[[str, Trace | None], None]
CandidateSourcesObserver = Callable[[str, tuple[str, ...] | None], None]


@dataclass(frozen=True, slots=True)
class _PreprocessWork:
    text: str


async def preprocess_batch(
    texts: Iterable[str],
    *,
    definition: PreprocessingDefinition,
    worker_count: int | None = None,
    on_trace: TraceObserver | None = None,
) -> dict[str, Trace]:
    """Preprocess distinct texts in parallel and return their traces.

    Each distinct text runs once as one trusted importable-JSON job on a
    dr-exec worker pool, which imports the preprocessing entry point once per
    worker and then runs jobs on real cores.

    Without `on_trace` every trace is retained and returned keyed by text.
    With `on_trace` each result is handed to the observer as it completes and
    nothing is retained, so a caller that only consumes per-text results does
    not hold the whole corpus in memory; the returned mapping is then empty.
    A text whose job did not return one valid trace is observed as `None` and
    is absent from the returned mapping.

    Callers that consume candidate sources rather than whole traces use
    `candidate_sources_batch` instead: a serialized trace is two orders of
    magnitude larger than the sources it carries, and the caller decodes and
    validates every byte that crosses the worker boundary.
    """

    return await _run_batch(
        texts,
        definition=definition,
        worker_count=worker_count,
        observer=on_trace,
        entry_point=PREPROCESS_TEXT_ENTRY_POINT,
        build_job=build_preprocess_text_job,
        parse_result=parse_preprocess_text_result,
    )


async def candidate_sources_batch(
    texts: Iterable[str],
    *,
    definition: PreprocessingDefinition,
    worker_count: int | None = None,
    on_sources: CandidateSourcesObserver | None = None,
) -> dict[str, tuple[str, ...]]:
    """Preprocess distinct texts and return only their candidate sources.

    Behaves like `preprocess_batch` except that each worker returns the
    candidate sources it extracted instead of the whole trace, so the trace
    never crosses the worker boundary.
    """

    return await _run_batch(
        texts,
        definition=definition,
        worker_count=worker_count,
        observer=on_sources,
        entry_point=CANDIDATE_SOURCES_ENTRY_POINT,
        build_job=build_candidate_sources_job,
        parse_result=parse_candidate_sources_result,
    )


async def _run_batch(
    texts: Iterable[str],
    *,
    definition: PreprocessingDefinition,
    worker_count: int | None,
    observer: Callable[[str, _ResultT | None], None] | None,
    entry_point: ImportableEntryPoint,
    build_job: Callable[[JobId, PreprocessTextJobRequest], ExecutionJob],
    parse_result: Callable[[CompletedExecution], _ResultT],
) -> dict[str, _ResultT]:
    distinct_texts = list(dict.fromkeys(texts))
    results: dict[str, _ResultT] = {}
    if not distinct_texts:
        return results

    with WorkerPoolImportableJsonExecutor(
        entry_point=entry_point,
        worker_count=worker_count,
    ) as executor:
        async with executor.open_pool() as pool:
            async for completion in pool.map_stream(
                _submissions(distinct_texts, definition, build_job)
            ):
                text = completion.context.text
                try:
                    parsed: _ResultT | None = parse_result(
                        completion.completed_execution
                    )
                except PreprocessTextExecutionError:
                    parsed = None
                if observer is not None:
                    observer(text, parsed)
                elif parsed is not None:
                    results[text] = parsed
    return results


async def _submissions(
    texts: Sequence[str],
    definition: PreprocessingDefinition,
    build_job: Callable[[JobId, PreprocessTextJobRequest], ExecutionJob],
) -> AsyncIterator[ExecutionSubmission[_PreprocessWork]]:
    for text in texts:
        request = preprocess_text_request(text, definition)
        yield ExecutionSubmission(
            job=build_job(JobId(uuid4()), request),
            context=_PreprocessWork(text=text),
        )


__all__ = [
    "CandidateSourcesObserver",
    "TraceObserver",
    "candidate_sources_batch",
    "preprocess_batch",
]
