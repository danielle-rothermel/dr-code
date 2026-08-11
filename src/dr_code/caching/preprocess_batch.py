from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from dataclasses import dataclass
from uuid import uuid4

from dr_exec import (
    ExecutionSubmission,
    JobId,
    WorkerPoolImportableJsonExecutor,
)

from dr_code.preprocessing.definition import PreprocessingDefinition
from dr_code.preprocessing.execution import (
    PreprocessTextExecutionError,
    build_preprocess_text_job,
    parse_preprocess_text_result,
    preprocess_text_request,
)
from dr_code.preprocessing.job import PREPROCESS_TEXT_ENTRY_POINT
from dr_code.trace import Trace

TraceObserver = Callable[[str, Trace | None], None]


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
    """Preprocess distinct texts in parallel across worker processes.

    Each distinct text runs once as one trusted importable-JSON job on a
    dr-exec worker pool, which imports the preprocessing entry point once per
    worker and then runs jobs on real cores.

    Without `on_trace` every trace is retained and returned keyed by text.
    With `on_trace` each result is handed to the observer as it completes and
    nothing is retained, so a caller that only consumes per-text results does
    not hold the whole corpus in memory; the returned mapping is then empty.
    A text whose job did not return one valid trace is observed as `None` and
    is absent from the returned mapping.
    """

    distinct_texts = list(dict.fromkeys(texts))
    results: dict[str, Trace] = {}
    if not distinct_texts:
        return results

    with WorkerPoolImportableJsonExecutor(
        entry_point=PREPROCESS_TEXT_ENTRY_POINT,
        worker_count=worker_count,
    ) as executor:
        async with executor.open_pool() as pool:
            async for completion in pool.map_stream(
                _submissions(distinct_texts, definition)
            ):
                text = completion.context.text
                try:
                    trace: Trace | None = parse_preprocess_text_result(
                        completion.completed_execution
                    )
                except PreprocessTextExecutionError:
                    trace = None
                if on_trace is not None:
                    on_trace(text, trace)
                elif trace is not None:
                    results[text] = trace
    return results


async def _submissions(
    texts: Sequence[str],
    definition: PreprocessingDefinition,
) -> AsyncIterator[ExecutionSubmission[_PreprocessWork]]:
    for text in texts:
        request = preprocess_text_request(text, definition)
        yield ExecutionSubmission(
            job=build_preprocess_text_job(JobId(uuid4()), request),
            context=_PreprocessWork(text=text),
        )


__all__ = [
    "TraceObserver",
    "preprocess_batch",
]
