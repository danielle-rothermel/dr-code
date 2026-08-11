from __future__ import annotations

import pytest

from dr_code.caching.preprocess_batch import preprocess_batch
from dr_code.preprocessing import EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION
from dr_code.trace import OUTPUT_KEY, InspectedCodeCandidateSetArtifact, Trace

pytestmark = pytest.mark.asyncio

_FENCED = "Here is the code:\n```python\ndef f(x):\n    return x + 1\n```\n"


async def test_preprocess_batch_returns_one_trace_per_distinct_text() -> None:
    duplicated = [_FENCED, _FENCED]

    results = await preprocess_batch(
        duplicated,
        definition=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
        worker_count=2,
    )

    assert set(results) == {_FENCED}
    output = results[_FENCED].value(OUTPUT_KEY)
    assert isinstance(output, InspectedCodeCandidateSetArtifact)


async def test_preprocess_batch_runs_every_distinct_text() -> None:
    texts = [
        f"```python\ndef f_{index}(x):\n    return x + {index}\n```"
        for index in range(8)
    ]

    results = await preprocess_batch(
        texts,
        definition=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
        worker_count=2,
    )

    assert set(results) == set(texts)


async def test_preprocess_batch_streams_to_the_observer_without_retaining() -> (
    None
):
    texts = [
        f"```python\ndef f_{index}(x):\n    return x + {index}\n```"
        for index in range(4)
    ]
    observed: dict[str, Trace | None] = {}

    results = await preprocess_batch(
        texts,
        definition=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
        worker_count=2,
        on_trace=observed.__setitem__,
    )

    assert results == {}
    assert set(observed) == set(texts)
    assert all(trace is not None for trace in observed.values())


async def test_preprocess_batch_returns_empty_for_no_texts() -> None:
    assert (
        await preprocess_batch(
            [],
            definition=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
        )
        == {}
    )
