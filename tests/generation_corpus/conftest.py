from __future__ import annotations

from pathlib import Path

import pytest

from dr_code.generation_corpus.tasks.human_eval import HumanEvalTaskAdapter

HUMANEVAL_SNAPSHOT = (
    Path(__file__).parents[1] / "corpus" / "humanevalplus_snapshot.json"
)


@pytest.fixture(scope="session")
def humaneval_task_adapter() -> HumanEvalTaskAdapter:
    return HumanEvalTaskAdapter(HUMANEVAL_SNAPSHOT)
