from __future__ import annotations


import pytest

from drc_generation_corpus.tasks.human_eval import HumanEvalTaskAdapter

from _paths import HUMANEVALPLUS_SNAPSHOT as HUMANEVAL_SNAPSHOT


@pytest.fixture(scope="session")
def humaneval_task_adapter() -> HumanEvalTaskAdapter:
    return HumanEvalTaskAdapter(HUMANEVAL_SNAPSHOT)
