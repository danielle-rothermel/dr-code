"""Session-scoped test boundaries for the mutant suite."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path

import pytest

from dr_code.synthetic.humaneval_loader import load_humaneval_plus
from mutants.humaneval_cache import (
    build_cached_loader,
    load_snapshot_tasks,
)


def _xdist_shared_temp_root(
    pytestconfig: pytest.Config,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path | None:
    if not hasattr(pytestconfig, "workerinput"):
        return None
    return tmp_path_factory.getbasetemp().parent


@pytest.fixture(scope="session", autouse=True)
def cache_humaneval_snapshot_for_mutant_tests(
    pytestconfig: pytest.Config,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
    """Patch only mutant modules that import the expensive snapshot loader."""
    snapshot_tasks = load_snapshot_tasks(
        original_loader=load_humaneval_plus,
        shared_temp_root=_xdist_shared_temp_root(
            pytestconfig, tmp_path_factory
        ),
    )
    cached_loader = build_cached_loader(
        snapshot_tasks=snapshot_tasks,
        original_loader=load_humaneval_plus,
    )
    generate_module = importlib.import_module("dr_code.mutants.generate")
    cli_module = importlib.import_module("dr_code.mutants.cli")

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            generate_module, "load_humaneval_plus", cached_loader
        )
        monkeypatch.setattr(cli_module, "load_humaneval_plus", cached_loader)
        yield
