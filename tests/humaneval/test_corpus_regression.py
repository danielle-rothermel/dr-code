"""Corruption-corpus regression baseline for the v1 best-effort parser.

Runs the ported whetstone v1 extraction over every sample in the
code-eval synthetic corruption corpus and pins per-recipe aggregate
outcomes. This is the measured baseline later parser improvements
(profile v2) must beat; a diff here means the v1 port's behavior moved.

Regenerate the baseline (only when intentionally re-baselining) with:
    uv run python tests/humaneval/corpus_baseline.py
"""

from __future__ import annotations

import json

from corpus_baseline import BASELINE_PATH, compute_corpus_baseline


def test_corpus_baseline_reproduces() -> None:
    assert BASELINE_PATH.exists(), (
        f"missing {BASELINE_PATH}; generate with "
        "`uv run python tests/humaneval/corpus_baseline.py`"
    )
    stored = json.loads(BASELINE_PATH.read_text())
    assert compute_corpus_baseline() == stored
