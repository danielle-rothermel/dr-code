"""Re-runnable LLM failure classifier for preprocessing runs.

This package labels preprocessing failures (parse/extraction failures and, when
candidate-evaluation artifacts exist, test failures) using a subscription LLM
lane and persists a per-task rollup through the viewer's task-annotation
machinery. It is additive: no schema migration, no example-level DB columns.
"""

from __future__ import annotations
