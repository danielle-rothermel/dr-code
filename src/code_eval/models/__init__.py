"""Pydantic models for code-eval.

All models inherit from `FrozenModel` (in `base.py`), which enforces the
project-wide config: `frozen=True, extra="forbid", validate_assignment=True`.

Models are intentionally split one-per-file. See individual files.
"""
