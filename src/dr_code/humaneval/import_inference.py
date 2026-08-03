"""HumanEval import inference backed by the preprocessing implementation.

``infer_necessary_imports`` and ``IMPORT_ALIAS_MAP`` expose the import handling
used by HumanEval parsing. Their implementation is shared with the atomic
preprocessing steps in ``dr_code.preprocessing.import_inference``.

The delegation is deferred to call time to avoid an import cycle: the
preprocessing package's ``definitions`` module imports coordinate constants
from ``code_parsing``, which imports this module at load time.
"""

from __future__ import annotations


def infer_necessary_imports(source: str) -> str:
    """Repair, infer, then deduplicate imports in ``source``."""
    from dr_code.preprocessing.import_inference import (
        infer_necessary_imports as _impl,
    )

    return _impl(source)


def __getattr__(name: str) -> object:
    # ``IMPORT_ALIAS_MAP`` is served lazily (same cycle constraint) while
    # staying importable from this module for HumanEval callers. It is
    # kept out of ``__all__`` because it is not a static module binding.
    if name == "IMPORT_ALIAS_MAP":
        from dr_code.preprocessing.import_inference import IMPORT_ALIAS_MAP

        return IMPORT_ALIAS_MAP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["infer_necessary_imports"]
