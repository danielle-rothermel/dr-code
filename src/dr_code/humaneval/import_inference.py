"""Old-pipeline import inference — delegates to the preprocessing package.

The implementation now lives in ``dr_code.preprocessing.import_inference``.
This module preserves the public surface the old pipeline depends on
(``infer_necessary_imports``, ``IMPORT_ALIAS_MAP``) so its callers are
unchanged; the shared fix to bound-name collection flows through here too.

The delegation is deferred to call time to avoid an import cycle: the
preprocessing package's ``definitions`` module imports coordinate constants
from ``code_parsing``, which imports this module at load time.
"""

from __future__ import annotations


def infer_necessary_imports(source: str) -> str:
    """Repair, infer, then dedupe imports — the full old-pipeline pass."""
    from dr_code.preprocessing.import_inference import (
        infer_necessary_imports as _impl,
    )

    return _impl(source)


def __getattr__(name: str) -> object:
    # ``IMPORT_ALIAS_MAP`` is served lazily (same cycle constraint) while
    # staying importable from this module for old-pipeline callers. It is
    # kept out of ``__all__`` because it is not a static module binding.
    if name == "IMPORT_ALIAS_MAP":
        from dr_code.preprocessing.import_inference import IMPORT_ALIAS_MAP

        return IMPORT_ALIAS_MAP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["infer_necessary_imports"]
