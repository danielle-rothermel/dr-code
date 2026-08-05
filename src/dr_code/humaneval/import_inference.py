"""HumanEval import inference backed by the preprocessing implementation.

``infer_necessary_imports`` delegates to
``dr_code.preprocessing.import_inference.infer_necessary_imports``, which the
atomic preprocessing steps also use.

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


__all__ = ["infer_necessary_imports"]
