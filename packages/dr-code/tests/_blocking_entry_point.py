"""A trusted importable-JSON entry point that never returns.

Importable by spawned dr-exec workers (the test that uses it puts this
directory on ``PYTHONPATH``, which workers inherit), so a wall-time budget
can be observed firing against a job that is genuinely wedged rather than
merely slow. Blocking on a pipe read whose write end stays open parks the
worker indefinitely without burning CPU and without depending on elapsed
time.
"""

from __future__ import annotations

import os
from typing import Any

_BLOCK_MARKER = "block"


def blocking_job(request: Any, /) -> Any:
    """Return immediately, or park forever when asked to block.

    A request carrying ``{"block": true}`` blocks on a pipe that no one ever
    writes to. The write end is deliberately held open by this process, so
    the read never reaches EOF and the only way this call ends is the worker
    being killed on its wall-time budget.
    """

    if isinstance(request, dict) and request.get(_BLOCK_MARKER):
        read_descriptor, write_descriptor = os.pipe()
        # `write_descriptor` is intentionally never closed and never written
        # to: an open write end is what makes this read block forever.
        del write_descriptor
        os.read(read_descriptor, 1)
    return {"echoed": request}
