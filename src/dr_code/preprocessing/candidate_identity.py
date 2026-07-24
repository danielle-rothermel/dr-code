"""Stable identity for final preprocessing candidates."""

from __future__ import annotations

import hashlib


def candidate_id_for_source(source: str) -> str:
    """Return a content-derived identity stable across processes and runs."""
    digest = hashlib.blake2b(
        source.encode("utf-8"), digest_size=16
    ).hexdigest()
    return f"candidate-{digest}"


__all__ = ["candidate_id_for_source"]
