from __future__ import annotations

from pathlib import Path

from dr_store import ObjectStore, RecordCache, SqliteBackend


def open_sqlite_record_cache(path: str | Path) -> RecordCache:
    """Open a SQLite cache; the caller owns its path and lifetime."""
    return RecordCache(ObjectStore(SqliteBackend(path)))


__all__ = ["open_sqlite_record_cache"]
