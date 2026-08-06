from __future__ import annotations

from typing import TYPE_CHECKING

from dr_store import ObjectStore, RecordCache, SqliteBackend

if TYPE_CHECKING:
    from pathlib import Path


def open_sqlite_record_cache(path: str | Path) -> RecordCache:
    """Open a record cache over the SQLite database at ``path``.

    The database is created when absent; the caller owns its location and
    lifetime.
    """
    return RecordCache(ObjectStore(SqliteBackend(path)))


__all__ = ["open_sqlite_record_cache"]
