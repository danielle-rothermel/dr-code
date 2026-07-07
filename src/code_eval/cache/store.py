"""Content-addressed disk cache.

Key layout: `{cache_dir}/{normalizer}/{first2}/{rest_of_hash}.json` where
the hash is computed over `(content + normalizer_name + tool_version_string)`.
The version string passed from `normalize_step` also includes
``subprocess_timeout_s`` for cache invalidation.

Values are JSON-serialized `NormalizedForm` instances. The cache is
append-only and safe to share across experiments.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final

from code_eval.models.base import FrozenModel
from code_eval.models.normalized_form import NormalizedForm

_SEP: Final[str] = "\x1f"  # ASCII Unit Separator — safe joiner for hash input


def _hash_key(content: str, normalizer_name: str, tool_version_string: str) -> str:
    raw = content + _SEP + normalizer_name + _SEP + tool_version_string
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class CacheStore(FrozenModel):
    """Disk-backed cache, keyed by `(content, normalizer, tool_version)`."""

    root: Path

    def ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, normalizer_name: str, key: str) -> Path:
        return self.root / normalizer_name / key[:2] / f"{key[2:]}.json"

    def get(
        self,
        content: str,
        normalizer_name: str,
        tool_version_string: str,
    ) -> NormalizedForm | None:
        key = _hash_key(content, normalizer_name, tool_version_string)
        p = self._path_for(normalizer_name, key)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return NormalizedForm.model_validate(data).model_copy(update={"from_cache": True})

    def put(
        self,
        content: str,
        normalizer_name: str,
        tool_version_string: str,
        form: NormalizedForm,
    ) -> None:
        key = _hash_key(content, normalizer_name, tool_version_string)
        p = self._path_for(normalizer_name, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Strip the from_cache flag before persisting.
        to_persist = form.model_copy(update={"from_cache": False})
        p.write_text(to_persist.model_dump_json(indent=2), encoding="utf-8")
