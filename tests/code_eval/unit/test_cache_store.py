"""Unit tests for content-addressed normalizer cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from code_eval.cache.store import CacheStore
from code_eval.models.normalized_form import NormalizedForm
from code_eval.names import NormalizerName


def _sample_form(source: str = "x = 1\n") -> NormalizedForm:
    return NormalizedForm(
        normalizer=NormalizerName.L2_RUFF_FORMAT,
        source=source,
        transformations_applied=(NormalizerName.L2_RUFF_FORMAT.value,),
        success=True,
    )


def test_cache_put_and_get_round_trip(tmp_path: Path) -> None:
    store = CacheStore(root=tmp_path)
    store.ensure_root()
    form = _sample_form()
    store.put("x = 1\n", NormalizerName.L2_RUFF_FORMAT.value, "ruff=0.8.4", form)
    hit = store.get("x = 1\n", NormalizerName.L2_RUFF_FORMAT.value, "ruff=0.8.4")
    assert hit is not None
    assert hit.from_cache
    assert hit.source == form.source


def test_cache_miss_on_different_timeout_key(tmp_path: Path) -> None:
    store = CacheStore(root=tmp_path)
    store.ensure_root()
    form = _sample_form()
    store.put(
        "x = 1\n",
        NormalizerName.L2_RUFF_FORMAT.value,
        "ruff=0.8.4|timeout_s=30.0",
        form,
    )
    miss = store.get(
        "x = 1\n",
        NormalizerName.L2_RUFF_FORMAT.value,
        "ruff=0.8.4|timeout_s=5.0",
    )
    assert miss is None


def test_cache_miss_on_corrupt_file(tmp_path: Path) -> None:
    store = CacheStore(root=tmp_path)
    store.ensure_root()
    form = _sample_form()
    store.put("x = 1\n", NormalizerName.L2_RUFF_FORMAT.value, "ruff=0.8.4", form)
    key_path = next(tmp_path.rglob("*.json"))
    key_path.write_text("{not valid json", encoding="utf-8")
    assert store.get("x = 1\n", NormalizerName.L2_RUFF_FORMAT.value, "ruff=0.8.4") is None


@pytest.mark.parametrize("content", ["x = 1\n", "y = 2\n"])
def test_cache_keys_are_content_addressed(tmp_path: Path, content: str) -> None:
    store = CacheStore(root=tmp_path)
    store.ensure_root()
    store.put(content, NormalizerName.L2_RUFF_FORMAT.value, "ruff=0.8.4", _sample_form(content))
    assert store.get(content, NormalizerName.L2_RUFF_FORMAT.value, "ruff=0.8.4") is not None
    assert store.get("other\n", NormalizerName.L2_RUFF_FORMAT.value, "ruff=0.8.4") is None
