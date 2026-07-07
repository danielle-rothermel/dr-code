"""Pipeline step 6 — normalize.

Run every configured normalizer against every valid candidate. Each
normalizer is computed independently from the source so any pair can be
compared without ordering bias. Subprocess-based normalizers are cached
by ``(content, normalizer_name, tool_version_string + timeout)``.
"""

from __future__ import annotations

from typing import Final

from code_eval.cache.store import CacheStore
from code_eval.config import ValidatorConfig
from code_eval.models.candidate import Candidate
from code_eval.models.normalized_form import NormalizedForm
from code_eval.models.tool_versions import ToolVersions
from code_eval.names import NormalizerName
from code_eval.normalizers import NORMALIZERS
from code_eval.subprocess_runner import SubprocessRunner

#: Which normalizers use a subprocess (and are therefore cache-worthy).
_SUBPROCESS_NORMALIZERS: Final[frozenset[NormalizerName]] = frozenset(
    {
        NormalizerName.L2_RUFF_FORMAT,
        NormalizerName.L3_RUFF_FIX_SAFE,
        NormalizerName.L4_RUFF_FIX_UNSAFE,
        NormalizerName.L5_TY_FIX,
        NormalizerName.IMPORT_SORT_DEDUP,
        NormalizerName.STRING_FORM_NORMALIZE,
    }
)


def _tool_version_string(versions: ToolVersions) -> str:
    """Stable joined-version string for cache keying."""
    parts = sorted(versions.as_dict().items())
    return "|".join(f"{k}={v}" for k, v in parts)


def _cache_key_string(versions: ToolVersions, timeout_s: float) -> str:
    """Cache invalidation string: tool versions plus subprocess timeout."""
    return f"{_tool_version_string(versions)}|timeout_s={timeout_s}"


def _make_normalizer(name: NormalizerName, runner: SubprocessRunner) -> object:
    """Construct a normalizer, threading the shared runner where applicable."""
    cls = NORMALIZERS[name]
    # Normalizers that take a runner accept it as their sole __init__ kwarg.
    # Pure-Python ones take no args. We detect by name set.
    if name in _SUBPROCESS_NORMALIZERS:
        return cls(runner=runner)  # type: ignore[call-arg]
    return cls()


def run_normalize(
    valid_candidates: tuple[Candidate, ...],
    config: ValidatorConfig,
    versions: ToolVersions,
    runner: SubprocessRunner,
) -> dict[str, dict[str, NormalizedForm]]:
    """Run every configured normalizer over every valid candidate.

    Returns ``{candidate_id: {normalizer_name: NormalizedForm}}``.
    """
    if not config.normalizers:
        return {}

    cache: CacheStore | None = None
    if config.cache_dir is not None:
        cache = CacheStore(root=config.cache_dir)
        cache.ensure_root()

    versions_key = _cache_key_string(versions, config.subprocess_timeout_s)
    out: dict[str, dict[str, NormalizedForm]] = {}
    #: Memoize per (source, normalizer) so candidates with identical source
    #: pay normalization cost only once per ``validate()`` call.
    memo: dict[tuple[str, NormalizerName], NormalizedForm] = {}

    for cand in valid_candidates:
        per_cand: dict[str, NormalizedForm] = {}
        for nname in config.normalizers:
            memo_key = (cand.source, nname)
            if memo_key in memo:
                per_cand[nname.value] = memo[memo_key]
                continue

            form: NormalizedForm | None = None
            if cache is not None and nname in _SUBPROCESS_NORMALIZERS:
                form = cache.get(cand.source, nname.value, versions_key)
            if form is None:
                normalizer = _make_normalizer(nname, runner)
                form = normalizer.normalize(cand.source)  # type: ignore[attr-defined]
                if cache is not None and nname in _SUBPROCESS_NORMALIZERS and form.success:
                    cache.put(cand.source, nname.value, versions_key, form)
            if form.success:
                memo[memo_key] = form
            per_cand[nname.value] = form
        out[cand.candidate_id] = per_cand
    return out
