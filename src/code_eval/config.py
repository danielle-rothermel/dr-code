"""ValidatorConfig and the canonical DEFAULT_CONFIG.

The config is part of the frozen public surface. Ad-hoc tweaks at experiment
time are discouraged by design — define a named variant if a different
configuration is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from pydantic import Field

from code_eval.models.base import FrozenModel
from code_eval.names import (
    DEFAULT_CACHE_DIR_NAME,
    DEFAULT_SUBPROCESS_TIMEOUT_S,
    DEFAULT_TAB_WIDTH,
    NormalizerName,
    ValidatorName,
)


def _default_cache_dir() -> Path:
    """Default to `.cache/` in the current working directory."""
    return Path.cwd() / DEFAULT_CACHE_DIR_NAME


#: Canonical default set of normalizers run on every valid candidate.
#: Includes the layered L0-L5 chain plus four orthogonal forms used for
#: cross-candidate equivalence checks (annotation_strip, name_normalize,
#: import_sort_dedup, string_form_normalize).
DEFAULT_NORMALIZERS: Final[tuple[NormalizerName, ...]] = (
    NormalizerName.L0_CANONICAL_AST,
    NormalizerName.L1_STRIP_COMMENTS_DOCSTRINGS,
    NormalizerName.L2_RUFF_FORMAT,
    NormalizerName.L3_RUFF_FIX_SAFE,
    NormalizerName.L4_RUFF_FIX_UNSAFE,
    NormalizerName.L5_TY_FIX,
    NormalizerName.IMPORT_SORT_DEDUP,
    NormalizerName.NAME_NORMALIZE,
    NormalizerName.ANNOTATION_STRIP,
    NormalizerName.STRING_FORM_NORMALIZE,
)

#: Canonical default set of validators.
DEFAULT_VALIDATORS: Final[tuple[ValidatorName, ...]] = (
    ValidatorName.AST_PARSE,
    ValidatorName.COMPILE_CHECK,
    ValidatorName.AST_SHAPE,
)


class ValidatorConfig(FrozenModel):
    """User-visible validator configuration. Frozen by base class."""

    #: Where subprocess-cached normalizer outputs are stored.
    cache_dir: Path = Field(default_factory=_default_cache_dir)

    #: Timeout for each subprocess invocation, in seconds.
    subprocess_timeout_s: float = DEFAULT_SUBPROCESS_TIMEOUT_S

    #: Tab width used by the text-normalization extractor.
    tab_width: int = DEFAULT_TAB_WIDTH

    #: Which validators to run.
    validators: tuple[ValidatorName, ...] = DEFAULT_VALIDATORS

    #: Which normalizers to run on every valid candidate.
    normalizers: tuple[NormalizerName, ...] = DEFAULT_NORMALIZERS

    #: If True, the optional ImportResolve validator is also run.
    enable_import_resolve_validator: bool = False


DEFAULT_CONFIG: Final[ValidatorConfig] = ValidatorConfig()

#: Recommended preset for downstream parse pipelines (e.g. dr-code stage 2).
#: Skips normalization subprocess work; callers only need Candidate Recovery.
EXTRACTION_CONFIG: Final[ValidatorConfig] = ValidatorConfig(normalizers=())
