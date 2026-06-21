"""code-eval configuration for dr-code parse stage."""

from __future__ import annotations

from functools import lru_cache

from code_eval import EXTRACTION_CONFIG, LLMCodeValidator

__all__ = ["EXTRACTION_CONFIG", "config_fingerprint", "default_validator"]


@lru_cache(maxsize=1)
def default_validator() -> LLMCodeValidator:
    """Shared validator using EXTRACTION_CONFIG (no subprocess normalizers)."""
    return LLMCodeValidator(config=EXTRACTION_CONFIG)


def config_fingerprint() -> str:
    """SHA-256 fingerprint for the active parse-stage validator config."""
    return default_validator().config_fingerprint
