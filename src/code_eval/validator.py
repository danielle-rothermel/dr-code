"""LLMCodeValidator - the single public entry point.

Orchestrates all six pipeline steps: capture, text-normalize, extract,
repair, validate, and normalize.
"""

from __future__ import annotations

import hashlib
import json
from typing import Final

from code_eval.__version__ import __version__
from code_eval.candidate_recovery import run_candidate_recovery
from code_eval.config import DEFAULT_CONFIG, ValidatorConfig
from code_eval.extraction import run_extraction
from code_eval.models.tool_versions import ToolVersions
from code_eval.models.validation_result import ValidationResult
from code_eval.pipeline import (
    backfill_extraction_log,
    run_normalize,
)
from code_eval.subprocess_runner import (
    SubprocessRunner,
    discover_version,
    python_version,
)

_CFG_FINGERPRINT_SEP: Final[str] = "\x1f"


def _compute_fingerprint(config: ValidatorConfig, versions: ToolVersions) -> str:
    """Stable hash over config + versions."""
    cfg_json = json.dumps(json.loads(config.model_dump_json()), sort_keys=True, default=str)
    ver_json = json.dumps(versions.model_dump(mode="json"), sort_keys=True)
    raw = cfg_json + _CFG_FINGERPRINT_SEP + ver_json
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _capture_tool_versions() -> ToolVersions:
    ruff = discover_version("ruff") or "unknown"
    ty = discover_version("ty")
    return ToolVersions(
        python=python_version(),
        code_eval=__version__,
        ruff=ruff,
        ty=ty,
    )


class LLMCodeValidator:
    """Validate raw LLM output as a Python program.

    Public API:
        validator = LLMCodeValidator()
        result = validator.validate(raw_text)

    The class captures tool versions and computes a `config_fingerprint` at
    construction time. The same fingerprint across runs guarantees that the
    output of `validate()` is reproducible.
    """

    def __init__(self, config: ValidatorConfig | None = None) -> None:
        self._config: ValidatorConfig = config if config is not None else DEFAULT_CONFIG
        self._tool_versions: ToolVersions = _capture_tool_versions()
        self._fingerprint: str = _compute_fingerprint(self._config, self._tool_versions)
        self._runner = SubprocessRunner(timeout_s=self._config.subprocess_timeout_s)

    # --- read-only public introspection -----------------------------------

    @property
    def config(self) -> ValidatorConfig:
        return self._config

    @property
    def tool_versions(self) -> ToolVersions:
        return self._tool_versions

    @property
    def config_fingerprint(self) -> str:
        return self._fingerprint

    # --- main entry point -------------------------------------------------

    def validate(self, raw_output: str, *, task_id: str | None = None) -> ValidationResult:
        """Validate raw LLM output.

        Implements all six pipeline steps (capture -> text-normalize ->
        extract -> repair -> validate -> normalize).
        """
        # Steps 2 + 3: Text Normalization + Extraction
        extraction = run_extraction(raw_output, self._config)

        # Steps 4 + 5: Candidate Recovery
        recovery = run_candidate_recovery(extraction.candidates, self._config)
        extraction_log = backfill_extraction_log(
            extraction.extraction_log,
            recovery.valid_candidates,
        )
        extraction = extraction.model_copy(update={"extraction_log": extraction_log})

        # Step 6: normalize every valid candidate.
        normalizations = run_normalize(
            recovery.valid_candidates,
            self._config,
            self._tool_versions,
            self._runner,
        )

        return ValidationResult(
            raw_input=raw_output,
            task_id=task_id,
            config_fingerprint=self._fingerprint,
            tool_versions=self._tool_versions.as_dict(),
            extraction=extraction,
            recovery=recovery,
            normalizations=normalizations,
            diagnostics=(),
        )
