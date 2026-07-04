"""ValidationResult - the single object every experiment serializes."""

from __future__ import annotations

from pydantic import Field

from code_eval.models.base import FrozenModel
from code_eval.models.candidate_recovery_result import CandidateRecoveryResult
from code_eval.models.diagnostic import Diagnostic
from code_eval.models.extraction_result import ExtractionResult
from code_eval.models.normalized_form import NormalizedForm


class ValidationResult(FrozenModel):
    """The full structured outcome of validating one LLM output."""

    raw_input: str
    task_id: str | None = None

    #: SHA-256 over the canonical config dump + tool_versions. Same fingerprint
    #: across runs guarantees reproducibility.
    config_fingerprint: str
    #: tool name -> version string. Always includes "python" and "code_eval"
    #: at minimum, plus any subprocess tools available.
    tool_versions: dict[str, str]

    extraction: ExtractionResult
    recovery: CandidateRecoveryResult

    #: candidate_id -> normalizer name -> NormalizedForm
    normalizations: dict[str, dict[str, NormalizedForm]] = Field(default_factory=dict)

    diagnostics: tuple[Diagnostic, ...] = ()
