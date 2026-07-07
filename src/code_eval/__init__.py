"""code-eval: validation harness for LLM-generated Python programs.

Public API (frozen surface):
    LLMCodeValidator — the validator class
    ValidatorConfig  — frozen configuration
    ValidationResult — structured outcome envelope
    Candidate        — one valid extraction with provenance
    CandidateRecoveryResult — recovery-stage outcome envelope
    DEFAULT_CONFIG   — full pipeline including all normalizers
    EXTRACTION_CONFIG — parse-only preset (no normalization)

Everything else is considered internal.
"""

from code_eval.__version__ import __version__
from code_eval.config import DEFAULT_CONFIG, EXTRACTION_CONFIG, ValidatorConfig
from code_eval.models.candidate import Candidate
from code_eval.models.candidate_rank import CandidateRank
from code_eval.models.candidate_recovery_attempt import CandidateRecoveryAttempt
from code_eval.models.candidate_recovery_result import CandidateRecoveryResult
from code_eval.models.candidate_selection import CandidateSelection
from code_eval.models.extraction_result import ExtractionResult
from code_eval.models.validation_result import ValidationResult
from code_eval.names import NormalizerName, ValidatorName
from code_eval.validator import LLMCodeValidator

__all__ = [
    "DEFAULT_CONFIG",
    "EXTRACTION_CONFIG",
    "Candidate",
    "CandidateRank",
    "CandidateRecoveryAttempt",
    "CandidateRecoveryResult",
    "CandidateSelection",
    "ExtractionResult",
    "LLMCodeValidator",
    "NormalizerName",
    "ValidationResult",
    "ValidatorConfig",
    "ValidatorName",
    "__version__",
]
