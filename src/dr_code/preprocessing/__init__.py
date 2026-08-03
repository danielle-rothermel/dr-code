"""Preprocessing: atomic, declared steps over typed artifacts.

Each ``PreprocessingDefinition`` chooses an ordered sequence of operations.
The runner applies those operations and records their artifacts, absences,
facts, and producer identity in a ``dr_code.trace.Trace``.

Core functions (``text_transforms``, ``text_analysis``, ``code_analysis``)
provide step bodies. ``preprocessing.import_inference`` owns the import repair,
inference, and deduplication logic used by preprocessing and HumanEval parsing.
"""

from __future__ import annotations

from dr_code.preprocessing.definition import (
    PreprocessingDefinition,
    StepSpec,
    preprocessing_definition_hash,
)
from dr_code.preprocessing.definitions import (
    BEST_EFFORT_HUMANEVAL_DEFINITION_ID,
    BEST_EFFORT_V2_DEFINITION,
    DEFINITION_VERSION,
    FIELD_MARKER_V2_DEFINITION,
    STRICT_FIELD_MARKER_DEFINITION_ID,
    SUPPORTED_DEFINITION_IDS,
    SUPPORTED_DEFINITION_VERSIONS,
    resolve_preprocessing_definition,
)
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.registry import REGISTRY
from dr_code.preprocessing.runner import (
    BoundStep,
    bind_definition,
    run_preprocessing,
)
from dr_code.preprocessing.steps.base import (
    AlternativesStep,
    CandidateMapStep,
    Step,
    StepFailedError,
    StepOutput,
    StepSettings,
)
from dr_code.preprocessing.steps.extract_candidates import (
    DEFAULT_STRATEGIES,
    STRATEGY_REGISTRY,
    ExtractionStrategy,
)

__all__ = [
    "BEST_EFFORT_HUMANEVAL_DEFINITION_ID",
    "BEST_EFFORT_V2_DEFINITION",
    "DEFAULT_STRATEGIES",
    "DEFINITION_VERSION",
    "FIELD_MARKER_V2_DEFINITION",
    "STRICT_FIELD_MARKER_DEFINITION_ID",
    "SUPPORTED_DEFINITION_IDS",
    "SUPPORTED_DEFINITION_VERSIONS",
    "AlternativesStep",
    "BoundStep",
    "CandidateMapStep",
    "ExtractionStrategy",
    "PreprocessingDefinition",
    "REGISTRY",
    "STRATEGY_REGISTRY",
    "Step",
    "StepFailedError",
    "StepName",
    "StepOutput",
    "StepSettings",
    "StepSpec",
    "bind_definition",
    "preprocessing_definition_hash",
    "resolve_preprocessing_definition",
    "run_preprocessing",
]
