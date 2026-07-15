"""Preprocessing: atomic, declared steps over typed artifacts.

Rebuilds the code-extraction pipeline as one-operation-per-step so a
``PreprocessingDefinition`` chooses exactly which operations run. The
runner produces a ``Trace`` from the ``dr_code.trace`` boundary-contract
package.

Core functions (``text_transforms``, ``text_analysis``, ``code_analysis``,
``humaneval/import_inference``) are reused as-is as step bodies.
"""

from __future__ import annotations

from dr_code.preprocessing.definition import (
    PreprocessingDefinition,
    StepSpec,
    preprocessing_definition_hash,
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
    "AlternativesStep",
    "BoundStep",
    "CandidateMapStep",
    "DEFAULT_STRATEGIES",
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
    "run_preprocessing",
]
