"""Preprocessing: atomic, declared steps over typed artifacts.

Rebuilds the code-extraction pipeline as one-operation-per-step so a
``PreprocessingDefinition`` chooses exactly which operations run. The
runner produces a ``Trace`` from the ``dr_code.trace`` boundary-contract
package.

Core functions (``text_transforms``, ``text_analysis``, ``code_analysis``)
are reused as-is as step bodies; import inference lives here in
``preprocessing.import_inference`` and the old pipeline delegates to it.
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
from dr_code.preprocessing.failures import PreprocessingFailureCode
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.registry import REGISTRY
from dr_code.preprocessing.runner import (
    BoundPreprocessingRunner,
    BoundStep,
    bind_definition,
    bind_preprocessing,
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
    "BoundPreprocessingRunner",
    "BoundStep",
    "CandidateMapStep",
    "ExtractionStrategy",
    "PreprocessingFailureCode",
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
    "bind_preprocessing",
    "preprocessing_definition_hash",
    "resolve_preprocessing_definition",
    "run_preprocessing",
]
