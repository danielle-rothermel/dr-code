"""Preprocessing: atomic, declared steps over typed artifacts.

Rebuilds the code-extraction pipeline as one-operation-per-step so a
``PreprocessingDefinition`` chooses exactly which operations run. The
runner produces a ``Trace`` from the ``dr_code.trace`` boundary-contract
package.

Core functions (``text_transforms``, ``text_analysis``, ``code_analysis``)
are reused as-is as step bodies; import inference is owned by
``preprocessing.import_inference``.
"""

from __future__ import annotations

from dr_code.preprocessing.definition import (
    PreprocessingDefinition,
    StepSpec,
    preprocessing_definition_hash,
)
from dr_code.preprocessing.definitions import (
    DEFINITION_VERSION,
    HUMANEVAL_FUNCTION_CANDIDATES_DEFINITION_ID,
    HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
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

__all__ = [
    "DEFINITION_VERSION",
    "HUMANEVAL_FUNCTION_CANDIDATES_DEFINITION_ID",
    "HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION",
    "SUPPORTED_DEFINITION_IDS",
    "SUPPORTED_DEFINITION_VERSIONS",
    "AlternativesStep",
    "BoundPreprocessingRunner",
    "BoundStep",
    "CandidateMapStep",
    "PreprocessingFailureCode",
    "PreprocessingDefinition",
    "REGISTRY",
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
