"""Preprocessing: atomic, declared steps over typed artifacts.

Each ``PreprocessingDefinition`` chooses an ordered sequence of operations.
The runner applies those operations and records their artifacts, absences,
facts, and producer identity in a ``dr_code.trace.Trace``.

Core functions (``text_transforms``, ``text_analysis``, ``code_analysis``)
provide step bodies; ``preprocessing.extraction`` owns candidate extraction,
``preprocessing.identification`` owns parse-once candidate identification,
and ``preprocessing.import_inference`` owns import inference. This package is
the single parsing implementation: HumanEval parser profiles are selection
policy over its output, never a second extractor.
"""

from __future__ import annotations

from dr_code.preprocessing.definition import (
    PreprocessingDefinition,
    StepSpec,
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
    bind_external_preprocessing,
    bind_preprocessing,
    run_external_preprocessing,
    run_preprocessing,
)
from dr_code.preprocessing.steps.base import (
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
    "BoundPreprocessingRunner",
    "BoundStep",
    "CandidateMapStep",
    "PreprocessingDefinition",
    "PreprocessingFailureCode",
    "REGISTRY",
    "Step",
    "StepFailedError",
    "StepName",
    "StepOutput",
    "StepSettings",
    "StepSpec",
    "bind_definition",
    "bind_external_preprocessing",
    "bind_preprocessing",
    "resolve_preprocessing_definition",
    "run_external_preprocessing",
    "run_preprocessing",
]
