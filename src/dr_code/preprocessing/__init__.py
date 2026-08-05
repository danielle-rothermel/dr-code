"""Preprocessing: atomic, declared steps over typed artifacts.

A ``PreprocessingDefinition`` names an ordered sequence of operations. The
runner applies them and records their artifacts, absences, facts, and
producer identity in a ``dr_code.trace.Trace``.

This facade is the whole supported surface: the definition models, the
resolver, the bound runner and the functions that produce one, and the
one-shot runners. Steps, the step registry, and individual step mechanics
are internal — a definition names steps by ``StepName``, and how a step is
implemented is not part of the contract. Code inside the package imports
those internals by their own module paths.

Core functions (``text_transforms``, ``text_analysis``, ``code_analysis``)
provide step bodies. ``preprocessing.import_inference`` owns the import
repair, inference, and deduplication logic the cleaning steps use.
"""

from __future__ import annotations

from dr_code.preprocessing.definition import (
    PreprocessingDefinition,
    StepSpec,
)
from dr_code.preprocessing.definitions import (
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_ID,
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_VERSION,
    resolve_preprocessing_definition,
)
from dr_code.preprocessing.failures import PreprocessingFailureCode
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.runner import (
    BoundPreprocessingRunner,
    bind_external_preprocessing,
    bind_preprocessing,
    run_external_preprocessing,
    run_preprocessing,
)

__all__ = [
    "EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION",
    "EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_ID",
    "EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_VERSION",
    "BoundPreprocessingRunner",
    "PreprocessingDefinition",
    "PreprocessingFailureCode",
    "StepName",
    "StepSpec",
    "bind_external_preprocessing",
    "bind_preprocessing",
    "resolve_preprocessing_definition",
    "run_external_preprocessing",
    "run_preprocessing",
]
