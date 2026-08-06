from __future__ import annotations

from types import MappingProxyType
from typing import Final

from dr_code.preprocessing.definition import (
    PreprocessingDefinition,
    StepSpec,
)
from dr_code.preprocessing.names import StepName

EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_ID: Final[str] = (
    "exhaustive-function-candidates"
)
EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_VERSION: Final[str] = "0"


def _spec(
    instance_name: str,
    step: StepName,
    **settings: object,
) -> StepSpec:
    from dr_code.preprocessing.registry import REGISTRY

    return StepSpec(
        instance_name=instance_name,
        step=step,
        settings=REGISTRY[step.value].Settings.model_validate(settings),
    )


_TEXT_NORMALIZATION: Final[tuple[StepSpec, ...]] = (
    _spec("normalize_line_endings", StepName.NORMALIZE_LINE_ENDINGS),
    _spec("normalize_unicode", StepName.NORMALIZE_UNICODE),
    _spec("expand_tabs", StepName.EXPAND_TABS),
    _spec("strip_trailing_whitespace", StepName.STRIP_TRAILING_WHITESPACE),
    _spec("collapse_blank_runs", StepName.COLLAPSE_BLANK_RUNS),
    _spec("trim_outer_blanks", StepName.TRIM_OUTER_BLANKS),
)

# All source mutation precedes inspection so facts describe the exact source.
_CANDIDATE_SHAPING: Final[tuple[StepSpec, ...]] = (
    _spec("strip_fences", StepName.STRIP_FENCES),
    _spec("dedent", StepName.DEDENT_CANDIDATES),
    _spec("normalize_smart_quotes", StepName.NORMALIZE_SMART_QUOTES),
    _spec("split_on_name_guard", StepName.SPLIT_ON_NAME_GUARD),
)

# Inference follows salvage because it no-ops on unparseable source.
_CANDIDATE_IMPORTS: Final[tuple[StepSpec, ...]] = (
    _spec("repair_import_lines", StepName.REPAIR_IMPORT_LINES),
    _spec("infer_missing_imports", StepName.INFER_MISSING_IMPORTS),
    _spec("dedupe_imports", StepName.DEDUPE_IMPORTS),
)

_CANDIDATE_FILTERS: Final[tuple[StepSpec, ...]] = (
    _spec("filter_plain_literal", StepName.FILTER_PLAIN_LITERAL),
    _spec("filter_code_repr", StepName.FILTER_CODE_REPR),
    _spec("filter_compilable", StepName.FILTER_COMPILABLE),
    _spec("filter_top_level_functions", StepName.FILTER_TOP_LEVEL_FUNCTIONS),
)


EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION: Final[PreprocessingDefinition] = (
    PreprocessingDefinition(
        definition_id=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_ID,
        version=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_VERSION,
        steps=(
            *_TEXT_NORMALIZATION,
            _spec("reject_blank_input", StepName.REJECT_BLANK_INPUT),
            _spec(
                "extract_all_representations",
                StepName.EXTRACT_ALL_REPRESENTATIONS,
            ),
            *_CANDIDATE_SHAPING,
            _spec("add_last_return_salvage", StepName.ADD_LAST_RETURN_SALVAGE),
            *_CANDIDATE_IMPORTS,
            _spec("drop_blank_candidates", StepName.DROP_BLANK_CANDIDATES),
            _spec("dedupe_candidates", StepName.DEDUPE_CANDIDATES),
            _spec("inspect_candidates", StepName.INSPECT_CANDIDATES),
            *_CANDIDATE_FILTERS,
            _spec(
                "materialize_candidate_set",
                StepName.MATERIALIZE_CANDIDATE_SET,
            ),
        ),
    )
)


_DEFINITIONS: Final = MappingProxyType(
    {
        (
            EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_ID,
            EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_VERSION,
        ): EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
    }
)


def resolve_preprocessing_definition(
    *,
    definition_id: str,
    version: str,
) -> PreprocessingDefinition:
    definition = _DEFINITIONS.get((definition_id, version))
    if definition is None:
        raise ValueError(
            "unsupported preprocessing definition: "
            f"({definition_id!r}, {version!r})"
        )
    return definition


__all__ = [
    "EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION",
    "EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_ID",
    "EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_VERSION",
    "resolve_preprocessing_definition",
]
