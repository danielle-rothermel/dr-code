"""Named preprocessing definition for exhaustive HumanEval candidates.

The definition interprets decoder text without imposing an expected function
name. It returns every distinct, compilable candidate containing at least one
top-level function; applications may add name-aware policy or rely on tests.
"""

from __future__ import annotations

from typing import Final

from pydantic import JsonValue

from dr_code.preprocessing.definition import PreprocessingDefinition, StepSpec
from dr_code.preprocessing.names import StepName

HUMANEVAL_FUNCTION_CANDIDATES_DEFINITION_ID: Final = (
    "humaneval-function-candidates"
)
DEFINITION_VERSION: Final = "v1"
SUPPORTED_DEFINITION_VERSIONS: Final[frozenset[str]] = frozenset(
    {DEFINITION_VERSION}
)


def _spec(
    instance_name: str,
    step: StepName,
    **settings: JsonValue,
) -> StepSpec:
    return StepSpec(
        instance_name=instance_name,
        step=step,
        settings=dict(settings),
    )


_TEXT_NORMALIZATION: Final[tuple[StepSpec, ...]] = (
    _spec("normalize_line_endings", StepName.NORMALIZE_LINE_ENDINGS),
    _spec("normalize_unicode", StepName.NORMALIZE_UNICODE),
    _spec("expand_tabs", StepName.EXPAND_TABS),
    _spec("strip_trailing_whitespace", StepName.STRIP_TRAILING_WHITESPACE),
    _spec("collapse_blank_runs", StepName.COLLAPSE_BLANK_RUNS),
    _spec("trim_outer_blanks", StepName.TRIM_OUTER_BLANKS),
    _spec("require_nonblank_text", StepName.REQUIRE_NONBLANK_TEXT),
)

_CANDIDATE_CLEANING: Final[tuple[StepSpec, ...]] = (
    _spec("strip_fences", StepName.STRIP_FENCES),
    _spec("dedent", StepName.DEDENT_CANDIDATES),
    _spec("normalize_smart_quotes", StepName.NORMALIZE_SMART_QUOTES),
    _spec("split_on_name_guard", StepName.SPLIT_ON_NAME_GUARD),
    _spec(
        "expand_last_return_salvage",
        StepName.EXPAND_LAST_RETURN_SALVAGE,
    ),
    _spec("repair_import_lines", StepName.REPAIR_IMPORT_LINES),
    _spec("dedupe_imports", StepName.DEDUPE_IMPORTS),
)

HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION: Final = PreprocessingDefinition(
    definition_id=HUMANEVAL_FUNCTION_CANDIDATES_DEFINITION_ID,
    version=DEFINITION_VERSION,
    steps=(
        *_TEXT_NORMALIZATION,
        _spec("extract_candidates", StepName.EXTRACT_CANDIDATES),
        *_CANDIDATE_CLEANING,
        _spec(
            "filter_nonblank_candidates",
            StepName.FILTER_NONBLANK_CANDIDATES,
        ),
        _spec("identify_candidates", StepName.IDENTIFY_CANDIDATES),
        _spec("filter_plain_literal", StepName.FILTER_PLAIN_LITERAL),
        _spec("filter_code_repr", StepName.FILTER_CODE_REPR),
        _spec("filter_compilable", StepName.FILTER_COMPILABLE),
        _spec(
            "filter_has_top_level_function",
            StepName.FILTER_HAS_TOP_LEVEL_FUNCTION,
        ),
        _spec("materialize_candidates", StepName.MATERIALIZE_CANDIDATES),
        _spec("return_all", StepName.RETURN_ALL),
    ),
)

_DEFINITIONS: Final = {
    (
        HUMANEVAL_FUNCTION_CANDIDATES_DEFINITION_ID,
        DEFINITION_VERSION,
    ): HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
}

SUPPORTED_DEFINITION_IDS: Final[frozenset[str]] = frozenset(
    {HUMANEVAL_FUNCTION_CANDIDATES_DEFINITION_ID}
)


def resolve_preprocessing_definition(
    *,
    definition_id: str,
    version: str,
) -> PreprocessingDefinition:
    """Resolve an exact public definition coordinate without aliases."""
    definition = _DEFINITIONS.get((definition_id, version))
    if definition is None:
        raise ValueError(
            "unsupported preprocessing definition: "
            f"({definition_id!r}, {version!r})"
        )
    return definition


__all__ = [
    "DEFINITION_VERSION",
    "HUMANEVAL_FUNCTION_CANDIDATES_DEFINITION_ID",
    "HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION",
    "SUPPORTED_DEFINITION_IDS",
    "SUPPORTED_DEFINITION_VERSIONS",
    "resolve_preprocessing_definition",
]
