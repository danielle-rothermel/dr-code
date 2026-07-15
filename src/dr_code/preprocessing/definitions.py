"""Named, frozen preprocessing definitions for the code-extraction pipeline.

Each ``PreprocessingDefinition`` here is the best final state of one
extraction path: the cleaning, candidate generation, and selection expressed
as atomic declared steps over typed artifacts. The definitions are pure
data — ordered ``StepSpec`` instances with explicit settings (no hidden
defaults); adding a step or changing a setting is a new definition whose
identity is the content hash.

These deliberately diverge from the old ``extract_code_with_profile`` path
where the old behaviour was wrong: string-aware smart-quote recovery, the
field-marker code-repr rejection, and the whitespace-only candidate drop.

``resolve_preprocessing_definition`` is an exact ``(definition_id, version)``
lookup that raises ``ValueError`` for any pair not in the table.

Step order and composition:

1. ``normalize_text`` — its six atomic constituents, one step each, so each
   is independently visible in the trace.
2. ``candidate_blocks`` — the ``extract_candidates`` strategy ladder.
3. Per-candidate cleaning — ``strip_code_fences``, ``textwrap.dedent``,
   string-aware smart-quote recovery, ``drop_if_name``,
   ``drop_after_last_return``, then ``infer_necessary_imports`` unbundled
   into ``repair_import_lines`` + ``infer_missing_imports`` +
   ``dedupe_imports``.
4. The ``candidate_selection`` checks — plain-literal, code-repr,
   compilable. Both best-effort and field-marker run all three.
5. ``select_first`` fixes the candidate set down to one code value.
"""

from __future__ import annotations

from typing import Final

from dr_code.humaneval.code_parsing import (
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
    FIELD_MARKER_NAME,
    PARSER_PROFILE_VERSION,
    STRICT_FIELD_MARKER_PARSER_PROFILE_ID,
)
from dr_code.preprocessing.definition import (
    PreprocessingDefinition,
    StepSpec,
)
from dr_code.preprocessing.names import StepName

#: Definition ids reuse ``code_parsing``'s so a definition is named by the
#: same coordinates as the extraction path it replaces.
BEST_EFFORT_HUMANEVAL_DEFINITION_ID: Final[str] = (
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID
)
STRICT_FIELD_MARKER_DEFINITION_ID: Final[str] = (
    STRICT_FIELD_MARKER_PARSER_PROFILE_ID
)
DEFINITION_VERSION: Final[str] = PARSER_PROFILE_VERSION
SUPPORTED_DEFINITION_VERSIONS: Final[frozenset[str]] = frozenset(
    {DEFINITION_VERSION}
)

#: The default extraction ladder: fenced blocks, then the markdown-wrapper
#: retry, then structural unescape, then the unescape + markdown-wrapper
#: retry.
_DEFAULT_EXTRACT_ALTERNATIVES: Final[tuple[str, ...]] = (
    "fenced_blocks",
    "markdown_wrapper",
    "escaped_python",
    "escaped_markdown_wrapper",
)


def _spec(
    instance_name: str,
    step: StepName,
    **settings: object,
) -> StepSpec:
    """Build a ``StepSpec``; settings pass through as JSON values."""
    return StepSpec(
        instance_name=instance_name,
        step=step,
        settings=dict(settings),
    )


#: ``normalize_text``'s constituents, one step per operation. Order matters:
#: line endings, NFKC unicode, tab expansion, trailing-whitespace strip,
#: blank-run collapse, then outer-blank trim.
_TEXT_NORMALIZATION: Final[tuple[StepSpec, ...]] = (
    _spec("normalize_line_endings", StepName.NORMALIZE_LINE_ENDINGS),
    _spec("normalize_unicode", StepName.NORMALIZE_UNICODE),
    _spec("expand_tabs", StepName.EXPAND_TABS),
    _spec("strip_trailing_whitespace", StepName.STRIP_TRAILING_WHITESPACE),
    _spec("collapse_blank_runs", StepName.COLLAPSE_BLANK_RUNS),
    _spec("trim_outer_blanks", StepName.TRIM_OUTER_BLANKS),
)

#: Per-candidate cleaning: strip fences, dedent, string-aware smart-quote
#: recovery (so smart-delimited code compiles while quote *contents* survive),
#: split on the ``if __name__`` guard, drop trailing prose after the last
#: return, then repair / infer / dedupe imports.
_CANDIDATE_CLEANING: Final[tuple[StepSpec, ...]] = (
    _spec("strip_fences", StepName.STRIP_FENCES),
    _spec("dedent", StepName.DEDENT_CANDIDATES),
    _spec("normalize_smart_quotes", StepName.NORMALIZE_SMART_QUOTES),
    _spec("split_on_name_guard", StepName.SPLIT_ON_NAME_GUARD),
    _spec("drop_after_last_return", StepName.DROP_AFTER_LAST_RETURN),
    _spec("repair_import_lines", StepName.REPAIR_IMPORT_LINES),
    _spec("infer_missing_imports", StepName.INFER_MISSING_IMPORTS),
    _spec("dedupe_imports", StepName.DEDUPE_IMPORTS),
)


#: best-effort: full normalization, the default extraction ladder,
#: per-candidate cleaning, then all three selection filters (plain-literal,
#: code-repr, compilable).
BEST_EFFORT_V2_DEFINITION: Final[PreprocessingDefinition] = (
    PreprocessingDefinition(
        definition_id=BEST_EFFORT_HUMANEVAL_DEFINITION_ID,
        version=DEFINITION_VERSION,
        steps=(
            *_TEXT_NORMALIZATION,
            _spec(
                "extract_candidates",
                StepName.EXTRACT_CANDIDATES,
                alternatives=list(_DEFAULT_EXTRACT_ALTERNATIVES),
            ),
            *_CANDIDATE_CLEANING,
            _spec("filter_plain_literal", StepName.FILTER_PLAIN_LITERAL),
            _spec("filter_code_repr", StepName.FILTER_CODE_REPR),
            _spec("filter_compilable", StepName.FILTER_COMPILABLE),
            _spec("select_first", StepName.SELECT_FIRST),
        ),
    )
)


#: strict field-marker: extract the ``[[ ## code ## ]]`` value, then the same
#: three selection filters as best-effort. The code-repr filter is included
#: so a ``code = "..."`` marker payload is rejected (symmetrical with
#: best-effort).
FIELD_MARKER_V2_DEFINITION: Final[PreprocessingDefinition] = (
    PreprocessingDefinition(
        definition_id=STRICT_FIELD_MARKER_DEFINITION_ID,
        version=DEFINITION_VERSION,
        steps=(
            _spec(
                "field_marker_extract",
                StepName.FIELD_MARKER_EXTRACT,
                field_name=FIELD_MARKER_NAME,
            ),
            _spec("filter_plain_literal", StepName.FILTER_PLAIN_LITERAL),
            _spec("filter_code_repr", StepName.FILTER_CODE_REPR),
            _spec("filter_compilable", StepName.FILTER_COMPILABLE),
            _spec("select_first", StepName.SELECT_FIRST),
        ),
    )
)


#: Lookup by (definition_id, version) — the resolver's single source of
#: truth. Only the registered pairs resolve.
_DEFINITIONS: Final[dict[tuple[str, str], PreprocessingDefinition]] = {
    (
        BEST_EFFORT_HUMANEVAL_DEFINITION_ID,
        DEFINITION_VERSION,
    ): BEST_EFFORT_V2_DEFINITION,
    (
        STRICT_FIELD_MARKER_DEFINITION_ID,
        DEFINITION_VERSION,
    ): FIELD_MARKER_V2_DEFINITION,
}

#: Supported definition ids.
SUPPORTED_DEFINITION_IDS: Final[frozenset[str]] = frozenset(
    {BEST_EFFORT_HUMANEVAL_DEFINITION_ID, STRICT_FIELD_MARKER_DEFINITION_ID}
)


def resolve_preprocessing_definition(
    *,
    definition_id: str,
    version: str,
) -> PreprocessingDefinition:
    """Return the named ``PreprocessingDefinition`` for an id x version.

    Keyword-only coordinates; an exact ``(definition_id, version)`` lookup
    that raises ``ValueError`` for any pair not in the table. The returned
    definition is frozen and shared — callers must never mutate it.
    """
    definition = _DEFINITIONS.get((definition_id, version))
    if definition is None:
        raise ValueError(
            "unsupported preprocessing definition: "
            f"({definition_id!r}, {version!r})"
        )
    return definition


__all__ = [
    "BEST_EFFORT_HUMANEVAL_DEFINITION_ID",
    "BEST_EFFORT_V2_DEFINITION",
    "DEFINITION_VERSION",
    "FIELD_MARKER_V2_DEFINITION",
    "STRICT_FIELD_MARKER_DEFINITION_ID",
    "SUPPORTED_DEFINITION_IDS",
    "SUPPORTED_DEFINITION_VERSIONS",
    "resolve_preprocessing_definition",
]
