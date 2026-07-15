"""Named, frozen preprocessing definitions that reproduce the old profiles.

Each ``PreprocessingDefinition`` here replaces one ``CodeParserProfile`` /
``extract_*`` path in ``dr_code.humaneval.code_parsing``: the same cleaning,
candidate generation, and selection, expressed as atomic declared steps over
typed artifacts. The definitions are pure data — ordered ``StepSpec``
instances with explicit settings (no hidden defaults); adding a step or
changing a setting is a new definition whose identity is the content hash.

``resolve_preprocessing_definition`` mirrors
``code_parsing.resolve_parser_profile``: keyword-only coordinates
(``definition_id`` x ``version``), the same supported ``(id, version)``
shape, and a ``ValueError`` on any unknown id or version — so callers swap
one resolver for the other.

Step order and composition mirror ``code_extraction.apply_cleaning_with_trace``
+ ``code_parsing.candidate_selection`` exactly (verified by the output-parity
suite against ``extract_code_with_profile``):

1. ``normalize_text`` — its six atomic constituents, one step each, so each
   is independently visible in the trace. Deliberately **no** smart-quote
   step: ``text_transforms.normalize_text`` does not normalize smart quotes,
   so adding one here would diverge from the old pipeline (it would recover
   smart-quoted string literals the old pipeline rejects).
2. ``candidate_blocks`` — the ``extract_candidates`` strategy ladder. The v1
   vs. v2 difference lives entirely here: v1 (legacy, ``unescape_fallback=
   False``) omits the escaped-Python rung; v2 keeps the full default ladder.
3. Per-candidate cleaning — ``strip_code_fences``, ``textwrap.dedent`` (the
   old pipeline runs ``apply_dedent=True``), ``drop_if_name``,
   ``drop_after_last_return``, then ``infer_necessary_imports`` unbundled into
   ``repair_import_lines`` + ``infer_missing_imports`` + ``dedupe_imports``.
4. The ``candidate_selection`` checks. Best-effort runs all three
   (plain-literal, code-repr, compilable — ``include_code_repr_check=True``);
   the field-marker path omits the code-repr filter
   (``include_code_repr_check=False``) — the one asymmetry between them.
5. ``select_first`` fixes the candidate set down to one code value.
"""

from __future__ import annotations

from typing import Final

from dr_code.humaneval.code_parsing import (
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
    FIELD_MARKER_NAME,
    LEGACY_PARSER_PROFILE_VERSION,
    PARSER_PROFILE_VERSION,
    STRICT_FIELD_MARKER_PARSER_PROFILE_ID,
    SUPPORTED_PARSER_PROFILE_VERSIONS,
)
from dr_code.preprocessing.definition import (
    PreprocessingDefinition,
    StepSpec,
)
from dr_code.preprocessing.names import StepName

#: Definition ids and versions reuse ``code_parsing``'s so a definition is
#: named by the same coordinates as the parser profile it replaces.
BEST_EFFORT_HUMANEVAL_DEFINITION_ID: Final[str] = (
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID
)
STRICT_FIELD_MARKER_DEFINITION_ID: Final[str] = (
    STRICT_FIELD_MARKER_PARSER_PROFILE_ID
)
LEGACY_DEFINITION_VERSION: Final[str] = LEGACY_PARSER_PROFILE_VERSION
DEFINITION_VERSION: Final[str] = PARSER_PROFILE_VERSION
SUPPORTED_DEFINITION_VERSIONS: Final[frozenset[str]] = frozenset(
    SUPPORTED_PARSER_PROFILE_VERSIONS
)

#: best-effort v1 omits the escaped-Python rung: the legacy profile ran with
#: ``unescape_fallback=False``, so it never recovered structurally escaped
#: payloads. Spelled out (not ``DEFAULT_STRATEGIES``) so v1's identity is
#: explicit and stable.
_LEGACY_EXTRACT_ALTERNATIVES: Final[tuple[str, ...]] = (
    "fenced_blocks",
    "markdown_wrapper",
)
#: best-effort v2 — the current default ladder, which recovers structurally
#: escaped payloads via the ``escaped_python`` rung.
_DEFAULT_EXTRACT_ALTERNATIVES: Final[tuple[str, ...]] = (
    "fenced_blocks",
    "markdown_wrapper",
    "escaped_python",
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

#: Per-candidate cleaning, mirroring the old candidate pass:
#: strip fences, dedent (``apply_dedent=True``), split on the ``if __name__``
#: guard, drop trailing prose after the last return, then repair / infer /
#: dedupe imports (``infer_necessary_imports`` unbundled).
_CANDIDATE_CLEANING: Final[tuple[StepSpec, ...]] = (
    _spec("strip_fences", StepName.STRIP_FENCES),
    _spec("dedent", StepName.DEDENT_CANDIDATES),
    _spec("split_on_name_guard", StepName.SPLIT_ON_NAME_GUARD),
    _spec("drop_after_last_return", StepName.DROP_AFTER_LAST_RETURN),
    _spec("repair_import_lines", StepName.REPAIR_IMPORT_LINES),
    _spec("infer_missing_imports", StepName.INFER_MISSING_IMPORTS),
    _spec("dedupe_imports", StepName.DEDUPE_IMPORTS),
)


def _best_effort_definition(
    *,
    version: str,
    alternatives: tuple[str, ...],
) -> PreprocessingDefinition:
    """Compose the best-effort pipeline.

    Factored so v1 and v2 share one step chain; the only axis that varies
    is the ``extract_candidates`` strategy ladder (v1 drops escaped-Python).
    The selection stage runs all three ``candidate_selection`` checks —
    ``include_code_repr_check=True`` for best-effort.
    """
    return PreprocessingDefinition(
        definition_id=BEST_EFFORT_HUMANEVAL_DEFINITION_ID,
        version=version,
        steps=(
            *_TEXT_NORMALIZATION,
            _spec(
                "extract_candidates",
                StepName.EXTRACT_CANDIDATES,
                alternatives=list(alternatives),
            ),
            *_CANDIDATE_CLEANING,
            _spec("filter_plain_literal", StepName.FILTER_PLAIN_LITERAL),
            _spec("filter_code_repr", StepName.FILTER_CODE_REPR),
            _spec("filter_compilable", StepName.FILTER_COMPILABLE),
            _spec("select_first", StepName.SELECT_FIRST),
        ),
    )


#: best-effort v2 — the current default (with escaped-Python recovery).
BEST_EFFORT_V2_DEFINITION: Final[PreprocessingDefinition] = (
    _best_effort_definition(
        version=DEFINITION_VERSION,
        alternatives=_DEFAULT_EXTRACT_ALTERNATIVES,
    )
)

#: best-effort v1 — the legacy default (no escaped-Python recovery).
BEST_EFFORT_V1_DEFINITION: Final[PreprocessingDefinition] = (
    _best_effort_definition(
        version=LEGACY_DEFINITION_VERSION,
        alternatives=_LEGACY_EXTRACT_ALTERNATIVES,
    )
)

def _field_marker_definition(*, version: str) -> PreprocessingDefinition:
    """Strict field-marker pipeline at a given version.

    Extract the ``[[ ## code ## ]]`` value, then the compile +
    plain-literal checks only (no code-repr filter, matching
    ``extract_strict_field_marker_code``'s ``include_code_repr_check=
    False``). The field-marker path never branches on version — a v1 and a
    v2 field-marker profile extract identically — so both versions share
    one step chain and differ only in stamped version / identity.
    """
    return PreprocessingDefinition(
        definition_id=STRICT_FIELD_MARKER_DEFINITION_ID,
        version=version,
        steps=(
            _spec(
                "field_marker_extract",
                StepName.FIELD_MARKER_EXTRACT,
                field_name=FIELD_MARKER_NAME,
            ),
            _spec("filter_plain_literal", StepName.FILTER_PLAIN_LITERAL),
            _spec("filter_compilable", StepName.FILTER_COMPILABLE),
            _spec("select_first", StepName.SELECT_FIRST),
        ),
    )


#: Strict field-marker, v2 (current) and v1 (legacy). Both extract
#: identically; the v1 entry exists so the resolver mirrors the old
#: ``resolve_parser_profile``, which accepts every ``(id, version)`` pair
#: whose id and version are individually supported — including the
#: off-menu ``(field-marker, v1)`` combination.
FIELD_MARKER_V2_DEFINITION: Final[PreprocessingDefinition] = (
    _field_marker_definition(version=DEFINITION_VERSION)
)
FIELD_MARKER_V1_DEFINITION: Final[PreprocessingDefinition] = (
    _field_marker_definition(version=LEGACY_DEFINITION_VERSION)
)


#: Lookup by (definition_id, version) — the resolver's single source of
#: truth. Every ``(supported id) x (supported version)`` pair resolves,
#: mirroring ``code_parsing.resolve_parser_profile`` (which constructs a
#: profile for any individually-valid id and version, never checking the
#: pair). ``(field-marker, v1)`` is therefore a real, resolvable entry.
_DEFINITIONS: Final[dict[tuple[str, str], PreprocessingDefinition]] = {
    (
        BEST_EFFORT_HUMANEVAL_DEFINITION_ID,
        DEFINITION_VERSION,
    ): BEST_EFFORT_V2_DEFINITION,
    (
        BEST_EFFORT_HUMANEVAL_DEFINITION_ID,
        LEGACY_DEFINITION_VERSION,
    ): BEST_EFFORT_V1_DEFINITION,
    (
        STRICT_FIELD_MARKER_DEFINITION_ID,
        DEFINITION_VERSION,
    ): FIELD_MARKER_V2_DEFINITION,
    (
        STRICT_FIELD_MARKER_DEFINITION_ID,
        LEGACY_DEFINITION_VERSION,
    ): FIELD_MARKER_V1_DEFINITION,
}

#: Supported definition ids — mirrors the id set accepted by
#: ``resolve_parser_profile`` before it constructs a profile.
SUPPORTED_DEFINITION_IDS: Final[frozenset[str]] = frozenset(
    {BEST_EFFORT_HUMANEVAL_DEFINITION_ID, STRICT_FIELD_MARKER_DEFINITION_ID}
)


def resolve_preprocessing_definition(
    *,
    definition_id: str,
    version: str,
) -> PreprocessingDefinition:
    """Return the named ``PreprocessingDefinition`` for an id x version.

    Mirrors ``code_parsing.resolve_parser_profile``: keyword-only
    coordinates, a ``ValueError`` for an unsupported version, and a
    ``ValueError`` for an unsupported id. Crucially, it matches the old
    resolver's *permissive* pair handling — the old resolver validates the
    id and version independently and then constructs a profile from both,
    so the off-menu ``(field-marker, v1)`` pair resolves (to a v1
    field-marker definition, which extracts identically to v2) rather than
    raising. Every ``(supported id) x (supported version)`` pair is
    therefore registered in ``_DEFINITIONS``.

    The returned definition is frozen and shared — callers must never
    mutate it.
    """
    if version not in SUPPORTED_DEFINITION_VERSIONS:
        raise ValueError(
            f"unsupported preprocessing definition version: {version}"
        )
    if definition_id not in SUPPORTED_DEFINITION_IDS:
        raise ValueError(
            f"unsupported preprocessing definition id: {definition_id}"
        )
    return _DEFINITIONS[(definition_id, version)]


__all__ = [
    "BEST_EFFORT_HUMANEVAL_DEFINITION_ID",
    "BEST_EFFORT_V1_DEFINITION",
    "BEST_EFFORT_V2_DEFINITION",
    "DEFINITION_VERSION",
    "FIELD_MARKER_V1_DEFINITION",
    "FIELD_MARKER_V2_DEFINITION",
    "LEGACY_DEFINITION_VERSION",
    "STRICT_FIELD_MARKER_DEFINITION_ID",
    "SUPPORTED_DEFINITION_IDS",
    "SUPPORTED_DEFINITION_VERSIONS",
    "resolve_preprocessing_definition",
]
