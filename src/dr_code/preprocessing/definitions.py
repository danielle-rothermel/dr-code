"""The registered preprocessing definitions and their exact resolver.

A ``PreprocessingDefinition`` is pure data: ordered ``StepSpec`` instances
with explicit settings and no hidden defaults. Component identity is the
explicit definition id plus manual version; the ordered steps and their
settings stay directly inspectable.

``resolve_preprocessing_definition`` is an exact ``(definition_id,
version)`` lookup that raises ``ValueError`` for any pair not in the table.

Definition ids are preprocessing's own. A definition describes how text is
turned into candidates, which is independent of the dataset a consumer
scores against, so a definition never borrows a consumer's coordinate.
"""

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
    """Build a ``StepSpec`` with the registered typed settings model."""
    from dr_code.preprocessing.registry import REGISTRY

    return StepSpec(
        instance_name=instance_name,
        step=step,
        settings=REGISTRY[step.value].Settings.model_validate(settings),
    )


#: ``normalize_text``'s constituents, one step per operation, so each is
#: independently visible in the trace. Order matters: line endings, NFKC
#: unicode, tab expansion, trailing-whitespace strip, blank-run collapse,
#: then outer-blank trim.
_TEXT_NORMALIZATION: Final[tuple[StepSpec, ...]] = (
    _spec("normalize_line_endings", StepName.NORMALIZE_LINE_ENDINGS),
    _spec("normalize_unicode", StepName.NORMALIZE_UNICODE),
    _spec("expand_tabs", StepName.EXPAND_TABS),
    _spec("strip_trailing_whitespace", StepName.STRIP_TRAILING_WHITESPACE),
    _spec("collapse_blank_runs", StepName.COLLAPSE_BLANK_RUNS),
    _spec("trim_outer_blanks", StepName.TRIM_OUTER_BLANKS),
)

#: Candidate-local cleaning, every step extending the lineage of the
#: candidate it rewrites: strip fences, dedent, string-aware smart-quote
#: recovery (so smart-delimited code compiles while quote *contents*
#: survive), split on the ``if __name__`` guard, then repair / infer /
#: dedupe imports.
#:
#: Import inference belongs here, with the other cleaning, and not after
#: inspection: it rewrites a candidate's source, and an inspection must
#: always describe the exact source it accompanies. Running it after
#: inspection would leave every stored inspection describing text the
#: candidate no longer holds, and the filters reading those inspections
#: would be answering questions about a source that no longer exists.
#: Placing every source-mutating step before ``inspect_candidates`` is what
#: makes one parse per candidate both correct and sufficient.
_CANDIDATE_CLEANING: Final[tuple[StepSpec, ...]] = (
    _spec("strip_fences", StepName.STRIP_FENCES),
    _spec("dedent", StepName.DEDENT_CANDIDATES),
    _spec("normalize_smart_quotes", StepName.NORMALIZE_SMART_QUOTES),
    _spec("split_on_name_guard", StepName.SPLIT_ON_NAME_GUARD),
    _spec("repair_import_lines", StepName.REPAIR_IMPORT_LINES),
    _spec("infer_missing_imports", StepName.INFER_MISSING_IMPORTS),
    _spec("dedupe_imports", StepName.DEDUPE_IMPORTS),
)

#: The structural filters, all reading candidates' stored inspections or
#: sources — never reparsing what inspection already established.
_CANDIDATE_FILTERS: Final[tuple[StepSpec, ...]] = (
    _spec("filter_plain_literal", StepName.FILTER_PLAIN_LITERAL),
    _spec("filter_code_repr", StepName.FILTER_CODE_REPR),
    _spec("filter_compilable", StepName.FILTER_COMPILABLE),
    _spec("filter_top_level_functions", StepName.FILTER_TOP_LEVEL_FUNCTIONS),
)


#: The one registered definition: read every representation, clean every
#: candidate, then narrow structurally and return everything that survived.
#:
#: Step order, and why:
#:
#: 1. Normalize the text (six atomic steps).
#: 2. Reject blank input, so "there was nothing here" is its own failure.
#: 3. Extract candidates additively from every supported representation —
#:    no representation shadows another, and nothing is chosen yet.
#: 4. Clean each candidate, extending its lineage. Import inference is part
#:    of cleaning, for the ordering reason documented above.
#: 5. Add last-return truncations as *additional* candidates, so the
#:    salvage never destroys the candidate it was salvaged from.
#: 6. Drop blank candidates that cleaning emptied.
#: 7. Merge exact-duplicate sources, concatenating their lineages.
#: 8. Inspect each remaining source exactly once — the last word on
#:    structure, and the last time any source is parsed.
#: 9. Filter on the stored inspections and sources.
#: 10. Materialize everything that survived, in order. ``candidate_ordinal``
#:     indexes this final set.
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
            *_CANDIDATE_CLEANING,
            _spec("add_last_return_salvage", StepName.ADD_LAST_RETURN_SALVAGE),
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


#: Lookup by (definition_id, version) — the resolver's single source of
#: truth. Only the registered pairs resolve.
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
    "EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION",
    "EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_ID",
    "EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_VERSION",
    "resolve_preprocessing_definition",
]
