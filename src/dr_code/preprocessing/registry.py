"""Step registry — name-keyed lookup, mirroring ``synthetic.corruptions``.

Add a step by:

1. Implement it as a subclass of ``Step`` in its own file.
2. Import and register it here in ``REGISTRY``.
3. Add a paired entry in ``StepName`` (kept in sync by the registry test).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from dr_code.preprocessing.steps.base import Step
from dr_code.preprocessing.steps.collapse_blank_runs import (
    CollapseBlankRuns,
)
from dr_code.preprocessing.steps.dedupe_imports import DedupeImports
from dr_code.preprocessing.steps.dedent import Dedent
from dr_code.preprocessing.steps.expand_tabs import ExpandTabs
from dr_code.preprocessing.steps.expand_last_return_salvage import (
    ExpandLastReturnSalvage,
)
from dr_code.preprocessing.steps.extract_candidates import (
    ExtractCandidates,
)
from dr_code.preprocessing.steps.filter_code_repr import FilterCodeRepr
from dr_code.preprocessing.steps.filter_compilable import (
    FilterCompilable,
)
from dr_code.preprocessing.steps.filter_plain_literal import (
    FilterPlainLiteral,
)
from dr_code.preprocessing.steps.filter_has_top_level_function import (
    FilterHasTopLevelFunction,
)
from dr_code.preprocessing.steps.filter_nonblank_candidates import (
    FilterNonblankCandidates,
)
from dr_code.preprocessing.steps.identify_candidates import IdentifyCandidates
from dr_code.preprocessing.steps.materialize_candidates import (
    MaterializeCandidates,
)
from dr_code.preprocessing.steps.normalize_line_endings import (
    NormalizeLineEndings,
)
from dr_code.preprocessing.steps.normalize_smart_quotes import (
    NormalizeSmartQuotes,
)
from dr_code.preprocessing.steps.normalize_unicode import (
    NormalizeUnicode,
)
from dr_code.preprocessing.steps.repair_import_lines import (
    RepairImportLines,
)
from dr_code.preprocessing.steps.require_nonblank_text import (
    RequireNonblankText,
)
from dr_code.preprocessing.steps.return_all import ReturnAll
from dr_code.preprocessing.steps.split_on_name_guard import (
    SplitOnNameGuard,
)
from dr_code.preprocessing.steps.strip_fences import StripFences
from dr_code.preprocessing.steps.strip_trailing_whitespace import (
    StripTrailingWhitespace,
)
from dr_code.preprocessing.steps.trim_outer_blanks import TrimOuterBlanks

REGISTRY: Mapping[str, type[Step]] = MappingProxyType(
    {
        NormalizeLineEndings.NAME: NormalizeLineEndings,
        NormalizeUnicode.NAME: NormalizeUnicode,
        NormalizeSmartQuotes.NAME: NormalizeSmartQuotes,
        ExpandTabs.NAME: ExpandTabs,
        StripTrailingWhitespace.NAME: StripTrailingWhitespace,
        CollapseBlankRuns.NAME: CollapseBlankRuns,
        TrimOuterBlanks.NAME: TrimOuterBlanks,
        RequireNonblankText.NAME: RequireNonblankText,
        ExtractCandidates.NAME: ExtractCandidates,
        StripFences.NAME: StripFences,
        Dedent.NAME: Dedent,
        SplitOnNameGuard.NAME: SplitOnNameGuard,
        ExpandLastReturnSalvage.NAME: ExpandLastReturnSalvage,
        RepairImportLines.NAME: RepairImportLines,
        DedupeImports.NAME: DedupeImports,
        FilterNonblankCandidates.NAME: FilterNonblankCandidates,
        IdentifyCandidates.NAME: IdentifyCandidates,
        FilterCompilable.NAME: FilterCompilable,
        FilterPlainLiteral.NAME: FilterPlainLiteral,
        FilterCodeRepr.NAME: FilterCodeRepr,
        FilterHasTopLevelFunction.NAME: FilterHasTopLevelFunction,
        MaterializeCandidates.NAME: MaterializeCandidates,
        ReturnAll.NAME: ReturnAll,
    }
)

__all__ = ["REGISTRY"]
