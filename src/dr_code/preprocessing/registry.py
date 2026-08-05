from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from dr_code.preprocessing.steps.add_last_return_salvage import (
    AddLastReturnSalvage,
)
from dr_code.preprocessing.steps.base import Step
from dr_code.preprocessing.steps.collapse_blank_runs import (
    CollapseBlankRuns,
)
from dr_code.preprocessing.steps.dedent_candidates import DedentCandidates
from dr_code.preprocessing.steps.dedupe_candidates import DedupeCandidates
from dr_code.preprocessing.steps.dedupe_imports import DedupeImports
from dr_code.preprocessing.steps.drop_blank_candidates import (
    DropBlankCandidates,
)
from dr_code.preprocessing.steps.expand_tabs import ExpandTabs
from dr_code.preprocessing.steps.extract_all_representations import (
    ExtractAllRepresentations,
)
from dr_code.preprocessing.steps.filter_code_repr import FilterCodeRepr
from dr_code.preprocessing.steps.filter_compilable import (
    FilterCompilable,
)
from dr_code.preprocessing.steps.filter_plain_literal import (
    FilterPlainLiteral,
)
from dr_code.preprocessing.steps.filter_top_level_functions import (
    FilterTopLevelFunctions,
)
from dr_code.preprocessing.steps.infer_missing_imports import (
    InferMissingImports,
)
from dr_code.preprocessing.steps.inspect_candidates import InspectCandidates
from dr_code.preprocessing.steps.materialize_candidate_set import (
    MaterializeCandidateSet,
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
from dr_code.preprocessing.steps.reject_blank_input import RejectBlankInput
from dr_code.preprocessing.steps.repair_import_lines import (
    RepairImportLines,
)
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
        ExpandTabs.NAME: ExpandTabs,
        StripTrailingWhitespace.NAME: StripTrailingWhitespace,
        CollapseBlankRuns.NAME: CollapseBlankRuns,
        TrimOuterBlanks.NAME: TrimOuterBlanks,
        RejectBlankInput.NAME: RejectBlankInput,
        ExtractAllRepresentations.NAME: ExtractAllRepresentations,
        NormalizeSmartQuotes.NAME: NormalizeSmartQuotes,
        StripFences.NAME: StripFences,
        DedentCandidates.NAME: DedentCandidates,
        SplitOnNameGuard.NAME: SplitOnNameGuard,
        RepairImportLines.NAME: RepairImportLines,
        InferMissingImports.NAME: InferMissingImports,
        DedupeImports.NAME: DedupeImports,
        AddLastReturnSalvage.NAME: AddLastReturnSalvage,
        DropBlankCandidates.NAME: DropBlankCandidates,
        DedupeCandidates.NAME: DedupeCandidates,
        InspectCandidates.NAME: InspectCandidates,
        FilterPlainLiteral.NAME: FilterPlainLiteral,
        FilterCodeRepr.NAME: FilterCodeRepr,
        FilterCompilable.NAME: FilterCompilable,
        FilterTopLevelFunctions.NAME: FilterTopLevelFunctions,
        MaterializeCandidateSet.NAME: MaterializeCandidateSet,
    }
)

__all__ = ["REGISTRY"]
