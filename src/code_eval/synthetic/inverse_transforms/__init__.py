"""Inverse transforms.

Each inverse transform is a pure function of (source, rng) that returns a
`CorruptedSample` whose `expected_recovery_steps` declares the recovery
contract the validator must satisfy.

Transforms live in individual files (one per file) and are aggregated in
the `REGISTRY` below. Add a transform by:

1. Implement it as a subclass of `InverseTransform` in its own file.
2. Import and register it here in `REGISTRY`.
3. Add a paired entry in `corruption_recipes.py` if it should appear in
   the default dataset.
"""

from code_eval.synthetic.inverse_transforms.add_blank_lines import AddBlankLines
from code_eval.synthetic.inverse_transforms.add_code_fences import AddCodeFences
from code_eval.synthetic.inverse_transforms.add_comments_noise import AddCommentsNoise
from code_eval.synthetic.inverse_transforms.add_crlf import AddCrlf
from code_eval.synthetic.inverse_transforms.add_dead_code import AddDeadCode
from code_eval.synthetic.inverse_transforms.add_indentation import AddIndentation
from code_eval.synthetic.inverse_transforms.add_inline_backticks import AddInlineBackticks
from code_eval.synthetic.inverse_transforms.add_markdown_wrappers import AddMarkdownWrappers
from code_eval.synthetic.inverse_transforms.add_multiple_solutions import AddMultipleSolutions
from code_eval.synthetic.inverse_transforms.add_prose_wrapper import AddProseWrapper
from code_eval.synthetic.inverse_transforms.add_smart_quotes import AddSmartQuotes
from code_eval.synthetic.inverse_transforms.add_tabs import AddTabs
from code_eval.synthetic.inverse_transforms.add_trailing_whitespace import AddTrailingWhitespace
from code_eval.synthetic.inverse_transforms.add_type_annotations import AddTypeAnnotations
from code_eval.synthetic.inverse_transforms.add_unicode_noise import AddUnicodeNoise
from code_eval.synthetic.inverse_transforms.base import InverseTransform
from code_eval.synthetic.inverse_transforms.change_quote_style import ChangeQuoteStyle
from code_eval.synthetic.inverse_transforms.change_string_form import ChangeStringForm
from code_eval.synthetic.inverse_transforms.duplicate_imports import DuplicateImports
from code_eval.synthetic.inverse_transforms.mangle_import_lines import MangleImportLines
from code_eval.synthetic.inverse_transforms.remove_imports import RemoveImports
from code_eval.synthetic.inverse_transforms.rename_locals import RenameLocals
from code_eval.synthetic.inverse_transforms.truncate import Truncate

REGISTRY: dict[str, type[InverseTransform]] = {
    AddCodeFences.NAME: AddCodeFences,
    AddProseWrapper.NAME: AddProseWrapper,
    AddSmartQuotes.NAME: AddSmartQuotes,
    AddIndentation.NAME: AddIndentation,
    AddTabs.NAME: AddTabs,
    AddTrailingWhitespace.NAME: AddTrailingWhitespace,
    AddCrlf.NAME: AddCrlf,
    AddUnicodeNoise.NAME: AddUnicodeNoise,
    AddBlankLines.NAME: AddBlankLines,
    AddMarkdownWrappers.NAME: AddMarkdownWrappers,
    AddInlineBackticks.NAME: AddInlineBackticks,
    Truncate.NAME: Truncate,
    RemoveImports.NAME: RemoveImports,
    MangleImportLines.NAME: MangleImportLines,
    DuplicateImports.NAME: DuplicateImports,
    AddMultipleSolutions.NAME: AddMultipleSolutions,
    AddCommentsNoise.NAME: AddCommentsNoise,
    AddDeadCode.NAME: AddDeadCode,
    ChangeQuoteStyle.NAME: ChangeQuoteStyle,
    ChangeStringForm.NAME: ChangeStringForm,
    AddTypeAnnotations.NAME: AddTypeAnnotations,
    RenameLocals.NAME: RenameLocals,
}

__all__ = [
    "REGISTRY",
    "InverseTransform",
]
