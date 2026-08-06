from collections.abc import Mapping
from types import MappingProxyType

from dr_code.synthetic.corruptions.add_blank_lines import AddBlankLines
from dr_code.synthetic.corruptions.add_code_fences import AddCodeFences
from dr_code.synthetic.corruptions.add_comments_noise import (
    AddCommentsNoise,
)
from dr_code.synthetic.corruptions.add_crlf import AddCrlf
from dr_code.synthetic.corruptions.add_dead_code import AddDeadCode
from dr_code.synthetic.corruptions.add_indentation import AddIndentation
from dr_code.synthetic.corruptions.add_inline_backticks import (
    AddInlineBackticks,
)
from dr_code.synthetic.corruptions.add_markdown_wrappers import (
    AddMarkdownWrappers,
)
from dr_code.synthetic.corruptions.add_multiple_solutions import (
    AddMultipleSolutions,
)
from dr_code.synthetic.corruptions.add_prose_wrapper import (
    AddProseWrapper,
)
from dr_code.synthetic.corruptions.add_smart_quotes import (
    AddSmartQuotes,
)
from dr_code.synthetic.corruptions.add_tabs import AddTabs
from dr_code.synthetic.corruptions.add_trailing_whitespace import (
    AddTrailingWhitespace,
)
from dr_code.synthetic.corruptions.add_type_annotations import (
    AddTypeAnnotations,
)
from dr_code.synthetic.corruptions.add_unicode_noise import (
    AddUnicodeNoise,
)
from dr_code.synthetic.corruptions.base import Corruption
from dr_code.synthetic.corruptions.change_quote_style import (
    ChangeQuoteStyle,
)
from dr_code.synthetic.corruptions.change_string_form import (
    ChangeStringForm,
)
from dr_code.synthetic.corruptions.duplicate_imports import (
    DuplicateImports,
)
from dr_code.synthetic.corruptions.mangle_import_lines import (
    MangleImportLines,
)
from dr_code.synthetic.corruptions.remove_imports import RemoveImports
from dr_code.synthetic.corruptions.rename_locals import RenameLocals
from dr_code.synthetic.corruptions.truncate import Truncate

REGISTRY: Mapping[str, type[Corruption]] = MappingProxyType(
    {
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
)
