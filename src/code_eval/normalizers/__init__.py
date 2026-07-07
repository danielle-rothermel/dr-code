"""Normalizer registry.

Each normalizer level (L0-L5) and orthogonal form is a separate class.
The ``NORMALIZERS`` mapping lets the pipeline look up a normalizer
class by its ``NormalizerName`` enum value.
"""

from typing import Final

from code_eval.names import NormalizerName
from code_eval.normalizers.annotation_strip import AnnotationStrip
from code_eval.normalizers.base import Normalizer
from code_eval.normalizers.import_sort_dedup import ImportSortDedup
from code_eval.normalizers.l0_canonical_ast import L0CanonicalAst
from code_eval.normalizers.l1_strip_comments_docstrings import L1StripCommentsDocstrings
from code_eval.normalizers.l2_ruff_format import L2RuffFormat
from code_eval.normalizers.l3_ruff_fix_safe import L3RuffFixSafe
from code_eval.normalizers.l4_ruff_fix_unsafe import L4RuffFixUnsafe
from code_eval.normalizers.l5_ty_fix import L5TyFix
from code_eval.normalizers.name_normalize import NameNormalize
from code_eval.normalizers.string_form_normalize import StringFormNormalize

NORMALIZERS: Final[dict[NormalizerName, type[Normalizer]]] = {
    NormalizerName.L0_CANONICAL_AST: L0CanonicalAst,
    NormalizerName.L1_STRIP_COMMENTS_DOCSTRINGS: L1StripCommentsDocstrings,
    NormalizerName.L2_RUFF_FORMAT: L2RuffFormat,
    NormalizerName.L3_RUFF_FIX_SAFE: L3RuffFixSafe,
    NormalizerName.L4_RUFF_FIX_UNSAFE: L4RuffFixUnsafe,
    NormalizerName.L5_TY_FIX: L5TyFix,
    NormalizerName.IMPORT_SORT_DEDUP: ImportSortDedup,
    NormalizerName.NAME_NORMALIZE: NameNormalize,
    NormalizerName.ANNOTATION_STRIP: AnnotationStrip,
    NormalizerName.STRING_FORM_NORMALIZE: StringFormNormalize,
}

__all__ = [
    "NORMALIZERS",
    "AnnotationStrip",
    "ImportSortDedup",
    "L0CanonicalAst",
    "L1StripCommentsDocstrings",
    "L2RuffFormat",
    "L3RuffFixSafe",
    "L4RuffFixUnsafe",
    "L5TyFix",
    "NameNormalize",
    "Normalizer",
    "StringFormNormalize",
]
