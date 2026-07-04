"""Per-validator outcome attached to a `Candidate`."""

from __future__ import annotations

from code_eval.models.base import FrozenModel
from code_eval.names import AstShapeKind, ValidatorName


class ValidationOutcome(FrozenModel):
    """Outcome of running a single validator against a candidate."""

    validator: ValidatorName
    passed: bool
    #: Free-form short message (e.g. the SyntaxError text). Empty on success.
    detail: str = ""
    #: AST shape only populated by the AST_SHAPE validator.
    ast_shape: AstShapeKind | None = None
