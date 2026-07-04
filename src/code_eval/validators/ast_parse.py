"""ast.parse validator."""

from __future__ import annotations

import ast
from typing import ClassVar

from code_eval.models.validation_outcome import ValidationOutcome
from code_eval.names import ValidatorName
from code_eval.validators.base import Validator


class AstParseValidator(Validator):
    NAME: ClassVar[ValidatorName] = ValidatorName.AST_PARSE

    def validate(self, source: str) -> ValidationOutcome:
        try:
            ast.parse(source)
        except SyntaxError as e:
            return ValidationOutcome(
                validator=self.NAME,
                passed=False,
                detail=str(e),
            )
        return ValidationOutcome(validator=self.NAME, passed=True)
