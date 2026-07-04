"""compile(..., mode='exec') validator."""

from __future__ import annotations

from typing import ClassVar

from code_eval.models.validation_outcome import ValidationOutcome
from code_eval.names import ValidatorName
from code_eval.validators.base import Validator


class CompileCheckValidator(Validator):
    NAME: ClassVar[ValidatorName] = ValidatorName.COMPILE_CHECK

    def validate(self, source: str) -> ValidationOutcome:
        try:
            compile(source, "<candidate>", "exec")
        except SyntaxError as e:
            return ValidationOutcome(
                validator=self.NAME,
                passed=False,
                detail=str(e),
            )
        except ValueError as e:  # e.g. null bytes
            return ValidationOutcome(
                validator=self.NAME,
                passed=False,
                detail=str(e),
            )
        return ValidationOutcome(validator=self.NAME, passed=True)
