"""Validator base class."""

from __future__ import annotations

from typing import ClassVar

from code_eval.models.validation_outcome import ValidationOutcome
from code_eval.names import ValidatorName


class Validator:
    """Base class for validators.

    Validators consume a source string and produce a single
    `ValidationOutcome`. They never raise.
    """

    NAME: ClassVar[ValidatorName]

    def validate(self, source: str) -> ValidationOutcome:
        raise NotImplementedError
