"""Import-resolve validator (optional).

Checks that every top-level `import` / `from ... import ...` in the
module can actually be located. Uses `importlib.util.find_spec` and is
off by default to avoid filesystem / installed-package side effects.
"""

from __future__ import annotations

import ast
import importlib.util
from typing import ClassVar

from code_eval.models.validation_outcome import ValidationOutcome
from code_eval.names import ValidatorName
from code_eval.validators.base import Validator


def _module_findable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


class ImportResolveValidator(Validator):
    NAME: ClassVar[ValidatorName] = ValidatorName.IMPORT_RESOLVE

    def validate(self, source: str) -> ValidationOutcome:
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return ValidationOutcome(
                validator=self.NAME,
                passed=False,
                detail=f"parse failed: {e}",
            )

        missing: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if not _module_findable(root):
                        missing.append(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                if node.level == 0 and not _module_findable(root):
                    missing.append(node.module)

        if missing:
            return ValidationOutcome(
                validator=self.NAME,
                passed=False,
                detail=f"missing: {', '.join(missing)}",
            )
        return ValidationOutcome(validator=self.NAME, passed=True)
