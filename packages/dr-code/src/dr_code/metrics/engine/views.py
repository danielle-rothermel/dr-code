from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _ParseOutcome:
    module: ast.Module | None
    error: str | None


class ViewCache:
    def __init__(self) -> None:
        self._parsed: dict[str, _ParseOutcome] = {}

    def _parse(self, source: str) -> _ParseOutcome:
        cached = self._parsed.get(source)
        if cached is not None:
            return cached
        try:
            outcome = _ParseOutcome(module=ast.parse(source), error=None)
        except (SyntaxError, ValueError) as exc:
            outcome = _ParseOutcome(
                module=None,
                error=f"{type(exc).__name__}: {exc}",
            )
        self._parsed[source] = outcome
        return outcome

    def parsed_module(self, source: str) -> ast.Module | None:
        return self._parse(source).module

    def parse_error(self, source: str) -> str | None:
        return self._parse(source).error
