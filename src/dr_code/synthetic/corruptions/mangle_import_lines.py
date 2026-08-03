"""Mangle import lines so they fail to parse without repair."""

from __future__ import annotations

import random
from typing import ClassVar, Final

from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import CorruptionName
from dr_code.synthetic.corruptions.base import Corruption

_TRAILING_PROSE: Final[tuple[str, ...]] = (
    "  -- we use this for math",
    "   (standard library)",
    "  ; remove if not needed",
)


def _mangle(source: str, rng: random.Random) -> str:
    """Mangle the first import line we encounter."""
    lines = source.splitlines(keepends=True)
    for i, raw in enumerate(lines):
        stripped = raw.lstrip()
        if stripped.startswith(("import ", "from ")):
            body = raw.rstrip("\n")
            mangler = rng.choice(
                ("trailing_prose", "unbalanced_paren", "trailing_comma")
            )
            if mangler == "trailing_prose":
                # Append non-comment trailing text. Pure prose makes
                # the line invalid.
                lines[i] = body + rng.choice(_TRAILING_PROSE) + "\n"
            elif mangler == "unbalanced_paren":
                lines[i] = body + " (extra_token_no_close\n"
            else:  # trailing_comma — only for `from X import a, b,` style
                if "import" in body and "," not in body:
                    # convert to `from X import (a,` form on the body
                    if body.startswith("import "):
                        mod = body[len("import ") :].strip()
                        lines[i] = f"from {mod} import (a,\n"
                    else:
                        lines[i] = body + ",\n"
                else:
                    lines[i] = body + ",\n"
            return "".join(lines)
    # No import found — fall back to injecting a broken one.
    return "from broken import (\n" + source


class MangleImportLines(Corruption):
    """Introduce a parse-blocking error in an import line."""

    NAME: ClassVar[CorruptionName] = CorruptionName.MANGLE_IMPORT_LINES
    VERSION: ClassVar[str] = "0"

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        return CorruptedSample(
            corrupted_source=_mangle(source, rng),
        )
