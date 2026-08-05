from __future__ import annotations

import random
from typing import ClassVar, Final

from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import CorruptionName, ImportMangleMode
from dr_code.synthetic.corruptions.base import Corruption

_TRAILING_PROSE: Final[tuple[str, ...]] = (
    "  -- we use this for math",
    "   (standard library)",
    "  ; remove if not needed",
)


def _mangle(source: str, rng: random.Random) -> str:
    lines = source.splitlines(keepends=True)
    for i, raw in enumerate(lines):
        stripped = raw.lstrip()
        if stripped.startswith(("import ", "from ")):
            body = raw.rstrip("\n")
            mangler = rng.choice(list(ImportMangleMode))
            if mangler is ImportMangleMode.TRAILING_PROSE:
                lines[i] = body + rng.choice(_TRAILING_PROSE) + "\n"
            elif mangler is ImportMangleMode.UNBALANCED_PAREN:
                lines[i] = body + " (extra_token_no_close\n"
            else:
                if "import" in body and "," not in body:
                    if body.startswith("import "):
                        mod = body[len("import ") :].strip()
                        lines[i] = f"from {mod} import (a,\n"
                    else:
                        lines[i] = body + ",\n"
                else:
                    lines[i] = body + ",\n"
            return "".join(lines)
    return "from broken import (\n" + source


class MangleImportLines(Corruption):
    NAME: ClassVar[CorruptionName] = CorruptionName.MANGLE_IMPORT_LINES
    VERSION: ClassVar[str] = "0"

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        return CorruptedSample(
            corrupted_source=_mangle(source, rng),
        )
