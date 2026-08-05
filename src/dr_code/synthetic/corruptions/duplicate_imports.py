from __future__ import annotations

import ast
import random
from typing import ClassVar

from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import CorruptionName
from dr_code.synthetic.corruptions.base import Corruption


def _duplicate(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    import_linenos: list[int] = []
    for node in tree.body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            import_linenos.append(node.lineno)
    if not import_linenos:
        return "import math\nimport math\n" + source
    lines = source.splitlines(keepends=True)
    out: list[str] = []
    for i, line in enumerate(lines, start=1):
        out.append(line)
        if i in import_linenos:
            out.append(line)
    return "".join(out)


class DuplicateImports(Corruption):
    NAME: ClassVar[CorruptionName] = CorruptionName.DUPLICATE_IMPORTS
    VERSION: ClassVar[str] = "0"

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        return CorruptedSample(
            corrupted_source=_duplicate(source),
        )
