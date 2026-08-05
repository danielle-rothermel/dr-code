from __future__ import annotations

import random
import tokenize
from io import StringIO
from typing import ClassVar

from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import CorruptionName
from dr_code.synthetic.corruptions.base import Corruption


def _flip_string_quotes(source: str) -> str:
    try:
        tokens = list(tokenize.generate_tokens(StringIO(source).readline))
    except tokenize.TokenError:
        return source
    out: list[tokenize.TokenInfo] = []
    for tok in tokens:
        if (
            tok.type == tokenize.STRING
            and tok.string.startswith('"')
            and not tok.string.startswith('"""')
        ):
            body = tok.string[1:-1]
            if "'" not in body:
                new = "'" + body + "'"
                out.append(tok._replace(string=new))
                continue
        out.append(tok)
    try:
        return tokenize.untokenize(out)
    except (ValueError, tokenize.TokenError):
        return source


class ChangeQuoteStyle(Corruption):
    NAME: ClassVar[CorruptionName] = CorruptionName.CHANGE_QUOTE_STYLE
    VERSION: ClassVar[str] = "0"

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        return CorruptedSample(
            corrupted_source=_flip_string_quotes(source),
        )
