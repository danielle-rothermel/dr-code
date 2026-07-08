"""Switch all ASCII string quotes from one style to the other."""

from __future__ import annotations

import random
import tokenize
from io import StringIO
from typing import ClassVar

from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import InverseTransformName
from dr_code.synthetic.inverse_transforms.base import InverseTransform


def _flip_string_quotes(source: str) -> str:
    """Re-emit every string literal with single quotes (so format flips them back)."""
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
            # Naively re-quote with single quotes. If body contains a single
            # quote we leave it alone to avoid producing invalid tokens.
            if "'" not in body:
                new = "'" + body + "'"
                out.append(tok._replace(string=new))
                continue
        out.append(tok)
    try:
        return tokenize.untokenize(out)
    except (ValueError, tokenize.TokenError):
        return source


class ChangeQuoteStyle(InverseTransform):
    """Flip double-quoted strings to single-quoted. Recovery is L2 ruff format."""

    NAME: ClassVar[InverseTransformName] = (
        InverseTransformName.CHANGE_QUOTE_STYLE
    )

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        return CorruptedSample(
            corrupted_source=_flip_string_quotes(source),
        )
