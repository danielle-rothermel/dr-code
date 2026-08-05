"""Make the evaluation test builders importable under importlib mode.

Test directories carry no package markers and pytest imports test modules
via importlib, so a sibling helper module is not importable by name unless
its directory is on ``sys.path``. ``_builders`` holds override-taking
constructors rather than fixtures because a test varies one field at a time
and inherits valid values for the rest, which fixtures cannot express.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = str(Path(__file__).parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
