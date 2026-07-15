"""The trace package imports only pydantic and the stdlib."""

from __future__ import annotations

import subprocess
import sys

# The trace package holds only shared vocabulary; a test asserts it never
# imports humaneval, preprocessing, metrics, or synthetic.
FORBIDDEN = ("humaneval", "preprocessing", "metrics", "synthetic")


def test_trace_does_not_import_sibling_systems() -> None:
    # Fresh subprocess so nothing else has warmed sys.modules.
    code = (
        "import sys\n"
        "import dr_code.trace  # noqa: F401\n"
        "forbidden = " + repr(FORBIDDEN) + "\n"
        "loaded = [\n"
        "    name for name in sys.modules\n"
        "    if name.startswith('dr_code.')\n"
        "    and any(name.startswith('dr_code.' + f) for f in forbidden)\n"
        "]\n"
        "print(';'.join(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == ""


def test_trace_modules_import_only_pydantic_and_stdlib() -> None:
    # The package's imports resolve to pydantic/stdlib only. Baseline is
    # a bare `import pydantic`, so pydantic's own transitive deps are
    # allowed automatically; the trace package must add nothing beyond
    # pydantic, stdlib, and dr_code itself.
    code = (
        "import sys\n"
        # Baseline pulls pydantic and its own transitive dep tree, so the
        # only unexplained additions are genuine third parties.
        "import pydantic  # noqa: F401\n"
        "from pydantic import (  # noqa: F401\n"
        "    BaseModel, ConfigDict, Field, JsonValue, TypeAdapter,\n"
        ")\n"
        "import annotated_types  # noqa: F401\n"
        "import typing_inspection  # noqa: F401\n"
        "baseline = {n.split('.')[0] for n in sys.modules}\n"
        "import dr_code.trace  # noqa: F401\n"
        "stdlib = set(sys.stdlib_module_names)\n"
        "added = {\n"
        "    n.split('.')[0] for n in sys.modules\n"
        "} - baseline - stdlib - {'dr_code'}\n"
        "added = {n for n in added if n and not n.startswith('_')}\n"
        "print(';'.join(sorted(added)))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    added = [n for n in result.stdout.strip().split(";") if n]
    assert added == [], f"unexpected non-pydantic imports: {added}"
