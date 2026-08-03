"""Isolated runtime report for the trace package import boundary."""

from __future__ import annotations

import json
import sys
from types import ModuleType


def _report() -> dict[str, list[str]]:
    """Describe trace boundary crossings in the current interpreter."""
    approved_dr_code_roots = {"dr_code.models", "dr_code.trace"}
    loaded_siblings = sorted(
        name
        for name in sys.modules
        if name.startswith("dr_code.")
        and not any(
            name == root or name.startswith(f"{root}.")
            for root in approved_dr_code_roots
        )
    )
    third_party_roots = sorted(
        {
            root
            for name in sys.modules
            if (root := name.partition(".")[0]) not in sys.stdlib_module_names
            and root not in sys.builtin_module_names
            and root not in {"__main__", "dr_code", "_virtualenv"}
            and not root.startswith("_sysconfigdata_")
        }
    )
    return {
        "loaded_siblings": loaded_siblings,
        "third_party_roots": third_party_roots,
    }


def main() -> None:
    """Print loaded sibling modules and non-stdlib package roots as JSON."""
    for module_name in sys.argv[1:]:
        sys.modules[module_name] = ModuleType(module_name)
    print(json.dumps(_report(), sort_keys=True))


if __name__ == "__main__":
    main()
