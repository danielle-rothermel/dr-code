"""Isolated runtime report for the trace package import boundary.

Run as a script in a fresh interpreter (see ``run_python_script``). Emits
raw facts only — every loaded ``dr_code.*`` module and every non-stdlib
package root. ``test_import_hygiene.py`` owns the approved-roots rule.
"""

from __future__ import annotations

import json
import sys
from types import ModuleType

# Imported at module load, before ``main`` injects fake modules, so the
# recorded facts describe a clean import of the trace façade.
import dr_code.trace  # noqa: F401


def _report() -> dict[str, list[str]]:
    """Describe module loading in the current interpreter."""
    loaded_dr_code_modules = sorted(
        name for name in sys.modules if name.startswith("dr_code.")
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
        "loaded_dr_code_modules": loaded_dr_code_modules,
        "third_party_roots": third_party_roots,
    }


def main() -> None:
    """Print loaded dr_code modules and non-stdlib package roots as JSON."""
    for module_name in sys.argv[1:]:
        sys.modules[module_name] = ModuleType(module_name)
    print(json.dumps(_report(), sort_keys=True))


if __name__ == "__main__":
    main()
