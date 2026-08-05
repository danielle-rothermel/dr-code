from __future__ import annotations

import json
import sys
from types import ModuleType


import dr_code.trace  # noqa: F401


def _report() -> dict[str, list[str]]:
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
    for module_name in sys.argv[1:]:
        sys.modules[module_name] = ModuleType(module_name)
    print(json.dumps(_report(), sort_keys=True))


if __name__ == "__main__":
    main()
