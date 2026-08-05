from __future__ import annotations

import json
import sys
import sysconfig
from pathlib import Path
from types import ModuleType


import dr_code.trace  # noqa: F401


STDLIB = Path(sysconfig.get_path("stdlib")).resolve()
THIRD_PARTY_DIRS = tuple(
    Path(path).resolve()
    for key in ("purelib", "platlib")
    if (path := sysconfig.get_path(key)) is not None
)


def _is_stdlib_module(name: str, module: ModuleType | None) -> bool:
    root = name.partition(".")[0]
    if (
        root in sys.stdlib_module_names
        or root in sys.builtin_module_names
        or root.startswith("_sysconfigdata_")
    ):
        return True

    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        return False
    module_path = Path(module_file).resolve()
    if any(
        module_path.is_relative_to(directory) for directory in THIRD_PARTY_DIRS
    ):
        return False
    if {"site-packages", "dist-packages"} & set(module_path.parts):
        return False
    return module_path.is_relative_to(STDLIB)


def _report() -> dict[str, list[str]]:
    loaded_dr_code_modules = sorted(
        name for name in sys.modules if name.startswith("dr_code.")
    )
    third_party_roots = sorted(
        {
            root
            for name, module in sys.modules.items()
            if not _is_stdlib_module(name, module)
            and (root := name.partition(".")[0])
            and root not in {"__main__", "dr_code", "_virtualenv"}
        }
    )
    return {
        "loaded_dr_code_modules": loaded_dr_code_modules,
        "third_party_roots": third_party_roots,
    }


def main() -> None:
    for module_spec in sys.argv[1:]:
        module_name, separator, module_file = module_spec.partition("=")
        module = ModuleType(module_name)
        if separator:
            module.__file__ = module_file
        sys.modules[module_name] = module
    print(json.dumps(_report(), sort_keys=True))


if __name__ == "__main__":
    main()
