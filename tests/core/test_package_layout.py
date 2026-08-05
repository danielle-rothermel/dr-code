"""Architecture contracts for the functional package layout."""

from __future__ import annotations

import ast
from importlib.util import resolve_name
from pathlib import Path

DR_CODE_PACKAGE = Path(__file__).parents[2] / "src" / "dr_code"
CORE_PACKAGE = DR_CODE_PACKAGE / "core"
FUNCTIONAL_PACKAGES = frozenset(
    {
        "evaluation",
        "humaneval",
        "metrics",
        "preprocessing",
        "synthetic",
        "trace",
    }
)


def _internal_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path)
    relative_parent = path.relative_to(DR_CODE_PACKAGE).parent
    current_package = ".".join(("dr_code", *relative_parent.parts))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(
                alias.name
                for alias in node.names
                if alias.name.startswith("dr_code.")
            )
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative_name = "." * node.level + (node.module or "")
                module = resolve_name(relative_name, current_package)
            else:
                module = node.module or ""
            if module.startswith("dr_code"):
                imports.add(module)
                imports.update(
                    f"{module}.{alias.name}" for alias in node.names
                )
    return imports


def test_python_root_contains_only_core_and_functional_packages() -> None:
    assert {
        path.name
        for path in DR_CODE_PACKAGE.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    } == {"core", *FUNCTIONAL_PACKAGES}
    assert {path.name for path in DR_CODE_PACKAGE.glob("*.py")} == {
        "__init__.py"
    }


def test_core_does_not_import_functional_packages() -> None:
    forbidden = tuple(f"dr_code.{name}" for name in FUNCTIONAL_PACKAGES)
    violations = {
        path.relative_to(DR_CODE_PACKAGE): sorted(
            target
            for target in _internal_imports(path)
            if any(
                target == prefix or target.startswith(f"{prefix}.")
                for prefix in forbidden
            )
        )
        for path in sorted(CORE_PACKAGE.rglob("*.py"))
    }
    assert not {
        path: imports for path, imports in violations.items() if imports
    }
