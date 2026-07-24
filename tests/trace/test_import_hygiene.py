"""Architecture boundary tests for the trace package."""

from __future__ import annotations

import ast
from importlib.util import resolve_name
from pathlib import Path

DR_CODE_PACKAGE = Path(__file__).parents[2] / "src" / "dr_code"
TRACE_PACKAGE = DR_CODE_PACKAGE / "trace"
SIBLING_SYSTEMS = frozenset(
    child.name
    for child in DR_CODE_PACKAGE.iterdir()
    if child.is_dir() and child.name not in {"__pycache__", TRACE_PACKAGE.name}
)


def _import_targets(source_path: Path) -> set[str]:
    tree = ast.parse(source_path.read_text(), filename=source_path)
    relative_parent = source_path.relative_to(DR_CODE_PACKAGE).parent
    current_package = ".".join(("dr_code", *relative_parent.parts))
    targets: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative_name = "." * node.level + (node.module or "")
                module = resolve_name(relative_name, current_package)
            else:
                module = node.module or ""
            targets.add(module)
            targets.update(f"{module}.{alias.name}" for alias in node.names)

    return targets


def test_trace_source_does_not_import_sibling_systems() -> None:
    forbidden_prefixes = tuple(
        f"dr_code.{system}" for system in sorted(SIBLING_SYSTEMS)
    )
    violations = {
        source_path.relative_to(DR_CODE_PACKAGE): sorted(
            target
            for target in _import_targets(source_path)
            if any(
                target == prefix or target.startswith(f"{prefix}.")
                for prefix in forbidden_prefixes
            )
        )
        for source_path in sorted(TRACE_PACKAGE.rglob("*.py"))
    }

    assert not {
        source_path: targets
        for source_path, targets in violations.items()
        if targets
    }
