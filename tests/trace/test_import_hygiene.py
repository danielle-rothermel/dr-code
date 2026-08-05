"""Architecture boundary tests for the trace package."""

from __future__ import annotations

import ast
import json
from importlib.util import resolve_name
from pathlib import Path

IMPORT_PROBE = Path(__file__).parent / "_import_probe.py"
DR_CODE_PACKAGE = Path(__file__).parents[2] / "src" / "dr_code"
TRACE_PACKAGE = DR_CODE_PACKAGE / "trace"
#: The only dr_code roots the trace package may load at runtime.
APPROVED_DR_CODE_ROOTS = frozenset({"dr_code.base", "dr_code.trace"})
SIBLING_SYSTEMS = frozenset(
    {
        child.name
        for child in DR_CODE_PACKAGE.iterdir()
        if child.is_dir()
        and child.name not in {"__pycache__", TRACE_PACKAGE.name}
    }
    | {
        child.stem
        for child in DR_CODE_PACKAGE.glob("*.py")
        if child.stem not in {"__init__", "base"}
    }
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


def _loaded_siblings(report: dict[str, list[str]]) -> list[str]:
    """Loaded dr_code modules outside the approved roots."""
    return [
        name
        for name in report["loaded_dr_code_modules"]
        if not any(
            name == root or name.startswith(f"{root}.")
            for root in APPROVED_DR_CODE_ROOTS
        )
    ]


def test_trace_runtime_import_loads_only_approved_boundaries(
    run_python_script,
) -> None:
    result = run_python_script(IMPORT_PROBE)

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert _loaded_siblings(report) == []
    assert set(report["third_party_roots"]) <= {
        "annotated_types",
        "pydantic",
        "pydantic_core",
        "typing_extensions",
        "typing_inspection",
    }


def test_trace_runtime_probe_reports_injected_boundary_crossings(
    run_python_script,
) -> None:
    result = run_python_script(
        IMPORT_PROBE,
        "dr_code.code_analysis",
        "unexpected_dependency",
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert _loaded_siblings(report) == ["dr_code.code_analysis"]
    assert "unexpected_dependency" in report["third_party_roots"]
