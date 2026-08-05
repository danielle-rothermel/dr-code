from __future__ import annotations

import importlib
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]
TERMS_PATH = ROOT / ".defs" / "terms.toml"
CONTRACTS_PATH = ROOT / ".defs" / "contracts.toml"
VIEWER_ENTRYPOINT = (
    ROOT / "viewer" / "packages" / "viewer" / "src" / "index.ts"
)
RELATIONSHIPS = ("is_a", "part_of")


def _load(path: Path) -> dict[str, object]:
    with path.open("rb") as file:
        return tomllib.load(file)


def _assert_acyclic(
    edges: dict[str, tuple[str, ...]], relationship: str
) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visiting:
            raise AssertionError(
                f"{relationship} relationship contains a cycle at {name!r}"
            )
        if name in visited:
            return
        visiting.add(name)
        for target in edges[name]:
            visit(target)
        visiting.remove(name)
        visited.add(name)

    for name in edges:
        visit(name)


def _resolve_python_symbol(symbol: str) -> object:
    parts = symbol.split(".")
    for boundary in range(len(parts), 0, -1):
        module_name = ".".join(parts[:boundary])
        try:
            value: object = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name:
                raise
            continue
        for attribute in parts[boundary:]:
            value = getattr(value, attribute)
        return value
    raise AssertionError(
        f"no importable module owns exported symbol {symbol!r}"
    )


def test_terms_graph_and_exports_are_valid() -> None:
    document = _load(TERMS_PATH)
    entries = document["terms"]
    assert isinstance(entries, list)

    names = [entry["name"] for entry in entries]
    assert len(names) == len(set(names)), "term names must be unique"
    name_set = set(names)

    for relationship in RELATIONSHIPS:
        edges: dict[str, tuple[str, ...]] = {}
        for entry in entries:
            targets = tuple(entry.get(relationship, ()))
            assert len(targets) == len(set(targets))
            assert entry["name"] not in targets
            assert set(targets) <= name_set
            edges[entry["name"]] = targets
        _assert_acyclic(edges, relationship)

    viewer_source = VIEWER_ENTRYPOINT.read_text(encoding="utf-8")
    for entry in entries:
        for symbol in entry.get("exported_symbols", ()):
            if symbol.startswith("@dr-code/viewer:"):
                viewer_symbol = symbol.partition(":")[2]
                assert re.search(
                    rf"\b{re.escape(viewer_symbol)}\b", viewer_source
                )
            else:
                _resolve_python_symbol(symbol)


def test_contract_entries_have_unique_complete_shapes() -> None:
    document = _load(CONTRACTS_PATH)
    entries = document["contracts"]
    assert isinstance(entries, list)

    required = {"title", "statement", "rationale", "date"}
    allowed = required | {"check"}
    titles = [entry["title"] for entry in entries]
    assert len(titles) == len(set(titles)), "contract titles must be unique"

    for entry in entries:
        assert required <= entry.keys()
        assert entry.keys() <= allowed
        assert all(
            isinstance(value, str) and value.strip()
            for value in entry.values()
        )
