from __future__ import annotations

import ast

import pytest

from dr_code.core.source.python_transforms import (
    RENAMED_LOCAL_PREFIX,
    alpha_rename_locals_in_tree,
    alpha_rename_locals,
    dedupe_imports,
    remove_top_level_imports,
    rename_locals_in_function,
    strip_docstrings,
    strip_type_annotations_in_tree,
    strip_type_annotations,
)

UNPARSEABLE = "def broken(:\n"

SOURCE_TO_SOURCE_TRANSFORMS = (
    alpha_rename_locals,
    dedupe_imports,
    remove_top_level_imports,
    strip_docstrings,
    strip_type_annotations,
)


def _run_entrypoint(source: str) -> object:
    namespace: dict[str, object] = {}
    exec(
        compile(source, "<alpha-rename-test>", "exec", dont_inherit=True),
        namespace,
    )
    entrypoint = namespace["run"]
    assert callable(entrypoint)
    return entrypoint()


def _assert_alpha_rename_preserves_result(source: str) -> None:
    transformed = alpha_rename_locals(source, rename_params=False)
    entrypoint = next(
        node
        for node in ast.parse(transformed).body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    assignment = next(
        node for node in entrypoint.body if isinstance(node, ast.Assign)
    )
    target = assignment.targets[0]
    assert isinstance(target, ast.Name)
    assert target.id.startswith(RENAMED_LOCAL_PREFIX)

    assert _run_entrypoint(source) == _run_entrypoint(transformed)


@pytest.mark.parametrize(
    "transform",
    SOURCE_TO_SOURCE_TRANSFORMS,
    ids=lambda fn: fn.__name__,
)
def test_code_transforms_raise_syntax_error_on_unparseable_input(
    transform,
) -> None:
    with pytest.raises(SyntaxError):
        transform(UNPARSEABLE)


def test_strip_docstrings_removes_module_function_and_class_docstrings() -> (
    None
):
    source = (
        '"""Module doc."""\n'
        "class C:\n"
        '    """Class doc."""\n'
        "    def m(self):\n"
        '        """Method doc."""\n'
        "        return 1\n"
    )
    out = strip_docstrings(source)
    assert '"""' not in out
    assert "return 1" in out


def test_strip_docstrings_keeps_docstring_only_bodies_parseable() -> None:
    out = strip_docstrings('def f():\n    """Only a docstring."""\n')
    assert "pass" in out


def test_strip_type_annotations_drops_args_returns_and_annassign() -> None:
    source = "def f(x: int, *args: str) -> bool:\n    y: int = 2\n    z: int\n    return x\n"
    out = strip_type_annotations(source)
    assert ":" not in out.split("\n")[0].removesuffix(":")
    assert "def f(x, *args):" in out
    assert "y = 2" in out
    assert "z:" not in out


def test_strip_type_annotations_in_tree_can_keep_selected_sites() -> None:
    tree = ast.parse("def f(x: int) -> bool:\n    y: int = 2\n    return x\n")
    strip_type_annotations_in_tree(
        tree,
        keep=lambda site: site.name == "x",
    )

    out = ast.unparse(tree)

    assert "def f(x: int):" in out
    assert "->" not in out
    assert "y = 2" in out


def test_alpha_rename_locals_can_rename_params() -> None:
    out = alpha_rename_locals(
        "def f(count):\n    total = count + 1\n    return total\n",
        rename_params=True,
    )
    assert "def f(_v0):" in out
    assert "_v1 = _v0 + 1" in out


def test_alpha_rename_locals_preserves_keyword_params_by_default() -> None:
    source = "def f(*, count):\n    total = count + 1\n    return total\n"
    transformed = alpha_rename_locals(source)
    namespace: dict[str, object] = {}
    transformed_namespace: dict[str, object] = {}

    exec(source, namespace)
    exec(transformed, transformed_namespace)

    assert "def f(*, count):" in transformed
    assert "_v0 = count + 1" in transformed
    function = namespace["f"]
    transformed_function = transformed_namespace["f"]
    assert callable(function)
    assert callable(transformed_function)
    assert function(count=2) == transformed_function(count=2) == 3


def test_alpha_rename_locals_preserves_nested_scope_semantics() -> None:
    source = (
        "def build(seed):\n"
        "    captured = seed + 10\n"
        "    shadowed = seed * 2\n"
        "    def calculate(shadowed):\n"
        "        inner_local = captured + shadowed\n"
        "        return inner_local\n"
        "    return calculate\n"
    )
    transformed = alpha_rename_locals(source, rename_params=False)
    transformed_tree = ast.parse(transformed)
    nested_function = next(
        node
        for node in ast.walk(transformed_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "calculate"
    )
    renamed_assignment_targets = {
        node.id
        for node in ast.walk(transformed_tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Store)
        and node.id.startswith("_v")
    }
    original_namespace: dict[str, object] = {}
    transformed_namespace: dict[str, object] = {}

    exec(source, original_namespace)
    exec(transformed, transformed_namespace)

    assert [argument.arg for argument in nested_function.args.args] == [
        "shadowed"
    ]
    assert renamed_assignment_targets
    assert original_namespace["build"](3)(5) == transformed_namespace["build"](
        3
    )(5)


def test_alpha_rename_locals_updates_nonlocal_bindings() -> None:
    source = (
        "def run():\n"
        "    value = 1\n"
        "    def increment():\n"
        "        nonlocal value\n"
        "        value += 1\n"
        "        return value\n"
        "    return increment(), value\n"
    )

    _assert_alpha_rename_preserves_result(source)


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            "def run():\n"
            "    T = 'outer'\n"
            "    def inner[T](value: T) -> T:\n"
            "        return value\n"
            "    return inner.__annotations__['return'].__name__\n",
            id="generic-function",
        ),
        pytest.param(
            "def run():\n"
            "    T = 'outer'\n"
            "    class Inner[T]:\n"
            "        value: T\n"
            "    return Inner.__annotations__['value'].__name__\n",
            id="generic-class",
        ),
        pytest.param(
            "def run():\n"
            "    T = str\n"
            "    type Alias[T] = tuple[T]\n"
            "    return Alias.__value__[0].__name__\n",
            id="generic-type-alias",
        ),
    ],
)
def test_alpha_rename_locals_respects_type_parameter_annotation_scopes(
    source: str,
) -> None:
    _assert_alpha_rename_preserves_result(source)


def test_alpha_rename_locals_preserves_method_class_cell_semantics() -> None:
    source = (
        "def run():\n"
        "    __class__ = 'outer'\n"
        "    class Base:\n"
        "        marker = 'base'\n"
        "    class Inner(Base):\n"
        "        def explicit_class(self):\n"
        "            return __class__.__name__\n"
        "        def inherited_marker(self):\n"
        "            return super().marker\n"
        "    instance = Inner()\n"
        "    return instance.explicit_class(), instance.inherited_marker()\n"
    )

    _assert_alpha_rename_preserves_result(source)


def test_alpha_rename_locals_can_rename_method_parameter_used_by_super() -> (
    None
):
    source = (
        "class Base:\n"
        "    marker = 'base'\n"
        "class Inner(Base):\n"
        "    def inherited_marker(self):\n"
        "        return super().marker\n"
        "def run():\n"
        "    return Inner().inherited_marker()\n"
    )

    transformed = alpha_rename_locals(source, rename_params=True)

    assert _run_entrypoint(source) == _run_entrypoint(transformed)
    assert "def inherited_marker(_v" in transformed


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            "def run():\n    value = 10\n    return (lambda _v0: value)(3)\n",
            id="descendant-lambda-parameter",
        ),
        pytest.param(
            "def run():\n"
            "    value = 10\n"
            "    return [value + _v0 for _v0 in (1, 2)]\n",
            id="comprehension-binder",
        ),
        pytest.param(
            "def run():\n"
            "    value = 10\n"
            "    class Box:\n"
            "        _v0 = 3\n"
            "        result = value + _v0\n"
            "    return Box.result\n",
            id="nested-class-binder",
        ),
        pytest.param(
            "def run():\n"
            "    value = 10\n"
            "    def _v0():\n"
            "        return 3\n"
            "    return value + _v0()\n",
            id="nested-function-name",
        ),
        pytest.param(
            "_v0: int = 10\n"
            "def run():\n"
            "    value = 3\n"
            "    return value + _v0\n",
            id="module-annassign-binder",
        ),
    ],
)
def test_alpha_rename_locals_generates_fresh_names_across_scopes(
    source: str,
) -> None:
    _assert_alpha_rename_preserves_result(source)


def test_rename_locals_in_function_applies_mapping_to_one_function() -> None:
    tree = ast.parse(
        "def f(count):\n    total = count + 1\n    return total\n"
    )
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)

    rename_locals_in_function(function, {"count": "value", "total": "result"})

    assert ast.unparse(tree) == (
        "def f(value):\n    result = value + 1\n    return result"
    )


def test_rename_locals_in_function_updates_captures_not_shadowing() -> None:
    source = (
        "def run():\n"
        "    value = 10\n"
        "    def capture():\n"
        "        return value\n"
        "    def shadow(value):\n"
        "        return value + 1\n"
        "    return capture(), shadow(3)\n"
    )
    tree = ast.parse(source)
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)

    rename_locals_in_function(function, {"value": "renamed"})

    shadow = next(
        node
        for node in function.body
        if isinstance(node, ast.FunctionDef) and node.name == "shadow"
    )
    assert [argument.arg for argument in shadow.args.args] == ["value"]
    assert _run_entrypoint(source) == _run_entrypoint(ast.unparse(tree))


@pytest.mark.parametrize(
    "mapping",
    [
        pytest.param({"value": "not valid"}, id="invalid-replacement"),
        pytest.param({"not valid": "renamed"}, id="invalid-source"),
        pytest.param({"value": "class"}, id="keyword-replacement"),
        pytest.param(
            {"value": "renamed", "other": "renamed"},
            id="non-injective",
        ),
    ],
)
def test_rename_locals_in_function_rejects_invalid_mapping_shape(
    mapping: dict[str, str],
) -> None:
    tree = ast.parse("def run(value, other):\n    return value + other\n")
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)

    with pytest.raises(
        ValueError,
        match=(
            "^unsafe local rename mapping: names must be valid Python "
            "identifiers, replacements must be unique and fresh$"
        ),
    ):
        rename_locals_in_function(function, mapping)


@pytest.mark.parametrize(
    "source,mapping",
    [
        pytest.param(
            "def run(value):\n    return value + existing\n",
            {"value": "existing"},
            id="unmapped-reference",
        ),
        pytest.param(
            "def run(value):\n"
            "    def inner(existing):\n"
            "        return value + existing\n"
            "    return inner(1)\n",
            {"value": "existing"},
            id="descendant-function-binder",
        ),
        pytest.param(
            "def run(value):\n"
            "    return (lambda existing: value + existing)(1)\n",
            {"value": "existing"},
            id="descendant-lambda-binder",
        ),
        pytest.param(
            "def run(value):\n"
            "    return [value + existing for existing in (1,)]\n",
            {"value": "existing"},
            id="descendant-comprehension-binder",
        ),
        pytest.param(
            "def run(value):\n"
            "    def inner[existing]():\n"
            "        return value, existing\n"
            "    return inner()\n",
            {"value": "existing"},
            id="descendant-type-parameter",
        ),
        pytest.param(
            "class Base:\n"
            "    pass\n"
            "class Inner(Base):\n"
            "    def run(self):\n"
            "        return super()\n",
            {"self": "__class__"},
            id="implicit-class-cell",
        ),
    ],
)
def test_rename_locals_in_function_rejects_capture_unsafe_mapping(
    source: str,
    mapping: dict[str, str],
) -> None:
    tree = ast.parse(source)
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    assert isinstance(function, ast.FunctionDef)

    with pytest.raises(
        ValueError,
        match=(
            "^unsafe local rename mapping: names must be valid Python "
            "identifiers, replacements must be unique and fresh$"
        ),
    ):
        rename_locals_in_function(function, mapping)


def test_alpha_rename_locals_in_tree_matches_source_transform() -> None:
    source = "def f(count):\n    total = count + 1\n    return total\n"
    tree = ast.parse(source)

    alpha_rename_locals_in_tree(tree)

    assert ast.unparse(tree) == (
        "def f(count):\n    _v0 = count + 1\n    return _v0"
    )
    assert ast.unparse(tree) == alpha_rename_locals(source)


def test_alpha_rename_locals_preserves_module_level_names() -> None:
    out = alpha_rename_locals(
        "import math\n\ndef f(x):\n    return math.sqrt(x)\n"
    )
    assert "math.sqrt" in out
    assert "def f(" in out


def test_remove_top_level_imports_deletes_only_import_lines() -> None:
    source = "import math\nfrom os import path\n\nx = 1\n"
    assert remove_top_level_imports(source) == "\nx = 1\n"


def test_remove_top_level_imports_no_imports_is_identity() -> None:
    source = "x = 1\n"
    assert remove_top_level_imports(source) == source


@pytest.mark.parametrize(
    "source",
    [
        "import os; sibling = 1\n",
        "sibling = 1; import os\n",
    ],
)
def test_remove_top_level_imports_preserves_same_line_sibling(
    source: str,
) -> None:
    transformed = remove_top_level_imports(source)
    namespace: dict[str, object] = {}

    exec(transformed, namespace)

    assert transformed == "sibling = 1\n"
    assert namespace["sibling"] == 1


def test_dedupe_imports_keeps_first_occurrence_and_trailing_newline() -> None:
    source = "import math\nimport math\nx = 1\n"
    assert dedupe_imports(source) == "import math\nx = 1\n"
