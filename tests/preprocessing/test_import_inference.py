from __future__ import annotations

import warnings

import pytest

from dr_code.humaneval.acceptance import extract_humaneval_code
from dr_code.preprocessing.import_inference import (
    dedupe_import_lines,
    infer_necessary_imports,
    repair_import_lines,
)


@pytest.mark.parametrize(
    ("source", "expected_prefix"),
    [
        (
            "def f(x):\n    return np.array(x)\n",
            "import numpy as np\n",
        ),
        (
            "def f():\n    return math.sqrt(2)\n",
            "import math\n",
        ),
        (
            "def f():\n    return Counter([1, 1])\n",
            "from collections import Counter\n",
        ),
        (
            "def f():\n    return nn.Linear(1, 1)\n",
            "import torch.nn as nn\n",
        ),
    ],
)
def test_infer_necessary_imports_prepends_missing_imports(
    source: str,
    expected_prefix: str,
) -> None:
    result = infer_necessary_imports(source)
    assert result.startswith(expected_prefix)
    assert source.strip() in result


def test_infer_necessary_imports_skips_existing_import() -> None:
    source = "import numpy as np\n\ndef f(x):\n    return np.array(x)\n"
    result = infer_necessary_imports(source)
    assert result.count("import numpy as np") == 1
    assert "def f(x):" in result


def test_infer_necessary_imports_skips_locally_bound_name() -> None:
    source = "np = 1\n\ndef f():\n    return np\n"
    result = infer_necessary_imports(source)
    assert "import numpy" not in result
    assert "return np" in result


def test_infer_necessary_imports_skips_function_parameter_name() -> None:
    source = "def solve(F):\n    return F + 1\n"
    result = infer_necessary_imports(source)
    assert "import torch.nn.functional as F" not in result
    assert "import" not in result


def test_infer_necessary_imports_respects_sibling_function_scopes() -> None:
    source = (
        "def passthrough(math):\n"
        "    return math\n\n"
        "def square_root(value):\n"
        "    return math.sqrt(value)\n"
    )

    result = infer_necessary_imports(source)

    assert result == f"import math\n{source.rstrip()}"


def test_infer_necessary_imports_skips_lambda_parameter_name() -> None:
    source = "g = lambda np: np + 1\n"
    result = infer_necessary_imports(source)
    assert "import numpy" not in result


def test_infer_necessary_imports_skips_nested_local_assignment() -> None:
    source = "def f():\n    Path = 1\n    return Path\n"
    result = infer_necessary_imports(source)
    assert "from pathlib import Path" not in result


def test_infer_necessary_imports_skips_comprehension_target() -> None:
    source = "def f(xs):\n    return [np for np in xs]\n"
    result = infer_necessary_imports(source)
    assert "import numpy" not in result


def test_infer_necessary_imports_skips_for_loop_target() -> None:
    source = "def f(xs):\n    for math in xs:\n        pass\n    return xs\n"
    result = infer_necessary_imports(source)
    assert "import math" not in result


def test_infer_necessary_imports_skips_walrus_target() -> None:
    source = "def f(xs):\n    return [y for y in xs if (Counter := y)]\n"
    result = infer_necessary_imports(source)
    assert "from collections import Counter" not in result


def test_infer_necessary_imports_skips_except_binding() -> None:
    source = (
        "def f():\n"
        "    try:\n"
        "        pass\n"
        "    except Exception as os:\n"
        "        return os\n"
    )
    result = infer_necessary_imports(source)
    assert "import os" not in result


def test_infer_necessary_imports_still_injects_free_name() -> None:
    source = "def solve(x):\n    return np.array(x)\n"
    result = infer_necessary_imports(source)
    assert result.startswith("import numpy as np\n")


def test_infer_necessary_imports_adds_multiple_missing_imports() -> None:
    source = "def f():\n    return np.zeros(1) + math.pi\n"
    result = infer_necessary_imports(source)
    assert "import numpy as np" in result
    assert "import math" in result
    assert result.index("import numpy as np") < result.index("import math")


def test_infer_necessary_imports_repairs_trailing_comment_on_import_line() -> (
    None
):
    source = (
        "from collections import (Counter,  # noqa\n"
        "\n"
        "def f():\n"
        "    return Counter([1])\n"
    )
    result = infer_necessary_imports(source)
    assert "from collections import (Counter)" in result
    assert "def f():" in result


def test_infer_necessary_imports_repairs_unbalanced_import_parens() -> None:
    source = (
        "from typing import (List, Dict\n\ndef f():\n    return List[int]\n"
    )
    result = infer_necessary_imports(source)
    assert "from typing import (List, Dict)" in result


def test_infer_necessary_imports_deduplicates_import_lines() -> None:
    source = "import math\nimport math\n\ndef f():\n    return math.pi\n"
    result = infer_necessary_imports(source)
    assert result.count("import math") == 1


def test_dedupe_import_lines_preserves_imports_in_sibling_scopes() -> None:
    source = (
        "def floor_value(x):\n"
        "    import math\n"
        "    return math.floor(x)\n\n"
        "def ceil_value(x):\n"
        "    import math\n"
        "    return math.ceil(x)\n"
    )

    assert dedupe_import_lines(source) == source.rstrip()


def test_infer_necessary_imports_passthrough_on_syntax_error() -> None:
    source = "def f(x\n    return np.array(x)\n"
    result = infer_necessary_imports(source)
    assert "return np.array(x)" in result
    assert "import numpy" not in result


def test_parse_or_none_treats_parser_overflow_as_unparseable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dr_code.preprocessing import import_inference

    def _overflow(*_args: object, **_kwargs: object) -> object:
        raise MemoryError(
            "Parser stack overflowed - Python source too complex to parse"
        )

    monkeypatch.setattr(import_inference.ast, "parse", _overflow)

    assert import_inference._parse_or_none("def f():\n    return 1\n") is None


def test_infer_necessary_imports_ignores_unmapped_names() -> None:
    source = "def f():\n    return random.randint(0, 1)\n"
    result = infer_necessary_imports(source)
    assert "random.randint" in result
    assert "import random" not in result


@pytest.mark.parametrize(
    "import_line",
    (
        "    import random",
        "    from functools import cmp_to_key",
        "    import hashlib",
    ),
)
def test_repair_import_lines_preserves_valid_indented_imports(
    import_line: str,
) -> None:
    source = f"def f():\n{import_line}\n    return None\n"

    repaired, changed = repair_import_lines(source)

    assert repaired == source.rstrip()
    assert not changed


@pytest.mark.parametrize(
    "import_lines",
    (
        "from collections import (\n    Counter,\n    defaultdict,\n)",
        "from collections import Counter, \\\n    defaultdict",
        "from os import (\n    path,\n); sibling = 1",
    ),
)
def test_repair_import_lines_preserves_valid_multiline_imports(
    import_lines: str,
) -> None:
    source = f"{import_lines}\n\ndef f():\n    return Counter()\n"

    repaired, changed = repair_import_lines(source)

    assert repaired == source.rstrip()
    assert not changed
    compile(repaired, "<repaired>", "exec")


def test_inferred_import_follows_and_preserves_module_docstring() -> None:
    source = (
        '"""Module documentation."""\n\n'
        "def circumference(radius):\n"
        "    return 2 * math.pi * radius\n"
    )

    result = infer_necessary_imports(source)

    assert result.startswith('"""Module documentation."""\nimport math\n')
    namespace: dict[str, object] = {}
    exec(compile(result, "<inferred>", "exec"), namespace)
    assert namespace["__doc__"] == "Module documentation."


def test_inferred_import_follows_docstring_and_contiguous_futures() -> None:
    source = (
        '"""Module documentation."""\n'
        "from __future__ import annotations\n"
        "from __future__ import generator_stop\n\n"
        "def circumference(radius: float) -> float:\n"
        "    return 2 * math.pi * radius\n"
    )

    result = infer_necessary_imports(source)

    assert result.startswith(
        '"""Module documentation."""\n'
        "from __future__ import annotations\n"
        "from __future__ import generator_stop\n"
        "import math\n"
    )
    compile(result, "<inferred>", "exec")


def test_inferred_import_leads_a_module_with_no_header() -> None:
    source = "def circumference(radius):\n    return 2 * math.pi * radius\n"

    assert infer_necessary_imports(source) == f"import math\n{source.strip()}"


def test_speculative_repair_parse_does_not_leak_syntax_warnings() -> None:
    invalid_escape = chr(92) + "q"
    source = f"import re  # noqa\nvalue = '{invalid_escape}'\n"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SyntaxWarning)
        repaired, _changed = repair_import_lines(source)

    assert caught == []
    assert "import re" in repaired


def test_future_import_candidate_survives_extraction() -> None:
    source = (
        "```python\n"
        "from __future__ import annotations\n\n"
        "def circumference(radius: float) -> float:\n"
        "    return 2 * math.pi * radius\n"
        "```"
    )

    result = extract_humaneval_code(source)

    assert result.succeeded
    assert result.accepted_code is not None
    assert result.accepted_code.startswith(
        "from __future__ import annotations\nimport math\n"
    )
