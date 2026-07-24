from __future__ import annotations

import ast
import warnings

import pytest

from dr_code.code_analysis import validate_python_source
from dr_code.preprocessing import (
    HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
    bind_preprocessing,
)
from dr_code.preprocessing.import_inference import (
    dedupe_import_lines,
    infer_missing_imports_from_tree,
    repair_import_lines,
)
from dr_code.trace import CodeCandidateSetArtifact, TextArtifact

RUNNER = bind_preprocessing(HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION)


def _infer(source: str) -> str:
    """Infer imports the way ``identify_candidates`` does: from a parsed tree."""
    return infer_missing_imports_from_tree(source, ast.parse(source))


# --- inference over a parsed tree ------------------------------------


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
def test_inference_prepends_missing_imports(
    source: str,
    expected_prefix: str,
) -> None:
    result = _infer(source)
    assert result.startswith(expected_prefix)
    assert source.strip() in result


def test_inference_skips_existing_import() -> None:
    source = "import numpy as np\n\ndef f(x):\n    return np.array(x)\n"
    result = _infer(source)
    assert result.count("import numpy as np") == 1
    assert "def f(x):" in result


def test_inference_skips_locally_bound_name() -> None:
    source = "np = 1\n\ndef f():\n    return np\n"
    result = _infer(source)
    assert "import numpy" not in result
    assert "return np" in result


def test_inference_skips_function_parameter_name() -> None:
    source = "def solve(F):\n    return F + 1\n"
    result = _infer(source)
    assert "import torch.nn.functional as F" not in result
    assert "import" not in result


def test_inference_skips_lambda_parameter_name() -> None:
    source = "g = lambda np: np + 1\n"
    result = _infer(source)
    assert "import numpy" not in result


def test_inference_skips_nested_local_assignment() -> None:
    source = "def f():\n    Path = 1\n    return Path\n"
    result = _infer(source)
    assert "from pathlib import Path" not in result


def test_inference_skips_comprehension_target() -> None:
    source = "def f(xs):\n    return [np for np in xs]\n"
    result = _infer(source)
    assert "import numpy" not in result


def test_inference_skips_for_loop_target() -> None:
    source = "def f(xs):\n    for math in xs:\n        pass\n    return xs\n"
    result = _infer(source)
    assert "import math" not in result


def test_inference_skips_walrus_target() -> None:
    source = "def f(xs):\n    return [y for y in xs if (Counter := y)]\n"
    result = _infer(source)
    assert "from collections import Counter" not in result


def test_inference_skips_except_binding() -> None:
    source = (
        "def f():\n"
        "    try:\n"
        "        pass\n"
        "    except Exception as os:\n"
        "        return os\n"
    )
    result = _infer(source)
    assert "import os" not in result


def test_inference_still_injects_free_name() -> None:
    # A genuinely free mapped name is still injected — the rule is
    # conservative about bound names, not about all names.
    source = "def solve(x):\n    return np.array(x)\n"
    result = _infer(source)
    assert result.startswith("import numpy as np\n")


def test_inference_adds_multiple_missing_imports() -> None:
    source = "def f():\n    return np.zeros(1) + math.pi\n"
    result = _infer(source)
    assert "import numpy as np" in result
    assert "import math" in result
    assert result.index("import numpy as np") < result.index("import math")


def test_inference_ignores_unmapped_names() -> None:
    source = "def f():\n    return random.randint(0, 1)\n"
    result = _infer(source)
    assert "random.randint" in result
    assert "import random" not in result


def test_inferred_import_follows_and_preserves_module_docstring() -> None:
    source = (
        '"""Module documentation."""\n\n'
        "def circumference(radius):\n"
        "    return 2 * math.pi * radius\n"
    )

    result = _infer(source)

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

    result = _infer(source)

    assert result.startswith(
        '"""Module documentation."""\n'
        "from __future__ import annotations\n"
        "from __future__ import generator_stop\n"
        "import math\n"
    )
    compile(result, "<inferred>", "exec")


def test_inferred_import_keeps_no_docstring_prefix_behavior() -> None:
    source = "def circumference(radius):\n    return 2 * math.pi * radius\n"

    assert _infer(source) == f"import math\n{source}"


# --- repair and dedupe step bodies ------------------------------------


def test_repair_trims_trailing_comment_on_import_line() -> None:
    source = (
        "from collections import (Counter,  # noqa\n"
        "\n"
        "def f():\n"
        "    return Counter([1])\n"
    )
    result, changed = repair_import_lines(source)
    assert changed
    assert "from collections import (Counter)" in result
    assert "def f():" in result


def test_repair_closes_unbalanced_import_parens() -> None:
    source = (
        "from typing import (List, Dict\n\ndef f():\n    return List[int]\n"
    )
    result, changed = repair_import_lines(source)
    assert changed
    assert "from typing import (List, Dict)" in result


def test_repair_leaves_parseable_import_lines_alone() -> None:
    source = "import math\n\ndef f():\n    return math.pi\n"
    result, changed = repair_import_lines(source)
    assert not changed
    assert result == source.rstrip("\n")


def test_repair_drops_an_unrepairable_import_line() -> None:
    source = "from collections import )\n\ndef f():\n    return 1\n"
    result, changed = repair_import_lines(source)
    assert changed
    assert "from collections import" not in result
    assert "def f():" in result


def test_dedupe_drops_later_duplicate_import_lines() -> None:
    source = "import math\nimport math\n\ndef f():\n    return math.pi\n"
    result = dedupe_import_lines(source)
    assert result.count("import math") == 1
    assert "def f():" in result


def test_dedupe_keeps_distinct_import_lines() -> None:
    source = "import math\nimport json\n\ndef f():\n    return math.pi\n"
    result = dedupe_import_lines(source)
    assert "import math" in result
    assert "import json" in result


def test_speculative_repair_parse_does_not_leak_syntax_warnings() -> None:
    invalid_escape = chr(92) + "q"
    source = f"import re  # noqa\nvalue = '{invalid_escape}'\n"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SyntaxWarning)
        repaired, _changed = repair_import_lines(source)

    assert caught == []
    assert "import re" in repaired


# --- the registered pipeline ------------------------------------------


def test_official_pipeline_infers_imports_for_compilable_candidate() -> None:
    source = "```python\ndef f(x):\n    return np.array([x])\n```"
    output = RUNNER.run(TextArtifact(text=source)).value("output")

    assert isinstance(output, CodeCandidateSetArtifact)
    assert output.candidates[0].startswith("import numpy as np")
    assert validate_python_source(output.candidates[0]).compile_ok
