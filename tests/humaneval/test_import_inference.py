from __future__ import annotations

import pytest

from dr_code.code_analysis import validate_python_source
from dr_code.humaneval.import_inference import infer_necessary_imports
from dr_code.preprocessing import (
    BEST_EFFORT_V2_DEFINITION,
    run_preprocessing,
)
from dr_code.trace import CodeArtifact, TextArtifact


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
    # A genuinely free mapped name is still injected — the fix is
    # conservative about bound names, not about all names.
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


def test_infer_necessary_imports_passthrough_on_syntax_error() -> None:
    source = "def f(x\n    return np.array(x)\n"
    result = infer_necessary_imports(source)
    assert "return np.array(x)" in result
    assert "import numpy" not in result


def test_infer_necessary_imports_ignores_unmapped_names() -> None:
    source = "def f():\n    return random.randint(0, 1)\n"
    result = infer_necessary_imports(source)
    assert "random.randint" in result
    assert "import random" not in result


def test_canonical_preprocessing_infers_imports_for_compilable_candidate() -> (
    None
):
    source = "```python\ndef f(x):\n    return np.array([x])\n```"
    output = run_preprocessing(
        BEST_EFFORT_V2_DEFINITION.materialize(),
        TextArtifact(text=source),
    ).value("output")
    assert isinstance(output, CodeArtifact)
    assert output.source.startswith("import numpy as np")
    assert validate_python_source(output.source).compile_ok
