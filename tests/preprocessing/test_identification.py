"""Parse-once identification, additive repair, and policy integration."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dr_code.corpus.preprocessing_run import run_preprocessing_corpus
from dr_code.preprocessing import identification
from dr_code.preprocessing.definitions import (
    HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
)
from dr_code.preprocessing.runner import bind_preprocessing
from dr_code.preprocessing.steps.expand_last_return_salvage import (
    ExpandLastReturnSalvage,
)
from dr_code.preprocessing.steps.filter_code_repr import FilterCodeRepr
from dr_code.preprocessing.steps.filter_compilable import FilterCompilable
from dr_code.preprocessing.steps.filter_has_top_level_function import (
    FilterHasTopLevelFunction,
)
from dr_code.preprocessing.steps.filter_plain_literal import FilterPlainLiteral
from dr_code.preprocessing.steps.materialize_candidates import (
    MaterializeCandidates,
)
from dr_code.trace import (
    CandidateLineage,
    CandidateOrigin,
    CodeCandidateSetArtifact,
    ExtractionOperation,
    IdentifiedCandidateSetArtifact,
    TextArtifact,
    is_absent,
)


def _origin(name: str) -> CandidateOrigin:
    return CandidateOrigin(
        path=(
            ExtractionOperation(
                kind="response_representation", details={"name": name}
            ),
        )
    )


def test_identification_inspects_each_unique_exact_source_once(
    monkeypatch,
) -> None:
    calls: list[str] = []
    original = identification.validate_python_source_with_ast

    def counting_inspection(source: str):
        calls.append(source)
        return original(source)

    monkeypatch.setattr(
        identification, "validate_python_source_with_ast", counting_inspection
    )
    source = "def f():\n    return math.sqrt(4)"
    value = CodeCandidateSetArtifact(candidates=(source, source))

    identified, facts = identification.identify_candidates(value)

    assert len(calls) == len(set(calls)) == 2
    assert calls == [source, f"import math\n{source}"]
    assert facts["inspection_count"] == 2
    assert [item.source for item in identified.candidates] == [
        f"import math\n{source}"
    ]


def test_identification_merges_distinct_paths_in_first_seen_order() -> None:
    source = "def f():\n    return 1"
    first = _origin("raw")
    second = _origin("json_code")
    value = CodeCandidateSetArtifact(
        candidates=(source, source, source),
        lineage=(
            CandidateLineage(origins=(first,)),
            CandidateLineage(origins=(second,)),
            CandidateLineage(origins=(first,)),
        ),
    )

    identified, _ = identification.identify_candidates(value)

    assert identified.candidates[0].lineage.origins == (first, second)


def test_bare_lambda_adds_named_function_after_original() -> None:
    source = "lambda x, y=2: x + y"
    identified, _ = identification.identify_candidates(
        CodeCandidateSetArtifact(candidates=(source,))
    )

    assert [item.source for item in identified.candidates] == [
        source,
        "def candidate(x, y=2):\n    return x + y",
    ]
    derived = identified.candidates[1]
    assert derived.inspection.top_level_function_names == ("candidate",)
    assert derived.lineage.origins[0].path[-1].kind == "lambda_to_function"


def test_single_name_lambda_assignment_adds_named_function() -> None:
    identified, _ = identification.identify_candidates(
        CodeCandidateSetArtifact(candidates=("solve = lambda x: x + 1",))
    )

    assert identified.candidates[1].source == (
        "def solve(x):\n    return x + 1"
    )
    assert identified.candidates[1].inspection.top_level_function_names == (
        "solve",
    )


def test_nonconservative_lambda_shapes_are_not_rendered() -> None:
    for source in (
        "left = right = lambda x: x",
        "holder.fn = lambda x: x",
        "x = 1\nfn = lambda x: x",
    ):
        identified, _ = identification.identify_candidates(
            CodeCandidateSetArtifact(candidates=(source,))
        )
        assert [item.source for item in identified.candidates] == [source]


def test_deep_valid_lambda_unparse_recursion_retains_raw_candidate() -> None:
    source = "lambda value: " + " + ".join(["value"] * 500)
    identified, facts = identification.identify_candidates(
        CodeCandidateSetArtifact(candidates=(source,))
    )

    assert [item.source for item in identified.candidates] == [source]
    assert identified.candidates[0].inspection.compile_ok is True
    assert facts["transformations"] == []


def test_bound_runner_contains_deep_lambda_unparse_recursion() -> None:
    source = "lambda value: " + " + ".join(["value"] * 500)
    runner = bind_preprocessing(HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION)

    output = runner.run(TextArtifact(text=source)).value("output")

    assert is_absent(output)
    assert output.failure_code == "no_top_level_function_candidate"


def test_corpus_resume_is_not_locked_by_deep_lambda(
    tmp_path: Path,
) -> None:
    source = "lambda value: " + " + ".join(["value"] * 500)
    input_path = tmp_path / "input.parquet"
    schema = pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("decoder_output", pa.string(), nullable=True),
        ]
    )
    pq.write_table(
        pa.Table.from_arrays(
            [
                pa.array(["ordinary", "deep-lambda"]),
                pa.array(["def f():\n    return 1", source]),
            ],
            schema=schema,
        ),
        input_path,
        row_group_size=1,
    )
    output_root = tmp_path / "runs"

    partial = run_preprocessing_corpus(
        input_path=input_path,
        output_root=output_root,
        run_id="deep-lambda",
        max_row_groups=1,
    )
    assert partial.name == "deep-lambda.partial"
    completed = run_preprocessing_corpus(
        input_path=input_path,
        output_root=output_root,
        run_id="deep-lambda",
    )

    manifest = json.loads((completed / "manifest.json").read_text())
    assert manifest["complete"] is True
    assert not partial.exists()
    outcomes = (
        pq.read_table(completed / "results.parquet")
        .column("outcome")
        .to_pylist()
    )
    assert outcomes == [
        "function_candidates_extracted",
        "no_top_level_function_candidate",
    ]


def test_last_return_salvage_is_additive_and_globally_appended() -> None:
    multiline = "def f(x):\n    return (\n        x + 1\n    )\ntrailing prose"
    trailing = "def g(x):\n    return x\ntrailing prose"
    output = ExpandLastReturnSalvage().apply(
        CodeCandidateSetArtifact(
            candidates=(multiline, trailing),
            lineage=(
                CandidateLineage(origins=(_origin("first"),)),
                CandidateLineage(origins=(_origin("second"),)),
            ),
        )
    )
    assert isinstance(output.value, CodeCandidateSetArtifact)
    assert output.value.candidates[:2] == (multiline, trailing)
    assert output.value.candidates[2:] == (
        "def f(x):\n    return (\n        x + 1\n    )\n",
        "def g(x):\n    return x\n",
    )
    multiline_operation = output.value.lineage[2].origins[0].path[-1]
    trailing_operation = output.value.lineage[3].origins[0].path[-1]
    assert multiline_operation.kind == "drop_after_last_return_salvage"
    assert multiline_operation.details == {
        "end_line": 4,
        "end_column": 6,
    }
    assert trailing_operation.kind == "drop_after_last_return_salvage"
    assert trailing_operation.details == {
        "end_line": 2,
        "end_column": 13,
    }


def test_identification_retains_distinct_salvage_for_compiling_intact_function(
    monkeypatch,
) -> None:
    source = "def f():\n    return 1\nprint(f())"
    intact_origins = (_origin("intact-first"), _origin("intact-second"))
    expanded = (
        ExpandLastReturnSalvage()
        .apply(
            CodeCandidateSetArtifact(
                candidates=(source, source),
                lineage=tuple(
                    CandidateLineage(origins=(origin,))
                    for origin in intact_origins
                ),
            )
        )
        .value
    )
    assert isinstance(expanded, CodeCandidateSetArtifact)
    calls: list[str] = []
    original_inspection = identification.validate_python_source_with_ast

    def count_inspection(candidate_source: str):
        calls.append(candidate_source)
        return original_inspection(candidate_source)

    monkeypatch.setattr(
        identification,
        "validate_python_source_with_ast",
        count_inspection,
    )

    identified, facts = identification.identify_candidates(expanded)

    assert [candidate.source for candidate in identified.candidates] == [
        source,
        "def f():\n    return 1\n",
    ]
    intact, salvaged = identified.candidates
    assert all(
        operation.kind != "drop_after_last_return_salvage"
        for origin in intact.lineage.origins
        for operation in origin.path
    )
    assert intact.lineage.origins == intact_origins
    assert [origin.path[:-1] for origin in salvaged.lineage.origins] == [
        origin.path for origin in intact_origins
    ]
    assert all(
        origin.path[-1].kind == "drop_after_last_return_salvage"
        for origin in salvaged.lineage.origins
    )
    assert all(
        origin.path[-1].details
        == {
            "end_line": 2,
            "end_column": 13,
        }
        for origin in salvaged.lineage.origins
    )
    assert len(calls) == len(set(calls)) == 2
    assert facts["unique_input_source_count"] == 2
    assert facts["transformations"] == []


def test_identification_retains_salvage_for_incomplete_json_representation() -> (
    None
):
    source = "def f():\n    return 1\ntruncated = True"
    expanded = (
        ExpandLastReturnSalvage()
        .apply(
            CodeCandidateSetArtifact(
                candidates=(source,),
                lineage=(
                    CandidateLineage(
                        origins=(_origin("completed_top_level_json_code"),)
                    ),
                ),
            )
        )
        .value
    )
    assert isinstance(expanded, CodeCandidateSetArtifact)

    identified, facts = identification.identify_candidates(expanded)

    assert [candidate.source for candidate in identified.candidates] == [
        source,
        "def f():\n    return 1\n",
    ]
    assert any(
        operation.kind == "drop_after_last_return_salvage"
        for origin in identified.candidates[1].lineage.origins
        for operation in origin.path
    )
    assert facts["transformations"] == []


def test_last_return_salvage_ignores_strings_docstrings_and_comments() -> None:
    sources = (
        (
            "def f():\n"
            "    return 1\n"
            '"""return is string content"""\n'
            "trailing prose"
        ),
        "def g():\n    return 2\n# return is a comment\ntrailing prose",
    )

    output = ExpandLastReturnSalvage().apply(
        CodeCandidateSetArtifact(candidates=sources)
    )

    assert isinstance(output.value, CodeCandidateSetArtifact)
    assert output.value.candidates == (
        *sources,
        "def f():\n    return 1\n",
        "def g():\n    return 2\n",
    )


def test_last_return_salvage_ignores_return_inside_trailing_prose() -> None:
    sources = (
        (
            "def find_value(mid):\n"
            "    if mid < 0:\n"
            "        return -1\n"
            "If you want the binary string to include the '0b' prefix, "
            "change the return line to: return bin(mid).\n"
        ),
        (
            "def rotate(s, shifted):\n"
            "    return shifted\n"
            "If you want a strictly greater-than rule, change the condition "
            "to if shift >= n: return s[::-1].\n"
        ),
        (
            "def count(values):\n"
            "    return len(values)\n"
            "- If there are no values, return 0.\n"
        ),
        (
            "def identity(value):\n"
            "    return value\n"
            "return the result as an integer.\n"
        ),
    )

    output = ExpandLastReturnSalvage().apply(
        CodeCandidateSetArtifact(candidates=sources)
    )

    assert isinstance(output.value, CodeCandidateSetArtifact)
    assert output.value.candidates == (
        *sources,
        "def find_value(mid):\n    if mid < 0:\n        return -1\n",
        "def rotate(s, shifted):\n    return shifted\n",
        "def count(values):\n    return len(values)\n",
        "def identity(value):\n    return value\n",
    )


def test_last_return_salvage_fails_closed_on_malformed_tokenization() -> None:
    source = "def f():\n    return (\n        1"

    output = ExpandLastReturnSalvage().apply(
        CodeCandidateSetArtifact(candidates=(source,))
    )

    assert isinstance(output.value, CodeCandidateSetArtifact)
    assert output.value.candidates == (source,)
    assert output.facts["salvage_candidate_count"] == 0


def test_named_pipeline_recovers_trailing_prose_after_complete_return() -> (
    None
):
    source = "def f(x):\n    return (\n        x + 1\n    )\ntrailing prose"
    runner = bind_preprocessing(HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION)

    output = runner.run(TextArtifact(text=source)).value("output")

    assert isinstance(output, CodeCandidateSetArtifact)
    assert "def f(x):\n    return (\n        x + 1\n    )" in output.candidates
    assert any(
        operation.kind == "drop_after_last_return_salvage"
        for lineage in output.lineage
        for origin in lineage.origins
        for operation in origin.path
    )


def test_policy_filters_use_stored_inspection_and_materialize(
    monkeypatch,
) -> None:
    identified, _ = identification.identify_candidates(
        CodeCandidateSetArtifact(
            candidates=("def f():\n    return 1", "def broken(:")
        )
    )

    def unexpected_parse(*args, **kwargs):
        raise AssertionError("policy filter reparsed candidate source")

    monkeypatch.setattr(
        identification, "validate_python_source_with_ast", unexpected_parse
    )
    current: IdentifiedCandidateSetArtifact = identified
    for step in (
        FilterPlainLiteral(),
        FilterCodeRepr(),
        FilterCompilable(),
        FilterHasTopLevelFunction(),
    ):
        result = step.apply(current)
        assert isinstance(result.value, IdentifiedCandidateSetArtifact)
        current = result.value
    materialized = MaterializeCandidates().apply(current).value
    assert isinstance(materialized, CodeCandidateSetArtifact)
    assert materialized.candidates == ("def f():\n    return 1",)
    assert materialized.lineage[0].candidate_id
