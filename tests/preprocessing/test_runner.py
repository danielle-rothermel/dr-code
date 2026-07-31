"""Tests for bind_definition wiring and run_preprocessing execution."""

from __future__ import annotations

import pytest

from dr_code.eval import (
    PreprocessingDefinition,
    PreprocessingStepBinding,
)
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.runner import (
    BoundStep,
    bind_definition,
    run_preprocessing,
)
from dr_code.trace import (
    CodeArtifact,
    TextArtifact,
    Trace,
    WiringError,
    is_absent,
)


def _def(
    steps: tuple[PreprocessingStepBinding, ...],
    definition_id: str = "d1",
) -> PreprocessingDefinition:
    return PreprocessingDefinition(
        definition_id=definition_id, version="1", steps=steps
    )


def _run(definition: PreprocessingDefinition, input_value):
    return run_preprocessing(definition.materialize(), input_value)


# --- bind-time wiring ------------------------------------------------


def test_bind_resolves_steps() -> None:
    definition = _def(
        (
            PreprocessingStepBinding(
                instance_name="n", step=StepName.NORMALIZE_UNICODE
            ),
        )
    )
    bound = bind_definition(definition.materialize())
    assert len(bound) == 1
    assert isinstance(bound[0], BoundStep)
    assert bound[0].instance_name == "n"


def test_bind_empty_definition() -> None:
    assert bind_definition(_def(()).materialize()) == ()


def test_materialize_detects_registry_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = _def(
        (
            PreprocessingStepBinding(
                instance_name="n", step=StepName.NORMALIZE_UNICODE
            ),
        )
    )
    # monkeypatch the registry to simulate an unregistered name
    import dr_code.preprocessing.runner as runner_mod

    config = definition.materialize()
    monkeypatch.setattr(runner_mod, "REGISTRY", {})
    with pytest.raises(WiringError, match="unknown step"):
        bind_definition(config)


def test_definition_rejects_bad_settings() -> None:
    with pytest.raises(ValueError):
        _def(
            (
                PreprocessingStepBinding(
                    instance_name="e",
                    step=StepName.EXPAND_TABS,
                    settings={"tab_width": "not-an-int"},
                ),
            )
        )


def test_definition_rejects_broken_kind_chain() -> None:
    # normalize_line_endings (Text->Text) then strip_fences
    # (CandidateSet->CandidateSet): Text != CandidateSet.
    with pytest.raises(ValueError, match="kind chain"):
        _def(
            (
                PreprocessingStepBinding(
                    instance_name="n",
                    step=StepName.NORMALIZE_LINE_ENDINGS,
                ),
                PreprocessingStepBinding(
                    instance_name="s", step=StepName.STRIP_FENCES
                ),
            )
        )


def test_bind_accepts_valid_kind_chain() -> None:
    definition = _def(
        (
            PreprocessingStepBinding(
                instance_name="e", step=StepName.EXTRACT_CANDIDATES
            ),
            PreprocessingStepBinding(
                instance_name="s", step=StepName.STRIP_FENCES
            ),
            PreprocessingStepBinding(
                instance_name="sel", step=StepName.SELECT_FIRST
            ),
        )
    )
    bound = bind_definition(definition.materialize())
    assert len(bound) == 3


# --- run: basic execution -------------------------------------------


def test_run_single_text_step() -> None:
    definition = _def(
        (
            PreprocessingStepBinding(
                instance_name="n", step=StepName.NORMALIZE_UNICODE
            ),
        )
    )
    trace = _run(definition, TextArtifact(text="ｄｅｆ"))
    assert trace.value("output") == TextArtifact(text="def")
    assert trace.value("input") == TextArtifact(text="ｄｅｆ")
    assert trace.value("n") == TextArtifact(text="def")


def test_run_produces_trace_with_producer_stamp() -> None:
    definition = _def(
        (
            PreprocessingStepBinding(
                instance_name="n", step=StepName.NORMALIZE_UNICODE
            ),
        )
    )
    trace = _run(definition, TextArtifact(text="x"))
    assert isinstance(trace, Trace)
    assert trace.producer.producer_id == "d1"
    assert trace.producer.version == "1"
    assert trace.producer.definition_hash is not None


def test_run_empty_definition_output_equals_input() -> None:
    trace = _run(_def(()), TextArtifact(text="x"))
    assert trace.value("output") == TextArtifact(text="x")


def test_run_input_kind_mismatch_raises_wiring_error() -> None:
    definition = _def(
        (
            PreprocessingStepBinding(
                instance_name="e", step=StepName.EXTRACT_CANDIDATES
            ),
        )
    )
    # extract_candidates expects Text; pass a CodeArtifact.
    with pytest.raises(WiringError):
        _run(definition, CodeArtifact(source="x = 1"))


# --- run: Absent propagation -----------------------------------------


def test_run_failed_step_yields_absent_and_complete_trace() -> None:
    definition = _def(
        (
            PreprocessingStepBinding(
                instance_name="e", step=StepName.EXTRACT_CANDIDATES
            ),
            PreprocessingStepBinding(
                instance_name="s", step=StepName.STRIP_FENCES
            ),
            PreprocessingStepBinding(
                instance_name="sel", step=StepName.SELECT_FIRST
            ),
        )
    )
    trace = _run(definition, TextArtifact(text=""))

    extract_val = trace.value("e")
    assert is_absent(extract_val)
    assert extract_val.failed_step == "e"
    assert extract_val.cause == "no alternative produced candidates"

    # downstream steps inherit the same Absent, propagated_through grows
    strip_val = trace.value("s")
    assert is_absent(strip_val)
    assert strip_val.failed_step == "e"
    assert "s" in strip_val.propagated_through

    out = trace.value("output")
    assert is_absent(out)
    assert out.failed_step == "e"

    # every instance name retained + input/output
    assert set(trace.values) == {"input", "e", "s", "sel", "output"}


def test_run_select_first_empty_set_yields_absent() -> None:
    definition = _def(
        (
            PreprocessingStepBinding(
                instance_name="e", step=StepName.EXTRACT_CANDIDATES
            ),
            PreprocessingStepBinding(
                instance_name="flt", step=StepName.FILTER_COMPILABLE
            ),
            PreprocessingStepBinding(
                instance_name="sel", step=StepName.SELECT_FIRST
            ),
        )
    )
    trace = _run(definition, TextArtifact(text="def broken(:"))
    out = trace.value("output")
    assert is_absent(out)
    assert out.failed_step == "sel"
    assert out.cause == "no candidate survived filtering"
    # rejection reason recorded as fact
    assert "rejected_0" in trace.step_facts["flt"]


def test_run_step_facts_merged() -> None:
    definition = _def(
        (
            PreprocessingStepBinding(
                instance_name="e", step=StepName.EXTRACT_CANDIDATES
            ),
            PreprocessingStepBinding(
                instance_name="sel", step=StepName.SELECT_FIRST
            ),
        )
    )
    trace = _run(
        definition,
        TextArtifact(text="```python\ndef f():\n    return 1\n```"),
    )
    assert trace.step_facts["e"] == {"alternative": "fenced_blocks"}


def test_run_propagated_absent_records_each_step() -> None:
    definition = _def(
        (
            PreprocessingStepBinding(
                instance_name="e", step=StepName.EXTRACT_CANDIDATES
            ),
            PreprocessingStepBinding(
                instance_name="a", step=StepName.STRIP_FENCES
            ),
            PreprocessingStepBinding(
                instance_name="b", step=StepName.DEDENT_CANDIDATES
            ),
            PreprocessingStepBinding(
                instance_name="c", step=StepName.SPLIT_ON_NAME_GUARD
            ),
        )
    )
    trace = _run(definition, TextArtifact(text=""))
    out = trace.value("output")
    assert is_absent(out)
    assert out.propagated_through == ("a", "b", "c")


# --- run: full fenced pipeline --------------------------------------


def test_run_full_fenced_pipeline() -> None:
    definition = _def(
        (
            PreprocessingStepBinding(
                instance_name="e", step=StepName.EXTRACT_CANDIDATES
            ),
            PreprocessingStepBinding(
                instance_name="sf", step=StepName.STRIP_FENCES
            ),
            PreprocessingStepBinding(
                instance_name="ng", step=StepName.SPLIT_ON_NAME_GUARD
            ),
            PreprocessingStepBinding(
                instance_name="rr", step=StepName.DROP_AFTER_LAST_RETURN
            ),
            PreprocessingStepBinding(
                instance_name="ri", step=StepName.REPAIR_IMPORT_LINES
            ),
            PreprocessingStepBinding(
                instance_name="ii", step=StepName.INFER_MISSING_IMPORTS
            ),
            PreprocessingStepBinding(
                instance_name="dd", step=StepName.DEDUPE_IMPORTS
            ),
            PreprocessingStepBinding(
                instance_name="fc", step=StepName.FILTER_COMPILABLE
            ),
            PreprocessingStepBinding(
                instance_name="sel", step=StepName.SELECT_FIRST
            ),
        )
    )
    raw = (
        "Here is the code:\n"
        "```python\n"
        "def f(x):\n    return np.array(x)\n```\n"
    )
    trace = _run(definition, TextArtifact(text=raw))
    out = trace.value("output")
    assert isinstance(out, CodeArtifact)
    assert "import numpy as np" in out.source
    assert "def f(x):" in out.source
    assert trace.step_facts["e"] == {"alternative": "fenced_blocks"}


# --- escaped-newline behavior cases, re-expressed against the pipeline


@pytest.mark.parametrize(
    "source",
    [
        r"Intro\n```python\ndef f():\n    return 1\n```",
        r"Explanation:\ndef f():\n\treturn 1",
        "Intro\n" + r"```python\ndef f():\n    return 1\n```",
        r'"Intro\n```python\ndef f():\n    return 1\n```"',
    ],
    ids=["escaped-fenced", "escaped-unfenced", "mixed", "json-string"],
)
def test_pipeline_recovers_escaped_newline_shapes(source: str) -> None:
    from dr_code.code_analysis import validate_python_source

    definition = _def(
        (
            PreprocessingStepBinding(
                instance_name="e", step=StepName.EXTRACT_CANDIDATES
            ),
            PreprocessingStepBinding(
                instance_name="sf", step=StepName.STRIP_FENCES
            ),
            PreprocessingStepBinding(
                instance_name="fc", step=StepName.FILTER_COMPILABLE
            ),
            PreprocessingStepBinding(
                instance_name="sel", step=StepName.SELECT_FIRST
            ),
        )
    )
    trace = _run(definition, TextArtifact(text=source))
    out = trace.value("output")
    assert isinstance(out, CodeArtifact)
    assert "def f():" in out.source
    # round-trip: recovered code must compile
    assert validate_python_source(out.source).compile_ok


def test_pipeline_preserves_string_literal_escapes() -> None:
    from dr_code.code_analysis import validate_python_source

    definition = _def(
        (
            PreprocessingStepBinding(
                instance_name="e", step=StepName.EXTRACT_CANDIDATES
            ),
            PreprocessingStepBinding(
                instance_name="sf", step=StepName.STRIP_FENCES
            ),
            PreprocessingStepBinding(
                instance_name="fc", step=StepName.FILTER_COMPILABLE
            ),
            PreprocessingStepBinding(
                instance_name="sel", step=StepName.SELECT_FIRST
            ),
        )
    )
    source = (
        r"Intro\n```python\ndef join_lines(lines):\n"
        r'    return "\n".join(lines)\n```'
    )
    expected = 'def join_lines(lines):\n    return "\\n".join(lines)'
    trace = _run(definition, TextArtifact(text=source))
    out = trace.value("output")
    assert isinstance(out, CodeArtifact)
    assert out.source == expected
    assert validate_python_source(out.source).compile_ok


def test_pipeline_prose_only_yields_absent() -> None:
    definition = _def(
        (
            PreprocessingStepBinding(
                instance_name="e", step=StepName.EXTRACT_CANDIDATES
            ),
            PreprocessingStepBinding(
                instance_name="sel", step=StepName.SELECT_FIRST
            ),
        )
    )
    trace = _run(definition, TextArtifact(text="just prose, no code at all"))
    assert is_absent(trace.value("output"))
