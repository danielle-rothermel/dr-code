"""Tests for bind_definition wiring and run_preprocessing execution."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from dr_code.eval import (
    PreprocessingConfig,
    PreprocessingDefinition,
    PreprocessingStepBinding,
)
from dr_code.preprocessing.failures import PreprocessingFailureCode
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.runner import (
    BoundPreprocessingRunner,
    BoundStep,
    bind_definition,
    bind_preprocessing,
    run_preprocessing,
)
from dr_code.preprocessing.steps.base import (
    Step,
    StepFailedError,
    StepSettings,
)
from dr_code.trace import (
    Artifact,
    ArtifactKind,
    CodeArtifact,
    CodeCandidateSetArtifact,
    TextArtifact,
    Trace,
    WiringError,
    is_absent,
)


class _FailingTextStep(Step[StepSettings]):
    NAME = StepName.NORMALIZE_UNICODE
    VERSION = "test"
    INPUT = ArtifactKind.TEXT
    OUTPUT = ArtifactKind.TEXT

    def apply(self, value: Artifact):  # noqa: ARG002
        raise StepFailedError(
            "cannot recover this input",
            failure_code=PreprocessingFailureCode.DECODER_OUTPUT_BLANK,
            facts={
                "candidates": ["candidate-0"],
                "reason": {"kind": "unrecoverable"},
            },
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
                instance_name="sel", step=StepName.RETURN_ALL
            ),
        )
    )
    bound = bind_definition(definition.materialize())
    assert len(bound) == 3


def test_bound_runner_is_immutable_and_uses_frozen_config() -> None:
    definition = _def(
        (
            PreprocessingStepBinding(
                instance_name="n", step=StepName.NORMALIZE_UNICODE
            ),
        )
    )
    config = definition.materialize()
    runner = bind_preprocessing(config)

    assert isinstance(runner, BoundPreprocessingRunner)
    with pytest.raises(FrozenInstanceError):
        setattr(runner, "bound_steps", ())
    assert config.steps[0].settings == ()
    assert runner.bound_steps[0].step.settings == StepSettings()


def test_bound_runner_matches_one_shot_execution() -> None:
    definition = _def(
        (
            PreprocessingStepBinding(
                instance_name="n", step=StepName.NORMALIZE_UNICODE
            ),
        )
    )
    input_value = TextArtifact(text="ｄｅｆ")
    config = definition.materialize()

    assert bind_preprocessing(config).run(input_value) == run_preprocessing(
        config, input_value
    )


def test_bound_runner_binds_steps_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = _def(
        (
            PreprocessingStepBinding(
                instance_name="n", step=StepName.NORMALIZE_UNICODE
            ),
        )
    )
    import dr_code.preprocessing.runner as runner_mod

    calls = 0
    original = runner_mod.bind_definition

    def count_bind(
        config: PreprocessingConfig,
    ) -> tuple[BoundStep, ...]:
        nonlocal calls
        calls += 1
        return original(config)

    monkeypatch.setattr(runner_mod, "bind_definition", count_bind)
    runner = runner_mod.bind_preprocessing(definition.materialize())

    assert runner.run(TextArtifact(text="ｄｅｆ")).value(
        "output"
    ) == TextArtifact(text="def")
    assert runner.run(TextArtifact(text="ｇ")).value("output") == TextArtifact(
        text="g"
    )
    assert calls == 1


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


def test_bound_runner_input_kind_mismatch_raises_wiring_error() -> None:
    definition = _def(
        (
            PreprocessingStepBinding(
                instance_name="e", step=StepName.EXTRACT_CANDIDATES
            ),
        )
    )
    # extract_candidates expects Text; pass a CodeArtifact.
    with pytest.raises(WiringError):
        bind_preprocessing(definition.materialize()).run(
            CodeArtifact(source="x = 1")
        )


def test_bound_runner_preserves_structured_step_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = _def(
        (
            PreprocessingStepBinding(
                instance_name="fail", step=StepName.NORMALIZE_UNICODE
            ),
            PreprocessingStepBinding(
                instance_name="downstream",
                step=StepName.NORMALIZE_LINE_ENDINGS,
            ),
        )
    )
    import dr_code.preprocessing.runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "REGISTRY",
        {
            **runner_mod.REGISTRY,
            StepName.NORMALIZE_UNICODE.value: _FailingTextStep,
        },
    )
    trace = bind_preprocessing(definition.materialize()).run(
        TextArtifact(text="x")
    )

    failure = trace.value("fail")
    assert is_absent(failure)
    assert (
        failure.failure_code == PreprocessingFailureCode.DECODER_OUTPUT_BLANK
    )
    assert trace.step_facts["fail"] == {
        "candidates": ["candidate-0"],
        "reason": {"kind": "unrecoverable"},
    }

    propagated = trace.value("downstream")
    assert is_absent(propagated)
    assert (
        propagated.failure_code
        == PreprocessingFailureCode.DECODER_OUTPUT_BLANK
    )


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
                instance_name="sel", step=StepName.RETURN_ALL
            ),
        )
    )
    trace = _run(definition, TextArtifact(text=""))

    extract_val = trace.value("e")
    assert is_absent(extract_val)
    assert extract_val.failed_step == "e"
    assert extract_val.cause == "no code candidates extracted"
    assert extract_val.failure_code == "no_code_candidates"

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


def test_filter_exhaustion_yields_absent_before_return_all() -> None:
    definition = _def(
        (
            PreprocessingStepBinding(
                instance_name="e", step=StepName.EXTRACT_CANDIDATES
            ),
            PreprocessingStepBinding(
                instance_name="id", step=StepName.IDENTIFY_CANDIDATES
            ),
            PreprocessingStepBinding(
                instance_name="flt", step=StepName.FILTER_COMPILABLE
            ),
            PreprocessingStepBinding(
                instance_name="materialize",
                step=StepName.MATERIALIZE_CANDIDATES,
            ),
            PreprocessingStepBinding(
                instance_name="sel", step=StepName.RETURN_ALL
            ),
        )
    )
    trace = _run(definition, TextArtifact(text="def broken(:"))
    out = trace.value("output")
    assert is_absent(out)
    assert out.failed_step == "flt"
    assert out.failure_code == "no_compilable_candidate"
    assert "sel" in out.propagated_through
    # rejection reason recorded as fact
    assert trace.step_facts["flt"]["rejections"][0]["input_index"] == 0


def test_run_step_facts_merged() -> None:
    definition = _def(
        (
            PreprocessingStepBinding(
                instance_name="e", step=StepName.EXTRACT_CANDIDATES
            ),
            PreprocessingStepBinding(
                instance_name="sel", step=StepName.RETURN_ALL
            ),
        )
    )
    trace = _run(
        definition,
        TextArtifact(text="```python\ndef f():\n    return 1\n```"),
    )
    assert trace.step_facts["e"]["candidate_count"] >= 1


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
                instance_name="ri", step=StepName.REPAIR_IMPORT_LINES
            ),
            PreprocessingStepBinding(
                instance_name="dd", step=StepName.DEDUPE_IMPORTS
            ),
            PreprocessingStepBinding(
                instance_name="id", step=StepName.IDENTIFY_CANDIDATES
            ),
            PreprocessingStepBinding(
                instance_name="fc", step=StepName.FILTER_COMPILABLE
            ),
            PreprocessingStepBinding(
                instance_name="materialize",
                step=StepName.MATERIALIZE_CANDIDATES,
            ),
            PreprocessingStepBinding(
                instance_name="sel", step=StepName.RETURN_ALL
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
    assert isinstance(out, CodeCandidateSetArtifact)
    assert any("import numpy as np" in source for source in out.candidates)
    assert any("def f(x):" in source for source in out.candidates)
    assert trace.step_facts["e"]["candidate_count"] >= 1


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
                instance_name="id", step=StepName.IDENTIFY_CANDIDATES
            ),
            PreprocessingStepBinding(
                instance_name="fc", step=StepName.FILTER_COMPILABLE
            ),
            PreprocessingStepBinding(
                instance_name="materialize",
                step=StepName.MATERIALIZE_CANDIDATES,
            ),
            PreprocessingStepBinding(
                instance_name="sel", step=StepName.RETURN_ALL
            ),
        )
    )
    trace = _run(definition, TextArtifact(text=source))
    out = trace.value("output")
    assert isinstance(out, CodeCandidateSetArtifact)
    assert all("def f():" in candidate for candidate in out.candidates)
    assert all(
        validate_python_source(candidate).compile_ok
        for candidate in out.candidates
    )


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
                instance_name="id", step=StepName.IDENTIFY_CANDIDATES
            ),
            PreprocessingStepBinding(
                instance_name="fc", step=StepName.FILTER_COMPILABLE
            ),
            PreprocessingStepBinding(
                instance_name="materialize",
                step=StepName.MATERIALIZE_CANDIDATES,
            ),
            PreprocessingStepBinding(
                instance_name="sel", step=StepName.RETURN_ALL
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
    assert isinstance(out, CodeCandidateSetArtifact)
    assert out.candidates == (expected,)
    assert validate_python_source(out.candidates[0]).compile_ok


def test_pipeline_prose_only_yields_absent() -> None:
    definition = _def(
        (
            PreprocessingStepBinding(
                instance_name="e", step=StepName.EXTRACT_CANDIDATES
            ),
            PreprocessingStepBinding(
                instance_name="sel", step=StepName.RETURN_ALL
            ),
        )
    )
    trace = _run(definition, TextArtifact(text="just prose, no code at all"))
    assert is_absent(trace.value("output"))
