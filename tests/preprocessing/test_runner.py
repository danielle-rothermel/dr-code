"""Tests for bind_definition wiring and preprocessing execution."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Final

import pytest

from dr_code.preprocessing.definition import (
    PreprocessingDefinition,
    StepSpec,
)
from dr_code.preprocessing.definitions import (
    _DEFINITIONS,
    DEFINITION_VERSION,
    HUMANEVAL_FUNCTION_CANDIDATES_DEFINITION_ID,
    HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
)
from dr_code.preprocessing.failures import PreprocessingFailureCode
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.runner import (
    BoundPreprocessingRunner,
    BoundStep,
    bind_definition,
    bind_external_preprocessing,
    run_external_preprocessing,
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
    coordinate_settings,
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
    steps: tuple[StepSpec, ...],
    definition_id: str = "d1",
) -> PreprocessingDefinition:
    return PreprocessingDefinition(
        definition_id=definition_id, version="1", steps=steps
    )


def _run(definition: PreprocessingDefinition, input_value):
    return run_external_preprocessing(definition, input_value)


# --- bind-time wiring ------------------------------------------------


def test_bind_resolves_steps() -> None:
    definition = _def(
        (StepSpec(instance_name="n", step=StepName.NORMALIZE_UNICODE),)
    )
    bound = bind_definition(definition)
    assert len(bound) == 1
    assert isinstance(bound[0], BoundStep)
    assert bound[0].instance_name == "n"


def test_bind_empty_definition() -> None:
    assert bind_definition(_def(())) == ()


def test_bind_detects_registry_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = _def(
        (StepSpec(instance_name="n", step=StepName.NORMALIZE_UNICODE),)
    )
    # monkeypatch the registry to simulate an unregistered name
    import dr_code.preprocessing.runner as runner_mod

    config = definition
    monkeypatch.setattr(runner_mod, "REGISTRY", {})
    with pytest.raises(WiringError, match="unknown step"):
        bind_definition(config)


def test_definition_rejects_bad_settings() -> None:
    with pytest.raises(ValueError):
        _def(
            (
                StepSpec(
                    instance_name="e",
                    step=StepName.EXPAND_TABS,
                    settings={"tab_width": "not-an-int"},
                ),
            )
        )


def test_bind_broken_kind_chain_raises_wiring_error() -> None:
    # normalize_line_endings (Text->Text) then strip_fences
    # (CandidateSet->CandidateSet): Text != CandidateSet. A definition may
    # name any step sequence; whether the sequence composes is decided at
    # bind time, before any input is processed.
    definition = _def(
        (
            StepSpec(instance_name="n", step=StepName.NORMALIZE_LINE_ENDINGS),
            StepSpec(instance_name="s", step=StepName.STRIP_FENCES),
        )
    )
    with pytest.raises(WiringError):
        bind_definition(definition)


def test_bind_accepts_valid_kind_chain() -> None:
    definition = _def(
        (
            StepSpec(instance_name="e", step=StepName.EXTRACT_CANDIDATES),
            StepSpec(instance_name="s", step=StepName.STRIP_FENCES),
            StepSpec(instance_name="sel", step=StepName.RETURN_ALL),
        )
    )
    bound = bind_definition(definition)
    assert len(bound) == 3


def test_bound_runner_is_immutable_and_uses_frozen_config() -> None:
    definition = _def(
        (StepSpec(instance_name="n", step=StepName.NORMALIZE_UNICODE),)
    )
    runner = bind_external_preprocessing(definition)

    assert isinstance(runner, BoundPreprocessingRunner)
    with pytest.raises(FrozenInstanceError):
        setattr(runner, "bound_steps", ())
    assert definition.steps[0].settings == StepSettings()
    assert runner.bound_steps[0].step.settings == StepSettings()


def test_bound_runner_matches_one_shot_execution() -> None:
    definition = _def(
        (StepSpec(instance_name="n", step=StepName.NORMALIZE_UNICODE),)
    )
    input_value = TextArtifact(text="ｄｅｆ")

    assert bind_external_preprocessing(definition).run(
        input_value
    ) == run_external_preprocessing(definition, input_value)


def test_bound_runner_binds_steps_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = _def(
        (StepSpec(instance_name="n", step=StepName.NORMALIZE_UNICODE),)
    )
    import dr_code.preprocessing.runner as runner_mod

    calls = 0
    original = runner_mod.bind_definition

    def count_bind(
        config: PreprocessingDefinition,
    ) -> tuple[BoundStep, ...]:
        nonlocal calls
        calls += 1
        return original(config)

    monkeypatch.setattr(runner_mod, "bind_definition", count_bind)
    runner = runner_mod.bind_external_preprocessing(definition)

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
        (StepSpec(instance_name="n", step=StepName.NORMALIZE_UNICODE),)
    )
    trace = _run(definition, TextArtifact(text="ｄｅｆ"))
    assert trace.value("output") == TextArtifact(text="def")
    assert trace.value("input") == TextArtifact(text="ｄｅｆ")
    assert trace.value("n") == TextArtifact(text="def")


def test_run_produces_trace_with_producer_stamp() -> None:
    definition = _def(
        (StepSpec(instance_name="n", step=StepName.NORMALIZE_UNICODE),)
    )
    trace = _run(definition, TextArtifact(text="x"))
    assert isinstance(trace, Trace)
    assert trace.producer.kind == "external_preprocessing"
    assert trace.producer.definition.definition_id == "d1"
    assert trace.producer.definition.version == definition.version
    assert trace.producer.definition.steps[0].component.registered_name == (
        "normalize_unicode"
    )
    assert trace.producer.definition.steps[0].component.version == "0"


def test_run_empty_definition_output_equals_input() -> None:
    trace = _run(_def(()), TextArtifact(text="x"))
    assert trace.value("output") == TextArtifact(text="x")


def test_bound_runner_input_kind_mismatch_raises_wiring_error() -> None:
    definition = _def(
        (StepSpec(instance_name="e", step=StepName.EXTRACT_CANDIDATES),)
    )
    # extract_candidates expects Text; pass a CodeArtifact.
    with pytest.raises(WiringError):
        bind_external_preprocessing(definition).run(
            CodeArtifact(source="x = 1")
        )


def test_bound_runner_preserves_structured_step_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = _def(
        (
            StepSpec(instance_name="fail", step=StepName.NORMALIZE_UNICODE),
            StepSpec(
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
    trace = bind_external_preprocessing(definition).run(TextArtifact(text="x"))

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
            StepSpec(instance_name="e", step=StepName.EXTRACT_CANDIDATES),
            StepSpec(instance_name="s", step=StepName.STRIP_FENCES),
            StepSpec(instance_name="sel", step=StepName.RETURN_ALL),
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
            StepSpec(instance_name="e", step=StepName.EXTRACT_CANDIDATES),
            StepSpec(instance_name="id", step=StepName.IDENTIFY_CANDIDATES),
            StepSpec(instance_name="flt", step=StepName.FILTER_COMPILABLE),
            StepSpec(
                instance_name="materialize",
                step=StepName.MATERIALIZE_CANDIDATES,
            ),
            StepSpec(instance_name="sel", step=StepName.RETURN_ALL),
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
            StepSpec(instance_name="e", step=StepName.EXTRACT_CANDIDATES),
            StepSpec(instance_name="sel", step=StepName.RETURN_ALL),
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
            StepSpec(instance_name="e", step=StepName.EXTRACT_CANDIDATES),
            StepSpec(instance_name="a", step=StepName.STRIP_FENCES),
            StepSpec(instance_name="b", step=StepName.DEDENT_CANDIDATES),
            StepSpec(instance_name="c", step=StepName.SPLIT_ON_NAME_GUARD),
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
            StepSpec(instance_name="e", step=StepName.EXTRACT_CANDIDATES),
            StepSpec(instance_name="sf", step=StepName.STRIP_FENCES),
            StepSpec(instance_name="ng", step=StepName.SPLIT_ON_NAME_GUARD),
            StepSpec(instance_name="ri", step=StepName.REPAIR_IMPORT_LINES),
            StepSpec(instance_name="dd", step=StepName.DEDUPE_IMPORTS),
            StepSpec(instance_name="id", step=StepName.IDENTIFY_CANDIDATES),
            StepSpec(instance_name="fc", step=StepName.FILTER_COMPILABLE),
            StepSpec(
                instance_name="materialize",
                step=StepName.MATERIALIZE_CANDIDATES,
            ),
            StepSpec(instance_name="sel", step=StepName.RETURN_ALL),
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


def test_pipeline_prose_only_yields_absent() -> None:
    definition = _def(
        (
            StepSpec(instance_name="e", step=StepName.EXTRACT_CANDIDATES),
            StepSpec(instance_name="sel", step=StepName.RETURN_ALL),
        )
    )
    trace = _run(definition, TextArtifact(text="just prose, no code at all"))
    assert is_absent(trace.value("output"))


# --- persisted producer coordinate (wire-format contract) ------------

# Contract pin. The dict below pins the exact persisted producer
# coordinate of the registered definition: definition id, version,
# instance names, registered component names, and every setting name and
# value. Setting names are derived from Python field names, so a field
# rename silently rewrites stored trace identity. A failure here means the
# wire format changed and must be a deliberate, versioned decision — never
# a mechanical test update.

_HUMANEVAL_FUNCTION_CANDIDATES_PRODUCER_JSON: Final[dict[str, object]] = {
    "kind": "preprocessing",
    "definition": {
        "definition_id": "humaneval-function-candidates",
        "version": "0",
        "steps": [
            {
                "instance_name": "normalize_line_endings",
                "component": {
                    "registered_name": "normalize_line_endings",
                    "version": "0",
                    "settings": [],
                },
            },
            {
                "instance_name": "normalize_unicode",
                "component": {
                    "registered_name": "normalize_unicode",
                    "version": "0",
                    "settings": [],
                },
            },
            {
                "instance_name": "expand_tabs",
                "component": {
                    "registered_name": "expand_tabs",
                    "version": "0",
                    "settings": [
                        {
                            "name": "tab_width",
                            "value": 4,
                        },
                    ],
                },
            },
            {
                "instance_name": "strip_trailing_whitespace",
                "component": {
                    "registered_name": "strip_trailing_whitespace",
                    "version": "0",
                    "settings": [],
                },
            },
            {
                "instance_name": "collapse_blank_runs",
                "component": {
                    "registered_name": "collapse_blank_runs",
                    "version": "0",
                    "settings": [],
                },
            },
            {
                "instance_name": "trim_outer_blanks",
                "component": {
                    "registered_name": "trim_outer_blanks",
                    "version": "0",
                    "settings": [],
                },
            },
            {
                "instance_name": "require_nonblank_text",
                "component": {
                    "registered_name": "require_nonblank_text",
                    "version": "0",
                    "settings": [],
                },
            },
            {
                "instance_name": "extract_candidates",
                "component": {
                    "registered_name": "extract_candidates",
                    "version": "0",
                    "settings": [],
                },
            },
            {
                "instance_name": "strip_fences",
                "component": {
                    "registered_name": "strip_fences",
                    "version": "0",
                    "settings": [],
                },
            },
            {
                "instance_name": "dedent",
                "component": {
                    "registered_name": "dedent_candidates",
                    "version": "0",
                    "settings": [],
                },
            },
            {
                "instance_name": "normalize_smart_quotes",
                "component": {
                    "registered_name": "normalize_smart_quotes",
                    "version": "0",
                    "settings": [],
                },
            },
            {
                "instance_name": "split_on_name_guard",
                "component": {
                    "registered_name": "split_on_name_guard",
                    "version": "0",
                    "settings": [],
                },
            },
            {
                "instance_name": "expand_last_return_salvage",
                "component": {
                    "registered_name": "expand_last_return_salvage",
                    "version": "0",
                    "settings": [],
                },
            },
            {
                "instance_name": "repair_import_lines",
                "component": {
                    "registered_name": "repair_import_lines",
                    "version": "0",
                    "settings": [],
                },
            },
            {
                "instance_name": "dedupe_imports",
                "component": {
                    "registered_name": "dedupe_imports",
                    "version": "0",
                    "settings": [],
                },
            },
            {
                "instance_name": "filter_nonblank_candidates",
                "component": {
                    "registered_name": "filter_nonblank_candidates",
                    "version": "0",
                    "settings": [],
                },
            },
            {
                "instance_name": "identify_candidates",
                "component": {
                    "registered_name": "identify_candidates",
                    "version": "0",
                    "settings": [],
                },
            },
            {
                "instance_name": "filter_plain_literal",
                "component": {
                    "registered_name": "filter_plain_literal",
                    "version": "0",
                    "settings": [],
                },
            },
            {
                "instance_name": "filter_code_repr",
                "component": {
                    "registered_name": "filter_code_repr",
                    "version": "0",
                    "settings": [],
                },
            },
            {
                "instance_name": "filter_compilable",
                "component": {
                    "registered_name": "filter_compilable",
                    "version": "0",
                    "settings": [],
                },
            },
            {
                "instance_name": "filter_has_top_level_function",
                "component": {
                    "registered_name": "filter_has_top_level_function",
                    "version": "0",
                    "settings": [],
                },
            },
            {
                "instance_name": "materialize_candidates",
                "component": {
                    "registered_name": "materialize_candidates",
                    "version": "0",
                    "settings": [],
                },
            },
            {
                "instance_name": "return_all",
                "component": {
                    "registered_name": "return_all",
                    "version": "0",
                    "settings": [],
                },
            },
        ],
    },
}


@pytest.mark.parametrize(
    ("definition", "expected"),
    [
        (
            HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
            _HUMANEVAL_FUNCTION_CANDIDATES_PRODUCER_JSON,
        ),
    ],
    ids=["humaneval-function-candidates"],
)
def test_registered_definition_producer_matches_persisted_coordinate(
    definition: PreprocessingDefinition,
    expected: dict[str, object],
) -> None:
    trace = run_preprocessing(
        definition, TextArtifact(text="def f():\n    return 1\n")
    )
    assert trace.producer.model_dump(mode="json") == expected


def test_every_registered_definition_has_a_pinned_producer_golden() -> None:
    # A new registered definition without a golden would persist producer
    # coordinates that nothing pins.
    assert set(_DEFINITIONS) == {
        (
            HUMANEVAL_FUNCTION_CANDIDATES_DEFINITION_ID,
            DEFINITION_VERSION,
        )
    }


def test_registered_run_rejects_definition_coordinate_impersonation() -> None:
    definition = _def(
        (StepSpec(instance_name="n", step=StepName.NORMALIZE_UNICODE),),
        definition_id=HUMANEVAL_FUNCTION_CANDIDATES_DEFINITION_ID,
    ).model_copy(update={"version": DEFINITION_VERSION})

    with pytest.raises(
        ValueError,
        match="does not match its registered coordinate",
    ):
        run_preprocessing(definition, TextArtifact(text="x"))


# --- settings projection: tuple support and rejected shapes ----------


def test_coordinate_settings_rejects_non_string_tuple() -> None:
    class _IntTupleSettings(StepSettings):
        alternatives: tuple[int, ...] = (1, 2)

    with pytest.raises(
        TypeError,
        match="unsupported persisted tuple setting for 'alternatives'",
    ):
        coordinate_settings(_IntTupleSettings())


def test_coordinate_settings_rejects_unsupported_value_type() -> None:
    class _MappingSettings(StepSettings):
        mapping: dict[str, str] = {}

    with pytest.raises(
        TypeError,
        match="unsupported persisted setting shape for 'mapping': dict",
    ):
        coordinate_settings(_MappingSettings())
