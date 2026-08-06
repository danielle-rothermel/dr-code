from __future__ import annotations

from typing import Final, Never

import pytest
from pydantic import ValidationError

from dr_code.preprocessing import (
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
    BoundPreprocessingRunner,
    PreprocessingDefinition,
    PreprocessingFailureCode,
    StepSpec,
    bind_external_preprocessing,
    bind_preprocessing,
    run_external_preprocessing,
    run_preprocessing,
)
from dr_code.preprocessing.names import StepName
from dr_code.trace import (
    Artifact,
    CodeArtifact,
    ComponentSetting,
    InspectedCodeCandidateSetArtifact,
    TextArtifact,
    Trace,
    WiringError,
    is_absent,
)


def _def(
    steps: tuple[StepSpec, ...],
    definition_id: str = "d1",
) -> PreprocessingDefinition:
    return PreprocessingDefinition(
        definition_id=definition_id,
        version="test-version",
        steps=steps,
    )


def test_bind_resolves_steps() -> None:
    definition = _def(
        (StepSpec(instance_name="n", step=StepName.NORMALIZE_UNICODE),)
    )
    bound = bind_external_preprocessing(definition)
    assert isinstance(bound, BoundPreprocessingRunner)
    assert len(bound.steps) == 1
    assert bound.steps[0].instance_name == "n"
    assert bound.definition == definition


def test_bind_empty_definition() -> None:
    assert bind_external_preprocessing(_def(())).steps == ()


def test_bind_unknown_step_raises_wiring_error() -> None:
    definition = _def(
        (StepSpec(instance_name="n", step=StepName.NORMALIZE_UNICODE),)
    )

    import dr_code.preprocessing.runner as runner_mod

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(runner_mod, "REGISTRY", {})
        with pytest.raises(WiringError):
            bind_external_preprocessing(definition)


def test_step_spec_rejects_bad_settings_at_definition_boundary() -> None:
    with pytest.raises(ValidationError) as exc_info:
        _def(
            (
                StepSpec(
                    instance_name="e",
                    step=StepName.EXPAND_TABS,
                    settings={"tab_width": "not-an-int"},
                ),
            )
        )
    error = exc_info.value.errors()[0]
    assert (error["type"], error["loc"]) == ("int_parsing", ("tab_width",))


def test_bind_broken_kind_chain_raises_wiring_error() -> None:
    definition = _def(
        (
            StepSpec(instance_name="n", step=StepName.NORMALIZE_LINE_ENDINGS),
            StepSpec(instance_name="s", step=StepName.STRIP_FENCES),
        )
    )
    with pytest.raises(WiringError):
        bind_external_preprocessing(definition)


def test_bind_rejects_filter_before_inspection() -> None:
    definition = _def(
        (
            StepSpec(
                instance_name="e",
                step=StepName.EXTRACT_ALL_REPRESENTATIONS,
            ),
            StepSpec(instance_name="f", step=StepName.FILTER_COMPILABLE),
        )
    )
    with pytest.raises(WiringError):
        bind_external_preprocessing(definition)


def test_bind_accepts_valid_kind_chain() -> None:
    definition = _def(
        (
            StepSpec(
                instance_name="e",
                step=StepName.EXTRACT_ALL_REPRESENTATIONS,
            ),
            StepSpec(instance_name="s", step=StepName.STRIP_FENCES),
            StepSpec(instance_name="i", step=StepName.INSPECT_CANDIDATES),
            StepSpec(
                instance_name="m", step=StepName.MATERIALIZE_CANDIDATE_SET
            ),
        )
    )
    assert len(bind_external_preprocessing(definition).steps) == 4


def test_one_binding_runs_many_inputs() -> None:
    bound = bind_external_preprocessing(
        _def((StepSpec(instance_name="n", step=StepName.NORMALIZE_UNICODE),))
    )
    first = bound.run(TextArtifact(text="ｄｅｆ"))
    second = bound.run(TextArtifact(text="ｃｌａｓｓ"))
    assert first.value("output") == TextArtifact(text="def")
    assert second.value("output") == TextArtifact(text="class")

    assert first.value("input") == TextArtifact(text="ｄｅｆ")


def test_one_shot_runner_matches_the_bound_path() -> None:
    definition = _def(
        (StepSpec(instance_name="n", step=StepName.NORMALIZE_UNICODE),)
    )
    value = TextArtifact(text="ｄｅｆ")
    one_shot = run_external_preprocessing(definition, value)
    bound = bind_external_preprocessing(definition).run(value)
    assert one_shot.values == bound.values
    assert one_shot.producer == bound.producer


def test_registered_one_shot_matches_its_binding() -> None:
    value = TextArtifact(text="def f():\n    return 1\n")
    one_shot = run_preprocessing(
        EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION, value
    )
    bound = bind_preprocessing(EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION).run(
        value
    )
    assert one_shot.values == bound.values
    assert one_shot.producer == bound.producer


def test_run_single_text_step() -> None:
    definition = _def(
        (StepSpec(instance_name="n", step=StepName.NORMALIZE_UNICODE),)
    )
    trace = run_external_preprocessing(definition, TextArtifact(text="ｄｅｆ"))
    assert trace.value("output") == TextArtifact(text="def")
    assert trace.value("input") == TextArtifact(text="ｄｅｆ")
    assert trace.value("n") == TextArtifact(text="def")


def test_run_produces_trace_with_producer_stamp() -> None:
    definition = _def(
        (StepSpec(instance_name="n", step=StepName.NORMALIZE_UNICODE),)
    )
    trace = run_external_preprocessing(definition, TextArtifact(text="x"))
    assert isinstance(trace, Trace)
    assert trace.producer.kind == "external_preprocessing"
    assert trace.producer.definition.definition_id == "d1"
    assert trace.producer.definition.version == definition.version
    assert trace.producer.definition.steps[0].component.registered_name == (
        "normalize_unicode"
    )
    assert trace.producer.definition.steps[0].component.version == "0"


def test_registered_bind_rejects_coordinate_impersonation() -> None:
    definition = _def(
        (StepSpec(instance_name="n", step=StepName.NORMALIZE_UNICODE),),
        definition_id="exhaustive-function-candidates",
    ).model_copy(update={"version": "0"})

    with pytest.raises(
        ValueError,
        match="does not match its registered coordinate",
    ):
        bind_preprocessing(definition)


def test_external_binding_accepts_an_unregistered_definition() -> None:
    bound = bind_external_preprocessing(
        _def(
            (StepSpec(instance_name="n", step=StepName.NORMALIZE_UNICODE),),
            definition_id="not-registered",
        )
    )
    assert bound.producer.kind == "external_preprocessing"
    assert bound.run(TextArtifact(text="x")).producer == bound.producer


def test_producer_coordinate_distinguishes_resolved_step_settings() -> None:
    first = _def(
        (
            StepSpec(
                instance_name="tabs",
                step=StepName.EXPAND_TABS,
                settings={"tab_width": 2},
            ),
        )
    )
    second = _def(
        (
            StepSpec(
                instance_name="tabs",
                step=StepName.EXPAND_TABS,
                settings={"tab_width": 8},
            ),
        )
    )

    first_producer = run_external_preprocessing(
        first, TextArtifact(text="\tvalue")
    ).producer
    second_producer = run_external_preprocessing(
        second, TextArtifact(text="\tvalue")
    ).producer

    assert first.definition_id == second.definition_id
    assert first.version == second.version
    assert first_producer != second_producer
    assert first_producer.definition.steps[0].component.settings == (
        ComponentSetting(name="tab_width", value=2),
    )


def test_run_empty_definition_output_equals_input() -> None:
    trace = run_external_preprocessing(_def(()), TextArtifact(text="x"))
    assert trace.value("output") == TextArtifact(text="x")


def test_run_input_kind_mismatch_raises_wiring_error() -> None:
    definition = _def(
        (
            StepSpec(
                instance_name="e",
                step=StepName.EXTRACT_ALL_REPRESENTATIONS,
            ),
        )
    )

    with pytest.raises(WiringError):
        run_external_preprocessing(definition, CodeArtifact(source="x = 1"))


def test_run_unexpected_step_exception_escapes_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound = bind_external_preprocessing(
        _def((StepSpec(instance_name="n", step=StepName.NORMALIZE_UNICODE),))
    )
    unexpected = RuntimeError("unexpected step defect")

    def raise_unexpected(_value: Artifact) -> Never:
        raise unexpected

    monkeypatch.setattr(bound.steps[0].step, "apply", raise_unexpected)

    with pytest.raises(RuntimeError) as exc_info:
        bound.run(TextArtifact(text="x"))

    assert exc_info.value is unexpected


def _short_definition() -> PreprocessingDefinition:
    return _def(
        (
            StepSpec(instance_name="guard", step=StepName.REJECT_BLANK_INPUT),
            StepSpec(
                instance_name="e",
                step=StepName.EXTRACT_ALL_REPRESENTATIONS,
            ),
            StepSpec(instance_name="s", step=StepName.STRIP_FENCES),
        )
    )


def test_run_failed_step_yields_absent_and_complete_trace() -> None:
    trace = run_external_preprocessing(
        _short_definition(), TextArtifact(text="")
    )

    guard_value = trace.value("guard")
    assert is_absent(guard_value)
    assert guard_value.failed_step == "guard"
    assert guard_value.failure_code == PreprocessingFailureCode.BLANK_INPUT
    assert guard_value.cause == "input text is empty or whitespace-only"

    extract_value = trace.value("e")
    assert is_absent(extract_value)
    assert extract_value.failed_step == "guard"

    assert extract_value.failure_code == PreprocessingFailureCode.BLANK_INPUT
    assert "e" in extract_value.propagated_through

    out = trace.value("output")
    assert is_absent(out)
    assert out.failed_step == "guard"
    assert out.propagated_through == ("e", "s")

    assert set(trace.values) == {"input", "guard", "e", "s", "output"}


def test_failure_evidence_lands_in_the_failing_steps_facts() -> None:
    trace = run_external_preprocessing(
        _short_definition(), TextArtifact(text="   ")
    )
    assert trace.step_facts["guard"] == {"input_length": 3}
    absent = trace.value("output")
    assert is_absent(absent)
    assert absent.failure_code == PreprocessingFailureCode.BLANK_INPUT


def test_extraction_failure_evidence_names_every_representation() -> None:
    definition = _def(
        (
            StepSpec(
                instance_name="e",
                step=StepName.EXTRACT_ALL_REPRESENTATIONS,
            ),
        )
    )
    trace = run_external_preprocessing(definition, TextArtifact(text="  \n "))
    assert is_absent(trace.value("output"))
    assert set(trace.step_facts["e"].values()) == {0}


def test_run_step_facts_merged() -> None:
    definition = _def(
        (
            StepSpec(
                instance_name="e",
                step=StepName.EXTRACT_ALL_REPRESENTATIONS,
            ),
        )
    )
    trace = run_external_preprocessing(
        definition,
        TextArtifact(text="```python\ndef f():\n    return 1\n```"),
    )
    assert trace.step_facts["e"]["candidate_count"] >= 1


def _registered_output(raw: str):
    return run_preprocessing(
        EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION, TextArtifact(text=raw)
    ).value("output")


def test_registered_definition_materializes_an_inspected_set() -> None:
    out = _registered_output(
        "Here is the code:\n```python\ndef f(x):\n    return np.array(x)\n```"
    )
    assert isinstance(out, InspectedCodeCandidateSetArtifact)
    assert out.candidates
    accepted = out.candidates[0]
    assert "import numpy as np" in accepted.candidate.source
    assert accepted.inspection.compiles
    assert accepted.inspection.top_level_function_names == ("f",)


def test_registered_definition_blank_input_is_its_own_failure() -> None:
    out = _registered_output("   \n\t ")
    assert is_absent(out)
    assert out.failure_code == PreprocessingFailureCode.BLANK_INPUT


def test_registered_definition_prose_survives_nothing() -> None:
    out = _registered_output("This is an explanation with no code at all.")
    assert is_absent(out)
    assert out.failure_code == (
        PreprocessingFailureCode.NO_CANDIDATE_SURVIVED_FILTERING
    )


def test_registered_definition_keeps_salvage_as_an_extra_candidate() -> None:
    out = _registered_output("def f():\n    return 1\nprint('trailing')\n")
    assert isinstance(out, InspectedCodeCandidateSetArtifact)
    sources = [item.candidate.source for item in out.candidates]
    assert "def f():\n    return 1\nprint('trailing')" in sources
    assert "def f():\n    return 1" in sources


def test_registered_definition_deduplicates_across_representations() -> None:
    out = _registered_output("def f():\n    return 1\n")
    assert isinstance(out, InspectedCodeCandidateSetArtifact)
    (only,) = out.candidates
    operations = [
        origin.operation.operation_name for origin in only.candidate.origins
    ]
    assert operations.count("raw_response") == 1
    assert "text_segments" in operations


# This literal pins the persisted producer coordinate and component versions.
_EXHAUSTIVE_PRODUCER_JSON: Final[dict[str, object]] = {
    "kind": "preprocessing",
    "definition": {
        "definition_id": "exhaustive-function-candidates",
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
                    "settings": [{"name": "tab_width", "value": 4}],
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
                "instance_name": "reject_blank_input",
                "component": {
                    "registered_name": "reject_blank_input",
                    "version": "0",
                    "settings": [],
                },
            },
            {
                "instance_name": "extract_all_representations",
                "component": {
                    "registered_name": "extract_all_representations",
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
                "instance_name": "add_last_return_salvage",
                "component": {
                    "registered_name": "add_last_return_salvage",
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
                "instance_name": "infer_missing_imports",
                "component": {
                    "registered_name": "infer_missing_imports",
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
                "instance_name": "drop_blank_candidates",
                "component": {
                    "registered_name": "drop_blank_candidates",
                    "version": "0",
                    "settings": [],
                },
            },
            {
                "instance_name": "dedupe_candidates",
                "component": {
                    "registered_name": "dedupe_candidates",
                    "version": "0",
                    "settings": [],
                },
            },
            {
                "instance_name": "inspect_candidates",
                "component": {
                    "registered_name": "inspect_candidates",
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
                "instance_name": "filter_top_level_functions",
                "component": {
                    "registered_name": "filter_top_level_functions",
                    "version": "0",
                    "settings": [],
                },
            },
            {
                "instance_name": "materialize_candidate_set",
                "component": {
                    "registered_name": "materialize_candidate_set",
                    "version": "0",
                    "settings": [],
                },
            },
        ],
    },
}


def test_registered_definition_producer_matches_persisted_coordinate() -> None:
    trace = run_preprocessing(
        EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
        TextArtifact(text="def f():\n    return 1\n"),
    )
    assert trace.producer.model_dump(mode="json") == (
        _EXHAUSTIVE_PRODUCER_JSON
    )
