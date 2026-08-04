"""Tests for bind_definition wiring and run_preprocessing execution."""

from __future__ import annotations

from typing import Final

import pytest

from dr_code.preprocessing.definition import (
    PreprocessingDefinition,
    StepSpec,
)
from dr_code.preprocessing.definitions import (
    BEST_EFFORT_DEFINITION,
    FIELD_MARKER_DEFINITION,
)
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.runner import (
    BoundStep,
    _coordinate_settings,
    bind_definition,
    run_external_preprocessing as run_preprocessing,
    run_preprocessing as run_registered_preprocessing,
)
from dr_code.preprocessing.steps.base import StepSettings
from dr_code.trace import (
    CodeArtifact,
    ComponentSetting,
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


def test_bind_unknown_step_raises_wiring_error() -> None:
    definition = _def(
        (StepSpec(instance_name="n", step=StepName.NORMALIZE_UNICODE),)
    )
    # monkeypatch the registry to simulate an unregistered name
    import dr_code.preprocessing.runner as runner_mod

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(runner_mod, "REGISTRY", {})
        with pytest.raises(WiringError):
            bind_definition(definition)


def test_step_spec_rejects_bad_settings_at_definition_boundary() -> None:
    with pytest.raises(Exception):
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
    # (CandidateSet->CandidateSet): Text != CandidateSet.
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
            StepSpec(instance_name="sel", step=StepName.SELECT_FIRST),
        )
    )
    bound = bind_definition(definition)
    assert len(bound) == 3


# --- run: basic execution -------------------------------------------


def test_run_single_text_step() -> None:
    definition = _def(
        (StepSpec(instance_name="n", step=StepName.NORMALIZE_UNICODE),)
    )
    trace = run_preprocessing(definition, TextArtifact(text="ｄｅｆ"))
    assert trace.value("output") == TextArtifact(text="def")
    assert trace.value("input") == TextArtifact(text="ｄｅｆ")
    assert trace.value("n") == TextArtifact(text="def")


def test_run_produces_trace_with_producer_stamp() -> None:
    definition = _def(
        (StepSpec(instance_name="n", step=StepName.NORMALIZE_UNICODE),)
    )
    trace = run_preprocessing(definition, TextArtifact(text="x"))
    assert isinstance(trace, Trace)
    assert trace.producer.kind == "external_preprocessing"
    assert trace.producer.definition.definition_id == "d1"
    assert trace.producer.definition.version == definition.version
    assert trace.producer.definition.steps[0].component.registered_name == (
        "normalize_unicode"
    )
    assert trace.producer.definition.steps[0].component.version == "0"


def test_registered_run_rejects_definition_coordinate_impersonation() -> None:
    definition = _def(
        (StepSpec(instance_name="n", step=StepName.NORMALIZE_UNICODE),),
        definition_id="humaneval-best-effort",
    ).model_copy(update={"version": "0"})

    with pytest.raises(
        ValueError,
        match="does not match its registered coordinate",
    ):
        run_registered_preprocessing(definition, TextArtifact(text="x"))


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

    first_producer = run_preprocessing(
        first, TextArtifact(text="\tvalue")
    ).producer
    second_producer = run_preprocessing(
        second, TextArtifact(text="\tvalue")
    ).producer

    assert first.definition_id == second.definition_id
    assert first.version == second.version
    assert first_producer != second_producer
    assert first_producer.kind == "external_preprocessing"
    assert first_producer.definition.steps[0].component.settings == (
        ComponentSetting(name="tab_width", value=2),
    )


def test_run_empty_definition_output_equals_input() -> None:
    trace = run_preprocessing(_def(()), TextArtifact(text="x"))
    assert trace.value("output") == TextArtifact(text="x")


def test_run_input_kind_mismatch_raises_wiring_error() -> None:
    definition = _def(
        (StepSpec(instance_name="e", step=StepName.EXTRACT_CANDIDATES),)
    )
    # extract_candidates expects Text; pass a CodeArtifact.
    with pytest.raises(WiringError):
        run_preprocessing(definition, CodeArtifact(source="x = 1"))


# --- run: Absent propagation -----------------------------------------


def test_run_failed_step_yields_absent_and_complete_trace() -> None:
    definition = _def(
        (
            StepSpec(instance_name="e", step=StepName.EXTRACT_CANDIDATES),
            StepSpec(instance_name="s", step=StepName.STRIP_FENCES),
            StepSpec(instance_name="sel", step=StepName.SELECT_FIRST),
        )
    )
    trace = run_preprocessing(definition, TextArtifact(text=""))

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
            StepSpec(instance_name="e", step=StepName.EXTRACT_CANDIDATES),
            StepSpec(instance_name="flt", step=StepName.FILTER_COMPILABLE),
            StepSpec(instance_name="sel", step=StepName.SELECT_FIRST),
        )
    )
    trace = run_preprocessing(definition, TextArtifact(text="def broken(:"))
    out = trace.value("output")
    assert is_absent(out)
    assert out.failed_step == "sel"
    assert out.cause == "no candidate survived filtering"
    # rejection reason recorded as fact
    assert "rejected_0" in trace.step_facts["flt"]


def test_run_step_facts_merged() -> None:
    definition = _def(
        (
            StepSpec(instance_name="e", step=StepName.EXTRACT_CANDIDATES),
            StepSpec(instance_name="sel", step=StepName.SELECT_FIRST),
        )
    )
    trace = run_preprocessing(
        definition,
        TextArtifact(text="```python\ndef f():\n    return 1\n```"),
    )
    assert trace.step_facts["e"] == {"alternative": "fenced_blocks"}


def test_run_propagated_absent_records_each_step() -> None:
    definition = _def(
        (
            StepSpec(instance_name="e", step=StepName.EXTRACT_CANDIDATES),
            StepSpec(instance_name="a", step=StepName.STRIP_FENCES),
            StepSpec(instance_name="b", step=StepName.DEDENT_CANDIDATES),
            StepSpec(instance_name="c", step=StepName.SPLIT_ON_NAME_GUARD),
        )
    )
    trace = run_preprocessing(definition, TextArtifact(text=""))
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
            StepSpec(instance_name="rr", step=StepName.DROP_AFTER_LAST_RETURN),
            StepSpec(instance_name="ri", step=StepName.REPAIR_IMPORT_LINES),
            StepSpec(instance_name="ii", step=StepName.INFER_MISSING_IMPORTS),
            StepSpec(instance_name="dd", step=StepName.DEDUPE_IMPORTS),
            StepSpec(instance_name="fc", step=StepName.FILTER_COMPILABLE),
            StepSpec(instance_name="sel", step=StepName.SELECT_FIRST),
        )
    )
    raw = (
        "Here is the code:\n"
        "```python\n"
        "def f(x):\n    return np.array(x)\n```\n"
    )
    trace = run_preprocessing(definition, TextArtifact(text=raw))
    out = trace.value("output")
    assert isinstance(out, CodeArtifact)
    assert "import numpy as np" in out.source
    assert "def f(x):" in out.source
    assert trace.step_facts["e"] == {"alternative": "fenced_blocks"}


def test_pipeline_prose_only_yields_absent() -> None:
    definition = _def(
        (
            StepSpec(instance_name="e", step=StepName.EXTRACT_CANDIDATES),
            StepSpec(instance_name="sel", step=StepName.SELECT_FIRST),
        )
    )
    trace = run_preprocessing(
        definition, TextArtifact(text="just prose, no code at all")
    )
    assert is_absent(trace.value("output"))


# --- persisted producer coordinate (wire-format contract) ------------

# The dicts below pin the exact persisted producer coordinate of every
# registered definition: definition ids, versions, instance names,
# registered component names, and every setting name and value. Setting
# names are derived from Python field names, so a field rename silently
# rewrites stored trace identity. A failure here means the wire format
# changed and must be a deliberate, versioned decision — never a
# mechanical test update.

_BEST_EFFORT_PRODUCER_JSON: Final[dict[str, object]] = {
    "kind": "preprocessing",
    "definition": {
        "definition_id": "humaneval-best-effort",
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
                "instance_name": "extract_candidates",
                "component": {
                    "registered_name": "extract_candidates",
                    "version": "0",
                    "settings": [
                        {
                            "name": "alternatives",
                            "value": [
                                "fenced_blocks",
                                "markdown_wrapper",
                                "escaped_python",
                                "escaped_markdown_wrapper",
                            ],
                        }
                    ],
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
                "instance_name": "drop_after_last_return",
                "component": {
                    "registered_name": "drop_after_last_return",
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
                "instance_name": "select_first",
                "component": {
                    "registered_name": "select_first",
                    "version": "0",
                    "settings": [],
                },
            },
        ],
    },
}

_FIELD_MARKER_PRODUCER_JSON: Final[dict[str, object]] = {
    "kind": "preprocessing",
    "definition": {
        "definition_id": "humaneval-field-marker",
        "version": "0",
        "steps": [
            {
                "instance_name": "field_marker_extract",
                "component": {
                    "registered_name": "field_marker_extract",
                    "version": "0",
                    "settings": [{"name": "field_name", "value": "code"}],
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
                "instance_name": "select_first",
                "component": {
                    "registered_name": "select_first",
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
        (BEST_EFFORT_DEFINITION, _BEST_EFFORT_PRODUCER_JSON),
        (FIELD_MARKER_DEFINITION, _FIELD_MARKER_PRODUCER_JSON),
    ],
    ids=["best-effort", "field-marker"],
)
def test_registered_definition_producer_matches_persisted_coordinate(
    definition: PreprocessingDefinition,
    expected: dict[str, object],
) -> None:
    trace = run_registered_preprocessing(
        definition, TextArtifact(text="def f():\n    return 1\n")
    )
    assert trace.producer.model_dump(mode="json") == expected


# --- settings projection: tuple support and rejected shapes ----------


def test_coordinate_settings_rejects_non_string_tuple() -> None:
    class _IntTupleSettings(StepSettings):
        alternatives: tuple[int, ...] = (1, 2)

    with pytest.raises(
        TypeError,
        match="unsupported persisted tuple setting for 'alternatives'",
    ):
        _coordinate_settings(_IntTupleSettings())


def test_coordinate_settings_rejects_unsupported_value_type() -> None:
    class _MappingSettings(StepSettings):
        mapping: dict[str, str] = {}

    with pytest.raises(
        TypeError,
        match="unsupported persisted setting shape for 'mapping': dict",
    ):
        _coordinate_settings(_MappingSettings())
