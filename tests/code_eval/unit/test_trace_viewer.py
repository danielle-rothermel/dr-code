"""Tests for the static trace viewer data export."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import code_eval

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "render_trace_viewer.py"


def _load_trace_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("render_trace_viewer", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_trace_links_attempts_to_validation_result() -> None:
    module = _load_trace_module()

    trace = module.build_trace(module.hello_sample(), code_eval.EXTRACTION_CONFIG)

    result = trace["result"]
    candidate_ids = {candidate["candidate_id"] for candidate in result["recovery"]["candidates"]}
    assert trace["extraction"]["candidates"]
    assert trace["attempts"]
    assert {attempt["candidate_id"] for attempt in trace["attempts"]}.issuperset(candidate_ids)
    assert any(attempt["validation"] for attempt in trace["attempts"])
    assert trace["selection"]["best_candidate_id"] in candidate_ids


def test_build_trace_best_candidate_matches_validator() -> None:
    module = _load_trace_module()
    sample = module.hello_sample()

    trace = module.build_trace(sample, code_eval.EXTRACTION_CONFIG)
    expected = code_eval.LLMCodeValidator(config=code_eval.EXTRACTION_CONFIG).validate(
        sample["raw_output"],
        task_id=sample["task_id"],
    )

    best = expected.recovery.selected_candidate()
    assert best is not None
    assert trace["selection"]["best_candidate_id"] == best.candidate_id


def test_trace_viewer_does_not_import_private_recovery_helpers() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "_repair_attempts_for" not in source
    assert "_make_candidate" not in source


def test_render_html_embeds_trace_payload() -> None:
    module = _load_trace_module()
    trace = module.build_trace(module.hello_sample(), code_eval.EXTRACTION_CONFIG)

    html = module.render_html([trace])

    assert 'id="trace-data"' in html
    assert "hello fenced add" in html
    assert "__TRACE_DATA__" not in html
