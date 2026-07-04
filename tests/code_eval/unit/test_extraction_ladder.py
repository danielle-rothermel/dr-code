"""Tests for the static extraction ladder data export."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from code_eval import EXTRACTION_CONFIG
from code_eval.extraction import EXTRACTION_CATALOG, run_extraction

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "render_extraction_ladder.py"


def _load_ladder_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("render_extraction_ladder", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row() -> dict[str, object]:
    return {
        "sample_id": "sample-1",
        "run_id": None,
        "task_id": "HumanEval/0",
        "entry_point": "has_close_elements",
        "raw_output": (
            "Here is code:\n```python\n"
            "def has_close_elements(numbers, threshold):\n"
            "    return False\n```"
        ),
        "provenance_source": "pool",
        "provenance_occurrence_count": 2,
    }


def test_ladder_data_shape_matches_first_stage_extraction() -> None:
    module = _load_ladder_module()
    row = _row()

    data = module.build_ladder_data([row], attempts_path=Path("attempts.parquet"), start_index=0)

    sample = data["samples"][0]
    raw_output = str(row["raw_output"])
    extraction = run_extraction(raw_output, EXTRACTION_CONFIG)

    assert sample["input"]["normalized_output"] == extraction.normalized_output
    assert [extractor_row["extractor"] for extractor_row in sample["passes"]] == [
        name.value for name, _ in EXTRACTION_CATALOG
    ]
    assert all("raw_candidates" in extractor_row for extractor_row in sample["passes"])
    assert all("normalized_candidates" in extractor_row for extractor_row in sample["passes"])
    assert sample["summary"]["total_candidate_count"] == len(extraction.candidates)

    candidates = [
        candidate
        for extractor_row in sample["passes"]
        for candidate in extractor_row["raw_candidates"] + extractor_row["normalized_candidates"]
    ]
    assert candidates
    assert all("is_valid_before_repair" in candidate for candidate in candidates)
    assert all("validation" in candidate for candidate in candidates)


def test_render_html_embeds_ladder_payload() -> None:
    module = _load_ladder_module()
    data = module.build_ladder_data([_row()], attempts_path=Path("attempts.parquet"), start_index=0)

    html = module.render_html(data)

    assert 'id="ladder-data"' in html
    assert "HumanEval/0" in html
    assert "__EXTRACTION_LADDER_JSON__" not in html
