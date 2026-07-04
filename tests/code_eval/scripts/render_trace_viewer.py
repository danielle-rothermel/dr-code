"""Render a static HTML trace viewer for validator runs.

Run::

    uv run python scripts/render_trace_viewer.py --out trace_viewer.html --full-normalization
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import code_eval
from code_eval.config import DEFAULT_CONFIG, EXTRACTION_CONFIG, ValidatorConfig
from code_eval.models.extracted_candidate import ExtractedCandidate

ROOT: Final[Path] = Path(__file__).resolve().parent.parent
POOL_PATH: Final[Path] = ROOT / "tests" / "corpus" / "pool_samples.jsonl"
SYNTHETIC_PATH: Final[Path] = ROOT / "tests" / "corpus" / "synthetic_dataset.jsonl"
TEMPLATE_PATH: Final[Path] = Path(__file__).with_name("trace_viewer_template.html")
DEFAULT_OUTPUT: Final[Path] = ROOT / "trace_viewer.html"
JSON_PLACEHOLDER: Final[str] = "__TRACE_DATA__"
DEFAULT_POOL_INDICES: Final[tuple[int, ...]] = (0, 36)
DEFAULT_RECIPE_SELECTORS: Final[tuple[str, ...]] = ("smart_quoted:0", "comments_noise:0")
HELLO_SOURCE: Final[str] = """```python
def add(a: int, b: int) -> int:
    return a + b
```"""


def hello_sample() -> dict[str, object]:
    return {
        "kind": "hello",
        "label": "hello fenced add",
        "raw_output": HELLO_SOURCE,
        "task_id": "hello",
    }


def parse_recipe_selector(selector: str) -> tuple[str, int]:
    recipe, sep, index_text = selector.partition(":")
    if not recipe:
        raise ValueError("recipe selector must start with a recipe name")
    if not sep:
        return recipe, 0
    return recipe, int(index_text)


def load_pool_sample(index: int) -> dict[str, object]:
    rows = [json.loads(line) for line in POOL_PATH.read_text().splitlines() if line.strip()]
    sample = rows[index]
    return {
        "kind": "pool",
        "label": f"pool[{index}] {sample['pattern']} {sample['task_id']}",
        "raw_output": sample["raw_output"],
        "task_id": sample["task_id"],
        "pattern": sample["pattern"],
        "expect_success": sample["expect_success"],
        "pool_index": index,
    }


def load_recipe_sample(selector: str) -> dict[str, object]:
    recipe, index = parse_recipe_selector(selector)
    rows = [json.loads(line) for line in SYNTHETIC_PATH.read_text().splitlines() if line.strip()]
    matches = [row for row in rows if row["recipe_name"] == recipe]
    if not matches:
        recipes = ", ".join(sorted({row["recipe_name"] for row in rows}))
        raise ValueError(f"unknown recipe {recipe!r}; available: {recipes}")
    sample = matches[index]
    return {
        "kind": "synthetic",
        "label": f"synthetic {recipe}[{index}]",
        "raw_output": sample["corrupted_source"],
        "task_id": sample["sample_id"],
        "recipe_name": sample["recipe_name"],
        "humaneval_task_id": sample["humaneval_task_id"],
        "ground_truth_source": sample["ground_truth_source"],
        "expected_recovery_steps": sample["expected_recovery_steps"],
        "expected_extractor_path_contains": sample["expected_extractor_path_contains"],
        "synthetic_index": index,
    }


def _union_steps(result: code_eval.ValidationResult) -> set[str]:
    steps: set[str] = set()
    for candidate in result.recovery.valid_candidates:
        steps.update(candidate.extractor_path)
        steps.update(candidate.repairs_applied)
        for form in result.normalizations.get(candidate.candidate_id, {}).values():
            if form.success and form.source != candidate.source:
                steps.update(form.transformations_applied)
    return steps


def _extracted_to_dict(index: int, candidate: ExtractedCandidate) -> dict[str, object]:
    return {
        "extracted_id": f"e{index:03d}",
        "extractor": candidate.extractor.value,
        "extractor_path": list(candidate.extractor_path),
        "notes": candidate.notes,
        "source": candidate.source,
        "source_chars": len(candidate.source),
    }


def build_trace(sample: dict[str, object], config: ValidatorConfig) -> dict[str, object]:
    raw_output = str(sample["raw_output"])
    task_id = str(sample.get("task_id") or sample["label"])
    validator = code_eval.LLMCodeValidator(config=config)
    result = validator.validate(raw_output, task_id=task_id)
    extraction = result.extraction
    normalized = extraction.normalized_output
    extracted = extraction.candidates

    extracted_items: list[dict[str, object]] = []
    for extracted_index, extracted_candidate in enumerate(extracted):
        extracted_items.append(_extracted_to_dict(extracted_index, extracted_candidate))

    attempts = [
        {
            **attempt.model_dump(mode="json"),
            "extracted_id": f"e{attempt.extracted_index:03d}",
        }
        for attempt in result.recovery.attempts
    ]

    best = result.recovery.selected_candidate()
    expected_steps = set(sample.get("expected_recovery_steps") or [])
    union_steps = _union_steps(result)
    rank_rows = [
        {
            "candidate_id": rank.candidate_id,
            "attempt_id": rank.attempt_id,
            "rank": list(rank.rank_key),
            "is_best": best is not None and rank.candidate_id == best.candidate_id,
        }
        for rank in result.recovery.selection.ranked_valid_candidates
    ]

    return {
        "meta": {
            "trace_id": task_id,
            "label": sample["label"],
            "source_kind": sample["kind"],
            "generated_at": datetime.now(UTC).isoformat(),
            "config_fingerprint": result.config_fingerprint,
            "tool_versions": result.tool_versions,
            "full_normalization": bool(config.normalizers),
        },
        "sample": {key: value for key, value in sample.items() if key != "raw_output"},
        "input": {
            "raw_source": raw_output,
            "text_normalized_source": normalized,
            "text_normalized_changed": raw_output != normalized,
        },
        "extraction": {
            "log": [step.model_dump(mode="json") for step in extraction.extraction_log],
            "candidates": extracted_items,
        },
        "attempts": attempts,
        "result": result.model_dump(mode="json"),
        "selection": {
            "best_candidate_id": best.candidate_id if best else None,
            "best_candidate_source": best.source if best else None,
            "ranked_valid_candidates": rank_rows,
        },
        "summary": {
            "overall_success": result.recovery.overall_success,
            "candidate_count": len(result.recovery.candidates),
            "valid_candidate_count": len(result.recovery.valid_candidates),
            "extracted_count": len(extracted),
            "attempt_count": len(attempts),
            "expected_steps": sorted(expected_steps),
            "observed_steps": sorted(union_steps),
            "missing_expected_steps": sorted(expected_steps - union_steps),
        },
    }


def select_samples(args: argparse.Namespace) -> list[dict[str, object]]:
    if not args.no_defaults and not args.pool and not args.recipe and not args.hello:
        samples = [hello_sample()]
        samples.extend(load_pool_sample(index) for index in DEFAULT_POOL_INDICES)
        samples.extend(load_recipe_sample(selector) for selector in DEFAULT_RECIPE_SELECTORS)
        return samples

    samples: list[dict[str, object]] = []
    if args.hello:
        samples.append(hello_sample())
    samples.extend(load_pool_sample(index) for index in args.pool)
    samples.extend(load_recipe_sample(selector) for selector in args.recipe)
    return samples


def render_html(traces: list[dict[str, object]]) -> str:
    template = TEMPLATE_PATH.read_text()
    payload = json.dumps({"traces": traces}, ensure_ascii=False).replace("</", "<\\/")
    return template.replace(JSON_PLACEHOLDER, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a static code-eval trace viewer.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pool", type=int, action="append", default=[])
    parser.add_argument("--recipe", action="append", default=[], help="recipe or recipe:index")
    parser.add_argument("--hello", action="store_true")
    parser.add_argument("--no-defaults", action="store_true")
    parser.add_argument("--full-normalization", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    samples = select_samples(args)
    if not samples:
        raise SystemExit("no samples selected")
    config = DEFAULT_CONFIG if args.full_normalization else EXTRACTION_CONFIG
    traces = [build_trace(sample, config) for sample in samples]
    args.out.write_text(render_html(traces))
    print(f"wrote {args.out} ({len(traces)} traces)")


if __name__ == "__main__":
    main()
