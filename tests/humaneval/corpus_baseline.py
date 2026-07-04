"""Compute the v1 best-effort parser baseline over the corruption corpus.

Shared by the regression test (recompute-and-compare) and the
regenerate entry point (``python tests/humaneval/corpus_baseline.py``).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from dr_code.humaneval.code_parsing import extract_best_effort_code

TESTS_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = TESTS_ROOT / "code_eval" / "corpus" / "synthetic_dataset.jsonl"
BASELINE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "corpus_baseline_v1.json"
)


def compute_corpus_baseline() -> dict[str, Any]:
    per_recipe: dict[str, Counter[str]] = {}
    total = 0
    with CORPUS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            sample = json.loads(line)
            total += 1
            recipe = sample["recipe_name"]
            counts = per_recipe.setdefault(recipe, Counter())
            counts["samples"] += 1
            result = extract_best_effort_code(sample["corrupted_source"])
            if result.extracted_code is not None:
                counts["extracted"] += 1
                counts[f"method:{result.extraction_method}"] += 1
                if result.extracted_code == sample["ground_truth_source"]:
                    counts["exact_ground_truth"] += 1
    return {
        "corpus": CORPUS_PATH.name,
        "total_samples": total,
        "per_recipe": {
            recipe: dict(sorted(counts.items()))
            for recipe, counts in sorted(per_recipe.items())
        },
    }


def main() -> None:
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    baseline = compute_corpus_baseline()
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n")
    extracted = sum(
        counts.get("extracted", 0)
        for counts in baseline["per_recipe"].values()
    )
    print(f"wrote {BASELINE_PATH}")
    print(
        f"baseline: {extracted}/{baseline['total_samples']} extracted: OK"
    )


if __name__ == "__main__":
    main()
