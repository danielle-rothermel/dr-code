"""Phase-2 property tests across the 4100-sample synthetic dataset.

These tests pin down three high-level properties:

1. **Recovery coverage.** The pipeline successfully validates the vast
   majority of samples. ``truncated_midfn`` and ``truncated_and_unfenced``
   are allowed to underperform (documented Phase-2 limitation - mid-string
   truncation can produce unterminated string literals that cannot be
   repaired without inventing content).

2. **Attribution.** For samples that *did* succeed, the union of
   ``extractor_path`` and ``repairs_applied`` on the best valid candidate
   must be a superset of ``expected_recovery_steps``.

3. **Determinism of the config fingerprint.** Two ``LLMCodeValidator``
   instances built with the default config produce the same fingerprint
   string, and validating the same input twice yields the same set of
   candidate ids.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

import code_eval
from code_eval.synthetic.equivalence import equivalent

DATASET_PATH = Path(__file__).parent.parent / "corpus" / "synthetic_dataset.jsonl"

#: Recipes that are *expected* to underperform in Phase 2 (documented).
#: Phase 3 may improve these but they are not blockers for Phase 2 sign-off.
KNOWN_HARD_RECIPES: frozenset[str] = frozenset(
    {
        "truncated_midfn",
        "truncated_and_unfenced",
    }
)

#: Lower bound on overall success rate for non-hard recipes.
MIN_OVERALL_SUCCESS_RATE = 0.98

#: Recipes that historically required Phase-3 normalizers (L0-L5 or
#: orthogonal forms). With Phase 3 wired in step 6 of the pipeline, these
#: are now attributable through ``result.normalizations`` and counted in
#: the main attribution property.
PHASE_3_RECIPES: frozenset[str] = frozenset(
    {
        "comments_noise",
        "dead_code",
        "extra_type_annotations",
        "quote_style_swap",
        "renamed_locals",
        "string_form_swap",
    }
)

#: Recipes where the corruption is already parseable as Python so
#: ``direct_parse`` succeeds without the targeted repair firing. These
#: are attribution misses by design - tracked separately as GitHub issues.
DIRECT_PARSE_BYPASS_RECIPES: frozenset[str] = frozenset(
    {
        "blank_lines_noise",
        "crlf_tabs",
        "duplicated_imports",
        "missing_np_import",
        "smart_quoted",
        "trailing_whitespace",
        "unicode_fullwidth",
    }
)

#: Multi-step corruptions where the earliest extractor wins attribution.
#: Tracked as GitHub issue #7.
KITCHEN_SINK_ATTRIBUTION_RECIPES: frozenset[str] = frozenset(
    {
        "kitchen_sink",
    }
)

#: Prose-wrapped fences where attribution often misses the prose extractor.
FENCED_WITH_PROSE_ATTRIBUTION_RECIPES: frozenset[str] = frozenset(
    {
        "fenced_with_prose",
    }
)

#: Truncation recipes where parseable-boundary samples bypass repair attribution.
TRUNCATION_ATTRIBUTION_RECIPES: frozenset[str] = frozenset(
    {
        "truncated_midfn",
        "truncated_and_unfenced",
    }
)

#: Recipes where semantic equivalence to ground truth is not guaranteed on
#: ``recovery.valid_candidates`` alone (normalization-only or approximate recovery).
SEMANTIC_RECOVERY_EXCLUDED_RECIPES: frozenset[str] = frozenset(
    {
        "dead_code",
        "renamed_locals",
        "string_form_swap",
    }
)

#: Lower bound on aggregate semantic recovery (non-excluded successful samples).
MIN_SEMANTIC_RECOVERY_RATE = 0.98

#: Lower bound on attribution match rate for successful samples on
#: recipes that are *not* in the excluded attribution buckets.
MIN_ATTRIBUTION_RATE = 0.90


pytestmark = pytest.mark.property


def _load_dataset() -> list[dict]:
    if not DATASET_PATH.exists():
        pytest.skip(f"synthetic dataset not found at {DATASET_PATH}")
    return [json.loads(line) for line in DATASET_PATH.read_text().splitlines() if line.strip()]


def _attribution_steps(result: code_eval.ValidationResult) -> set[str]:
    """Union of extractor_path + repairs_applied + successful normalizer
    transformations across all valid candidates."""
    steps: set[str] = set()
    for c in result.recovery.valid_candidates:
        steps.update(c.extractor_path)
        steps.update(c.repairs_applied)
        for form in result.normalizations.get(c.candidate_id, {}).values():
            if form.success:
                if form.transformations_applied:
                    steps.update(form.transformations_applied)
                else:
                    steps.add(form.normalizer.value)
    return steps


def _semantically_recovered(result: code_eval.ValidationResult, ground_truth: str) -> bool:
    """True if any valid candidate or its successful normalization matches ground truth."""
    for c in result.recovery.valid_candidates:
        if equivalent(c.source, ground_truth):
            return True
    for c in result.recovery.valid_candidates:
        for form in result.normalizations.get(c.candidate_id, {}).values():
            if form.success and equivalent(form.source, ground_truth):
                return True
    return False


@pytest.fixture(scope="module")
def dataset() -> list[dict]:
    return _load_dataset()


@pytest.fixture(scope="module")
def validator() -> code_eval.LLMCodeValidator:
    return code_eval.LLMCodeValidator()


@pytest.fixture(scope="module")
def results(
    dataset: list[dict], validator: code_eval.LLMCodeValidator
) -> list[tuple[dict, code_eval.ValidationResult]]:
    return [(sample, validator.validate(sample["corrupted_source"])) for sample in dataset]


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def test_overall_success_rate(
    results: list[tuple[dict, code_eval.ValidationResult]],
) -> None:
    """At least MIN_OVERALL_SUCCESS_RATE of samples produce a valid candidate."""
    total = len(results)
    succeeded = sum(1 for _, r in results if r.recovery.overall_success)
    rate = succeeded / total
    assert rate >= MIN_OVERALL_SUCCESS_RATE, (
        f"only {succeeded}/{total} = {rate:.3f} samples succeeded; "
        f"required >= {MIN_OVERALL_SUCCESS_RATE}"
    )


def test_per_recipe_coverage(
    results: list[tuple[dict, code_eval.ValidationResult]],
) -> None:
    """Each non-hard recipe achieves 100% success. Hard recipes need >= 80%."""
    by_recipe: dict[str, list[bool]] = defaultdict(list)
    for sample, result in results:
        by_recipe[sample["recipe_name"]].append(result.recovery.overall_success)

    failures: list[str] = []
    for recipe, oks in sorted(by_recipe.items()):
        rate = sum(oks) / len(oks)
        floor = 0.80 if recipe in KNOWN_HARD_RECIPES else 1.0
        if rate < floor:
            failures.append(f"{recipe}: {sum(oks)}/{len(oks)} = {rate:.3f} (floor {floor})")
    assert not failures, "Per-recipe coverage regressions:\n  " + "\n  ".join(failures)


def test_semantic_recovery_on_success(
    results: list[tuple[dict, code_eval.ValidationResult]],
) -> None:
    """Successful samples must recover a program equivalent to ground truth."""
    excluded = SEMANTIC_RECOVERY_EXCLUDED_RECIPES | KNOWN_HARD_RECIPES
    hits = 0
    total = 0
    for sample, result in results:
        if sample["recipe_name"] in excluded:
            continue
        if not result.recovery.overall_success:
            continue
        total += 1
        if _semantically_recovered(result, sample["ground_truth_source"]):
            hits += 1
    rate = hits / total if total else 1.0
    assert rate >= MIN_SEMANTIC_RECOVERY_RATE, (
        f"semantic recovery too low: {hits}/{total} = {rate:.3f}; "
        f"required >= {MIN_SEMANTIC_RECOVERY_RATE}"
    )


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


def test_attribution_property_on_successful_samples(
    results: list[tuple[dict, code_eval.ValidationResult]],
) -> None:
    """For successful samples, at least one valid candidate's
    (extractor_path union repairs_applied) is a superset of the expected steps.

    The property test allows MIN_ATTRIBUTION_RATE strictness because some
    recipes (kitchen_sink, two_solutions) intentionally exercise alternate
    routes. The full attribution metric lives in Phase 3.
    """
    by_recipe_hits: dict[str, list[bool]] = defaultdict(list)
    for sample, result in results:
        if not result.recovery.overall_success:
            continue
        expected = set(sample["expected_recovery_steps"])
        if not expected:
            # Clean recipes have no expectations - trivially pass.
            by_recipe_hits[sample["recipe_name"]].append(True)
            continue
        # Compute on selected candidate AND the union across all valid
        # candidates - either may satisfy expectations.
        best = result.recovery.selected_candidate()
        assert best is not None
        best_steps = set(best.extractor_path) | set(best.repairs_applied)
        union_steps = _attribution_steps(result)
        hit = expected.issubset(best_steps) or expected.issubset(union_steps)
        by_recipe_hits[sample["recipe_name"]].append(hit)

    # Aggregate rate across recipes that are attributable. Phase 3
    # normalizers are now wired (step 6), so PHASE_3_RECIPES are counted.
    excluded = (
        DIRECT_PARSE_BYPASS_RECIPES
        | KITCHEN_SINK_ATTRIBUTION_RECIPES
        | FENCED_WITH_PROSE_ATTRIBUTION_RECIPES
        | TRUNCATION_ATTRIBUTION_RECIPES
        | {"clean"}
    )
    total = 0
    hits = 0
    for recipe, oks in by_recipe_hits.items():
        if recipe in excluded:
            continue
        total += len(oks)
        hits += sum(oks)
    rate = hits / total if total else 1.0
    assert rate >= MIN_ATTRIBUTION_RATE, (
        f"attribution rate too low: {hits}/{total} = {rate:.3f}; "
        f"required >= {MIN_ATTRIBUTION_RATE}"
    )


def test_phase3_recipes_attribute_via_normalizers(
    results: list[tuple[dict, code_eval.ValidationResult]],
) -> None:
    """For each Phase-3 recipe, the documented normalizer step appears in
    the attribution union for the majority of successful samples.

    This pins the gain Phase 3 delivered: previously these recipes were
    excluded from attribution; now they should hit via ``normalizations``.
    """
    by_recipe_hits: dict[str, list[bool]] = defaultdict(list)
    for sample, result in results:
        if sample["recipe_name"] not in PHASE_3_RECIPES:
            continue
        if not result.recovery.overall_success:
            continue
        expected = set(sample["expected_recovery_steps"])
        union = _attribution_steps(result)
        by_recipe_hits[sample["recipe_name"]].append(expected.issubset(union))

    failures: list[str] = []
    for recipe, oks in sorted(by_recipe_hits.items()):
        if not oks:
            continue
        rate = sum(oks) / len(oks)
        if rate < 0.80:
            failures.append(f"{recipe}: {sum(oks)}/{len(oks)} = {rate:.3f}")
    assert not failures, "Phase-3 normalizer attribution below 80% for:\n  " + "\n  ".join(failures)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_fingerprint_is_stable_across_instances() -> None:
    v1 = code_eval.LLMCodeValidator()
    v2 = code_eval.LLMCodeValidator()
    assert v1.config_fingerprint == v2.config_fingerprint


def test_repeat_validate_yields_same_candidate_ids(
    validator: code_eval.LLMCodeValidator, dataset: list[dict]
) -> None:
    """Sample 0 (clean) validated twice produces the same candidate ids."""
    sample = dataset[0]
    r1 = validator.validate(sample["corrupted_source"])
    r2 = validator.validate(sample["corrupted_source"])
    ids1 = tuple(c.candidate_id for c in r1.recovery.candidates)
    ids2 = tuple(c.candidate_id for c in r2.recovery.candidates)
    assert ids1 == ids2
