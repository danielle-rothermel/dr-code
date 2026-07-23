"""End-to-end generation: oracle correctness, determinism, distinctness.

These run the real subprocess oracle over a few pinned HumanEval+ tasks from
the offline snapshot. Kept to a small task/seed/input budget for speed.
"""

from __future__ import annotations

from pathlib import Path

from dr_code.mutants.dataset import load_records, save_dataset
from dr_code.mutants.generate import generate_mutants
from dr_code.mutants.oracle import (
    OutcomeKind,
    distinct_input_indices,
    run_program_on_inputs,
)
from dr_code.mutants.operators import ALL_FAMILIES, OperatorFamily

_TASKS = ("HumanEval/0", "HumanEval/2", "HumanEval/7")


def _small_generate(**overrides):
    kwargs = dict(
        families=ALL_FAMILIES,
        seeds=2,
        max_inputs_per_mutant=20,
        timeout_seconds=5.0,
        task_filter=_TASKS,
        prefer_snapshot=True,
    )
    kwargs.update(overrides)
    return generate_mutants(**kwargs)


def test_generation_produces_accepted_distinct_mutants() -> None:
    records, manifest = _small_generate()
    assert manifest.accepted_count == len(records)
    assert records, "expected at least one accepted mutant on pinned tasks"
    for record in records:
        # Acceptance gate: behaviorally distinct on at least one input.
        assert record.distinct_input_count >= 1
        assert len(record.mutant_expected) == len(record.input_reprs)
        assert len(record.canonical_expected) == len(record.input_reprs)


def test_mutant_matches_its_own_oracle_and_diverges_from_canonical() -> None:
    # Core oracle-derivation correctness: re-running the mutant reproduces its
    # recorded expected outputs (its own suite), and on the distinct indices it
    # differs from the canonical expected outputs.
    records, _ = _small_generate()
    record = records[0]
    inputs = tuple(eval(r) for r in record.input_reprs)  # noqa: S307
    replay = run_program_on_inputs(
        program=record.mutated_full_source,
        entry_point=record.entry_point,
        inputs=tuple(tuple(args) for args in inputs),
        timeout_seconds=5.0,
    )
    replay_reprs = [o.output_repr for o in replay.outcomes]
    assert replay_reprs == [o.output_repr for o in record.mutant_expected]
    # On each distinct index, mutant expected != canonical expected.
    for index in record.distinct_input_indices:
        assert (
            record.mutant_expected[index].output_repr
            != record.canonical_expected[index].output_repr
            or record.mutant_expected[index].kind
            != record.canonical_expected[index].kind
        )


def test_generation_is_deterministic_across_two_runs() -> None:
    first_records, first_manifest = _small_generate()
    second_records, second_manifest = _small_generate()
    assert [r.model_dump() for r in first_records] == [
        r.model_dump() for r in second_records
    ]
    assert first_manifest.config_identity == second_manifest.config_identity
    assert first_manifest.accepted_by_family == (
        second_manifest.accepted_by_family
    )


def test_save_regeneration_is_byte_identical(tmp_path: Path) -> None:
    records, manifest = _small_generate()
    first_dir = tmp_path / "run_a"
    second_dir = tmp_path / "run_b"
    m1, mani1 = save_dataset(
        output_dir=first_dir, records=records, manifest=manifest
    )
    # Regenerate from scratch under the same pinned config.
    records2, manifest2 = _small_generate()
    m2, mani2 = save_dataset(
        output_dir=second_dir, records=records2, manifest=manifest2
    )
    assert m1.read_bytes() == m2.read_bytes()
    assert mani1.read_bytes() == mani2.read_bytes()


def test_saved_records_round_trip(tmp_path: Path) -> None:
    records, manifest = _small_generate()
    mutants_path, _ = save_dataset(
        output_dir=tmp_path / "run", records=records, manifest=manifest
    )
    loaded = load_records(mutants_path)
    assert [r.model_dump() for r in loaded] == [
        r.model_dump() for r in records
    ]


def test_skip_recorded_when_family_has_no_site() -> None:
    # truncate_number (HumanEval/2) is `return number - int(number)`: no
    # comparison/range/min-max/branch sites for most families.
    _, manifest = generate_mutants(
        families=(OperatorFamily.AGGREGATION_SWAP,),
        seeds=1,
        max_inputs_per_mutant=5,
        timeout_seconds=5.0,
        task_filter=("HumanEval/2",),
        prefer_snapshot=True,
    )
    reasons = {s.reason for s in manifest.skipped}
    assert "no applicable site" in reasons
    assert manifest.accepted_count == 0


def test_compose_rename_changes_entry_point_preserving_behavior() -> None:
    plain_records, _ = _small_generate(seeds=1)
    renamed_records, _ = _small_generate(seeds=1, compose_rename=True)
    assert plain_records and renamed_records
    renamed = renamed_records[0]
    assert renamed.entry_point == "target_fxn"
    assert renamed.optional_identifier_rename is not None
    assert "target_fxn" in renamed.mutated_full_source
    # Behavior preserved: re-executing the renamed mutant reproduces the same
    # expected outputs as recorded (rename does not change outcomes).
    inputs = tuple(tuple(eval(r)) for r in renamed.input_reprs)  # noqa: S307
    replay = run_program_on_inputs(
        program=renamed.mutated_full_source,
        entry_point=renamed.entry_point,
        inputs=inputs,
        timeout_seconds=5.0,
    )
    assert [o.output_repr for o in replay.outcomes] == [
        o.output_repr for o in renamed.mutant_expected
    ]
    assert any(o.kind is OutcomeKind.VALUE for o in replay.outcomes)


def test_manifest_config_identity_is_stable_sha256() -> None:
    _, manifest = _small_generate()
    assert len(manifest.config_identity) == 64
    assert manifest.config_identity == manifest.config.identity_hash()
