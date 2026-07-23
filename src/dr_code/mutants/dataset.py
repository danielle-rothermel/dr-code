"""Mutant dataset records, manifest, and serialization.

One :class:`MutantRecord` per accepted mutant, HumanEval+-loader-compatible
(carries ``task_id``, ``prompt``, ``entry_point`` and a mutated program that
defines ``entry_point``), plus the execution-derived expected outputs, a diff
summary vs. canonical, the count of behaviorally-differing inputs, and a
cross-reference to the canonical test suite for downstream dual/attractor-pull
scoring. The :class:`MutantManifest` pins the generation config identity.
"""

from __future__ import annotations

import json
from pathlib import Path

from dr_code.eval.identity import identity_hash_for
from dr_code.models import FrozenModel

DATASET_SCHEMA_VERSION = 1
GENERATOR_VERSION = "mutants@v1"
SCHEMA_MUTANT_CONFIG = "dr_code.mutants.generation_config"


class ExpectedOutcome(FrozenModel):
    """One input's execution-derived expected outcome for the mutant."""

    kind: str
    output_repr: str


class MutantRecord(FrozenModel):
    """One accepted behavioral mutant: loader-compatible, self-describing."""

    # Loader-compatible identity + code.
    task_id: str
    entry_point: str
    prompt: str
    canonical_full_source: str
    mutated_full_source: str

    # Seeded provenance: mutant = f(task_id, family, seed, site).
    operator_family: str
    seed: int
    site_node_path: int
    site_description: str

    # Execution-derived oracle over the task's HumanEval+ test inputs.
    input_reprs: tuple[str, ...]
    mutant_expected: tuple[ExpectedOutcome, ...]
    canonical_expected: tuple[ExpectedOutcome, ...]
    distinct_input_indices: tuple[int, ...]

    # Diff + cross-reference for downstream scoring.
    diff_summary: str
    canonical_test: str
    optional_identifier_rename: str | None = None

    @property
    def distinct_input_count(self) -> int:
        return len(self.distinct_input_indices)


class GenerationConfig(FrozenModel):
    """The identity-bearing configuration of one generation run."""

    generator_version: str
    dataset_schema_version: int
    dataset_id: str
    hf_revision: str
    operator_families: tuple[str, ...]
    seeds: int
    max_inputs_per_mutant: int
    timeout_seconds: float
    compose_rename: bool = False
    task_filter: tuple[str, ...] = ()

    def identity_payload(self) -> dict[str, object]:
        return {
            "generator_version": self.generator_version,
            "dataset_schema_version": self.dataset_schema_version,
            "dataset_id": self.dataset_id,
            "hf_revision": self.hf_revision,
            "operator_families": list(self.operator_families),
            "seeds": self.seeds,
            "max_inputs_per_mutant": self.max_inputs_per_mutant,
            "timeout_seconds": self.timeout_seconds,
            "compose_rename": self.compose_rename,
            "task_filter": list(self.task_filter),
        }

    def identity_hash(self) -> str:
        return identity_hash_for(
            schema=SCHEMA_MUTANT_CONFIG, payload=self.identity_payload()
        )


class SkippedFamily(FrozenModel):
    """A (task, family) pair with no applicable site or no distinct mutant."""

    task_id: str
    operator_family: str
    reason: str


class MutantManifest(FrozenModel):
    """Run manifest: config identity, per-family counts, and the search log."""

    config: GenerationConfig
    config_identity: str
    accepted_count: int
    accepted_by_family: tuple[tuple[str, int], ...]
    skipped: tuple[SkippedFamily, ...]


def dataset_filenames() -> tuple[str, str]:
    """The two artifact filenames: mutants JSONL and manifest JSON."""

    return ("mutants.jsonl", "manifest.json")


def save_dataset(
    *,
    output_dir: Path,
    records: tuple[MutantRecord, ...],
    manifest: MutantManifest,
) -> tuple[Path, Path]:
    """Write ``mutants.jsonl`` + ``manifest.json`` deterministically.

    Records are emitted in stable order (task_id, family, seed) with sorted
    JSON keys and a trailing newline per line, so regeneration under a pinned
    config is byte-identical.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    mutants_name, manifest_name = dataset_filenames()
    mutants_path = output_dir / mutants_name
    manifest_path = output_dir / manifest_name

    lines = [
        json.dumps(record.model_dump(mode="json"), sort_keys=True)
        for record in records
    ]
    mutants_path.write_text(
        "".join(f"{line}\n" for line in lines), encoding="utf-8"
    )
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return mutants_path, manifest_path


def load_records(path: Path) -> tuple[MutantRecord, ...]:
    """Load mutant records from a JSONL artifact."""

    records: list[MutantRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(MutantRecord.model_validate_json(line))
    return tuple(records)


__all__ = [
    "DATASET_SCHEMA_VERSION",
    "GENERATOR_VERSION",
    "ExpectedOutcome",
    "GenerationConfig",
    "MutantManifest",
    "MutantRecord",
    "SkippedFamily",
    "dataset_filenames",
    "load_records",
    "save_dataset",
]
