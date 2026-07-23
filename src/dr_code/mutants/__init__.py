"""Seeded behavioral-mutant suite for HumanEval+.

A contamination instrument: rule-based, seeded, behavior-altering mutations of
the canonical HumanEval+ ground-truth solutions, validated by execution-derived
oracles over the task's existing test inputs. See the design
doc at ``docs/behavioral-mutants.md``.
"""

from __future__ import annotations

from dr_code.mutants.dataset import (
    GenerationConfig,
    MutantManifest,
    MutantRecord,
    load_records,
    save_dataset,
)
from dr_code.mutants.generate import generate_mutants
from dr_code.mutants.operators import (
    ALL_FAMILIES,
    MutationSite,
    OperatorFamily,
    apply_site,
    iter_sites,
)

__all__ = [
    "ALL_FAMILIES",
    "GenerationConfig",
    "MutantManifest",
    "MutantRecord",
    "MutationSite",
    "OperatorFamily",
    "apply_site",
    "generate_mutants",
    "iter_sites",
    "load_records",
    "save_dataset",
]
