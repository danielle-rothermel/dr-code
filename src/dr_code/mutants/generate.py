"""Deterministic seeded generation of the behavioral-mutant suite.

For each (task, family, seed) the generator:

1. enumerates applicable sites in stable order (:mod:`operators`);
2. selects a site by a seeded index derived from ``(task_id, family, seed)``;
3. applies the mutation to the full ground-truth program;
4. derives the execution oracle over the task's HumanEval+ inputs and runs the
   validation gates (:mod:`oracle`);
5. on rejection (no site, non-distinct, non-deterministic, non-terminating),
   advances deterministically to the next candidate site and records the
   search; if no site yields an accepted mutant, records a skip.

Determinism: the only randomness is the seeded site permutation, derived by
hashing ``(task_id, family, seed)``. Given a pinned config the output is
byte-identical (asserted by the regeneration golden).
"""

from __future__ import annotations

import ast
import difflib
import hashlib
from collections.abc import Iterable, Sequence

from dr_code.humaneval.parsed_tests import (
    UnsupportedTestFormatError,
    parse_human_eval_tests,
)
from dr_code.humaneval.subprocess_runner import (
    SubprocessRunner,
    run_python_subprocess,
)
from dr_code.mutants.dataset import (
    DATASET_SCHEMA_VERSION,
    GENERATOR_VERSION,
    ExpectedOutcome,
    GenerationConfig,
    MutantManifest,
    MutantRecord,
    SkippedFamily,
)
from dr_code.mutants.oracle import (
    OracleError,
    ProgramOutcomes,
    distinct_input_indices,
    evaluate_gates,
    run_program_on_inputs,
)
from dr_code.mutants.operators import (
    MutationSite,
    OperatorFamily,
    apply_site,
    iter_sites,
)
from dr_code.mutants.rename import DEFAULT_TARGET_NAME, rename_entry_point
from dr_code.synthetic.humaneval_loader import (
    HF_DATASET_ID,
    HF_REVISION,
    HumanEvalPlusTask,
    load_humaneval_plus,
)


def _seeded_site_order(
    task_id: str, family: OperatorFamily, seed: int, site_count: int
) -> tuple[int, ...]:
    """A deterministic permutation of site indices for seeded selection.

    Each site gets a stable sort key from ``sha256(task_id|family|seed|i)``;
    sorting by that key yields a reproducible, seed-dependent visit order.
    """

    def key(index: int) -> str:
        payload = f"{task_id}|{family}|{seed}|{index}".encode()
        return hashlib.sha256(payload).hexdigest()

    return tuple(sorted(range(site_count), key=key))


def _canonical_inputs(
    task: HumanEvalPlusTask, max_inputs: int
) -> tuple[tuple[object, ...], ...]:
    parsed = parse_human_eval_tests(task.test)
    inputs = [tuple(case.args) for case in parsed.cases]
    # Deterministic head subsample keeps oracle cost bounded for preliminary
    # results while preserving input order (publication-hardening TODO: a
    # coverage-aware or difficulty-stratified input selection).
    return tuple(inputs[:max_inputs])


def _diff_summary(canonical: str, mutant: str) -> str:
    diff = difflib.unified_diff(
        canonical.splitlines(),
        mutant.splitlines(),
        fromfile="canonical",
        tofile="mutant",
        lineterm="",
        n=1,
    )
    return "\n".join(diff)


def _to_expected(outcomes: ProgramOutcomes) -> tuple[ExpectedOutcome, ...]:
    return tuple(
        ExpectedOutcome(kind=str(o.kind), output_repr=o.output_repr)
        for o in outcomes.outcomes
    )


def _normalized_canonical_source(task: HumanEvalPlusTask) -> str:
    # Round-trip through unparse so canonical and mutant differ only by the
    # mutation, not by incidental formatting (the mutant is also unparsed).
    return ast.unparse(ast.parse(task.full_source))


class _CanonicalCache:
    """Per-task canonical outcomes, computed once and reused per family."""

    def __init__(self, run_in_subprocess: SubprocessRunner) -> None:
        self._run = run_in_subprocess
        self._cache: dict[str, ProgramOutcomes] = {}

    def outcomes(
        self,
        *,
        task: HumanEvalPlusTask,
        source: str,
        inputs: tuple[tuple[object, ...], ...],
        timeout_seconds: float,
    ) -> ProgramOutcomes:
        if task.task_id not in self._cache:
            self._cache[task.task_id] = run_program_on_inputs(
                program=source,
                entry_point=task.entry_point,
                inputs=inputs,
                timeout_seconds=timeout_seconds,
                run_in_subprocess=self._run,
            )
        return self._cache[task.task_id]


def _try_family(
    *,
    task: HumanEvalPlusTask,
    family: OperatorFamily,
    seed: int,
    canonical_source: str,
    canonical_outcomes: ProgramOutcomes,
    inputs: tuple[tuple[object, ...], ...],
    input_reprs: tuple[str, ...],
    timeout_seconds: float,
    run_in_subprocess: SubprocessRunner,
    compose_rename: bool,
) -> MutantRecord | SkippedFamily:
    sites = iter_sites(canonical_source, family)
    if not sites:
        return SkippedFamily(
            task_id=task.task_id,
            operator_family=str(family),
            reason="no applicable site",
        )

    order = _seeded_site_order(task.task_id, family, seed, len(sites))
    last_reason = "no site produced a valid distinct mutant"
    for site_index in order:
        site = sites[site_index]
        try:
            mutant_source = apply_site(canonical_source, site)
        except Exception:  # noqa: BLE001 - malformed edit, try next site
            last_reason = "mutation edit failed to unparse"
            continue
        if mutant_source == canonical_source:
            last_reason = "mutation was a no-op"
            continue

        record = _oracle_accept(
            task=task,
            family=family,
            seed=seed,
            site=site,
            canonical_source=canonical_source,
            mutant_source=mutant_source,
            canonical_outcomes=canonical_outcomes,
            inputs=inputs,
            input_reprs=input_reprs,
            timeout_seconds=timeout_seconds,
            run_in_subprocess=run_in_subprocess,
            compose_rename=compose_rename,
        )
        if isinstance(record, MutantRecord):
            return record
        last_reason = record

    return SkippedFamily(
        task_id=task.task_id,
        operator_family=str(family),
        reason=last_reason,
    )


def _oracle_accept(
    *,
    task: HumanEvalPlusTask,
    family: OperatorFamily,
    seed: int,
    site: MutationSite,
    canonical_source: str,
    mutant_source: str,
    canonical_outcomes: ProgramOutcomes,
    inputs: tuple[tuple[object, ...], ...],
    input_reprs: tuple[str, ...],
    timeout_seconds: float,
    run_in_subprocess: SubprocessRunner,
    compose_rename: bool,
) -> MutantRecord | str:
    try:
        first = run_program_on_inputs(
            program=mutant_source,
            entry_point=task.entry_point,
            inputs=inputs,
            timeout_seconds=timeout_seconds,
            run_in_subprocess=run_in_subprocess,
        )
        second = run_program_on_inputs(
            program=mutant_source,
            entry_point=task.entry_point,
            inputs=inputs,
            timeout_seconds=timeout_seconds,
            run_in_subprocess=run_in_subprocess,
        )
    except OracleError as exc:
        return f"oracle failure: {exc}"

    report = evaluate_gates(
        canonical=canonical_outcomes,
        mutant_first=first,
        mutant_second=second,
    )
    reason = report.rejection_reason()
    if reason is not None:
        return reason

    distinct = distinct_input_indices(canonical_outcomes, first)

    record_entry_point = task.entry_point
    record_source = mutant_source
    rename_note: str | None = None
    if compose_rename:
        # Behavior-preserving rename of the entry point; oracle outputs are
        # unchanged, so mutant_expected still holds. Applied after acceptance
        # so distinctness is measured on the un-renamed behavior.
        record_source = rename_entry_point(
            mutant_source,
            entry_point=task.entry_point,
            target_name=DEFAULT_TARGET_NAME,
        )
        record_entry_point = DEFAULT_TARGET_NAME
        rename_note = f"{task.entry_point}->{DEFAULT_TARGET_NAME}"

    return MutantRecord(
        task_id=task.task_id,
        entry_point=record_entry_point,
        prompt=task.prompt,
        canonical_full_source=canonical_source,
        mutated_full_source=record_source,
        operator_family=str(family),
        seed=seed,
        site_node_path=site.node_path,
        site_description=site.description,
        input_reprs=input_reprs,
        mutant_expected=_to_expected(first),
        canonical_expected=_to_expected(canonical_outcomes),
        distinct_input_indices=distinct,
        diff_summary=_diff_summary(canonical_source, mutant_source),
        canonical_test=task.test,
        optional_identifier_rename=rename_note,
    )


def generate_mutants(
    *,
    families: Sequence[OperatorFamily],
    seeds: int,
    max_inputs_per_mutant: int,
    timeout_seconds: float,
    task_filter: Sequence[str] = (),
    compose_rename: bool = False,
    prefer_snapshot: bool = True,
    run_in_subprocess: SubprocessRunner = run_python_subprocess,
) -> tuple[tuple[MutantRecord, ...], MutantManifest]:
    """Generate the mutant suite deterministically; return records + manifest.

    ``seeds`` runs seed indices ``0..seeds-1`` per (task, family). A given
    (task, family, seed) yields at most one accepted mutant; duplicate mutated
    programs across seeds are de-duplicated so the suite has no exact repeats.
    """

    tasks = _selected_tasks(
        task_filter=task_filter, prefer_snapshot=prefer_snapshot
    )
    canonical_cache = _CanonicalCache(run_in_subprocess)

    records: list[MutantRecord] = []
    skipped: list[SkippedFamily] = []
    seen_programs: set[tuple[str, str]] = set()

    for task in tasks:
        try:
            inputs = _canonical_inputs(task, max_inputs_per_mutant)
        except UnsupportedTestFormatError:
            skipped.append(
                SkippedFamily(
                    task_id=task.task_id,
                    operator_family="*",
                    reason="unsupported test format",
                )
            )
            continue
        if not inputs:
            skipped.append(
                SkippedFamily(
                    task_id=task.task_id,
                    operator_family="*",
                    reason="no test inputs",
                )
            )
            continue

        canonical_source = _normalized_canonical_source(task)
        input_reprs = tuple(repr(list(args)) for args in inputs)
        try:
            canonical_outcomes = canonical_cache.outcomes(
                task=task,
                source=canonical_source,
                inputs=inputs,
                timeout_seconds=timeout_seconds,
            )
        except OracleError:
            skipped.append(
                SkippedFamily(
                    task_id=task.task_id,
                    operator_family="*",
                    reason="canonical did not execute cleanly",
                )
            )
            continue

        for family in families:
            for seed in range(seeds):
                outcome = _try_family(
                    task=task,
                    family=family,
                    seed=seed,
                    canonical_source=canonical_source,
                    canonical_outcomes=canonical_outcomes,
                    inputs=inputs,
                    input_reprs=input_reprs,
                    timeout_seconds=timeout_seconds,
                    run_in_subprocess=run_in_subprocess,
                    compose_rename=compose_rename,
                )
                if isinstance(outcome, SkippedFamily):
                    skipped.append(outcome)
                    continue
                key = (outcome.task_id, outcome.mutated_full_source)
                if key in seen_programs:
                    skipped.append(
                        SkippedFamily(
                            task_id=task.task_id,
                            operator_family=str(family),
                            reason="duplicate of an earlier accepted mutant",
                        )
                    )
                    continue
                seen_programs.add(key)
                records.append(outcome)

    ordered = _stable_order(records)
    config = GenerationConfig(
        generator_version=GENERATOR_VERSION,
        dataset_schema_version=DATASET_SCHEMA_VERSION,
        dataset_id=HF_DATASET_ID,
        hf_revision=HF_REVISION,
        operator_families=tuple(str(f) for f in families),
        seeds=seeds,
        max_inputs_per_mutant=max_inputs_per_mutant,
        timeout_seconds=timeout_seconds,
        compose_rename=compose_rename,
        task_filter=tuple(task_filter),
    )
    manifest = MutantManifest(
        config=config,
        config_identity=config.identity_hash(),
        accepted_count=len(ordered),
        accepted_by_family=_counts_by_family(ordered, families),
        skipped=tuple(skipped),
    )
    return ordered, manifest


def _stable_order(
    records: Iterable[MutantRecord],
) -> tuple[MutantRecord, ...]:
    return tuple(
        sorted(
            records,
            key=lambda r: (
                _task_sort_key(r.task_id),
                r.operator_family,
                r.seed,
            ),
        )
    )


def _task_sort_key(task_id: str) -> tuple[str, int]:
    # "HumanEval/12" sorts numerically after "HumanEval/2".
    prefix, _, number = task_id.rpartition("/")
    try:
        return (prefix, int(number))
    except ValueError:
        return (task_id, -1)


def _counts_by_family(
    records: Sequence[MutantRecord], families: Sequence[OperatorFamily]
) -> tuple[tuple[str, int], ...]:
    counts = {str(family): 0 for family in families}
    for record in records:
        counts[record.operator_family] = (
            counts.get(record.operator_family, 0) + 1
        )
    return tuple((name, counts[name]) for name in sorted(counts))


def _selected_tasks(
    *, task_filter: Sequence[str], prefer_snapshot: bool
) -> tuple[HumanEvalPlusTask, ...]:
    tasks = load_humaneval_plus(prefer_snapshot=prefer_snapshot)
    if not task_filter:
        return tuple(tasks)
    wanted = set(task_filter)
    return tuple(task for task in tasks if task.task_id in wanted)


__all__ = ["generate_mutants"]
