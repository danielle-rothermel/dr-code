"""Seeded search and execution validation for behavioral mutants."""

from __future__ import annotations

import difflib
import hashlib
import math
from collections.abc import Sequence

from dr_code.execution.subprocess import (
    PythonSubprocessRunner,
    run_python_subprocess,
)
from dr_code.mutants.dataset import (
    FamilyCount,
    GeneratedDataset,
    GenerationConfig,
    MutantRecord,
    SkippedMutation,
    build_record,
    record_order_key,
    skip_order_key,
)
from dr_code.mutants.operators import (
    ALL_FAMILIES,
    MutationError,
    MutationSite,
    OperatorFamily,
    apply_site,
    iter_sites,
)
from dr_code.mutants.oracle import (
    OracleError,
    ProgramOutcomes,
    distinct_input_indices,
    evaluate_gates,
    oracle_runner_source,
    run_program_on_inputs,
)
from dr_code.mutants.provenance import (
    CanonicalTask,
    canonical_suite_digest,
    capture_production_runner,
    is_production_runner_identity,
    resolve_canonical_suite,
)
from dr_code.synthetic.humaneval_loader import (
    HF_DATASET_ID,
    HF_REVISION,
    HumanEvalSource,
)


def generate_mutants(
    *,
    families: Sequence[OperatorFamily] = ALL_FAMILIES,
    seeds: int = 1,
    max_inputs_per_mutant: int = 50,
    timeout_seconds: float = 5.0,
    task_ids: Sequence[str] | None = None,
    dataset_source: HumanEvalSource = HumanEvalSource.SNAPSHOT,
    runner: PythonSubprocessRunner = run_python_subprocess,
    runner_identity: str | None = None,
    runtime_identity: str | None = None,
) -> GeneratedDataset:
    """Generate records in stable order from a pinned HumanEval+ source."""

    normalized_families = _normalize_families(families)
    _validate_limits(
        seeds=seeds,
        max_inputs=max_inputs_per_mutant,
        timeout_seconds=timeout_seconds,
    )
    tasks = resolve_canonical_suite(
        task_ids=task_ids,
        max_inputs=max_inputs_per_mutant,
        source=dataset_source,
    )
    if not tasks:
        raise ValueError("at least one task is required")
    captured_runner_source = oracle_runner_source()
    if runner is run_python_subprocess:
        if runner_identity is not None or runtime_identity is not None:
            raise ValueError(
                "production runner identities are assigned automatically"
            )
        production_runner = capture_production_runner(captured_runner_source)
        resolved_runner_identity = production_runner.runner_identity
        resolved_runtime_identity = production_runner.runtime_identity
        captured_runner_source = production_runner.runner_source
    else:
        if not runner_identity or not runtime_identity:
            raise ValueError(
                "injected runners require explicit runner and runtime identities"
            )
        if is_production_runner_identity(runner_identity):
            raise ValueError("injected runner identity must be non-production")
        resolved_runner_identity = runner_identity
        resolved_runtime_identity = runtime_identity
    config = GenerationConfig(
        dataset_source=dataset_source,
        dataset_id=HF_DATASET_ID,
        dataset_revision=HF_REVISION,
        operator_families=normalized_families,
        seeds=seeds,
        max_inputs_per_mutant=max_inputs_per_mutant,
        timeout_seconds=timeout_seconds,
        task_ids=tuple(task.task_id for task in tasks),
        canonical_suite_digest=canonical_suite_digest(tasks),
        runner_identity=resolved_runner_identity,
        runtime_identity=resolved_runtime_identity,
    )

    records: list[MutantRecord] = []
    skipped: list[SkippedMutation] = []
    seen_programs: set[tuple[str, str]] = set()

    for task in tasks:
        if task.preparation_failure is not None:
            skipped.append(
                SkippedMutation(
                    task_id=task.task_id,
                    operator_family="*",
                    seed=None,
                    reason=task.preparation_failure,
                )
            )
            continue
        canonical_source = task.canonical_full_source
        input_reprs = task.input_reprs
        try:
            canonical_first = run_program_on_inputs(
                program=canonical_source,
                entry_point=task.entry_point,
                input_reprs=input_reprs,
                timeout_seconds=timeout_seconds,
                runner=runner,
                runner_source=captured_runner_source,
            )
            canonical_second = run_program_on_inputs(
                program=canonical_source,
                entry_point=task.entry_point,
                input_reprs=input_reprs,
                timeout_seconds=timeout_seconds,
                runner=runner,
                runner_source=captured_runner_source,
            )
        except OracleError:
            _skip_task_coordinates(
                skipped=skipped,
                task_id=task.task_id,
                families=normalized_families,
                seeds=seeds,
                reason="canonical execution did not complete",
            )
            continue
        if canonical_first != canonical_second:
            _skip_task_coordinates(
                skipped=skipped,
                task_id=task.task_id,
                families=normalized_families,
                seeds=seeds,
                reason="canonical is non-deterministic across two runs",
            )
            continue
        if any(
            outcome.kind == "error" for outcome in canonical_first.outcomes
        ):
            _skip_task_coordinates(
                skipped=skipped,
                task_id=task.task_id,
                families=normalized_families,
                seeds=seeds,
                reason="canonical execution raised on a test input",
            )
            continue

        for family in normalized_families:
            sites = iter_sites(canonical_source, family)
            for seed in range(seeds):
                record, reason = _search_coordinate(
                    task=task,
                    family=family,
                    seed=seed,
                    sites=sites,
                    canonical_source=canonical_source,
                    canonical_outcomes=canonical_first,
                    input_reprs=input_reprs,
                    timeout_seconds=timeout_seconds,
                    seen_programs=seen_programs,
                    runner=runner,
                    runner_source=captured_runner_source,
                )
                if record is None:
                    skipped.append(
                        SkippedMutation(
                            task_id=task.task_id,
                            operator_family=family,
                            seed=seed,
                            reason=reason,
                        )
                    )
                    continue
                seen_programs.add((record.task_id, record.mutated_full_source))
                records.append(record)

    ordered = tuple(sorted(records, key=record_order_key))
    counts = tuple(
        FamilyCount(
            operator_family=family,
            count=sum(record.operator_family is family for record in ordered),
        )
        for family in sorted(
            normalized_families,
            key=lambda value: value.value,
        )
    )
    return GeneratedDataset(
        config=config,
        canonical_suite=tasks,
        records=ordered,
        accepted_by_family=counts,
        skipped=tuple(
            sorted(
                skipped,
                key=lambda skip: skip_order_key(skip, config),
            )
        ),
    )


def seeded_site_order(
    *,
    task_id: str,
    family: OperatorFamily,
    seed: int,
    site_count: int,
) -> tuple[int, ...]:
    """Return the stable seeded permutation searched for one coordinate."""

    if seed < 0 or site_count < 0:
        raise ValueError("seed and site_count must be non-negative")

    def sort_key(index: int) -> tuple[str, int]:
        payload = f"{task_id}|{family.value}|{seed}|{index}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest(), index

    return tuple(sorted(range(site_count), key=sort_key))


def _search_coordinate(
    *,
    task: CanonicalTask,
    family: OperatorFamily,
    seed: int,
    sites: tuple[MutationSite, ...],
    canonical_source: str,
    canonical_outcomes: ProgramOutcomes,
    input_reprs: tuple[str, ...],
    timeout_seconds: float,
    seen_programs: set[tuple[str, str]],
    runner: PythonSubprocessRunner,
    runner_source: str,
) -> tuple[MutantRecord | None, str]:
    if not sites:
        return None, "no applicable site"
    last_reason = "no site produced a valid distinct mutant"
    for site_index in seeded_site_order(
        task_id=task.task_id,
        family=family,
        seed=seed,
        site_count=len(sites),
    ):
        site = sites[site_index]
        try:
            mutant_source = apply_site(canonical_source, site)
        except MutationError:
            last_reason = "mutation edit was invalid"
            continue
        if mutant_source == canonical_source:
            last_reason = "mutation edit was a no-op"
            continue
        if (task.task_id, mutant_source) in seen_programs:
            last_reason = "duplicate of an earlier accepted mutant"
            continue
        try:
            first = run_program_on_inputs(
                program=mutant_source,
                entry_point=task.entry_point,
                input_reprs=input_reprs,
                timeout_seconds=timeout_seconds,
                runner=runner,
                runner_source=runner_source,
            )
            second = run_program_on_inputs(
                program=mutant_source,
                entry_point=task.entry_point,
                input_reprs=input_reprs,
                timeout_seconds=timeout_seconds,
                runner=runner,
                runner_source=runner_source,
            )
            report = evaluate_gates(
                canonical_first=canonical_outcomes,
                canonical_second=canonical_outcomes,
                mutant_first=first,
                mutant_second=second,
            )
        except OracleError:
            last_reason = "mutant execution did not complete"
            continue
        reason = report.rejection_reason()
        if reason is not None:
            last_reason = reason
            continue
        distinct = distinct_input_indices(canonical_outcomes, first)
        return (
            _record(
                task=task,
                family=family,
                seed=seed,
                site=site,
                canonical_source=canonical_source,
                mutant_source=mutant_source,
                input_reprs=input_reprs,
                canonical_outcomes=canonical_outcomes,
                mutant_outcomes=first,
                distinct=distinct,
            ),
            "",
        )
    return None, last_reason


def _record(
    *,
    task: CanonicalTask,
    family: OperatorFamily,
    seed: int,
    site: MutationSite,
    canonical_source: str,
    mutant_source: str,
    input_reprs: tuple[str, ...],
    canonical_outcomes: ProgramOutcomes,
    mutant_outcomes: ProgramOutcomes,
    distinct: tuple[int, ...],
) -> MutantRecord:
    return build_record(
        task_id=task.task_id,
        entry_point=task.entry_point,
        prompt=task.prompt,
        canonical_full_source=canonical_source,
        mutated_full_source=mutant_source,
        operator_family=family,
        seed=seed,
        site_node_path=site.node_path,
        site_target_index=site.target_index,
        site_description=site.description,
        input_reprs=input_reprs,
        mutant_expected=mutant_outcomes.outcomes,
        canonical_expected=canonical_outcomes.outcomes,
        distinct_input_indices=distinct,
        diff_summary=_diff_summary(canonical_source, mutant_source),
        canonical_test=task.canonical_test,
    )


def _diff_summary(canonical: str, mutant: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            canonical.splitlines(),
            mutant.splitlines(),
            fromfile="canonical",
            tofile="mutant",
            lineterm="",
            n=1,
        )
    )


def _normalize_families(
    families: Sequence[OperatorFamily],
) -> tuple[OperatorFamily, ...]:
    selected = tuple(families)
    if not selected:
        raise ValueError("at least one operator family is required")
    if len(set(selected)) != len(selected):
        raise ValueError("operator families contain duplicates")
    return tuple(family for family in ALL_FAMILIES if family in selected)


def _skip_task_coordinates(
    *,
    skipped: list[SkippedMutation],
    task_id: str,
    families: tuple[OperatorFamily, ...],
    seeds: int,
    reason: str,
) -> None:
    skipped.extend(
        SkippedMutation(
            task_id=task_id,
            operator_family=family,
            seed=seed,
            reason=reason,
        )
        for family in families
        for seed in range(seeds)
    )


def _validate_limits(
    *,
    seeds: int,
    max_inputs: int,
    timeout_seconds: float,
) -> None:
    if seeds < 1:
        raise ValueError("seeds must be at least 1")
    if max_inputs < 1:
        raise ValueError("max_inputs_per_mutant must be at least 1")
    if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
        raise ValueError("timeout_seconds must be finite and positive")


__all__ = ("generate_mutants", "seeded_site_order")
