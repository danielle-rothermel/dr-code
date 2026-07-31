"""Canonical-suite and execution identities for mutant generation."""

from __future__ import annotations

import ast
import platform
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from dr_code.eval.identity import identity_hash_for
from dr_code.humaneval.task import (
    HUMAN_EVAL_OVERRIDES,
    apply_human_eval_override,
)
from dr_code.humaneval.parsed_tests import (
    UnsupportedTestFormatError,
    parse_human_eval_tests,
)
from dr_code.synthetic.humaneval_loader import (
    HumanEvalSource,
    HumanEvalPlusTask,
    load_humaneval_plus,
)

_SUITE_SCHEMA: Final = "dr_code.mutants.canonical_suite"
_RUNTIME_SCHEMA: Final = "dr_code.mutants.python_runtime"


@dataclass(frozen=True, slots=True)
class CanonicalTask:
    """One exact canonical task and extracted input suite."""

    task_id: str
    prompt: str
    entry_point: str
    canonical_full_source: str
    canonical_test: str
    input_reprs: tuple[str, ...]
    preparation_failure: str | None


def resolve_canonical_suite(
    *,
    task_ids: Sequence[str] | None,
    max_inputs: int,
    source: HumanEvalSource,
) -> tuple[CanonicalTask, ...]:
    """Resolve selected tasks in source order and reject unknown ids."""

    tasks = tuple(
        _apply_humaneval_override(task)
        for task in load_humaneval_plus(source=source)
    )
    available = {task.task_id: task for task in tasks}
    if task_ids is None:
        selected = tasks
    else:
        requested = tuple(task_ids)
        if len(set(requested)) != len(requested):
            raise ValueError("task ids contain duplicates")
        missing = sorted(set(requested) - available.keys())
        if missing:
            raise ValueError(
                f"unknown HumanEval+ task id(s): {', '.join(missing)}"
            )
        wanted = set(requested)
        selected = tuple(task for task in tasks if task.task_id in wanted)
    return tuple(
        _prepare_task(task, max_inputs=max_inputs) for task in selected
    )


def _apply_humaneval_override(task: HumanEvalPlusTask) -> HumanEvalPlusTask:
    """Apply evaluator-owned benchmark corrections to a raw loader task."""

    updated = apply_human_eval_override(
        task.model_dump(mode="python"),
        HUMAN_EVAL_OVERRIDES,
    )
    return HumanEvalPlusTask(
        task_id=str(updated["task_id"]),
        prompt=str(updated["prompt"]),
        canonical_solution=str(updated["canonical_solution"]),
        entry_point=str(updated["entry_point"]),
        test=str(updated["test"]),
    )


def canonical_suite_digest(tasks: Sequence[CanonicalTask]) -> str:
    """Authenticate complete ordered canonical task/input content."""

    return identity_hash_for(
        schema=_SUITE_SCHEMA,
        payload=[
            {
                "canonical_full_source": task.canonical_full_source,
                "canonical_test": task.canonical_test,
                "entry_point": task.entry_point,
                "input_reprs": list(task.input_reprs),
                "preparation_failure": task.preparation_failure,
                "prompt": task.prompt,
                "task_id": task.task_id,
            }
            for task in tasks
        ],
    )


def current_runtime_identity() -> str:
    """Return the identity of execution-relevant Python runtime coordinates."""

    return identity_hash_for(
        schema=_RUNTIME_SCHEMA,
        payload={
            "byteorder": sys.byteorder,
            "cache_tag": sys.implementation.cache_tag,
            "implementation": platform.python_implementation(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "system": platform.system(),
        },
    )


def _prepare_task(
    task: HumanEvalPlusTask,
    *,
    max_inputs: int,
) -> CanonicalTask:
    failure: str | None = None
    try:
        canonical_source = ast.unparse(ast.parse(task.full_source))
    except (SyntaxError, ValueError):
        canonical_source = task.full_source
        failure = "canonical source is malformed"

    inputs: tuple[str, ...] = ()
    if failure is None:
        try:
            parsed = parse_human_eval_tests(task.test)
        except SyntaxError:
            failure = "canonical test is malformed"
        except UnsupportedTestFormatError:
            failure = "canonical test inputs use an unsupported format"
        else:
            inputs = tuple(
                repr(tuple(case.args)) for case in parsed.cases[:max_inputs]
            )
            if not inputs:
                failure = "canonical test has no inputs"
    return CanonicalTask(
        task_id=task.task_id,
        prompt=task.prompt,
        entry_point=task.entry_point,
        canonical_full_source=canonical_source,
        canonical_test=task.test,
        input_reprs=inputs,
        preparation_failure=failure,
    )


__all__ = (
    "CanonicalTask",
    "canonical_suite_digest",
    "current_runtime_identity",
    "resolve_canonical_suite",
)
