"""Canonical-suite and execution identities for mutant generation."""

from __future__ import annotations

import ast
import hashlib
import os
import platform
import stat
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
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
from dr_code.implementation_identity import package_source_digest

PRODUCTION_RUNNER_IDENTITY_PREFIX: Final = "python-subprocess-oracle@v2:"
_PRODUCTION_RUNNER_IDENTITY_NAMESPACE: Final = "python-subprocess-oracle@"
_SUITE_SCHEMA: Final = "dr_code.mutants.canonical_suite"
_RUNNER_SCHEMA: Final = "dr_code.mutants.production_runner"
_RUNTIME_SCHEMA: Final = "dr_code.mutants.python_runtime"
_READ_CHUNK_BYTES: Final = 1024 * 1024


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


class RunnerProvenanceError(ValueError):
    """Production execution provenance could not be captured safely."""


@dataclass(frozen=True, slots=True)
class CapturedProductionRunner:
    """One immutable runner source and the identities derived from it."""

    runner_source: str
    runner_identity: str
    runtime_identity: str


@dataclass(frozen=True, slots=True)
class _PythonRuntimeCoordinates:
    """Concrete coordinates for the interpreter selected by the executor."""

    byteorder: str
    implementation_cache_tag: str | None
    implementation_hexversion: int
    implementation_name: str
    machine: str
    python_executable_invoked_path: str
    python_executable_real_path: str
    python_executable_sha256: str
    python_executable_size: int
    python_version: str
    system: str
    system_release: str

    def identity_payload(self) -> dict[str, object]:
        # Persisted identity payload contract. Keep these literal keys explicit;
        # never derive them from Python field names.
        return {
            "byteorder": self.byteorder,
            "implementation_cache_tag": self.implementation_cache_tag,
            "implementation_hexversion": self.implementation_hexversion,
            "implementation_name": self.implementation_name,
            "machine": self.machine,
            "python_executable_invoked_path": (
                self.python_executable_invoked_path
            ),
            "python_executable_real_path": self.python_executable_real_path,
            "python_executable_sha256": self.python_executable_sha256,
            "python_executable_size": self.python_executable_size,
            "python_version": self.python_version,
            "system": self.system,
            "system_release": self.system_release,
        }


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


def capture_production_runner(source: str) -> CapturedProductionRunner:
    """Capture one source string and derive all production identities from it.

    The returned ``runner_source`` is the same immutable value whose UTF-8 bytes
    feed ``runner_identity``. Callers must pass that value to every subprocess
    call rather than reading or rebuilding the module-level source again.

    The source is byte-bound to execution. The package and executable digests
    are fail-closed pre-execution observations: the current subprocess API still
    resolves loaded implementation globals and the interpreter path later, so
    those resources are not immutable execution snapshots.
    """

    if not isinstance(source, str) or "\0" in source:
        raise RunnerProvenanceError(
            "production runner source must be NUL-free text"
        )
    try:
        source_utf8 = source.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RunnerProvenanceError(
            "production runner source is not valid UTF-8 text"
        ) from exc
    try:
        package_digest = package_source_digest()
    except (OSError, ValueError) as exc:
        raise RunnerProvenanceError(
            "installed dr_code Python source evidence is unavailable"
        ) from exc
    _require_sha256(
        package_digest,
        label="installed dr_code Python source evidence",
    )
    runner_payload = _production_runner_identity_payload(
        runner_source_utf8=source_utf8,
        dr_code_python_package_sha256=package_digest,
    )
    runtime_payload = _current_runtime_coordinates().identity_payload()
    return CapturedProductionRunner(
        runner_source=source,
        runner_identity=_production_runner_identity(runner_payload),
        runtime_identity=_runtime_identity(runtime_payload),
    )


def is_production_runner_identity(identity: object) -> bool:
    """Return whether an identity claims the reserved production runner."""

    return isinstance(identity, str) and identity.startswith(
        _PRODUCTION_RUNNER_IDENTITY_NAMESPACE
    )


def _production_runner_identity_payload(
    *,
    runner_source_utf8: bytes,
    dr_code_python_package_sha256: str,
) -> dict[str, object]:
    _require_sha256(
        dr_code_python_package_sha256,
        label="installed dr_code Python source evidence",
    )
    # Persisted identity payload contract. The source digest is computed from
    # the captured bytes that the caller's stored source value represents.
    return {
        "dr_code_python_package_sha256": dr_code_python_package_sha256,
        "python_argv_prefix": ["-I", "-c"],
        "runner_source_utf8_sha256": hashlib.sha256(
            runner_source_utf8
        ).hexdigest(),
        "runner_source_utf8_size": len(runner_source_utf8),
    }


def _production_runner_identity(payload: dict[str, object]) -> str:
    return PRODUCTION_RUNNER_IDENTITY_PREFIX + identity_hash_for(
        schema=_RUNNER_SCHEMA,
        payload=payload,
    )


def _runtime_identity(payload: dict[str, object]) -> str:
    return identity_hash_for(schema=_RUNTIME_SCHEMA, payload=payload)


def _current_runtime_coordinates() -> _PythonRuntimeCoordinates:
    invoked_path = sys.executable
    if (
        not isinstance(invoked_path, str)
        or not invoked_path
        or "\0" in invoked_path
    ):
        raise RunnerProvenanceError("Python executable path is unavailable")
    try:
        real_path = Path(invoked_path).expanduser().resolve(strict=True)
        executable_sha256, executable_size = _hash_stable_regular_file(
            real_path
        )
    except (OSError, ValueError) as exc:
        raise RunnerProvenanceError(
            "Python executable evidence is unavailable"
        ) from exc
    return _PythonRuntimeCoordinates(
        byteorder=sys.byteorder,
        implementation_cache_tag=sys.implementation.cache_tag,
        implementation_hexversion=sys.implementation.hexversion,
        implementation_name=sys.implementation.name,
        machine=platform.machine(),
        python_executable_invoked_path=invoked_path,
        python_executable_real_path=str(real_path),
        python_executable_sha256=executable_sha256,
        python_executable_size=executable_size,
        python_version=sys.version,
        system=platform.system(),
        system_release=platform.release(),
    )


def _hash_stable_regular_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(
                f"Python executable is not a regular file: {path}"
            )
        size = 0
        while chunk := stream.read(_READ_CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(stream.fileno())
    before_coordinates = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_coordinates = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_coordinates != after_coordinates or size != before.st_size:
        raise ValueError(f"Python executable changed while hashing: {path}")
    return digest.hexdigest(), size


def _require_sha256(value: str, *, label: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise RunnerProvenanceError(f"{label} is invalid")


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
    "PRODUCTION_RUNNER_IDENTITY_PREFIX",
    "CanonicalTask",
    "CapturedProductionRunner",
    "RunnerProvenanceError",
    "canonical_suite_digest",
    "capture_production_runner",
    "is_production_runner_identity",
    "resolve_canonical_suite",
)
