"""Operator verbs for dr-code's standalone validation flows."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import typer
from dr_exec import ExecutionPoolConfig, Executor, FixedPoolCapacity
from dr_store import (
    ArtifactBundlePublication,
    ObjectStore,
    SqliteBackend,
    SqliteRecordCache,
)

from dr_code.caching import WindowedExecutionCache
from dr_code.core.execution.executor import host_process_executor
from dr_code.evaluation.batch import EvaluationBatchRequest
from dr_code.evaluation.flows import (
    AttemptVerdict,
    validate_preprocessing,
    validate_testing,
)

CACHE_RESIDENT_ENTRIES = 4096
CACHE_PENDING_ENTRIES = 4096

validate_preprocessing_app = typer.Typer(
    name="dr-code-validate-preprocessing",
    help="Validate a preprocessing change over one evaluation request.",
    add_completion=False,
)
validate_testing_app = typer.Typer(
    name="dr-code-validate-testing",
    help="Validate a testing change over one evaluation request.",
    add_completion=False,
)

_REQUEST_OPTION = typer.Option(
    ...,
    "--request",
    exists=True,
    dir_okay=False,
    readable=True,
    resolve_path=True,
    help="Evaluation batch request JSON document.",
)
_RUN_ROOT_OPTION = typer.Option(
    ...,
    "--run-root",
    file_okay=False,
    resolve_path=True,
    help="Directory for the bundle, execution records, and caches.",
)
_RUNTIME_OPTION = typer.Option(
    ...,
    "--runtime",
    exists=True,
    dir_okay=False,
    resolve_path=True,
    help="Python executable that runs candidate jobs.",
)
_WORKERS_OPTION = typer.Option(
    ...,
    "--workers",
    min=1,
    help="Concurrency bound for the pooled leg.",
)


@dataclass(frozen=True, slots=True)
class _RunPaths:
    bundle_root: Path
    record_root: Path
    execution_cache: Path
    object_store: Path

    @classmethod
    def under(cls, run_root: Path, /) -> _RunPaths:
        paths = cls(
            bundle_root=run_root / "bundle",
            record_root=run_root / "execution-records",
            execution_cache=run_root / "execution-cache.sqlite",
            object_store=run_root / "object-store.sqlite",
        )
        paths.bundle_root.mkdir(parents=True, exist_ok=True)
        paths.record_root.mkdir(parents=True, exist_ok=True)
        return paths


@validate_preprocessing_app.command()
def validate_preprocessing_command(
    request: Path = _REQUEST_OPTION,
    run_root: Path = _RUN_ROOT_OPTION,
    runtime: Path = _RUNTIME_OPTION,
    workers: int = _WORKERS_OPTION,
) -> None:
    """Run the preprocessing validation flow and print its verdict."""

    batch_request = _load_request(request)
    verdict, coverage = asyncio.run(
        _run_preprocessing(
            batch_request,
            run_root=run_root,
            runtime=runtime,
            workers=workers,
        )
    )
    typer.echo(
        f"texts_with_candidates={coverage[0]} "
        f"texts_without_candidates={coverage[1]}"
    )
    _echo_verdict(verdict)


@validate_testing_app.command()
def validate_testing_command(
    request: Path = _REQUEST_OPTION,
    run_root: Path = _RUN_ROOT_OPTION,
    runtime: Path = _RUNTIME_OPTION,
    workers: int = _WORKERS_OPTION,
) -> None:
    """Run the testing validation flow and print its verdict."""

    batch_request = _load_request(request)
    verdict = asyncio.run(
        _run_testing(
            batch_request,
            run_root=run_root,
            runtime=runtime,
            workers=workers,
        )
    )
    _echo_verdict(verdict)


async def _run_preprocessing(
    request: EvaluationBatchRequest,
    *,
    run_root: Path,
    runtime: Path,
    workers: int,
) -> tuple[AttemptVerdict, tuple[int, int]]:
    paths = _RunPaths.under(run_root)
    async with _resources(request, paths, runtime=runtime) as resources:
        validation = await validate_preprocessing(
            request,
            definition=request.plan.procedure.preprocessing,
            executor=resources.executor,
            execution_cache=resources.execution_cache,
            object_store=resources.object_store,
            publication=resources.publication,
            pool_config=_pool_config(workers),
            worker_count=workers,
        )
    return validation.verdict, (
        validation.texts_with_candidates,
        validation.texts_without_candidates,
    )


async def _run_testing(
    request: EvaluationBatchRequest,
    *,
    run_root: Path,
    runtime: Path,
    workers: int,
) -> AttemptVerdict:
    paths = _RunPaths.under(run_root)
    async with _resources(request, paths, runtime=runtime) as resources:
        validation = await validate_testing(
            request,
            executor=resources.executor,
            execution_cache=resources.execution_cache,
            object_store=resources.object_store,
            publication=resources.publication,
            pool_config=_pool_config(workers),
        )
    return validation.verdict


@dataclass(frozen=True, slots=True)
class _FlowResources:
    executor: Executor
    execution_cache: WindowedExecutionCache
    object_store: ObjectStore
    publication: ArtifactBundlePublication


@asynccontextmanager
async def _resources(
    request: EvaluationBatchRequest,
    paths: _RunPaths,
    *,
    runtime: Path,
) -> AsyncIterator[_FlowResources]:
    backend = await SqliteBackend.open(paths.object_store)
    try:
        async with await SqliteRecordCache.open(
            paths.execution_cache
        ) as cache_store:
            execution_cache = WindowedExecutionCache(
                cache_store,
                runtime=request.runtime,
                max_resident_entries=CACHE_RESIDENT_ENTRIES,
                max_pending_checkpoint_entries=CACHE_PENDING_ENTRIES,
            )
            try:
                yield _FlowResources(
                    executor=host_process_executor(
                        paths.record_root,
                        runtime_executable=runtime,
                    ),
                    execution_cache=execution_cache,
                    object_store=ObjectStore(backend),
                    publication=ArtifactBundlePublication.allocate(
                        paths.bundle_root,
                        prefix="evaluation",
                    ),
                )
            finally:
                await execution_cache.close()
    finally:
        await backend.aclose()


def _pool_config(workers: int) -> ExecutionPoolConfig:
    return ExecutionPoolConfig(
        capacity=FixedPoolCapacity(max_active_jobs=workers)
    )


def _load_request(path: Path) -> EvaluationBatchRequest:
    return EvaluationBatchRequest.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def _echo_verdict(verdict: AttemptVerdict) -> None:
    typer.echo(
        f"completeness={verdict.completeness.value} "
        f"validity={verdict.validity.value}"
    )
    exhaustion = verdict.limit_exhaustion
    if exhaustion is not None:
        typer.echo(
            f"limit_exhaustion={exhaustion.limit.value} "
            f"configured={exhaustion.configured} "
            f"observed={exhaustion.observed}"
        )


__all__ = [
    "validate_preprocessing_app",
    "validate_testing_app",
]
