#!/usr/bin/env python3

"""Evaluate the selected historical candidates through evaluate_batch."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

import polars as pl
from dr_exec import ExecutionPoolConfig, FixedPoolCapacity
from dr_store import (
    ArtifactBundlePublication,
    ObjectStore,
    SqliteBackend,
    SqliteRecordCache,
)

from dr_code.caching import WindowedExecutionCache
from dr_code.core.execution.executor import host_process_executor
from dr_code.evaluation import AttemptCompleteness, evaluate_batch
from dr_code.humaneval import HumanEvalTask

from corpus_loader import manifest_sha256
from eval_batch import (
    attempt_id,
    build_preflight_batch_request_for_task,
    build_task_difficulty_batch_request,
    bundle_is_complete,
    evaluation_read_limits,
    export_candidate_results,
    load_humaneval_tasks,
    load_run_manifest,
    manifest_matches,
    probe_runtime_packages,
    runtime_id_from_executor,
    runtime_id_json,
    runtime_id_with_packages,
    settings_fingerprint,
    write_run_manifest,
)
from workflow_settings import (
    HUMANEVAL_SNAPSHOT,
    SELECTED_SAMPLE,
    EvalPaths,
    eval_paths,
    generation_corpus_bundle_path,
    parse_eval_args,
    prepare_run_directory,
)

_RUNTIME_ENVIRONMENT_VARIABLE = "DR_CODE_EVAL_PYTHON"
_PREFLIGHT_TASK_ID = "HumanEval/0"
_CACHE_RESIDENT_ENTRIES = 256
_CACHE_PENDING_ENTRIES = 64


def _configure_logging(path: Path) -> logging.Logger:
    logger = logging.getLogger("task_difficulty.evaluate")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(message)s")
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(path, mode="a", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _runtime_executable_from_environment() -> Path:
    value = os.environ.get(_RUNTIME_ENVIRONMENT_VARIABLE)
    if not value:
        raise SystemExit(
            f"{_RUNTIME_ENVIRONMENT_VARIABLE} must name a copied Python "
            "executable with dr-code's runtime dependencies installed"
        )
    executable = Path(value).expanduser().absolute()
    if not executable.is_file():
        raise SystemExit(f"evaluation Python is not a file: {executable}")
    if executable.is_symlink():
        raise SystemExit(
            "evaluation Python must be a copied executable, not a symlink: "
            f"{executable}"
        )
    return executable


def _bundle_path_from_manifest(
    manifest: dict[str, object] | None,
) -> Path | None:
    if manifest is None:
        return None
    bundle_path = manifest.get("bundle_path")
    if not isinstance(bundle_path, str) or not bundle_path:
        return None
    return Path(bundle_path)


async def _open_object_store(path: Path) -> tuple[ObjectStore, SqliteBackend]:
    backend = await SqliteBackend.open(path)
    return ObjectStore(backend), backend


async def _preflight_runtime(
    *,
    runtime_executable: Path,
    preflight_task: HumanEvalTask,
    paths: EvalPaths,
    settings,
    manifest_sha: str,
    object_store: ObjectStore,
    execution_cache: WindowedExecutionCache,
) -> None:
    record_directory = paths.execution_records / "preflight"
    record_directory.mkdir(parents=True, exist_ok=True)
    executor = host_process_executor(
        record_directory,
        runtime_executable=runtime_executable,
    )
    runtime = runtime_id_with_packages(
        runtime_id_from_executor(executor),
        probe_runtime_packages(executor),
    )
    preflight_root = paths.bundle_root / "preflight"
    if preflight_root.exists():
        shutil.rmtree(preflight_root)
    preflight_root.mkdir(parents=True, exist_ok=True)
    publication = ArtifactBundlePublication.allocate(
        preflight_root,
        prefix="preflight",
    )
    request = build_preflight_batch_request_for_task(
        preflight_task,
        settings=settings,
        runtime=runtime,
        manifest_sha256=manifest_sha,
    )
    result = await evaluate_batch(
        request,
        executor=executor,
        execution_cache=execution_cache,
        object_store=object_store,
        publication=publication,
        pool_config=ExecutionPoolConfig(
            capacity=FixedPoolCapacity(max_active_jobs=1)
        ),
    )
    if result.attempt.completeness is not AttemptCompleteness.COMPLETE:
        raise RuntimeError("runtime preflight did not complete")


async def _evaluate_selected_sample(
    *,
    paths: EvalPaths,
    settings,
    logger: logging.Logger,
) -> Path:
    runtime_executable = _runtime_executable_from_environment()
    selected = pl.read_parquet(SELECTED_SAMPLE)
    corpus_bundle = generation_corpus_bundle_path()
    manifest_sha = manifest_sha256(corpus_bundle)
    fingerprint = settings_fingerprint(
        settings=settings,
        manifest_sha256=manifest_sha,
        selected_sample_path=SELECTED_SAMPLE,
    )
    stored_manifest = load_run_manifest(paths.run_manifest)
    limits = evaluation_read_limits(
        sample_count=selected.height,
        candidate_count=int(selected.get_column("candidate_count").sum()),
    )

    paths.bundle_root.mkdir(parents=True, exist_ok=True)
    paths.execution_records.mkdir(parents=True, exist_ok=True)

    object_store_backend = await SqliteBackend.open(
        paths.evaluation_object_store
    )
    object_store = ObjectStore(object_store_backend)
    try:
        existing_bundle = _bundle_path_from_manifest(stored_manifest)
        if (
            stored_manifest is not None
            and manifest_matches(stored_manifest, fingerprint=fingerprint)
            and existing_bundle is not None
            and await bundle_is_complete(
                existing_bundle,
                object_store=object_store,
                limits=limits,
            )
        ):
            logger.info(
                "Skipping evaluation; complete bundle already present at %s",
                existing_bundle,
            )
            runtime_json = stored_manifest.get("runtime_identity_json")
            if not isinstance(runtime_json, str):
                raise RuntimeError(
                    "run manifest is missing runtime_identity_json"
                )
            export_candidate_results(
                existing_bundle,
                selected,
                paths.candidate_results,
                settings=settings,
                runtime_id_json=runtime_json,
                limits=limits,
                object_store=object_store,
            )
            return existing_bundle

        record_directory = paths.execution_records / "batch"
        record_directory.mkdir(parents=True, exist_ok=True)
        executor = host_process_executor(
            record_directory,
            runtime_executable=runtime_executable,
        )
        runtime = runtime_id_with_packages(
            runtime_id_from_executor(executor),
            probe_runtime_packages(executor),
        )
        preflight_task = load_humaneval_tasks(
            HUMANEVAL_SNAPSHOT,
            (_PREFLIGHT_TASK_ID,),
        )[_PREFLIGHT_TASK_ID]

        async with await SqliteRecordCache.open(
            paths.execution_cache
        ) as cache_store:
            execution_cache = WindowedExecutionCache(
                cache_store,
                runtime=runtime,
                max_resident_entries=_CACHE_RESIDENT_ENTRIES,
                max_pending_checkpoint_entries=_CACHE_PENDING_ENTRIES,
            )
            try:
                await _preflight_runtime(
                    runtime_executable=runtime_executable,
                    preflight_task=preflight_task,
                    paths=paths,
                    settings=settings,
                    manifest_sha=manifest_sha,
                    object_store=object_store,
                    execution_cache=execution_cache,
                )
                logger.info(
                    "Validated evaluation runtime %s",
                    runtime_executable,
                )

                publication_root = paths.bundle_root / "run"
                if publication_root.exists():
                    shutil.rmtree(publication_root)
                publication_root.mkdir(parents=True, exist_ok=True)
                publication = ArtifactBundlePublication.allocate(
                    publication_root,
                    prefix="evaluation",
                )
                attempt = attempt_id(fingerprint)
                request = build_task_difficulty_batch_request(
                    selected,
                    snapshot_path=HUMANEVAL_SNAPSHOT,
                    manifest_sha256=manifest_sha,
                    settings=settings,
                    runtime=runtime,
                    attempt=attempt,
                )
                logger.info(
                    "Starting evaluate_batch for %d generations across %d tasks",
                    selected.height,
                    selected.get_column("task_id").n_unique(),
                )
                started = perf_counter()
                result = await evaluate_batch(
                    request,
                    executor=executor,
                    execution_cache=execution_cache,
                    object_store=object_store,
                    publication=publication,
                    pool_config=ExecutionPoolConfig(
                        capacity=FixedPoolCapacity(
                            max_active_jobs=settings.worker_count
                        )
                    ),
                )
                elapsed = perf_counter() - started
                if (
                    result.attempt.completeness
                    is not AttemptCompleteness.COMPLETE
                ):
                    raise RuntimeError(
                        "evaluate_batch returned a partial attempt; re-run "
                        "stage 3 with the same workers, timeout, and corpus "
                        "pins to resume from the execution cache"
                    )
                if result.bundle_path is None:
                    raise RuntimeError(
                        "evaluate_batch did not publish an evaluation bundle"
                    )
                runtime_json = runtime_id_json(runtime)
                export_candidate_results(
                    result.bundle_path,
                    selected,
                    paths.candidate_results,
                    settings=settings,
                    runtime_id_json=runtime_json,
                    limits=limits,
                    object_store=object_store,
                )
                write_run_manifest(
                    paths.run_manifest,
                    {
                        "settings_fingerprint": fingerprint,
                        "attempt_id": str(attempt.attempt_id),
                        "bundle_path": str(result.bundle_path),
                        "runtime_identity_json": runtime_json,
                        "status": "complete",
                    },
                )
                logger.info(
                    "Evaluation finished in %.1f seconds; bundle at %s",
                    elapsed,
                    result.bundle_path,
                )
                return result.bundle_path
            finally:
                await execution_cache.close()
    finally:
        await object_store_backend.aclose()


async def _async_main(
    settings, paths: EvalPaths, logger: logging.Logger
) -> int:
    bundle_path = await _evaluate_selected_sample(
        paths=paths,
        settings=settings,
        logger=logger,
    )
    logger.info("Evaluation bundle: %s", bundle_path)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    run_started = perf_counter()
    evaluation_settings = parse_eval_args(__doc__, argv)
    paths = eval_paths(evaluation_settings)
    prepare_run_directory()
    paths.root.mkdir(parents=True, exist_ok=True)
    logger = _configure_logging(paths.evaluation_log)
    logger.info(
        "Eval config: workers=%d timeout_seconds=%g root=%s",
        evaluation_settings.worker_count,
        evaluation_settings.timeout_seconds,
        paths.root,
    )
    status = asyncio.run(_async_main(evaluation_settings, paths, logger))
    logger.info(
        "Evaluation stage finished in %.3f seconds",
        perf_counter() - run_started,
    )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
