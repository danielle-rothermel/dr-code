"""Resumable Parquet runner for exhaustive preprocessing traces."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, cast

import pyarrow as pa
import pyarrow.parquet as pq

from dr_code.corpus.preprocessing_artifacts import (
    AtomicProjectedPartWriter,
    PROJECTED_ARTIFACT_SCHEMAS,
    ProjectedArtifacts,
    combine_projected_parts,
    project_preprocessing_result,
)
from dr_code.preprocessing.definition import preprocessing_definition_hash
from dr_code.preprocessing.definitions import (
    HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
)
from dr_code.preprocessing.runner import (
    BoundPreprocessingRunner,
    bind_preprocessing,
)
from dr_code.trace import TextArtifact


MANIFEST_SCHEMA_VERSION: Final = 1
RELATION_NAMES: Final = tuple(PROJECTED_ARTIFACT_SCHEMAS)
REQUIRED_INPUT_COLUMNS: Final = ("sample_id", "decoder_output")


class CorpusRunError(ValueError):
    """The corpus input, checkpoint, or output does not meet the contract."""


def run_preprocessing_corpus(
    *,
    input_path: Path | str,
    output_root: Path | str,
    run_id: str | None = None,
    batch_size: int = 1_000,
    max_row_groups: int | None = None,
) -> Path:
    """Preprocess an input Parquet corpus, safely resuming by row group.

    A partial run is left under ``<run_id>.partial`` until every input row
    group has been written, combined, and validated. The completed directory
    is then atomically renamed to ``<run_id>``.
    """
    if batch_size < 1:
        raise CorpusRunError("batch_size must be at least 1")
    if max_row_groups is not None and max_row_groups < 1:
        raise CorpusRunError("max_row_groups must be at least 1 when set")

    input_file = Path(input_path).expanduser().resolve(strict=True)
    destination_root = Path(output_root).expanduser().resolve()
    parquet_file = pq.ParquetFile(input_file)
    _validate_input_schema(parquet_file.schema_arrow)
    fingerprint = _input_fingerprint(input_file, parquet_file)
    resolved_run_id = run_id or _generated_run_id(fingerprint)
    if (
        not resolved_run_id
        or resolved_run_id in {".", ".."}
        or Path(resolved_run_id).name != resolved_run_id
    ):
        raise CorpusRunError("run_id must be a single non-empty path segment")
    completed_dir = destination_root / resolved_run_id
    partial_dir = destination_root / f"{resolved_run_id}.partial"
    if completed_dir.exists():
        raise FileExistsError(f"completed run already exists: {completed_dir}")
    bound_runner = bind_preprocessing(
        HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION
    )
    immutable_manifest = _immutable_manifest(
        run_id=resolved_run_id,
        input_fingerprint=fingerprint,
        batch_size=batch_size,
        bound_runner=bound_runner,
    )
    manifest = _load_or_create_manifest(partial_dir, immutable_manifest)

    completed_row_groups = _completed_row_groups(manifest)
    _validate_completed_parts(partial_dir, completed_row_groups)
    processed_this_run = 0
    for row_group_index in range(parquet_file.num_row_groups):
        if row_group_index in completed_row_groups:
            continue
        if max_row_groups is not None and processed_this_run >= max_row_groups:
            break

        part_id = _part_id(row_group_index)
        _remove_orphan_part(partial_dir, part_id)
        _write_projected_row_group(
            parquet_file=parquet_file,
            row_group_index=row_group_index,
            batch_size=batch_size,
            bound_runner=bound_runner,
            partial_dir=partial_dir,
            part_id=part_id,
        )
        completed_row_groups.add(row_group_index)
        manifest["completed_row_groups"] = sorted(completed_row_groups)
        manifest["relation_totals"] = _relation_part_totals(
            partial_dir, completed_row_groups
        )
        manifest["outcome_totals"] = _outcome_part_totals(
            partial_dir, completed_row_groups
        )
        manifest["updated_at"] = _timestamp()
        _write_manifest(partial_dir, manifest)
        processed_this_run += 1

    if len(completed_row_groups) != parquet_file.num_row_groups:
        return partial_dir

    part_ids = [_part_id(index) for index in sorted(completed_row_groups)]
    relation_paths = combine_projected_parts(partial_dir, part_ids)
    totals = _validate_completed_relations(
        relation_paths=relation_paths,
        expected_rows=fingerprint["expected_rows"],
        input_parquet=parquet_file,
    )
    manifest["relation_totals"] = totals["relation_totals"]
    manifest["outcome_totals"] = totals["outcome_totals"]
    manifest["complete"] = True
    manifest["completed_at"] = _timestamp()
    manifest["updated_at"] = manifest["completed_at"]
    _write_manifest(partial_dir, manifest)
    destination_root.mkdir(parents=True, exist_ok=True)
    if completed_dir.exists():
        raise FileExistsError(f"completed run already exists: {completed_dir}")
    os.replace(partial_dir, completed_dir)
    return completed_dir


def _write_projected_row_group(
    *,
    parquet_file: pq.ParquetFile,
    row_group_index: int,
    batch_size: int,
    bound_runner: BoundPreprocessingRunner,
    partial_dir: Path,
    part_id: str,
) -> None:
    with AtomicProjectedPartWriter(partial_dir, part_id) as writer:
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            row_groups=[row_group_index],
            columns=list(REQUIRED_INPUT_COLUMNS),
        ):
            writer.append(_project_batch(batch, bound_runner))


def _project_batch(
    batch: pa.RecordBatch, bound_runner: BoundPreprocessingRunner
) -> ProjectedArtifacts:
    projected = ProjectedArtifacts(
        results=[], candidates=[], step_facts=[], rejections=[]
    )
    sample_ids = batch.column("sample_id").to_pylist()
    decoder_outputs = batch.column("decoder_output").to_pylist()
    for sample_id, decoder_output in zip(sample_ids, decoder_outputs):
        assert isinstance(sample_id, str)
        assert decoder_output is None or isinstance(decoder_output, str)
        trace = (
            None
            if decoder_output is None
            else bound_runner.run(TextArtifact(text=decoder_output))
        )
        sample_projection = project_preprocessing_result(
            sample_id, decoder_output, trace
        )
        projected.results.extend(sample_projection.results)
        projected.candidates.extend(sample_projection.candidates)
        projected.step_facts.extend(sample_projection.step_facts)
        projected.rejections.extend(sample_projection.rejections)
    return projected


def _validate_input_schema(schema: pa.Schema) -> None:
    missing = [
        name for name in REQUIRED_INPUT_COLUMNS if name not in schema.names
    ]
    if missing:
        raise CorpusRunError(
            "input Parquet is missing required column(s): "
            + ", ".join(missing)
        )
    for name in REQUIRED_INPUT_COLUMNS:
        if not pa.types.is_string(schema.field(name).type):
            raise CorpusRunError(
                f"input column {name!r} must have Arrow string type"
            )
    if schema.field("sample_id").nullable:
        raise CorpusRunError("input column 'sample_id' must be non-nullable")


def _input_fingerprint(
    input_file: Path, parquet_file: pq.ParquetFile
) -> dict[str, object]:
    expected_rows = parquet_file.metadata.num_rows
    return {
        "path": str(input_file),
        "sha256": _file_sha256(input_file),
        "schema": parquet_file.schema_arrow.serialize().to_pybytes().hex(),
        "expected_rows": expected_rows,
        "expected_row_groups": parquet_file.num_row_groups,
    }


def _immutable_manifest(
    *,
    run_id: str,
    input_fingerprint: dict[str, object],
    batch_size: int,
    bound_runner: BoundPreprocessingRunner,
) -> dict[str, object]:
    definition = HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION
    serialized_definition = definition.model_dump(mode="json")
    resolved_step_versions = [
        {
            "instance_name": bound_step.instance_name,
            "step": bound_step.step.NAME.value,
            "version": bound_step.step.VERSION,
        }
        for bound_step in bound_runner.bound_steps
    ]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "input": input_fingerprint,
        "definition": serialized_definition,
        "definition_hash": preprocessing_definition_hash(definition),
        "source": _source_fingerprint(),
        "resolved_step_versions": resolved_step_versions,
        "batch_size": batch_size,
    }


def _load_or_create_manifest(
    partial_dir: Path, immutable_manifest: dict[str, object]
) -> dict[str, object]:
    manifest_path = partial_dir / "manifest.json"
    if manifest_path.exists():
        manifest = _read_manifest(manifest_path)
        for key, expected in immutable_manifest.items():
            if manifest.get(key) != expected:
                raise CorpusRunError(
                    f"partial run is incompatible at manifest field {key!r}"
                )
        return manifest
    if partial_dir.exists():
        raise CorpusRunError(
            f"partial run directory lacks a manifest: {partial_dir}"
        )
    manifest = {
        **immutable_manifest,
        "started_at": _timestamp(),
        "updated_at": _timestamp(),
        "completed_row_groups": [],
        "relation_totals": {relation: 0 for relation in RELATION_NAMES},
        "outcome_totals": {},
        "complete": False,
    }
    _create_partial_run(partial_dir, manifest)
    return manifest


def _create_partial_run(
    partial_dir: Path, manifest: Mapping[str, object]
) -> None:
    """Publish the initial directory only after its manifest is durable."""
    partial_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = partial_dir.parent / f".{partial_dir.name}.tmp"
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir()
    try:
        _write_manifest(temporary_dir, manifest)
        os.replace(temporary_dir, partial_dir)
    except BaseException:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise


def _validate_completed_parts(
    partial_dir: Path, completed_row_groups: set[int]
) -> None:
    for row_group_index in completed_row_groups:
        part_id = _part_id(row_group_index)
        for relation in RELATION_NAMES:
            part_path = partial_dir / "parts" / part_id / f"{relation}.parquet"
            if not part_path.is_file():
                raise CorpusRunError(
                    f"missing {relation!r} part for completed row group "
                    f"{row_group_index}"
                )


def _remove_orphan_part(partial_dir: Path, part_id: str) -> None:
    """Remove an uncheckpointed part before safely replacing the row group."""
    parts_dir = partial_dir / "parts"
    for path in (parts_dir / part_id, parts_dir / f".{part_id}.tmp"):
        if path.exists():
            shutil.rmtree(path)


def _completed_row_groups(manifest: Mapping[str, object]) -> set[int]:
    raw_indexes = manifest.get("completed_row_groups")
    if not isinstance(raw_indexes, list) or not all(
        isinstance(index, int) and index >= 0 for index in raw_indexes
    ):
        raise CorpusRunError("manifest completed_row_groups must be int list")
    return set(cast(list[int], raw_indexes))


def _relation_part_totals(
    partial_dir: Path, completed_row_groups: set[int]
) -> dict[str, int]:
    return {
        relation: sum(
            pq.ParquetFile(
                partial_dir
                / "parts"
                / _part_id(row_group_index)
                / f"{relation}.parquet"
            ).metadata.num_rows
            for row_group_index in completed_row_groups
        )
        for relation in RELATION_NAMES
    }


def _outcome_part_totals(
    partial_dir: Path, completed_row_groups: set[int]
) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for row_group_index in completed_row_groups:
        parquet_file = pq.ParquetFile(
            partial_dir
            / "parts"
            / _part_id(row_group_index)
            / "results.parquet"
        )
        for batch in parquet_file.iter_batches(
            batch_size=65_536, columns=["outcome"]
        ):
            totals.update(_outcome_labels(batch))
    return dict(sorted(totals.items()))


def _validate_completed_relations(
    *,
    relation_paths: Mapping[str, Path],
    expected_rows: object,
    input_parquet: pq.ParquetFile,
) -> dict[str, dict[str, int]]:
    if set(relation_paths) != set(RELATION_NAMES):
        raise CorpusRunError(
            "combined relations do not match the artifact schema"
        )
    if not isinstance(expected_rows, int) or expected_rows < 0:
        raise CorpusRunError(
            "manifest expected row count must be non-negative"
        )
    relation_totals = {
        relation: pq.ParquetFile(path).metadata.num_rows
        for relation, path in relation_paths.items()
    }
    if relation_totals["results"] != expected_rows:
        raise CorpusRunError(
            "result row count does not match the input corpus row count"
        )
    outcomes = _validate_relations_against_input(
        input_parquet=input_parquet,
        results_path=relation_paths["results"],
        candidates_path=relation_paths["candidates"],
        step_facts_path=relation_paths["step_facts"],
        rejections_path=relation_paths["rejections"],
    )
    if sum(outcomes.values()) != relation_totals["results"]:
        raise CorpusRunError(
            "outcome totals do not reconcile with result rows"
        )
    return {
        "relation_totals": relation_totals,
        "outcome_totals": dict(sorted(outcomes.items())),
    }


def _validate_relations_against_input(
    *,
    input_parquet: pq.ParquetFile,
    results_path: Path,
    candidates_path: Path,
    step_facts_path: Path,
    rejections_path: Path,
) -> Counter[str]:
    """Validate source-ordered relations while retaining one sample at a time."""
    input_rows = _parquet_rows(input_parquet, REQUIRED_INPUT_COLUMNS)
    result_rows = _parquet_rows(
        pq.ParquetFile(results_path),
        (
            "sample_id",
            "decoder_output_presence",
            "raw_output_sha256",
            "outcome",
            "final_candidate_count",
        ),
    )
    candidate_rows = _parquet_rows(
        pq.ParquetFile(candidates_path),
        ("sample_id", "candidate_id", "candidate_index"),
    )
    step_fact_rows = _parquet_rows(
        pq.ParquetFile(step_facts_path), ("sample_id",)
    )
    rejection_rows = _parquet_rows(
        pq.ParquetFile(rejections_path), ("sample_id",)
    )
    outcomes: Counter[str] = Counter()
    candidate_row = next(candidate_rows, None)
    step_fact_row = next(step_fact_rows, None)
    rejection_row = next(rejection_rows, None)
    result_row = next(result_rows, None)
    with _disk_backed_sample_ids() as sample_ids:
        for input_row in input_rows:
            if result_row is None:
                raise CorpusRunError("results end before the input corpus")
            sample_id = input_row["sample_id"]
            decoder_output = input_row["decoder_output"]
            _validate_result_row(
                input_sample_id=sample_id,
                decoder_output=decoder_output,
                result_row=result_row,
            )
            result_sample_id = result_row["sample_id"]
            expected_count = result_row["final_candidate_count"]
            outcome = result_row["outcome"]
            assert isinstance(result_sample_id, str)
            assert isinstance(expected_count, int)
            assert isinstance(outcome, str)
            try:
                sample_ids.execute(
                    "INSERT INTO sample_ids (sample_id) VALUES (?)",
                    (result_sample_id,),
                )
            except sqlite3.IntegrityError as exc:
                raise CorpusRunError(
                    "results contain duplicate sample_id values"
                ) from exc
            outcomes[outcome] += 1
            candidate_row = _validate_current_sample_candidates(
                sample_id=result_sample_id,
                expected_count=expected_count,
                candidate_row=candidate_row,
                candidate_rows=candidate_rows,
            )
            step_fact_row = _consume_child_rows(
                relation_name="step_facts",
                sample_id=result_sample_id,
                current_row=step_fact_row,
                rows=step_fact_rows,
            )
            rejection_row = _consume_child_rows(
                relation_name="rejections",
                sample_id=result_sample_id,
                current_row=rejection_row,
                rows=rejection_rows,
            )
            result_row = next(result_rows, None)
    if result_row is not None:
        raise CorpusRunError(
            "results contain rows absent from the input corpus"
        )
    for relation_name, row in (
        ("candidates", candidate_row),
        ("step_facts", step_fact_row),
        ("rejections", rejection_row),
    ):
        if row is not None:
            raise CorpusRunError(
                f"{relation_name} contains sample_id absent from results"
            )
    return outcomes


@contextmanager
def _disk_backed_sample_ids() -> Iterator[sqlite3.Connection]:
    """Track exact result uniqueness without corpus-sized Python state."""
    with tempfile.TemporaryDirectory(
        prefix="dr-code-corpus-validation-"
    ) as root:
        connection = sqlite3.connect(Path(root) / "sample-ids.sqlite3")
        try:
            connection.execute(
                "CREATE TABLE sample_ids (sample_id TEXT PRIMARY KEY)"
            )
            yield connection
        finally:
            connection.close()


def _parquet_rows(
    parquet_file: pq.ParquetFile, columns: tuple[str, ...]
) -> Iterator[dict[str, object]]:
    for batch in parquet_file.iter_batches(batch_size=65_536, columns=columns):
        values = [batch.column(column).to_pylist() for column in columns]
        for row_values in zip(*values):
            yield dict(zip(columns, row_values))


def _validate_result_row(
    *,
    input_sample_id: object,
    decoder_output: object,
    result_row: Mapping[str, object],
) -> None:
    sample_id = result_row["sample_id"]
    presence = result_row["decoder_output_presence"]
    raw_output_sha256 = result_row["raw_output_sha256"]
    outcome = result_row["outcome"]
    candidate_count = result_row["final_candidate_count"]
    if not isinstance(input_sample_id, str) or sample_id != input_sample_id:
        raise CorpusRunError(
            "results sample_id does not match the input corpus"
        )
    expected_presence = "missing" if decoder_output is None else "present"
    if presence != expected_presence:
        raise CorpusRunError(
            "results decoder_output_presence does not match the input corpus"
        )
    expected_sha256 = (
        None if decoder_output is None else _text_sha256(str(decoder_output))
    )
    if raw_output_sha256 != expected_sha256:
        raise CorpusRunError(
            "results raw_output_sha256 does not match the input corpus"
        )
    if not isinstance(outcome, str) or not outcome:
        raise CorpusRunError("results must record a non-empty outcome")
    if not isinstance(candidate_count, int) or candidate_count < 0:
        raise CorpusRunError(
            "results final_candidate_count must be non-negative"
        )


def _validate_current_sample_candidates(
    *,
    sample_id: str,
    expected_count: int,
    candidate_row: dict[str, object] | None,
    candidate_rows: Iterator[dict[str, object]],
) -> dict[str, object] | None:
    candidate_ids: set[str] = set()
    candidate_indexes: set[int] = set()
    while (
        candidate_row is not None and candidate_row["sample_id"] == sample_id
    ):
        candidate_id = candidate_row["candidate_id"]
        candidate_index = candidate_row["candidate_index"]
        if not isinstance(candidate_id, str) or not candidate_id:
            raise CorpusRunError("candidate_id must be non-null")
        if not isinstance(candidate_index, int) or candidate_index < 0:
            raise CorpusRunError("candidate_index must be non-negative")
        if candidate_id in candidate_ids:
            raise CorpusRunError(
                f"candidate_id is not unique within sample {sample_id!r}"
            )
        candidate_ids.add(candidate_id)
        candidate_indexes.add(candidate_index)
        candidate_row = next(candidate_rows, None)
    if len(candidate_ids) != expected_count:
        raise CorpusRunError(
            f"candidate count does not match results for {sample_id!r}"
        )
    if candidate_indexes != set(range(expected_count)):
        raise CorpusRunError(
            f"candidate indexes are not contiguous for {sample_id!r}"
        )
    return candidate_row


def _consume_child_rows(
    *,
    relation_name: str,
    sample_id: str,
    current_row: dict[str, object] | None,
    rows: Iterator[dict[str, object]],
) -> dict[str, object] | None:
    while current_row is not None and current_row["sample_id"] == sample_id:
        current_row = next(rows, None)
    return current_row


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusRunError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CorpusRunError(f"manifest {path} must contain a JSON object")
    return payload


def _write_manifest(directory: Path, manifest: Mapping[str, object]) -> None:
    destination = directory / "manifest.json"
    temporary = directory / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)


def _source_fingerprint() -> dict[str, object]:
    return {
        "git_commit": _git_output("rev-parse", "HEAD"),
        "source_diff_sha256": _source_tree_sha256(),
        "python_implementation": platform.python_implementation(),
        "python_version": sys.version,
    }


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _source_tree_sha256() -> str:
    """Hash the runnable source tree, including untracked implementation files."""
    repository_root = Path(__file__).resolve().parents[3]
    paths = [
        *sorted((repository_root / "src").rglob("*.py")),
        *sorted((repository_root / "scripts").rglob("*.py")),
        repository_root / "pyproject.toml",
        repository_root / "uv.lock",
    ]
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            continue
        digest.update(path.relative_to(repository_root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _outcome_labels(table: pa.Table | pa.RecordBatch) -> list[str]:
    outcomes = table.column("outcome").to_pylist()
    if any(
        not isinstance(outcome, str) or not outcome for outcome in outcomes
    ):
        raise CorpusRunError("results must record a non-empty outcome")
    return cast(list[str], outcomes)


def _part_id(row_group_index: int) -> str:
    return f"row_group_{row_group_index:08d}"


def _generated_run_id(input_fingerprint: Mapping[str, object]) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    input_prefix = str(input_fingerprint["sha256"])[:8]
    definition_prefix = preprocessing_definition_hash(
        HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION
    )[:8]
    return f"preprocessing-{timestamp}-{input_prefix}-{definition_prefix}"


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


__all__ = ["CorpusRunError", "run_preprocessing_corpus"]
