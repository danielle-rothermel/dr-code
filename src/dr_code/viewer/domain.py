"""Validated domain contracts for the local preprocessing viewer."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, cast

import pyarrow as pa
import pyarrow.parquet as pq

from dr_code.corpus.candidate_evaluation import (
    MANIFEST_FILENAME as EVALUATION_MANIFEST_FILENAME,
)
from dr_code.corpus.candidate_evaluation import (
    MEMBERSHIP_FILENAME,
    MEMBERSHIP_SCHEMA,
    RESULTS_FILENAME as EVALUATION_RESULTS_FILENAME,
    RESULTS_SCHEMA as EVALUATION_RESULTS_SCHEMA,
)
from dr_code.corpus.preprocessing_artifacts import (
    normalize_persisted_origins,
    projected_artifact_schemas,
)


PREPROCESSING_MANIFEST_FILENAME: Final = "manifest.json"
_PREPROCESSING_SCHEMA_VERSIONS: Final = frozenset({1, 2})
_PREPROCESSING_ARTIFACT_NAMES: Final = tuple(
    f"{name}.parquet" for name in projected_artifact_schemas(2)
)
_SHA256_LENGTH: Final = 64
_INT64_MAX: Final = 2**63 - 1
_VIEWER_STAGE_STEPS: Final = (
    "require_nonblank_text",
    "extract_candidates",
    "filter_compilable",
    "filter_has_top_level_function",
)


class ViewerError(ValueError):
    """Base error for invalid viewer state or requests."""


class RunValidationError(ViewerError):
    """A registered immutable artifact bundle violates its contract."""


class RunNotFoundError(ViewerError):
    """A requested run ID is not registered."""


class InvalidQueryError(ViewerError):
    """A named analytical query received invalid parameters."""


class IncompatibleRunsError(ViewerError):
    """Two runs cannot be compared without producing misleading results."""


class Verdict(StrEnum):
    SHOULD_BE_PARSEABLE = "should_be_parseable"
    EXPECTED_NO_CODE = "expected_no_code"


@dataclass(frozen=True, slots=True)
class RunDescriptor:
    """A fully resolved and fingerprinted immutable run bundle."""

    run_id: str
    label: str
    corpus_path: Path
    corpus_sha256: str
    preprocessing_manifest_path: Path
    preprocessing_manifest_sha256: str
    results_path: Path
    candidates_path: Path
    step_facts_path: Path
    rejections_path: Path
    artifact_sha256: dict[str, str]
    preprocessing_schema_version: int
    definition_id: str
    definition_version: str
    definition_hash: str
    evaluation_manifest_path: Path | None = None
    evaluation_manifest_sha256: str | None = None
    candidate_membership_path: Path | None = None
    candidate_results_path: Path | None = None
    evaluation_coordinates: dict[str, object] | None = None

    @classmethod
    def from_paths(
        cls,
        *,
        label: str,
        corpus_path: str | Path,
        preprocessing: str | Path,
        candidate_evaluation: str | Path | None = None,
    ) -> RunDescriptor:
        """Validate and resolve an explicit corpus and manifest-backed run."""
        normalized_label = label.strip()
        if not normalized_label:
            raise RunValidationError("run label must not be blank")
        corpus = _required_file(corpus_path, "corpus")
        preprocessing_manifest = _resolve_manifest(
            preprocessing,
            PREPROCESSING_MANIFEST_FILENAME,
            "preprocessing",
        )
        preprocessing_root = preprocessing_manifest.parent
        manifest = _read_json_object(
            preprocessing_manifest, "preprocessing manifest"
        )
        preprocessing_schema_version = _validate_preprocessing_manifest(
            manifest, corpus
        )

        paths = {
            name.removesuffix(".parquet"): _required_file(
                preprocessing_root / name, name
            )
            for name in _PREPROCESSING_ARTIFACT_NAMES
        }
        _validate_preprocessing_artifacts(
            paths,
            manifest,
            schema_version=preprocessing_schema_version,
        )
        _validate_stage_facts(
            paths["step_facts"],
            results_path=paths["results"],
            definition=cast(dict[str, object], manifest["definition"]),
        )

        corpus_sha256 = _sha256_file(corpus)
        if (
            cast(dict[str, object], manifest["input"])["sha256"]
            != corpus_sha256
        ):
            raise RunValidationError(
                "preprocessing manifest corpus fingerprint does not match "
                "the explicit corpus"
            )
        manifest_sha256 = _sha256_file(preprocessing_manifest)
        artifact_sha256 = {
            name: _sha256_file(path) for name, path in paths.items()
        }

        evaluation_manifest_path: Path | None = None
        evaluation_manifest_sha256: str | None = None
        candidate_membership_path: Path | None = None
        candidate_results_path: Path | None = None
        evaluation_coordinates: dict[str, object] | None = None
        if candidate_evaluation is not None:
            evaluation_manifest_path = _resolve_manifest(
                candidate_evaluation,
                EVALUATION_MANIFEST_FILENAME,
                "candidate evaluation",
            )
            evaluation_root = evaluation_manifest_path.parent
            candidate_membership_path = _required_file(
                evaluation_root / MEMBERSHIP_FILENAME,
                "candidate membership",
            )
            candidate_results_path = _required_file(
                evaluation_root / EVALUATION_RESULTS_FILENAME,
                "candidate results",
            )
            evaluation_manifest = _read_json_object(
                evaluation_manifest_path, "candidate evaluation manifest"
            )
            evaluation_coordinates = _validate_evaluation_bundle(
                evaluation_manifest,
                evaluation_manifest_path=evaluation_manifest_path,
                membership_path=candidate_membership_path,
                results_path=candidate_results_path,
                corpus_sha256=corpus_sha256,
                preprocessing_manifest_sha256=manifest_sha256,
                candidates_sha256=artifact_sha256["candidates"],
                results_sha256=artifact_sha256["results"],
            )
            evaluation_manifest_sha256 = _sha256_file(evaluation_manifest_path)
            artifact_sha256["candidate_membership"] = _sha256_file(
                candidate_membership_path
            )
            artifact_sha256["candidate_results"] = _sha256_file(
                candidate_results_path
            )

        definition = cast(dict[str, object], manifest["definition"])
        return cls(
            run_id=cast(str, manifest["run_id"]),
            label=normalized_label,
            corpus_path=corpus,
            corpus_sha256=corpus_sha256,
            preprocessing_manifest_path=preprocessing_manifest,
            preprocessing_manifest_sha256=manifest_sha256,
            results_path=paths["results"],
            candidates_path=paths["candidates"],
            step_facts_path=paths["step_facts"],
            rejections_path=paths["rejections"],
            artifact_sha256=artifact_sha256,
            preprocessing_schema_version=preprocessing_schema_version,
            definition_id=cast(str, definition["definition_id"]),
            definition_version=cast(str, definition["version"]),
            definition_hash=cast(str, manifest["definition_hash"]),
            evaluation_manifest_path=evaluation_manifest_path,
            evaluation_manifest_sha256=evaluation_manifest_sha256,
            candidate_membership_path=candidate_membership_path,
            candidate_results_path=candidate_results_path,
            evaluation_coordinates=evaluation_coordinates,
        )

    @classmethod
    def from_file(
        cls, path: str | Path, *, label: str | None = None
    ) -> RunDescriptor:
        """Load the small startup descriptor format and validate its bundle."""
        descriptor_path = _required_file(path, "run descriptor")
        value = _read_json_object(descriptor_path, "run descriptor")
        allowed = {
            "label",
            "corpus",
            "corpus_path",
            "preprocessing",
            "preprocessing_manifest",
            "preprocessing_manifest_path",
            "candidate_evaluation",
            "candidate_evaluation_manifest",
            "candidate_evaluation_manifest_path",
        }
        unknown = sorted(set(value).difference(allowed))
        if unknown:
            raise RunValidationError(
                "run descriptor contains unknown field(s): "
                + ", ".join(unknown)
            )
        configured_label = label if label is not None else value.get("label")
        if not isinstance(configured_label, str):
            raise RunValidationError("run descriptor requires string 'label'")
        corpus = _one_path(value, ("corpus", "corpus_path"), "corpus")
        preprocessing = _one_path(
            value,
            (
                "preprocessing",
                "preprocessing_manifest",
                "preprocessing_manifest_path",
            ),
            "preprocessing",
        )
        evaluation = _optional_one_path(
            value,
            (
                "candidate_evaluation",
                "candidate_evaluation_manifest",
                "candidate_evaluation_manifest_path",
            ),
            "candidate evaluation",
        )
        return cls.from_paths(
            label=configured_label,
            corpus_path=_relative_to(descriptor_path, corpus),
            preprocessing=_relative_to(descriptor_path, preprocessing),
            candidate_evaluation=(
                _relative_to(descriptor_path, evaluation)
                if evaluation is not None
                else None
            ),
        )

    @property
    def has_evaluation(self) -> bool:
        return self.evaluation_manifest_path is not None

    def to_json(self) -> str:
        """Serialize stable registration state for the local catalog."""
        value = asdict(self)
        for key, item in tuple(value.items()):
            if isinstance(item, Path):
                value[key] = str(item)
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class RunSummary:
    run_id: str
    label: str
    manifest_sha256: str
    corpus_sha256: str
    definition_id: str
    definition_version: str
    has_evaluation: bool
    definition_hash: str | None = None


@dataclass(frozen=True, slots=True)
class WaterfallStage:
    stage_id: str
    label: str
    unit: str
    order: int
    count: int
    denominator: int
    rate: float | None


@dataclass(frozen=True, slots=True)
class Waterfall:
    run: RunSummary
    stages: tuple[WaterfallStage, ...]


@dataclass(frozen=True, slots=True)
class FailureGroup:
    failure_code: str
    failed_step: str
    cause: str | None
    count: int


@dataclass(frozen=True, slots=True)
class Failures:
    run: RunSummary
    groups: tuple[FailureGroup, ...]
    total_count: int


@dataclass(frozen=True, slots=True)
class ExampleSummary:
    sample_id: str
    task_id: str | None
    decoder_output_sha256: str | None
    outcome: str
    failure_code: str | None
    failed_step: str | None
    decoder_output: str | None
    annotation_verdict: Verdict | None


@dataclass(frozen=True, slots=True)
class Page:
    items: tuple[ExampleSummary, ...]
    total: int
    limit: int
    offset: int


@dataclass(frozen=True, slots=True)
class Tag:
    tag_id: str
    name: str


@dataclass(frozen=True, slots=True)
class Annotation:
    corpus_sha256: str
    sample_id: str
    decoder_output_sha256: str
    verdict: Verdict | None
    note: str | None
    tags: tuple[Tag, ...]


@dataclass(frozen=True, slots=True)
class ExampleDetail:
    sample_id: str
    corpus_sha256: str
    decoder_output_sha256: str | None
    context: dict[str, object]
    outcome: str
    failure_code: str | None
    failed_step: str | None
    cause: str | None
    raw_decoder_output: str | None
    candidates: tuple[dict[str, object], ...]
    facts: tuple[dict[str, object], ...]
    rejections: tuple[dict[str, object], ...]
    annotation: Annotation | None


@dataclass(frozen=True, slots=True)
class ReviewPage:
    items: tuple[ExampleDetail, ...]
    total: int
    limit: int
    offset: int


@dataclass(frozen=True, slots=True)
class ComparisonStage:
    stage_id: str
    label: str
    unit: str
    baseline_count: int
    baseline_denominator_count: int
    candidate_count: int
    candidate_denominator_count: int
    count_delta: int
    baseline_rate: float | None
    candidate_rate: float | None
    rate_delta: float | None


@dataclass(frozen=True, slots=True)
class OutcomeTransition:
    baseline_outcome: str
    candidate_outcome: str
    count: int


@dataclass(frozen=True, slots=True)
class RunComparison:
    baseline: RunSummary
    candidate: RunSummary
    stages: tuple[ComparisonStage, ...]
    transitions: tuple[OutcomeTransition, ...]


def validate_sha256(value: str, label: str) -> str:
    if len(value) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise InvalidQueryError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _required_file(value: str | Path, label: str) -> Path:
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise RunValidationError(
            f"{label} file does not exist: {value}"
        ) from exc
    if not path.is_file():
        raise RunValidationError(f"{label} is not a file: {path}")
    return path


def _resolve_manifest(value: str | Path, filename: str, label: str) -> Path:
    path = Path(value).expanduser()
    if path.is_dir():
        path = path / filename
    return _required_file(path, f"{label} manifest")


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RunValidationError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RunValidationError(f"{label} must contain a JSON object")
    return cast(dict[str, object], value)


def _validate_preprocessing_manifest(
    manifest: dict[str, object], corpus: Path
) -> int:
    schema_version = manifest.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version not in _PREPROCESSING_SCHEMA_VERSIONS
    ):
        raise RunValidationError(
            "unsupported preprocessing manifest schema_version"
        )
    if manifest.get("complete") is not True:
        raise RunValidationError("preprocessing manifest is incomplete")
    _required_string(manifest, "run_id", "preprocessing manifest")
    _required_digest(manifest, "definition_hash", "preprocessing manifest")
    definition = manifest.get("definition")
    if not isinstance(definition, dict):
        raise RunValidationError(
            "preprocessing manifest definition must be an object"
        )
    definition = cast(dict[str, object], definition)
    _required_string(definition, "definition_id", "preprocessing definition")
    _required_string(definition, "version", "preprocessing definition")
    _validate_stage_contract(definition)
    input_value = manifest.get("input")
    if not isinstance(input_value, dict):
        raise RunValidationError(
            "preprocessing manifest input must be an object"
        )
    input_value = cast(dict[str, object], input_value)
    _required_digest(input_value, "sha256", "preprocessing manifest input")
    expected_rows = input_value.get("expected_rows")
    expected_groups = input_value.get("expected_row_groups")
    parquet = _parquet_file(corpus, "corpus")
    if expected_rows != parquet.metadata.num_rows:
        raise RunValidationError(
            "preprocessing manifest corpus row count does not match corpus"
        )
    if expected_groups != parquet.num_row_groups:
        raise RunValidationError(
            "preprocessing manifest corpus row-group count does not match corpus"
        )
    if (
        input_value.get("schema")
        != parquet.schema_arrow.serialize().to_pybytes().hex()
    ):
        raise RunValidationError(
            "preprocessing manifest corpus schema fingerprint does not match corpus"
        )
    _require_corpus_schema(parquet.schema_arrow)
    return schema_version


def _validate_preprocessing_artifacts(
    paths: dict[str, Path],
    manifest: dict[str, object],
    *,
    schema_version: int,
) -> None:
    totals = manifest.get("relation_totals")
    if not isinstance(totals, dict):
        raise RunValidationError(
            "preprocessing manifest relation_totals must be an object"
        )
    for name, expected_schema in projected_artifact_schemas(
        schema_version
    ).items():
        parquet = _parquet_file(paths[name], name)
        if not parquet.schema_arrow.equals(expected_schema):
            raise RunValidationError(
                f"{name}.parquet has an unexpected schema"
            )
        if totals.get(name) != parquet.metadata.num_rows:
            raise RunValidationError(
                f"preprocessing manifest row count does not match {name}.parquet"
            )
    _validate_candidate_origins(paths["candidates"], schema_version)
    input_value = cast(dict[str, object], manifest["input"])
    if totals.get("results") != input_value.get("expected_rows"):
        raise RunValidationError(
            "preprocessing results do not cover every corpus row"
        )


def _validate_candidate_origins(path: Path, schema_version: int) -> None:
    parquet = _parquet_file(path, "candidates")
    for batch in parquet.iter_batches(
        batch_size=65_536,
        columns=["sample_id", "candidate_id", "origins"],
    ):
        for row in batch.to_pylist():
            try:
                normalize_persisted_origins(row["origins"], schema_version)
            except ValueError as exc:
                raise RunValidationError(
                    "candidate origins are invalid for "
                    f"{row['sample_id']!r}/{row['candidate_id']!r}: {exc}"
                ) from exc


def _validate_evaluation_bundle(
    manifest: dict[str, object],
    *,
    evaluation_manifest_path: Path,
    membership_path: Path,
    results_path: Path,
    corpus_sha256: str,
    preprocessing_manifest_sha256: str,
    candidates_sha256: str,
    results_sha256: str,
) -> dict[str, object]:
    if manifest.get("schema_version") != 1:
        raise RunValidationError(
            "unsupported candidate evaluation manifest schema_version"
        )
    if manifest.get("complete") is not True:
        raise RunValidationError("candidate evaluation manifest is incomplete")
    expected_hashes = {
        "corpus_sha256": corpus_sha256,
        "preprocessing_manifest_sha256": preprocessing_manifest_sha256,
        "preprocessing_candidates_sha256": candidates_sha256,
        "preprocessing_results_sha256": results_sha256,
    }
    for field, expected in expected_hashes.items():
        if manifest.get(field) != expected:
            raise RunValidationError(
                f"candidate evaluation manifest {field} mismatch"
            )
    membership = _parquet_file(membership_path, "candidate membership")
    results = _parquet_file(results_path, "candidate results")
    if not membership.schema_arrow.equals(MEMBERSHIP_SCHEMA):
        raise RunValidationError(
            "candidate_membership.parquet has an unexpected schema"
        )
    if not results.schema_arrow.equals(EVALUATION_RESULTS_SCHEMA):
        raise RunValidationError(
            "candidate_results.parquet has an unexpected schema"
        )
    if manifest.get("membership_rows") != membership.metadata.num_rows:
        raise RunValidationError(
            "candidate evaluation membership row count mismatch"
        )
    if manifest.get("result_rows") != results.metadata.num_rows:
        raise RunValidationError(
            "candidate evaluation result row count mismatch"
        )
    _validate_evaluation_artifact_hashes(
        manifest,
        membership_path=membership_path,
        results_path=results_path,
    )
    coordinates: dict[str, object] = {}
    for field in (
        "metrics_profile",
        "operator",
        "metrics_definition_hash",
        "snapshot_sha256",
        "runner_identity",
        "execution_fingerprint",
        "operator_settings",
    ):
        if field not in manifest:
            raise RunValidationError(
                f"candidate evaluation manifest is missing {field!r}"
            )
        coordinates[field] = manifest[field]
    for field in (
        "metrics_profile",
        "operator",
        "metrics_definition_hash",
        "snapshot_sha256",
        "runner_identity",
        "execution_fingerprint",
    ):
        if not isinstance(coordinates[field], str) or not coordinates[field]:
            raise RunValidationError(
                f"candidate evaluation manifest has invalid {field!r}"
            )
    if not isinstance(coordinates["operator_settings"], dict):
        raise RunValidationError(
            "candidate evaluation manifest operator_settings must be an object"
        )
    # Include this in diagnostics without making a machine-local path semantic.
    coordinates["manifest_name"] = evaluation_manifest_path.name
    return coordinates


def _validate_evaluation_artifact_hashes(
    manifest: dict[str, object],
    *,
    membership_path: Path,
    results_path: Path,
) -> None:
    fields = {
        "candidate_membership_sha256": membership_path,
        "candidate_results_sha256": results_path,
    }
    present = [field for field in fields if field in manifest]
    if not present:
        return
    if len(present) != len(fields):
        missing = sorted(set(fields) - set(present))
        raise RunValidationError(
            "candidate evaluation manifest is missing artifact hash field(s): "
            + ", ".join(missing)
        )
    for field, path in fields.items():
        expected = _required_sha256(
            manifest, field, "candidate evaluation manifest"
        )
        if expected != _sha256_file(path):
            raise RunValidationError(
                f"candidate evaluation manifest {field} mismatch"
            )


def _parquet_file(path: Path, label: str) -> pq.ParquetFile:
    try:
        return pq.ParquetFile(path)
    except (OSError, pa.ArrowException) as exc:
        raise RunValidationError(
            f"{label} is not a readable Parquet file"
        ) from exc


def _require_corpus_schema(schema: pa.Schema) -> None:
    required = {
        "sample_id": (pa.string(), False),
        "decoder_output": (pa.string(), True),
    }
    for name, (data_type, nullable) in required.items():
        index = schema.get_field_index(name)
        if index < 0:
            raise RunValidationError(
                f"corpus is missing required column {name!r}"
            )
        field = schema.field(index)
        if field.type != data_type or field.nullable != nullable:
            raise RunValidationError(
                f"corpus column {name!r} has an unexpected schema"
            )


def _validate_stage_contract(definition: dict[str, object]) -> None:
    steps = definition.get("steps")
    if not isinstance(steps, list) or not all(
        isinstance(step, dict) for step in steps
    ):
        raise RunValidationError(
            "preprocessing definition steps must be an object list"
        )
    typed_steps = cast(list[dict[str, object]], steps)
    names = [
        step.get("instance_name")
        for step in typed_steps
        if isinstance(step.get("instance_name"), str)
    ]
    required = _VIEWER_STAGE_STEPS
    missing = sorted(set(required).difference(names))
    if missing:
        raise RunValidationError(
            "preprocessing definition lacks viewer stage step(s): "
            + ", ".join(missing)
        )
    repeated = sorted(name for name in required if names.count(name) > 1)
    if repeated:
        raise RunValidationError(
            "preprocessing definition repeats viewer stage step(s): "
            + ", ".join(repeated)
        )
    indexes = [names.index(name) for name in required]
    if indexes != sorted(indexes):
        raise RunValidationError(
            "preprocessing definition viewer stage steps are out of order"
        )


def _validate_stage_facts(
    path: Path,
    *,
    results_path: Path,
    definition: dict[str, object],
) -> None:
    contracts: dict[str, tuple[str, type[object]]] = {
        "require_nonblank_text": ("is_nonblank", bool),
        "extract_candidates": ("candidate_count", int),
        "filter_compilable": ("survivor_candidate_count", int),
        "filter_has_top_level_function": (
            "survivor_candidate_count",
            int,
        ),
    }
    actual: dict[str, set[str]] = {step_name: set() for step_name in contracts}
    parquet = _parquet_file(path, "step facts")
    for batch in parquet.iter_batches(
        batch_size=65_536,
        columns=["sample_id", "step_name", "facts_json"],
    ):
        for row in batch.to_pylist():
            step_name = row["step_name"]
            if step_name not in contracts:
                continue
            sample_id = row["sample_id"]
            assert isinstance(sample_id, str)
            assert isinstance(step_name, str)
            if sample_id in actual[step_name]:
                raise RunValidationError(
                    "step_facts contains duplicate viewer stage fact: "
                    f"{sample_id!r}/{step_name!r}"
                )
            actual[step_name].add(sample_id)
            try:
                facts = json.loads(row["facts_json"])
            except (TypeError, json.JSONDecodeError) as exc:
                raise RunValidationError(
                    "step_facts contains invalid viewer stage facts JSON: "
                    f"{sample_id!r}/{step_name!r}"
                ) from exc
            if not isinstance(facts, dict):
                raise RunValidationError(
                    "viewer stage facts must contain a JSON object: "
                    f"{sample_id!r}/{step_name!r}"
                )
            key, expected_type = contracts[step_name]
            value = facts.get(key)
            if expected_type is bool:
                valid = isinstance(value, bool)
            else:
                valid = (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and 0 <= value <= _INT64_MAX
                )
            if not valid:
                raise RunValidationError(
                    f"viewer stage fact {sample_id!r}/{step_name!r} "
                    f"requires valid {key!r}"
                )
    expected = _expected_stage_fact_samples(results_path, definition)
    for step_name in _VIEWER_STAGE_STEPS:
        missing = expected[step_name].difference(actual[step_name])
        extra = actual[step_name].difference(expected[step_name])
        if missing or extra:
            detail = []
            if missing:
                detail.append(
                    f"missing={len(missing)} (first={min(missing)!r})"
                )
            if extra:
                detail.append(f"extra={len(extra)} (first={min(extra)!r})")
            raise RunValidationError(
                f"viewer stage fact coverage mismatch for {step_name!r}: "
                + ", ".join(detail)
            )


def _expected_stage_fact_samples(
    results_path: Path, definition: dict[str, object]
) -> dict[str, set[str]]:
    steps = cast(list[dict[str, object]], definition["steps"])
    order = {
        cast(str, step["instance_name"]): index
        for index, step in enumerate(steps)
        if isinstance(step.get("instance_name"), str)
    }
    expected = {step_name: set() for step_name in _VIEWER_STAGE_STEPS}
    parquet = _parquet_file(results_path, "results")
    for batch in parquet.iter_batches(
        batch_size=65_536,
        columns=[
            "sample_id",
            "decoder_output_presence",
            "failed_step",
        ],
    ):
        for row in batch.to_pylist():
            if row["decoder_output_presence"] != "present":
                continue
            sample_id = row["sample_id"]
            failed_step = row["failed_step"]
            assert isinstance(sample_id, str)
            if failed_step is None:
                reached_through = len(steps)
            elif failed_step in order:
                reached_through = order[failed_step]
            else:
                raise RunValidationError(
                    "preprocessing result names a failed_step absent from "
                    f"the definition: {sample_id!r}/{failed_step!r}"
                )
            for step_name in _VIEWER_STAGE_STEPS:
                if order[step_name] <= reached_through:
                    expected[step_name].add(sample_id)
    return expected


def _required_string(value: dict[str, object], field: str, label: str) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item:
        raise RunValidationError(f"{label} has invalid {field!r}")
    return item


def _required_digest(value: dict[str, object], field: str, label: str) -> str:
    item = _required_string(value, field, label)
    if len(item) not in {_SHA256_LENGTH, 128} or any(
        character not in "0123456789abcdef" for character in item
    ):
        raise RunValidationError(f"{label} has invalid {field!r}")
    return item


def _required_sha256(value: dict[str, object], field: str, label: str) -> str:
    item = _required_string(value, field, label)
    if len(item) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in item
    ):
        raise RunValidationError(f"{label} has invalid {field!r}")
    return item


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _one_path(
    value: dict[str, object], fields: tuple[str, ...], label: str
) -> str:
    selected = [value[field] for field in fields if field in value]
    if len(selected) != 1 or not isinstance(selected[0], str):
        raise RunValidationError(
            f"run descriptor requires exactly one string {label} path"
        )
    return selected[0]


def _optional_one_path(
    value: dict[str, object], fields: tuple[str, ...], label: str
) -> str | None:
    selected = [value[field] for field in fields if field in value]
    if not selected:
        return None
    if len(selected) != 1 or not isinstance(selected[0], str):
        raise RunValidationError(
            f"run descriptor accepts at most one string {label} path"
        )
    return selected[0]


def _relative_to(descriptor_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else descriptor_path.parent / path


__all__ = (
    "Annotation",
    "ComparisonStage",
    "ExampleDetail",
    "ExampleSummary",
    "FailureGroup",
    "Failures",
    "IncompatibleRunsError",
    "InvalidQueryError",
    "OutcomeTransition",
    "Page",
    "RunComparison",
    "RunDescriptor",
    "RunNotFoundError",
    "RunSummary",
    "ReviewPage",
    "RunValidationError",
    "Tag",
    "Verdict",
    "ViewerError",
    "Waterfall",
    "WaterfallStage",
    "validate_sha256",
)
