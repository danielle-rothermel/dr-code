"""Transport-neutral values for the preprocessing viewer."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Final, Mapping, TypeAlias, cast

from dr_code.corpus.run_descriptor import (
    RunDescriptor,
    RunValidationError,
)

_SHA256_LENGTH: Final = 64
TASK_IDENTITY_MAX_LENGTH: Final = 256
TASK_CATEGORY_MAX_LENGTH: Final = 256
TASK_NOTE_MAX_LENGTH: Final = 10_000
TASK_TAG_IDS_MAX_ITEMS: Final = 100
TASK_PROVENANCE_TEXT_MAX_LENGTH: Final = 256
TASK_PROVENANCE_REPEATS_MAX: Final = 10_000
TASK_PROVENANCE_JSON_MAX_LENGTH: Final = 100_000
TASK_PROVENANCE_JSON_SAFE_INTEGER_MAX: Final = 9_007_199_254_740_991
_TASK_PROVENANCE_INTEGER_MAX_DIGITS: Final = 4_300

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
FrozenJsonValue: TypeAlias = (
    JsonScalar | tuple[object, ...] | Mapping[str, object]
)


class ViewerError(ValueError):
    """Base error for invalid viewer state or requests."""


class RunNotFoundError(ViewerError):
    """A requested run ID is not registered."""


class InvalidQueryError(ViewerError):
    """An analytical query received invalid parameters."""


class InvalidTaskAnnotationError(InvalidQueryError):
    """A task annotation failed its shared domain contract."""


class TaskNotFoundError(ViewerError):
    """A task identity is absent from every registered corpus."""


class IncompatibleRunsError(ViewerError):
    """Two runs cannot be compared without misleading results."""


class Verdict(StrEnum):
    SHOULD_BE_PARSEABLE = "should_be_parseable"
    EXPECTED_NO_CODE = "expected_no_code"


@dataclass(frozen=True, slots=True)
class RunSummary:
    run_id: str
    label: str
    dataset_id: str
    manifest_sha256: str
    corpus_sha256: str
    definition_id: str
    definition_version: str
    has_evaluation: bool
    definition_identity: str


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
class ParseFailureClassificationInput:
    """One preprocessing failure exposed to a classifier."""

    sample_id: str
    dataset_id: str
    task_id: str | None
    task_identity: str | None
    decoder_output: str
    failure_code: str
    failed_step: str
    cause: str | None
    task_context: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ParseFailureClassificationInputs:
    """A deterministic, optionally capped parse-failure population."""

    items: tuple[ParseFailureClassificationInput, ...]
    total: int


@dataclass(frozen=True, slots=True)
class CandidateTestFailureClassificationInput:
    """One measured candidate test failure exposed to a classifier."""

    sample_id: str
    dataset_id: str
    task_id: str
    task_identity: str
    candidate_id: str
    evaluation_key: str
    cleaned_source: str
    outcome: str
    function_count: int
    best_function_name: str | None
    total_cases: int
    passed_count: int
    failed_count: int
    error_count: int
    timeout_count: int
    coverage_complete: bool
    task_context: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CandidateTestFailureClassificationInputs:
    """A deterministic, optionally capped candidate-test population."""

    items: tuple[CandidateTestFailureClassificationInput, ...]
    total: int


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


class TaskAnnotationOrigin(StrEnum):
    HUMAN = "human"
    MACHINE = "machine"


@dataclass(frozen=True, slots=True)
class TaskIdentity:
    """An authenticated benchmark task identity independent of any run."""

    dataset_id: str
    task_id: str
    task_identity: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dataset_id",
            _validate_task_identity_field(self.dataset_id, "dataset_id"),
        )
        object.__setattr__(
            self,
            "task_id",
            _validate_task_identity_field(self.task_id, "task_id"),
        )
        object.__setattr__(
            self,
            "task_identity",
            _validate_task_identity_sha256(self.task_identity),
        )


@dataclass(frozen=True, slots=True)
class TaskAnnotationProvenance:
    """Strict, immutable machine-classification provenance."""

    model: str | None = None
    taxonomy_version: str | None = None
    repeats: int | None = None
    agreement: float | None = None
    extra: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "model", _validate_provenance_text(self.model, "model")
        )
        object.__setattr__(
            self,
            "taxonomy_version",
            _validate_provenance_text(
                self.taxonomy_version, "taxonomy_version"
            ),
        )
        repeats = self.repeats
        if repeats is not None and (
            isinstance(repeats, bool)
            or not isinstance(repeats, int)
            or not 1 <= repeats <= TASK_PROVENANCE_REPEATS_MAX
        ):
            raise InvalidTaskAnnotationError(
                "task annotation provenance repeats must be an integer "
                f"between 1 and {TASK_PROVENANCE_REPEATS_MAX}"
            )
        agreement = self.agreement
        if agreement is not None:
            normalized_agreement = _finite_number(agreement, "agreement")
            if not 0 <= normalized_agreement <= 1:
                raise InvalidTaskAnnotationError(
                    "task annotation provenance agreement must be a finite "
                    "number between 0 and 1"
                )
            object.__setattr__(self, "agreement", normalized_agreement)
        try:
            frozen_extra = _freeze_json_object(self.extra, "extra")
        except RecursionError as exc:
            raise InvalidTaskAnnotationError(
                "task annotation provenance extra is cyclic or too deeply "
                "nested"
            ) from exc
        object.__setattr__(self, "extra", frozen_extra)
        if len(encode_task_annotation_provenance(self)) > (
            TASK_PROVENANCE_JSON_MAX_LENGTH
        ):
            raise InvalidTaskAnnotationError(
                "task annotation provenance JSON must be at most "
                f"{TASK_PROVENANCE_JSON_MAX_LENGTH} characters"
            )


@dataclass(frozen=True, slots=True)
class TaskAnnotation:
    identity: TaskIdentity
    origin: TaskAnnotationOrigin
    category: str | None
    note: str | None
    tags: tuple[Tag, ...]
    provenance: TaskAnnotationProvenance | None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, TaskIdentity):
            raise InvalidTaskAnnotationError(
                "task annotation identity must be a TaskIdentity"
            )
        try:
            origin = TaskAnnotationOrigin(self.origin)
        except (TypeError, ValueError) as exc:
            raise InvalidTaskAnnotationError(
                f"unsupported task annotation origin: {self.origin}"
            ) from exc
        object.__setattr__(self, "origin", origin)
        object.__setattr__(
            self, "category", _normalize_task_category(self.category)
        )
        object.__setattr__(self, "note", _normalize_task_note(self.note))
        tags = tuple(self.tags)
        if len(tags) > TASK_TAG_IDS_MAX_ITEMS:
            raise InvalidTaskAnnotationError(
                "task annotation tags must contain at most "
                f"{TASK_TAG_IDS_MAX_ITEMS} items"
            )
        if any(not isinstance(tag, Tag) for tag in tags):
            raise InvalidTaskAnnotationError(
                "task annotation tags must contain Tag values"
            )
        object.__setattr__(self, "tags", tags)
        if self.provenance is not None and not isinstance(
            self.provenance, TaskAnnotationProvenance
        ):
            raise InvalidTaskAnnotationError(
                "task annotation provenance must be a "
                "TaskAnnotationProvenance or null"
            )
        if (
            origin is TaskAnnotationOrigin.HUMAN
            and self.provenance is not None
        ):
            raise InvalidTaskAnnotationError(
                "human task annotations must not contain provenance"
            )
        if origin is TaskAnnotationOrigin.MACHINE and self.provenance is None:
            raise InvalidTaskAnnotationError(
                "machine task annotations require provenance"
            )


class MachineTaskAnnotationWriteOutcome(StrEnum):
    WRITTEN = "written"
    PROTECTED = "protected"


@dataclass(frozen=True, slots=True)
class MachineTaskAnnotationWriteResult:
    outcome: MachineTaskAnnotationWriteOutcome
    annotation: TaskAnnotation


@dataclass(frozen=True, slots=True)
class MachineTaskAnnotationBatchResult:
    written: int
    protected: int
    removed: int


@dataclass(frozen=True, slots=True)
class TaskAnnotationPublicationIntent:
    """Durable handoff between one artifact and its machine rollups."""

    producer: str
    experiment_identity: str
    output_path: str
    staged_path: str
    prior_sha256: str | None
    intended_sha256: str


@dataclass(frozen=True, slots=True)
class ExampleDetail:
    sample_id: str
    dataset_id: str | None
    task_identity: str | None
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


def validate_task_identity(
    dataset_id: str,
    task_id: str,
    task_identity: str,
) -> TaskIdentity:
    """Validate and defensively copy an authenticated task identity."""
    return TaskIdentity(
        dataset_id=dataset_id,
        task_id=task_id,
        task_identity=task_identity,
    )


def validate_task_tag_ids(tag_ids: object) -> tuple[str, ...]:
    """Validate, deduplicate, and canonically order task tag IDs."""
    if isinstance(tag_ids, (str, bytes)):
        raise InvalidTaskAnnotationError(
            "task annotation tag_ids must be a collection"
        )
    try:
        values = tuple(cast(Iterable[object], tag_ids))
    except TypeError as exc:
        raise InvalidTaskAnnotationError(
            "task annotation tag_ids must be a collection"
        ) from exc
    if len(values) > TASK_TAG_IDS_MAX_ITEMS:
        raise InvalidTaskAnnotationError(
            "task annotation tag_ids must contain at most "
            f"{TASK_TAG_IDS_MAX_ITEMS} items"
        )
    if any(
        not isinstance(tag_id, str)
        or not tag_id
        or tag_id.strip() != tag_id
        or len(tag_id) > TASK_IDENTITY_MAX_LENGTH
        for tag_id in values
    ):
        raise InvalidTaskAnnotationError(
            "task annotation tag_ids must contain nonblank strings"
        )
    return tuple(sorted(set(values)))


def validate_task_annotation(
    *,
    identity: TaskIdentity,
    origin: TaskAnnotationOrigin | str,
    category: str | None,
    note: str | None,
    tags: tuple[Tag, ...],
    provenance: TaskAnnotationProvenance | None,
) -> TaskAnnotation:
    """Apply the one domain boundary used by persistence and services."""
    return TaskAnnotation(
        identity=validate_task_identity(
            identity.dataset_id,
            identity.task_id,
            identity.task_identity,
        ),
        origin=cast(TaskAnnotationOrigin, origin),
        category=category,
        note=note,
        tags=tuple(tags),
        provenance=(
            decode_task_annotation_provenance(
                encode_task_annotation_provenance(provenance)
            )
            if provenance is not None
            else None
        ),
    )


def encode_task_annotation_provenance(
    provenance: TaskAnnotationProvenance,
) -> str:
    """Return canonical JSON with the exact persisted provenance shape."""
    if not isinstance(provenance, TaskAnnotationProvenance):
        raise InvalidTaskAnnotationError(
            "task annotation provenance must be a TaskAnnotationProvenance"
        )
    try:
        payload = task_annotation_provenance_json(provenance)
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (OverflowError, RecursionError, ValueError) as exc:
        raise InvalidTaskAnnotationError(
            "task annotation provenance contains an unsupported numeric value"
        ) from exc


def decode_task_annotation_provenance(
    value: str,
) -> TaskAnnotationProvenance:
    """Reject corrupt stored JSON and return immutable provenance."""
    if not isinstance(value, str):
        raise InvalidTaskAnnotationError(
            "task annotation provenance must be stored as JSON text"
        )
    if len(value) > TASK_PROVENANCE_JSON_MAX_LENGTH:
        raise InvalidTaskAnnotationError(
            "task annotation provenance JSON must be at most "
            f"{TASK_PROVENANCE_JSON_MAX_LENGTH} characters"
        )
    try:
        payload = json.loads(
            value,
            parse_constant=lambda constant: _raise_invalid_json(constant),
            parse_int=_parse_json_int,
            object_pairs_hook=_strict_json_object,
        )
    except (json.JSONDecodeError, RecursionError, UnicodeError) as exc:
        raise InvalidTaskAnnotationError(
            "task annotation provenance is not valid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise InvalidTaskAnnotationError(
            "task annotation provenance must be a JSON object"
        )
    expected = {
        "model",
        "taxonomy_version",
        "repeats",
        "agreement",
        "extra",
    }
    unknown = sorted(set(payload).difference(expected))
    missing = sorted(expected.difference(payload))
    if missing or unknown:
        details = []
        if missing:
            details.append("missing field(s): " + ", ".join(missing))
        if unknown:
            details.append("unknown field(s): " + ", ".join(unknown))
        raise InvalidTaskAnnotationError(
            "task annotation provenance must contain exactly model, "
            "taxonomy_version, repeats, agreement, and extra ("
            + "; ".join(details)
            + ")"
        )
    extra = payload["extra"]
    if not isinstance(extra, dict):
        raise InvalidTaskAnnotationError(
            "task annotation provenance extra must be a JSON object"
        )
    return TaskAnnotationProvenance(
        model=_optional_str(payload["model"], "model"),
        taxonomy_version=_optional_str(
            payload["taxonomy_version"], "taxonomy_version"
        ),
        repeats=_optional_int(payload["repeats"], "repeats"),
        agreement=_optional_number(payload["agreement"], "agreement"),
        extra=extra,
    )


def task_annotation_provenance_json(
    provenance: TaskAnnotationProvenance,
) -> dict[str, JsonValue]:
    """Return a mutable defensive copy for serialized boundaries."""
    return {
        "model": provenance.model,
        "taxonomy_version": provenance.taxonomy_version,
        "repeats": provenance.repeats,
        "agreement": provenance.agreement,
        "extra": cast(dict[str, JsonValue], _thaw_json(provenance.extra)),
    }


def _validate_task_identity_field(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise InvalidTaskAnnotationError(f"{label} must be a string")
    if not value or value.strip() != value:
        raise InvalidTaskAnnotationError(
            f"{label} must not be blank or surrounded by whitespace"
        )
    if len(value) > TASK_IDENTITY_MAX_LENGTH:
        raise InvalidTaskAnnotationError(
            f"{label} must be at most {TASK_IDENTITY_MAX_LENGTH} characters"
        )
    return value


def _validate_task_identity_sha256(value: object) -> str:
    if not isinstance(value, str):
        raise InvalidTaskAnnotationError("task_identity must be a string")
    if len(value) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise InvalidTaskAnnotationError(
            "task_identity must be a lowercase SHA-256 digest"
        )
    return value


def _normalize_task_category(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidTaskAnnotationError(
            "task annotation category must be a string or null"
        )
    normalized = value.strip()
    if not normalized:
        return None
    if len(normalized) > TASK_CATEGORY_MAX_LENGTH:
        raise InvalidTaskAnnotationError(
            "task annotation category must be at most "
            f"{TASK_CATEGORY_MAX_LENGTH} characters"
        )
    return normalized


def _normalize_task_note(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidTaskAnnotationError(
            "task annotation note must be a string or null"
        )
    if len(value) > TASK_NOTE_MAX_LENGTH:
        raise InvalidTaskAnnotationError(
            "task annotation note must be at most "
            f"{TASK_NOTE_MAX_LENGTH} characters"
        )
    return value


def _validate_provenance_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidTaskAnnotationError(
            f"task annotation provenance {label} must be a string or null"
        )
    if not value or value.strip() != value:
        raise InvalidTaskAnnotationError(
            f"task annotation provenance {label} must not be blank or "
            "surrounded by whitespace"
        )
    if len(value) > TASK_PROVENANCE_TEXT_MAX_LENGTH:
        raise InvalidTaskAnnotationError(
            f"task annotation provenance {label} must be at most "
            f"{TASK_PROVENANCE_TEXT_MAX_LENGTH} characters"
        )
    return value


def _freeze_json_object(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise InvalidTaskAnnotationError(
            f"task annotation provenance {label} must be a JSON object"
        )
    frozen: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise InvalidTaskAnnotationError(
                "task annotation provenance JSON object keys must be strings"
            )
        frozen[key] = _freeze_json_value(item)
    return MappingProxyType(frozen)


def _freeze_json_value(value: object) -> FrozenJsonValue:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        if (
            not -TASK_PROVENANCE_JSON_SAFE_INTEGER_MAX
            <= value
            <= (TASK_PROVENANCE_JSON_SAFE_INTEGER_MAX)
        ):
            raise InvalidTaskAnnotationError(
                "task annotation provenance JSON integers must be within "
                "the JavaScript safe integer range"
            )
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidTaskAnnotationError(
                "task annotation provenance JSON numbers must be finite"
            )
        return value
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    if isinstance(value, Mapping):
        return _freeze_json_object(value, "extra")
    raise InvalidTaskAnnotationError(
        "task annotation provenance extra must contain only JSON values"
    )


def _thaw_json(value: object) -> JsonValue:
    if isinstance(value, Mapping):
        thawed: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise InvalidTaskAnnotationError(
                    "task annotation provenance JSON object keys must "
                    "be strings"
                )
            thawed[key] = _thaw_json(item)
        return thawed
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return cast(JsonScalar, value)


def _optional_str(value: object, label: str) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise InvalidTaskAnnotationError(
        f"task annotation provenance {label} must be a string or null"
    )


def _optional_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidTaskAnnotationError(
            f"task annotation provenance {label} must be an integer or null"
        )
    return value


def _optional_number(value: object, label: str) -> float | None:
    if value is None:
        return None
    return _finite_number(value, label)


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidTaskAnnotationError(
            f"task annotation provenance {label} must be a number or null"
        )
    try:
        normalized = float(value)
    except OverflowError as exc:
        raise InvalidTaskAnnotationError(
            f"task annotation provenance {label} must be a finite number"
        ) from exc
    if not math.isfinite(normalized):
        raise InvalidTaskAnnotationError(
            f"task annotation provenance {label} must be a finite number"
        )
    return normalized


def _raise_invalid_json(constant: str) -> object:
    raise InvalidTaskAnnotationError(
        f"task annotation provenance contains invalid JSON number {constant}"
    )


def _parse_json_int(value: str) -> int:
    digits = value.removeprefix("-")
    if len(digits) > _TASK_PROVENANCE_INTEGER_MAX_DIGITS:
        raise InvalidTaskAnnotationError(
            "task annotation provenance JSON integer exceeds "
            f"{_TASK_PROVENANCE_INTEGER_MAX_DIGITS} digits"
        )
    try:
        return int(value)
    except ValueError as exc:
        raise InvalidTaskAnnotationError(
            "task annotation provenance contains an invalid JSON integer"
        ) from exc


def _strict_json_object(
    pairs: list[tuple[str, JsonValue]],
) -> dict[str, JsonValue]:
    value: dict[str, JsonValue] = {}
    for key, item in pairs:
        if key in value:
            raise InvalidTaskAnnotationError(
                "task annotation provenance JSON object keys must be unique"
            )
        value[key] = item
    return value


__all__ = (
    "Annotation",
    "CandidateTestFailureClassificationInput",
    "CandidateTestFailureClassificationInputs",
    "ExampleDetail",
    "ExampleSummary",
    "FailureGroup",
    "Failures",
    "IncompatibleRunsError",
    "InvalidQueryError",
    "InvalidTaskAnnotationError",
    "MachineTaskAnnotationWriteOutcome",
    "MachineTaskAnnotationWriteResult",
    "MachineTaskAnnotationBatchResult",
    "OutcomeTransition",
    "Page",
    "ParseFailureClassificationInput",
    "ParseFailureClassificationInputs",
    "ReviewPage",
    "RunComparison",
    "RunDescriptor",
    "RunNotFoundError",
    "RunSummary",
    "RunValidationError",
    "Tag",
    "TaskAnnotation",
    "TaskAnnotationOrigin",
    "TaskAnnotationProvenance",
    "TaskIdentity",
    "TaskNotFoundError",
    "Verdict",
    "ViewerError",
    "Waterfall",
    "WaterfallStage",
    "validate_sha256",
    "decode_task_annotation_provenance",
    "encode_task_annotation_provenance",
    "task_annotation_provenance_json",
    "validate_task_annotation",
    "validate_task_identity",
    "validate_task_tag_ids",
)
