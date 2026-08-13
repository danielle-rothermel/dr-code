from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import UNIQUE, StrEnum, verify
from pathlib import Path, PurePosixPath
from typing import Annotated, Final, Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from dr_code.core.models import FrozenModel
from drc_generation_corpus.models import (
    DatasetName,
    TaskMaterialFidelity,
    TaskRecord,
)
from drc_generation_corpus.pool_dump import canonical_json, content_sha256

_ARCHIVE_IDENTITY: Final = (
    "lla-2026.02.15_08.54p_nl_latents@720bcba473f084bf54ad87f2197b181724a29c96"
)
_PRIMARY_ROOT: Final = "data/tasks"
_SEED41_ROOT: Final = "data/tasks_seed41_u5"
_WORKSHOP_ROOT: Final = "data/tasks_workshop_core_f3_d2to5_seed41_u5"
_AMBIGUOUS_TASK_ID: Final = "stateful_225d71b320455b55"
_SMOKE_TASK_ID: Final = "check_02_add_one"
_TASK_ID_PATTERNS: Final = {
    "stateful": re.compile(r"^stateful_[0-9a-f]{16}$"),
    "bitops": re.compile(r"^bitops_[0-9a-f]{16}$"),
    "stringrules": re.compile(r"^stringrules_[0-9a-f]{16}$"),
    "humanevalpp": re.compile(r"^humanevalpp_humaneval_[0-9]+$"),
    "smoke": re.compile(r"^check_02_add_one$"),
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: JsonValue) -> str:
    return _sha256_text(canonical_json(value))


@verify(UNIQUE)
class NlLatentsFamily(StrEnum):
    STATEFUL = "stateful"
    HUMANEVALPP = "humanevalpp"
    BITOPS = "bitops"
    STRINGRULES = "stringrules"
    SMOKE = "smoke"


@verify(UNIQUE)
class NlLatentsLanguage(StrEnum):
    PYTHON = "python"
    JAVA = "java"
    RUST = "rust"


@verify(UNIQUE)
class NlLatentsSplit(StrEnum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


@verify(UNIQUE)
class NlLatentsTaskRoot(StrEnum):
    PRIMARY = _PRIMARY_ROOT
    SEED41 = _SEED41_ROOT
    EMBEDDED_SMOKE = "embedded_smoke_task"


class NlLatentsTaskCoordinate(FrozenModel):
    task_data_version: Literal[
        "tasks_v1_pre_resample_2026_02_10",
        "tasks_v2_resampled_2026_02_11",
    ]
    family: NlLatentsFamily
    difficulty: Literal[3, 4]
    split: NlLatentsSplit
    language: NlLatentsLanguage
    task_id: Annotated[str, Field(min_length=1)]

    @field_validator("language", mode="before")
    @classmethod
    def _normalize_smoke_language(cls, value: object) -> object:
        return "python" if value == "Python" else value

    @model_validator(mode="after")
    def _validate_family_grammar(self) -> NlLatentsTaskCoordinate:
        if not _TASK_ID_PATTERNS[self.family.value].fullmatch(self.task_id):
            raise ValueError(
                f"task_id {self.task_id!r} does not match family "
                f"{self.family.value!r}"
            )

        coordinate = (
            self.difficulty,
            self.split,
            self.language,
        )
        allowed: dict[
            NlLatentsFamily,
            set[tuple[int, NlLatentsSplit, NlLatentsLanguage]],
        ] = {
            NlLatentsFamily.STATEFUL: {
                (3, NlLatentsSplit.TRAIN, NlLatentsLanguage.PYTHON),
                (3, NlLatentsSplit.DEV, NlLatentsLanguage.PYTHON),
                (3, NlLatentsSplit.TEST, NlLatentsLanguage.PYTHON),
                (3, NlLatentsSplit.TEST, NlLatentsLanguage.JAVA),
                (3, NlLatentsSplit.TEST, NlLatentsLanguage.RUST),
                (4, NlLatentsSplit.TRAIN, NlLatentsLanguage.PYTHON),
            },
            NlLatentsFamily.HUMANEVALPP: {
                (3, NlLatentsSplit.TEST, NlLatentsLanguage.PYTHON)
            },
            NlLatentsFamily.BITOPS: {
                (3, NlLatentsSplit.TEST, NlLatentsLanguage.PYTHON)
            },
            NlLatentsFamily.STRINGRULES: {
                (3, NlLatentsSplit.TRAIN, NlLatentsLanguage.PYTHON),
                (4, NlLatentsSplit.TRAIN, NlLatentsLanguage.PYTHON),
            },
            NlLatentsFamily.SMOKE: {
                (3, NlLatentsSplit.TRAIN, NlLatentsLanguage.PYTHON)
            },
        }
        if coordinate not in allowed[self.family]:
            raise ValueError(
                "unsupported NL Latents family coordinate: "
                f"{self.family.value}/d{self.difficulty}/"
                f"{self.language.value}/{self.split.value}"
            )

        expected_version = (
            "tasks_v1_pre_resample_2026_02_10"
            if self.family is NlLatentsFamily.SMOKE
            else "tasks_v2_resampled_2026_02_11"
        )
        if self.task_data_version != expected_version:
            raise ValueError(
                f"family {self.family.value!r} requires task data version "
                f"{expected_version!r}"
            )
        return self

    def serialize(self) -> str:
        """Return the stable full persisted source coordinate."""

        return canonical_json(self.model_dump(mode="json"))

    def relative_jsonl_path(self) -> PurePosixPath | None:
        if self.family is NlLatentsFamily.SMOKE:
            return None
        return PurePosixPath(
            self.family.value,
            f"d{self.difficulty}",
            self.language.value,
            f"{self.split.value}.jsonl",
        )


class NlLatentsTaskMapping(FrozenModel):
    coordinate: NlLatentsTaskCoordinate
    archive_identity: Annotated[str, Field(min_length=1)]
    task_root: NlLatentsTaskRoot
    relative_jsonl_path: str | None
    task_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    code_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    query_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    task: dict[str, JsonValue]

    @property
    def task_record_id(self) -> str:
        return content_sha256(self.persisted_material())

    def persisted_material(self) -> dict[str, JsonValue]:
        return {
            "archive_identity": self.archive_identity,
            "task_root": self.task_root.value,
            "relative_jsonl_path": self.relative_jsonl_path,
            "task_sha256": self.task_sha256,
            "code_sha256": self.code_sha256,
            "query_sha256": self.query_sha256,
            "source_coordinate": self.coordinate.model_dump(mode="json"),
            "entry_point": "f",
            "execution_ready": False,
            "task": self.task,
        }

    @property
    def code(self) -> str:
        value = self.task.get("code")
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"task {self.coordinate.task_id!r} has invalid code"
            )
        return value

    @property
    def queries(self) -> list[JsonValue]:
        value = self.task.get("queries")
        if not isinstance(value, list):
            raise ValueError(
                f"task {self.coordinate.task_id!r} has invalid queries"
            )
        return value

    def to_task_record(self) -> TaskRecord:
        payload = self.persisted_material()
        task_json = canonical_json(payload)
        record_id = content_sha256(payload)
        source_dataset = self.task.get("source_dataset")
        dataset_id = (
            source_dataset if isinstance(source_dataset, str) else None
        )
        return TaskRecord(
            task_record_id=record_id,
            dataset=DatasetName.NL_LATENTS,
            source_variant=self.coordinate.family.value,
            task_id=self.coordinate.task_id,
            language=self.coordinate.language.value,
            dataset_id=dataset_id,
            split=self.coordinate.split.value,
            data_sample_id=None,
            source_digest=self.task_sha256,
            dataset_revision=None,
            evaluator_kind="nl_latents_stored_queries_unpinned_runtime",
            material_fidelity=TaskMaterialFidelity.PINNED_SNAPSHOT,
            task_json=task_json,
            content_sha256=record_id,
        )


def _smoke_task() -> dict[str, JsonValue]:
    return {
        "task_id": _SMOKE_TASK_ID,
        "family": "smoke",
        "difficulty": 3,
        "code": "def f(x):\n    return x + 1",
        "queries": [
            {"input": 1, "output": 2, "tag": "smoke"},
            {"input": 5, "output": 6, "tag": "smoke"},
        ],
        "description": "Returns the input incremented by one.",
    }


@dataclass(slots=True)
class NlLatentsTaskAdapter:
    """Resolve archived NL Latents tasks without retaining machine-local paths."""

    archive_base: Path
    dataset: DatasetName = field(init=False, default=DatasetName.NL_LATENTS)
    _file_cache: dict[tuple[str, str], dict[str, dict[str, JsonValue]]] = (
        field(init=False, default_factory=dict)
    )
    _mappings: dict[str, NlLatentsTaskMapping] = field(
        init=False, default_factory=dict
    )
    _ambiguous_resolutions: set[str] = field(init=False, default_factory=set)

    def __post_init__(self) -> None:
        self.archive_base = self.archive_base.resolve()
        if not self.archive_base.is_dir():
            raise ValueError(
                f"NL Latents archive base is not a directory: {self.archive_base}"
            )

    def records(self) -> Iterable[TaskRecord]:
        for coordinate in sorted(self._mappings):
            yield self._mappings[coordinate].to_task_record()

    def resolve(self, data_sample_id: str) -> TaskRecord | None:
        try:
            coordinate = NlLatentsTaskCoordinate.model_validate_json(
                data_sample_id
            )
        except ValueError:
            return None
        mapping = self.resolve_coordinate(coordinate)
        return mapping.to_task_record() if mapping is not None else None

    def resolve_coordinate(
        self,
        coordinate: NlLatentsTaskCoordinate,
        *,
        validation_json: str | None = None,
    ) -> NlLatentsTaskMapping | None:
        coordinate_id = coordinate.serialize()
        existing = self._mappings.get(coordinate_id)
        if existing is not None:
            self._validate_ambiguous_selection(existing, validation_json)
            return existing

        if coordinate.family is NlLatentsFamily.SMOKE:
            task = _smoke_task()
            mapping = self._mapping(
                coordinate,
                task_root=NlLatentsTaskRoot.EMBEDDED_SMOKE,
                relative_path=None,
                task=task,
            )
            self._mappings[coordinate_id] = mapping
            return mapping

        relative_path = coordinate.relative_jsonl_path()
        if relative_path is None:
            raise RuntimeError(
                "non-smoke NL Latents task lacks an archive path"
            )
        primary = self._task_at(
            _PRIMARY_ROOT, relative_path, coordinate.task_id
        )
        seed41 = self._task_at(_SEED41_ROOT, relative_path, coordinate.task_id)
        workshop = self._task_at(
            _WORKSHOP_ROOT, relative_path, coordinate.task_id
        )

        if seed41 is not None and workshop is not None:
            if _sha256_json(seed41) != _sha256_json(workshop):
                raise ValueError(
                    "seed-41 task differs from its workshop copy for "
                    f"{coordinate.serialize()}"
                )

        if primary is not None:
            selected = primary
            root = NlLatentsTaskRoot.PRIMARY
        elif seed41 is not None:
            if workshop is None:
                raise ValueError(
                    "seed-41 fallback has no matching workshop task for "
                    f"{coordinate.serialize()}"
                )
            selected = seed41
            root = NlLatentsTaskRoot.SEED41
        else:
            return None

        mapping = self._mapping(
            coordinate,
            task_root=root,
            relative_path=relative_path,
            task=selected,
        )
        self._mappings[coordinate_id] = mapping
        if (
            primary is not None
            and seed41 is not None
            and _sha256_json(primary.get("queries"))
            != _sha256_json(seed41.get("queries"))
            and coordinate.task_id != _AMBIGUOUS_TASK_ID
        ):
            raise ValueError(
                "unexpected primary/seed-41 query ambiguity for "
                f"{coordinate.serialize()}"
            )
        self._validate_ambiguous_selection(mapping, validation_json)
        return mapping

    def assert_ambiguous_resolutions_validated(self) -> None:
        unresolved = {
            coordinate_id
            for coordinate_id, mapping in self._mappings.items()
            if mapping.coordinate.task_id == _AMBIGUOUS_TASK_ID
            and coordinate_id not in self._ambiguous_resolutions
        }
        if unresolved:
            raise ValueError(
                "known ambiguous NL Latents task lacks matching stored "
                f"validation cases: {sorted(unresolved)!r}"
            )

    def _mapping(
        self,
        coordinate: NlLatentsTaskCoordinate,
        *,
        task_root: NlLatentsTaskRoot,
        relative_path: PurePosixPath | None,
        task: dict[str, JsonValue],
    ) -> NlLatentsTaskMapping:
        actual_task_id = task.get("task_id")
        actual_family = task.get("family")
        if actual_task_id != coordinate.task_id:
            raise ValueError(
                f"resolved task ID mismatch: expected={coordinate.task_id!r}, "
                f"actual={actual_task_id!r}"
            )
        if actual_family != coordinate.family.value:
            raise ValueError(
                f"resolved task family mismatch: expected={coordinate.family.value!r}, "
                f"actual={actual_family!r}"
            )
        code = task.get("code")
        queries = task.get("queries")
        if not isinstance(code, str) or not code:
            raise ValueError(f"task {coordinate.task_id!r} has invalid code")
        if not isinstance(queries, list):
            raise ValueError(
                f"task {coordinate.task_id!r} has invalid queries"
            )
        return NlLatentsTaskMapping(
            coordinate=coordinate,
            archive_identity=_ARCHIVE_IDENTITY,
            task_root=task_root,
            relative_jsonl_path=(
                relative_path.as_posix() if relative_path is not None else None
            ),
            task_sha256=_sha256_json(task),
            code_sha256=_sha256_text(code),
            query_sha256=_sha256_json(queries),
            task=task,
        )

    def _task_at(
        self,
        root: str,
        relative_path: PurePosixPath,
        task_id: str,
    ) -> dict[str, JsonValue] | None:
        key = (root, relative_path.as_posix())
        tasks = self._file_cache.get(key)
        if tasks is None:
            tasks = self._read_task_file(root, relative_path)
            self._file_cache[key] = tasks
        return tasks.get(task_id)

    def _read_task_file(
        self, root: str, relative_path: PurePosixPath
    ) -> dict[str, dict[str, JsonValue]]:
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"invalid task relative path {relative_path}")
        path = self.archive_base / root / Path(relative_path)
        if not path.is_file():
            return {}
        records: dict[str, dict[str, JsonValue]] = {}
        try:
            with path.open(encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    if not line.strip():
                        raise ValueError(
                            f"blank archived task row at {path}:{line_number}"
                        )
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(
                            f"archived task row is not an object at "
                            f"{path}:{line_number}"
                        )
                    task_id = value.get("task_id")
                    if not isinstance(task_id, str) or not task_id:
                        raise ValueError(
                            f"archived task has invalid task_id at "
                            f"{path}:{line_number}"
                        )
                    if task_id in records:
                        raise ValueError(
                            f"duplicate archived task {task_id!r} in {path}"
                        )
                    records[task_id] = value
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cannot read archived task file {path}: {exc}"
            ) from exc
        return records

    def _validate_ambiguous_selection(
        self,
        mapping: NlLatentsTaskMapping,
        validation_json: str | None,
    ) -> None:
        if mapping.coordinate.task_id != _AMBIGUOUS_TASK_ID:
            return
        if validation_json is None or not validation_json.strip():
            return
        try:
            validation = json.loads(validation_json)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "invalid stored validation JSON for known ambiguous task"
            ) from exc
        if not isinstance(validation, dict):
            raise ValueError(
                "stored validation for known ambiguous task is not an object"
            )
        raw_cases = validation.get("test_case_results")
        if not isinstance(raw_cases, list):
            raise ValueError(
                "stored validation for known ambiguous task has no test cases"
            )
        stored_cases: list[JsonValue] = []
        for raw_case in raw_cases:
            if not isinstance(raw_case, dict):
                raise ValueError(
                    "stored validation test case is not an object"
                )
            if (
                "input_value" not in raw_case
                or "expected_output" not in raw_case
            ):
                raise ValueError(
                    "stored validation test case lacks input or expected output"
                )
            stored_cases.append(
                [raw_case["input_value"], raw_case["expected_output"]]
            )

        selected_cases: list[JsonValue] = []
        for raw_query in mapping.queries:
            if not isinstance(raw_query, dict):
                raise ValueError("archived task query is not an object")
            if "input" not in raw_query or "output" not in raw_query:
                raise ValueError("archived task query lacks input or output")
            selected_cases.append([raw_query["input"], raw_query["output"]])
        if canonical_json(stored_cases) != canonical_json(selected_cases):
            raise ValueError(
                "stored validation cases do not match the selected primary "
                f"task for {mapping.coordinate.serialize()}"
            )
        self._ambiguous_resolutions.add(mapping.coordinate.serialize())


__all__ = [
    "NlLatentsFamily",
    "NlLatentsLanguage",
    "NlLatentsSplit",
    "NlLatentsTaskAdapter",
    "NlLatentsTaskCoordinate",
    "NlLatentsTaskMapping",
    "NlLatentsTaskRoot",
]
