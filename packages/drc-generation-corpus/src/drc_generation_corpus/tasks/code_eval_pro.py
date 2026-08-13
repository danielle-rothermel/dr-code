from __future__ import annotations

import ast
import gzip
import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, cast

from pydantic import JsonValue, ValidationError

from dr_code.core.models import FrozenModel
from drc_generation_corpus.models import (
    DatasetName,
    TaskMaterialFidelity,
    TaskRecord,
)
from drc_generation_corpus.pool_dump import canonical_json, content_sha256


@dataclass(frozen=True, slots=True)
class CodeCompDatasetDefinition:
    dataset: DatasetName
    namespace: str
    task_id_pattern: str
    primary_source_variant: str
    encoder_pool_source_variant: str
    encoder_pool_has_task_material: bool

    def parse_data_sample_id(self, value: object) -> tuple[str, str] | None:
        if not isinstance(value, str) or not value.startswith(
            f"{self.namespace}/"
        ):
            return None
        match = re.fullmatch(
            rf"{re.escape(self.namespace)}/({self.task_id_pattern})/"
            r"gt_solution@([0-9a-f]{16})",
            value,
        )
        if match is None:
            raise ValueError(
                f"invalid {self.dataset.value} data_sample_id: {value!r}"
            )
        return match.group(1), match.group(2)


MBPP_PRO_DEFINITION = CodeCompDatasetDefinition(
    dataset=DatasetName.MBPP_PRO,
    namespace="mbpp_pro",
    task_id_pattern=r"MbppPro/[0-9]+",
    primary_source_variant="canonical",
    encoder_pool_source_variant="opaque_encoder_pool_t1",
    encoder_pool_has_task_material=False,
)
HUMANEVAL_PRO_DEFINITION = CodeCompDatasetDefinition(
    dataset=DatasetName.HUMANEVAL_PRO,
    namespace="humaneval_pro",
    task_id_pattern=r"HumanEvalPro/[0-9]+",
    primary_source_variant="canonical",
    encoder_pool_source_variant="opaque_encoder_pool_t1",
    encoder_pool_has_task_material=False,
)
CLASS_EVAL_DEFINITION = CodeCompDatasetDefinition(
    dataset=DatasetName.CLASS_EVAL,
    namespace="class_eval",
    task_id_pattern=r"ClassEval_[0-9]+",
    primary_source_variant="gt_code",
    encoder_pool_source_variant="gt_code_with_comments",
    encoder_pool_has_task_material=True,
)
BIGCODEBENCH_LITE_PRO_DEFINITION = CodeCompDatasetDefinition(
    dataset=DatasetName.BIGCODEBENCH_LITE_PRO,
    namespace="bigcodebench_lite_pro",
    task_id_pattern=r"BigCodeBenchLitePro/[0-9]+",
    primary_source_variant="canonical",
    encoder_pool_source_variant="cleaned",
    encoder_pool_has_task_material=True,
)


class _CacheManifest(FrozenModel):
    dataset_id: str
    split: str
    source_revision: str
    cache_schema_version: int
    built_at: str
    raw_sample_count: int
    flawed_count: int
    task_count: int


@dataclass(frozen=True, slots=True)
class _TaskDatasetSpec:
    definition: CodeCompDatasetDefinition
    dataset_id: str
    split: str
    revision: str
    cache_schema_version: int
    raw_sample_count: int
    flawed_ids: tuple[str, ...]
    task_count: int
    payload_sha256: str
    accepted_ids_sha256: str
    source_digests_sha256: str
    evaluator_kind: str
    source_variants: tuple[str, ...]


_MBPP_PRO_TASK_SPEC = _TaskDatasetSpec(
    definition=MBPP_PRO_DEFINITION,
    dataset_id="CodeEval-Pro/mbpp-pro",
    split="train",
    revision="50f18448e09a8383226e1a5cd3654d2a454fe333",
    cache_schema_version=3,
    raw_sample_count=375,
    flawed_ids=("MbppPro/117", "MbppPro/260", "MbppPro/36"),
    task_count=375,
    payload_sha256="b936ae8f9310c518bb12bf983e06bd3085b7d805f2ae0df6680c711c68f7cf93",
    accepted_ids_sha256="352caddebc08dede80a8526642c741dd9e2562c0a96605ed7cc088629e82494c",
    source_digests_sha256="aaa09f0693f47af4736dbc5de88561e579dd1dcb181b32f4f34a5f92cc23426b",
    evaluator_kind="python_assertions",
    source_variants=("canonical",),
)
_HUMANEVAL_PRO_TASK_SPEC = _TaskDatasetSpec(
    definition=HUMANEVAL_PRO_DEFINITION,
    dataset_id="CodeEval-Pro/humaneval-pro",
    split="train",
    revision="cd078f93d57d1902b5c3e4ae330166b2ca0e0e80",
    cache_schema_version=3,
    raw_sample_count=163,
    flawed_ids=("HumanEvalPro/24",),
    task_count=163,
    payload_sha256="3fe6788dc4ab2714683f2cfa4aef1fd2d4b6cf68eb6f2f51e4ecc8d8ab134857",
    accepted_ids_sha256="c008e7bf64b4c4b7457d3758ecf9f51345718d55ce3fd10388222c5c28968a33",
    source_digests_sha256="2cdbccb4423f9fb084b961ee714a725f98465b2bb9ceb6f441d9dcbad144e51a",
    evaluator_kind="python_assertions",
    source_variants=("canonical",),
)
_CLASS_EVAL_TASK_SPEC = _TaskDatasetSpec(
    definition=CLASS_EVAL_DEFINITION,
    dataset_id="FudanSELab/ClassEval",
    split="test",
    revision="fef204b34e221f207f47904ee660bb920d4c5d1d",
    cache_schema_version=1,
    raw_sample_count=98,
    flawed_ids=("ClassEval_48", "ClassEval_58"),
    task_count=98,
    payload_sha256="8bebf31afc375df6c5ffb343c5a7c3fa9635582c0838c4cbf488f6ef45856398",
    accepted_ids_sha256="2d00189972f8da04e5a88ada9f47c06df796284a58ef327a7a2a675346c87238",
    source_digests_sha256="16d5ce0663da7e26ffa504c6a51368662ceb95818b499900c580b5ed935e6c4c",
    evaluator_kind="python_unittest",
    source_variants=("gt_code", "gt_code_with_comments"),
)
_BIGCODEBENCH_LITE_PRO_TASK_SPEC = _TaskDatasetSpec(
    definition=BIGCODEBENCH_LITE_PRO_DEFINITION,
    dataset_id="CodeEval-Pro/bigcodebench-lite-pro",
    split="train",
    revision="f70ee47b5701ae8b240c64bd4d4077e46b4c9278",
    cache_schema_version=3,
    raw_sample_count=54,
    flawed_ids=(
        "BigCodeBenchLitePro/201",
        "BigCodeBenchLitePro/551",
        "BigCodeBenchLitePro/679",
    ),
    task_count=54,
    payload_sha256="9f150a453dfc0e3694e3d5d8fe6015b9f9fd4837cd93dc12088dad29bc1038fe",
    accepted_ids_sha256="7071759160f0641fb3d03132704ae7f9904911047a89c1936f01ccc2abe52072",
    source_digests_sha256="c17b7cba946a649203c945c4848c33219147db9bc59c52c8646b422f109edde6",
    evaluator_kind="python_assertions_scientific",
    source_variants=("canonical", "cleaned"),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(
            f"cannot read task cache payload {path}: {exc}"
        ) from exc
    return digest.hexdigest()


def _canonical_sha256(value: JsonValue) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise ValueError(f"{context} must be a JSON object with string keys")
    return cast(Mapping[str, object], value)


def _string(value: object, *, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string")
    return value


def _clean_python_source(source_code: str) -> str:
    try:
        parsed = ast.parse(source_code)
    except SyntaxError as exc:
        raise ValueError(
            f"failed to parse pinned Python source: {exc}"
        ) from exc

    for node in ast.walk(parsed):
        if not isinstance(
            node,
            (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            continue
        if not node.body:
            continue
        first = node.body[0]
        if not (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            continue
        node.body = node.body[1:] or [ast.Pass()]
    ast.fix_missing_locations(parsed)
    return ast.unparse(parsed) + "\n"


class _PinnedCacheTaskAdapter:
    _spec: ClassVar[_TaskDatasetSpec]

    def __init__(self, cache_directory: Path) -> None:
        self._cache_directory = cache_directory
        self._records, self._by_data_sample_id = self._load()

    @property
    def dataset(self) -> DatasetName:
        return self._spec.definition.dataset

    def records(self) -> Iterable[TaskRecord]:
        return self._records

    def resolve(self, data_sample_id: str) -> TaskRecord | None:
        parsed = self._spec.definition.parse_data_sample_id(data_sample_id)
        if parsed is None:
            return None
        return self._by_data_sample_id.get(data_sample_id)

    def _load(self) -> tuple[tuple[TaskRecord, ...], dict[str, TaskRecord]]:
        manifest_path = self._cache_directory / "manifest.json"
        try:
            manifest = _CacheManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8"), strict=True
            )
        except (OSError, ValidationError) as exc:
            raise ValueError(
                f"invalid task cache manifest {manifest_path}: {exc}"
            ) from exc
        self._validate_manifest(manifest)

        payload_path = self._cache_directory / "payload.json.gz"
        actual_payload_sha256 = _sha256_file(payload_path)
        if actual_payload_sha256 != self._spec.payload_sha256:
            raise ValueError(
                "task cache payload SHA-256 mismatch: "
                f"expected={self._spec.payload_sha256}, "
                f"actual={actual_payload_sha256}"
            )
        try:
            with gzip.open(payload_path, "rt", encoding="utf-8") as file:
                payload_object: object = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid task cache payload {payload_path}: {exc}"
            ) from exc

        payload = _mapping(payload_object, context="task cache payload")
        expected_payload_keys = {"raw_samples", "flawed_raw_samples", "tasks"}
        if set(payload) != expected_payload_keys:
            raise ValueError(
                "task cache payload keys mismatch: "
                f"expected={sorted(expected_payload_keys)!r}, "
                f"actual={sorted(payload)!r}"
            )
        raw_samples = _mapping(
            payload["raw_samples"], context="task cache raw_samples"
        )
        flawed_samples = _mapping(
            payload["flawed_raw_samples"],
            context="task cache flawed_raw_samples",
        )
        tasks = _mapping(payload["tasks"], context="task cache tasks")
        self._validate_populations(raw_samples, flawed_samples, tasks)
        return self._build_records(raw_samples, tasks)

    def _validate_manifest(self, manifest: _CacheManifest) -> None:
        expected = (
            self._spec.dataset_id,
            self._spec.split,
            self._spec.revision,
            self._spec.cache_schema_version,
            self._spec.raw_sample_count,
            len(self._spec.flawed_ids),
            self._spec.task_count,
        )
        actual = (
            manifest.dataset_id,
            manifest.split,
            manifest.source_revision,
            manifest.cache_schema_version,
            manifest.raw_sample_count,
            manifest.flawed_count,
            manifest.task_count,
        )
        if actual != expected:
            raise ValueError(
                f"task cache manifest coordinate mismatch: "
                f"expected={expected!r}, actual={actual!r}"
            )

    def _validate_populations(
        self,
        raw_samples: Mapping[str, object],
        flawed_samples: Mapping[str, object],
        tasks: Mapping[str, object],
    ) -> None:
        task_ids = sorted(tasks)
        if set(raw_samples) != set(tasks):
            raise ValueError("task cache raw/task accepted ID sets differ")
        if len(raw_samples) != self._spec.raw_sample_count:
            raise ValueError(
                "task cache raw sample count mismatch: "
                f"expected={self._spec.raw_sample_count}, actual={len(raw_samples)}"
            )
        if len(tasks) != self._spec.task_count:
            raise ValueError(
                "task cache task count mismatch: "
                f"expected={self._spec.task_count}, actual={len(tasks)}"
            )
        actual_flawed_ids = tuple(sorted(flawed_samples))
        if actual_flawed_ids != self._spec.flawed_ids:
            raise ValueError(
                "task cache flawed ID set mismatch: "
                f"expected={self._spec.flawed_ids!r}, actual={actual_flawed_ids!r}"
            )
        if set(tasks) & set(flawed_samples):
            raise ValueError("task cache accepted and flawed ID sets overlap")
        accepted_ids_sha256 = _canonical_sha256(cast(JsonValue, task_ids))
        if accepted_ids_sha256 != self._spec.accepted_ids_sha256:
            raise ValueError(
                "task cache accepted ID set digest mismatch: "
                f"expected={self._spec.accepted_ids_sha256}, "
                f"actual={accepted_ids_sha256}"
            )

    def _build_records(
        self,
        raw_samples: Mapping[str, object],
        tasks: Mapping[str, object],
    ) -> tuple[tuple[TaskRecord, ...], dict[str, TaskRecord]]:
        records: list[TaskRecord] = []
        by_data_sample_id: dict[str, TaskRecord] = {}
        digest_entries: list[list[str]] = []
        for task_id in sorted(tasks):
            raw_sample = _mapping(
                raw_samples[task_id], context=f"raw sample {task_id}"
            )
            task = _mapping(tasks[task_id], context=f"task {task_id}")
            if (
                _string(raw_sample.get("task_id"), context="raw task_id")
                != task_id
            ):
                raise ValueError(f"raw task ID mismatch for {task_id}")
            if _string(task.get("task_id"), context="task task_id") != task_id:
                raise ValueError(f"task ID mismatch for {task_id}")
            if (
                _string(task.get("dataset"), context="task dataset")
                != self._spec.dataset_id
            ):
                raise ValueError(f"task dataset mismatch for {task_id}")

            for source_variant in self._spec.source_variants:
                source = self._source_for_variant(
                    raw_sample=raw_sample,
                    task=task,
                    source_variant=source_variant,
                )
                source_digest = hashlib.sha256(
                    f"gt_solution\0{source}".encode("utf-8")
                ).hexdigest()
                digest_entries.append([task_id, source_variant, source_digest])
                data_sample_id = (
                    f"{self._spec.definition.namespace}/{task_id}/"
                    f"gt_solution@{source_digest[:16]}"
                )
                task_payload: JsonValue = cast(
                    JsonValue,
                    {
                        "cache_schema_version": self._spec.cache_schema_version,
                        "dataset_id": self._spec.dataset_id,
                        "dataset_revision": self._spec.revision,
                        "raw_sample": raw_sample,
                        "snapshot_payload_sha256": self._spec.payload_sha256,
                        "source_variant": source_variant,
                        "split": self._spec.split,
                        "task": task,
                    },
                )
                task_json = canonical_json(task_payload)
                task_record_id = content_sha256(task_payload)
                record = TaskRecord(
                    task_record_id=task_record_id,
                    dataset=self.dataset,
                    source_variant=source_variant,
                    task_id=task_id,
                    language="python",
                    dataset_id=self._spec.dataset_id,
                    split=self._spec.split,
                    data_sample_id=data_sample_id,
                    source_digest=source_digest,
                    dataset_revision=self._spec.revision,
                    evaluator_kind=self._spec.evaluator_kind,
                    material_fidelity=TaskMaterialFidelity.PINNED_SNAPSHOT,
                    task_json=task_json,
                    content_sha256=task_record_id,
                )
                records.append(record)
                by_data_sample_id[data_sample_id] = record

        actual_source_digest = _canonical_sha256(
            cast(JsonValue, digest_entries)
        )
        if actual_source_digest != self._spec.source_digests_sha256:
            raise ValueError(
                "task cache full source digest set mismatch: "
                f"expected={self._spec.source_digests_sha256}, "
                f"actual={actual_source_digest}"
            )
        if len(by_data_sample_id) != len(records):
            raise ValueError("task cache source identities are not unique")
        return tuple(records), by_data_sample_id

    def _source_for_variant(
        self,
        *,
        raw_sample: Mapping[str, object],
        task: Mapping[str, object],
        source_variant: str,
    ) -> str:
        if source_variant == "canonical":
            source = _mapping(task.get("source"), context="task source")
            return _string(source.get("code"), context="task source code")
        if source_variant in {"gt_code", "gt_code_with_comments"}:
            return _string(
                raw_sample.get(source_variant), context=source_variant
            )
        if source_variant == "cleaned":
            source = _mapping(task.get("source"), context="task source")
            code = _string(source.get("code"), context="task source code")
            return _clean_python_source(code)
        raise AssertionError(f"unknown source variant {source_variant!r}")


class MbppProTaskAdapter(_PinnedCacheTaskAdapter):
    _spec = _MBPP_PRO_TASK_SPEC


class HumanEvalProTaskAdapter(_PinnedCacheTaskAdapter):
    _spec = _HUMANEVAL_PRO_TASK_SPEC


class ClassEvalTaskAdapter(_PinnedCacheTaskAdapter):
    _spec = _CLASS_EVAL_TASK_SPEC


class BigCodeBenchLiteProTaskAdapter(_PinnedCacheTaskAdapter):
    _spec = _BIGCODEBENCH_LITE_PRO_TASK_SPEC


__all__ = [
    "BIGCODEBENCH_LITE_PRO_DEFINITION",
    "CLASS_EVAL_DEFINITION",
    "HUMANEVAL_PRO_DEFINITION",
    "MBPP_PRO_DEFINITION",
    "BigCodeBenchLiteProTaskAdapter",
    "ClassEvalTaskAdapter",
    "CodeCompDatasetDefinition",
    "HumanEvalProTaskAdapter",
    "MbppProTaskAdapter",
]
