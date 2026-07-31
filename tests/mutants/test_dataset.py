"""Deterministic and authenticated dataset publication contracts."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from dr_exec import EXECUTOR_IDENTITY

from dr_code.mutants.dataset import (
    DatasetManifest,
    FamilyCount,
    GeneratedDataset,
    GenerationConfig,
    DatasetValidationError,
    SkippedMutation,
    build_record,
    load_dataset,
    publish_dataset,
)
from dr_code.mutants import dataset as dataset_module
from dr_code.mutants.outcomes import ValueOutcome
from dr_code.mutants.generate import generate_mutants
from dr_code.eval.identity import identity_hash_for
from dr_code.mutants.operators import OperatorFamily
from dr_code.mutants.provenance import (
    canonical_suite_digest,
    resolve_canonical_suite,
)
from dr_code.mutants import provenance as provenance_module
from dr_code.humaneval.task import parse_human_eval_dataset
from dr_code.synthetic import humaneval_loader as humaneval_loader_module
from dr_code.synthetic.humaneval_loader import (
    HF_DATASET_ID,
    HF_REVISION,
    HumanEvalSource,
    HumanEvalPlusTask,
)

_TEST = """def check(candidate):
    inputs = [[1]]
    results = [False]
    for i, (inp, exp) in enumerate(zip(inputs, results)):
        assertion(candidate(*inp), exp, 0)
"""

# Loader budgets come from this fixture suite's deliberately small generation
# envelope, not from the untrusted artifacts it reads.
_MANIFEST_BYTE_CEILING = 128 * 1024
_RECORDS_BYTE_CEILING = 128 * 1024


@pytest.fixture(autouse=True)
def pinned_fixture_suite(monkeypatch) -> None:
    tasks = [
        HumanEvalPlusTask(
            task_id=f"HumanEval/{suffix}",
            prompt="def f(x):\n",
            canonical_solution="    return x < 1\n",
            entry_point="f",
            test=_TEST,
        )
        for suffix in ("0", "1", "2", "10")
    ]
    monkeypatch.setattr(
        provenance_module,
        "load_humaneval_plus",
        lambda source: tasks,
    )


@pytest.mark.parametrize("source", HumanEvalSource)
def test_canonical_suite_matches_evaluator_humaneval_32_overrides(
    monkeypatch,
    source: HumanEvalSource,
) -> None:
    raw_task = next(
        task
        for task in humaneval_loader_module.load_humaneval_plus(
            source=HumanEvalSource.SNAPSHOT
        )
        if task.task_id == "HumanEval/32"
    )
    evaluator_task = parse_human_eval_dataset(
        [raw_task.model_dump(mode="json")]
    )[0]
    sources: list[HumanEvalSource] = []
    monkeypatch.setattr(
        provenance_module,
        "load_humaneval_plus",
        lambda source: sources.append(source) or [raw_task],
    )

    canonical_task = resolve_canonical_suite(
        task_ids=(raw_task.task_id,),
        max_inputs=10,
        source=source,
    )[0]

    assert sources == [source]
    assert canonical_task.prompt == evaluator_task.prompt
    assert canonical_task.canonical_full_source == ast.unparse(
        ast.parse(evaluator_task.ground_truth_code)
    )
    assert canonical_task.canonical_test == evaluator_task.test


def _record(
    *,
    task_id: str = "HumanEval/0",
    seed: int = 0,
    mutant_value: str = "2",
    prompt: str = "def f(x):\n",
    input_reprs: tuple[str, ...] = ("(1,)",),
):
    return build_record(
        task_id=task_id,
        entry_point="f",
        prompt=prompt,
        canonical_full_source="def f(x):\n    return x < 1",
        mutated_full_source="def f(x):\n    return x <= 1",
        operator_family=OperatorFamily.COMPARISON_FLIP,
        seed=seed,
        site_node_path=5,
        site_target_index=0,
        site_description="line 2: comparison operand 0 <",
        input_reprs=input_reprs,
        mutant_expected=tuple(
            ValueOutcome(value_repr=mutant_value) for _ in input_reprs
        ),
        canonical_expected=tuple(
            ValueOutcome(value_repr="1") for _ in input_reprs
        ),
        distinct_input_indices=tuple(range(len(input_reprs))),
        diff_summary="changed",
        canonical_test=_TEST,
    )


def _generated(*records) -> GeneratedDataset:
    task_ids = tuple(
        sorted(
            {record.task_id for record in records},
            key=lambda task_id: int(task_id.rpartition("/")[2]),
        )
    )
    seeds = max((record.seed for record in records), default=0) + 1
    accepted = {
        (record.task_id, record.operator_family, record.seed)
        for record in records
    }
    skipped = tuple(
        SkippedMutation(
            task_id=task_id,
            operator_family=OperatorFamily.COMPARISON_FLIP,
            seed=seed,
            reason="no applicable distinct site",
        )
        for task_id in task_ids
        for seed in range(seeds)
        if (
            task_id,
            OperatorFamily.COMPARISON_FLIP,
            seed,
        )
        not in accepted
    )
    suite = resolve_canonical_suite(
        task_ids=task_ids,
        max_inputs=10,
        source=HumanEvalSource.SNAPSHOT,
    )
    config = GenerationConfig(
        dataset_source=HumanEvalSource.SNAPSHOT,
        dataset_id=HF_DATASET_ID,
        dataset_revision=HF_REVISION,
        operator_families=(OperatorFamily.COMPARISON_FLIP,),
        seeds=seeds,
        max_inputs_per_mutant=10,
        timeout_seconds=5.0,
        task_ids=task_ids,
        canonical_suite_digest=canonical_suite_digest(suite),
        runner_identity="fixture-runner@v1",
        runtime_identity="fixture-runtime@v1",
    )
    return GeneratedDataset(
        config=config,
        canonical_suite=suite,
        records=tuple(records),
        accepted_by_family=(
            FamilyCount(
                operator_family=OperatorFamily.COMPARISON_FLIP,
                count=len(records),
            ),
        ),
        skipped=skipped,
    )


def _failed_generated(
    *,
    task: HumanEvalPlusTask,
    records=(),
    skipped=(),
) -> GeneratedDataset:
    suite = resolve_canonical_suite(
        task_ids=(task.task_id,),
        max_inputs=10,
        source=HumanEvalSource.SNAPSHOT,
    )
    return GeneratedDataset(
        config=GenerationConfig(
            dataset_source=HumanEvalSource.SNAPSHOT,
            dataset_id=HF_DATASET_ID,
            dataset_revision=HF_REVISION,
            operator_families=(OperatorFamily.COMPARISON_FLIP,),
            seeds=1,
            max_inputs_per_mutant=10,
            timeout_seconds=5.0,
            task_ids=(task.task_id,),
            canonical_suite_digest=canonical_suite_digest(suite),
            runner_identity="fixture-runner@v1",
            runtime_identity="fixture-runtime@v1",
        ),
        canonical_suite=suite,
        records=tuple(records),
        accepted_by_family=(
            FamilyCount(
                operator_family=OperatorFamily.COMPARISON_FLIP,
                count=len(records),
            ),
        ),
        skipped=tuple(skipped),
    )


def _malformed_task(*, malformed_source: bool) -> HumanEvalPlusTask:
    return HumanEvalPlusTask(
        task_id="HumanEval/0",
        prompt="def f(:\n" if malformed_source else "def f(x):\n",
        canonical_solution="" if malformed_source else "    return x < 1\n",
        entry_point="f",
        test=_TEST if malformed_source else "def check(:\n",
    )


def test_publication_round_trips_and_authenticates_all_artifacts(
    tmp_path: Path,
) -> None:
    generated = _generated(
        _record(),
        _record(task_id="HumanEval/1", seed=1, mutant_value="3"),
    )
    artifacts = publish_dataset(
        output_dir=tmp_path / "dataset",
        generated=generated,
    )

    loaded = load_dataset(
        tmp_path / "dataset",
        expected_dataset_identity=artifacts.manifest.dataset_identity,
        max_manifest_bytes=_MANIFEST_BYTE_CEILING,
        max_records_bytes=_RECORDS_BYTE_CEILING,
        expected_config_identity=generated.config.identity_hash(),
    )

    assert loaded.records == generated.records
    assert loaded.manifest == artifacts.manifest
    assert {path.name for path in (tmp_path / "dataset").iterdir()} == {
        "manifest.json",
        "mutants.jsonl",
    }
    assert len(loaded.manifest.records_sha256) == 64
    assert len(loaded.manifest.dataset_identity) == 64


def test_regeneration_is_byte_identical(tmp_path: Path) -> None:
    generated = _generated(_record())
    first = publish_dataset(
        output_dir=tmp_path / "first",
        generated=generated,
    )
    second = publish_dataset(
        output_dir=tmp_path / "second",
        generated=generated,
    )

    assert first.records_path.read_bytes() == second.records_path.read_bytes()
    assert first.manifest_path.read_bytes() == (
        second.manifest_path.read_bytes()
    )


def test_publication_is_atomic_and_does_not_clobber(
    tmp_path: Path,
) -> None:
    generated = _generated(_record())
    destination = tmp_path / "dataset"
    first = publish_dataset(output_dir=destination, generated=generated)
    before = first.manifest_path.read_bytes()

    with pytest.raises(FileExistsError):
        publish_dataset(output_dir=destination, generated=generated)

    assert first.manifest_path.read_bytes() == before
    assert not list(tmp_path.glob(".dataset.*.tmp"))


def test_loader_rejects_records_hash_corruption(tmp_path: Path) -> None:
    destination = tmp_path / "dataset"
    artifacts = publish_dataset(
        output_dir=destination,
        generated=_generated(_record()),
    )
    content = artifacts.records_path.read_text(encoding="utf-8")
    artifacts.records_path.write_text(
        content.replace('"value_repr":"2"', '"value_repr":"9"'),
        encoding="utf-8",
    )

    with pytest.raises(DatasetValidationError, match="SHA-256"):
        load_dataset(
            destination,
            expected_dataset_identity=artifacts.manifest.dataset_identity,
            max_manifest_bytes=_MANIFEST_BYTE_CEILING,
            max_records_bytes=_RECORDS_BYTE_CEILING,
        )


def test_loader_rejects_records_hash_before_decoding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "dataset"
    artifacts = publish_dataset(
        output_dir=destination,
        generated=_generated(_record()),
    )
    artifacts.records_path.write_bytes(b"unauthenticated")
    original_read_bytes = Path.read_bytes
    read_paths: list[Path] = []

    def read_bytes(path: Path) -> bytes:
        read_paths.append(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(
        dataset_module,
        "_decode_records",
        lambda content: (_ for _ in ()).throw(
            AssertionError("unauthenticated records must not be decoded")
        ),
    )

    with pytest.raises(DatasetValidationError, match="SHA-256"):
        load_dataset(
            destination,
            expected_dataset_identity=artifacts.manifest.dataset_identity,
            max_manifest_bytes=_MANIFEST_BYTE_CEILING,
            max_records_bytes=_RECORDS_BYTE_CEILING,
        )

    # The manifest is parsed from its captured snapshot; the unauthenticated
    # records snapshot is never read after its hash check fails.
    assert [path.name for path in read_paths] == ["snapshot"]


def test_loader_rejects_oversized_manifest_before_pydantic_parsing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "dataset"
    artifacts = publish_dataset(
        output_dir=destination,
        generated=_generated(_record()),
    )
    manifest_limit = 64
    artifacts.manifest_path.write_bytes(b"{" * (manifest_limit + 1))
    monkeypatch.setattr(
        DatasetManifest,
        "model_validate_json",
        lambda content: (_ for _ in ()).throw(
            AssertionError("oversized manifests must not be parsed")
        ),
    )

    with pytest.raises(
        DatasetValidationError,
        match="manifest.json exceeds maximum size",
    ):
        load_dataset(
            destination,
            expected_dataset_identity=artifacts.manifest.dataset_identity,
            max_manifest_bytes=manifest_limit,
            max_records_bytes=_RECORDS_BYTE_CEILING,
        )


def test_loader_stops_sparse_wrong_hash_at_records_ceiling_before_decode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "dataset"
    artifacts = publish_dataset(
        output_dir=destination,
        generated=_generated(_record()),
    )
    records_limit = 256
    with artifacts.records_path.open("wb") as stream:
        stream.truncate(records_limit * 1024)

    original_open = Path.open
    read_sizes: list[int] = []

    class CountingReader:
        def __init__(self, stream) -> None:
            self._stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            self._stream.close()

        def read(self, size: int = -1) -> bytes:
            read_sizes.append(size)
            return self._stream.read(size)

        def __getattr__(self, name: str) -> object:
            return getattr(self._stream, name)

    def open(path: Path, mode: str = "r"):
        stream = original_open(path, mode)
        if path == artifacts.records_path and mode == "rb":
            return CountingReader(stream)
        return stream

    monkeypatch.setattr(Path, "open", open)
    monkeypatch.setattr(
        dataset_module,
        "_decode_records",
        lambda content: (_ for _ in ()).throw(
            AssertionError("wrong-hash records must not be decoded")
        ),
    )

    with pytest.raises(
        DatasetValidationError,
        match="mutants.jsonl exceeds maximum size",
    ):
        load_dataset(
            destination,
            expected_dataset_identity=artifacts.manifest.dataset_identity,
            max_manifest_bytes=_MANIFEST_BYTE_CEILING,
            max_records_bytes=records_limit,
        )

    assert read_sizes == [records_limit + 1]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("config_identity", "0" * 64, "config identity"),
        ("accepted_count", 99, "dataset identity"),
        ("dataset_identity", "0" * 64, "dataset identity"),
    ],
)
def test_loader_rejects_manifest_corruption(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    destination = tmp_path / field
    artifacts = publish_dataset(
        output_dir=destination,
        generated=_generated(_record()),
    )
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    artifacts.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetValidationError, match=message):
        load_dataset(
            destination,
            expected_dataset_identity=artifacts.manifest.dataset_identity,
            max_manifest_bytes=_MANIFEST_BYTE_CEILING,
            max_records_bytes=_RECORDS_BYTE_CEILING,
        )


def test_loader_rejects_unexpected_config_identity(tmp_path: Path) -> None:
    destination = tmp_path / "dataset"
    artifacts = publish_dataset(
        output_dir=destination,
        generated=_generated(_record()),
    )

    with pytest.raises(DatasetValidationError, match="unexpected"):
        load_dataset(
            destination,
            expected_dataset_identity=artifacts.manifest.dataset_identity,
            max_manifest_bytes=_MANIFEST_BYTE_CEILING,
            max_records_bytes=_RECORDS_BYTE_CEILING,
            expected_config_identity="f" * 64,
        )


def test_loader_requires_caller_pinned_dataset_identity(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "dataset"
    publish_dataset(
        output_dir=destination,
        generated=_generated(_record()),
    )

    with pytest.raises(
        DatasetValidationError,
        match="unexpected mutant dataset identity",
    ):
        load_dataset(
            destination,
            expected_dataset_identity="f" * 64,
            max_manifest_bytes=_MANIFEST_BYTE_CEILING,
            max_records_bytes=_RECORDS_BYTE_CEILING,
        )


def test_loader_rejects_caller_pin_before_decoding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "dataset"
    publish_dataset(
        output_dir=destination,
        generated=_generated(_record()),
    )
    monkeypatch.setattr(
        dataset_module,
        "_decode_records",
        lambda content: (_ for _ in ()).throw(
            AssertionError("untrusted datasets must not be decoded")
        ),
    )

    with pytest.raises(
        DatasetValidationError,
        match="unexpected mutant dataset identity",
    ):
        load_dataset(
            destination,
            expected_dataset_identity="f" * 64,
            max_manifest_bytes=_MANIFEST_BYTE_CEILING,
            max_records_bytes=_RECORDS_BYTE_CEILING,
        )


def test_loader_authenticates_cross_runtime_production_dataset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    producer_runtime_identity = "a" * 64
    consumer_runtime_identity = "b" * 64
    generated = _generated(_record())
    generated = replace(
        generated,
        config=generated.config.model_copy(
            update={
                "runner_identity": EXECUTOR_IDENTITY,
                "runtime_identity": producer_runtime_identity,
            }
        ),
    )
    monkeypatch.setattr(
        dataset_module,
        "current_runtime_identity",
        lambda: producer_runtime_identity,
    )
    artifacts = publish_dataset(
        output_dir=tmp_path / "dataset",
        generated=generated,
    )
    monkeypatch.setattr(
        dataset_module,
        "current_runtime_identity",
        lambda: consumer_runtime_identity,
    )

    loaded = load_dataset(
        tmp_path / "dataset",
        expected_dataset_identity=artifacts.manifest.dataset_identity,
        max_manifest_bytes=_MANIFEST_BYTE_CEILING,
        max_records_bytes=_RECORDS_BYTE_CEILING,
    )

    assert loaded.manifest.config.runtime_identity == producer_runtime_identity
    assert loaded.manifest.dataset_identity == (
        artifacts.manifest.dataset_identity
    )


def test_publication_requires_current_production_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    generated = _generated(_record())
    generated = replace(
        generated,
        config=generated.config.model_copy(
            update={
                "runner_identity": EXECUTOR_IDENTITY,
                "runtime_identity": "a" * 64,
            }
        ),
    )
    monkeypatch.setattr(
        dataset_module,
        "current_runtime_identity",
        lambda: "b" * 64,
    )

    with pytest.raises(DatasetValidationError, match="runtime identity"):
        publish_dataset(
            output_dir=tmp_path / "dataset",
            generated=generated,
        )


def test_loader_rejects_malformed_production_runtime_identity(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "dataset"
    artifacts = publish_dataset(
        output_dir=destination,
        generated=_generated(_record()),
    )
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    manifest["config"]["runner_identity"] = EXECUTOR_IDENTITY
    manifest["config"]["runtime_identity"] = "not-a-runtime-digest"
    config = GenerationConfig.model_validate_json(
        json.dumps(manifest["config"])
    )
    manifest["config_identity"] = config.identity_hash()
    _rewrite_manifest_identity(manifest)
    artifacts.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetValidationError, match="runtime identity"):
        load_dataset(
            destination,
            expected_dataset_identity=manifest["dataset_identity"],
            max_manifest_bytes=_MANIFEST_BYTE_CEILING,
            max_records_bytes=_RECORDS_BYTE_CEILING,
        )


def test_tampered_dataset_source_is_rejected_without_source_access(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "dataset"
    artifacts = publish_dataset(
        output_dir=destination,
        generated=_generated(_record()),
    )
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    manifest["config"]["dataset_source"] = "hf"
    artifacts.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        humaneval_loader_module,
        "_load_from_hf",
        lambda: (_ for _ in ()).throw(
            AssertionError("tampered source must not trigger network access")
        ),
    )

    with pytest.raises(DatasetValidationError, match="config identity"):
        load_dataset(
            destination,
            expected_dataset_identity=artifacts.manifest.dataset_identity,
            max_manifest_bytes=_MANIFEST_BYTE_CEILING,
            max_records_bytes=_RECORDS_BYTE_CEILING,
        )


def test_loader_rejects_extra_files(tmp_path: Path) -> None:
    destination = tmp_path / "dataset"
    artifacts = publish_dataset(
        output_dir=destination,
        generated=_generated(_record()),
    )
    (destination / "extra.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="exactly"):
        load_dataset(
            destination,
            expected_dataset_identity=artifacts.manifest.dataset_identity,
            max_manifest_bytes=_MANIFEST_BYTE_CEILING,
            max_records_bytes=_RECORDS_BYTE_CEILING,
        )


def test_publication_rejects_bad_record_content_identity(
    tmp_path: Path,
) -> None:
    bad = _record().model_copy(update={"content_identity": "0" * 64})

    with pytest.raises(DatasetValidationError, match="content identity"):
        publish_dataset(
            output_dir=tmp_path / "dataset",
            generated=_generated(bad),
        )


def test_publication_rejects_unstable_record_order(tmp_path: Path) -> None:
    later = _record(task_id="HumanEval/10", seed=1, mutant_value="3")
    earlier = _record(task_id="HumanEval/2")

    with pytest.raises(DatasetValidationError, match="stable order"):
        publish_dataset(
            output_dir=tmp_path / "dataset",
            generated=replace(
                _generated(earlier, later),
                records=(later, earlier),
            ),
        )


def test_partition_rejects_accepted_and_skipped_overlap(
    tmp_path: Path,
) -> None:
    generated = _generated(_record())
    overlap = SkippedMutation(
        task_id="HumanEval/0",
        operator_family=OperatorFamily.COMPARISON_FLIP,
        seed=0,
        reason="overlap",
    )

    with pytest.raises(DatasetValidationError, match="overlap"):
        publish_dataset(
            output_dir=tmp_path / "dataset",
            generated=replace(generated, skipped=(overlap,)),
        )


def test_partition_rejects_missing_seed(tmp_path: Path) -> None:
    generated = _generated(_record(seed=1))

    with pytest.raises(DatasetValidationError, match="incomplete"):
        publish_dataset(
            output_dir=tmp_path / "dataset",
            generated=replace(generated, skipped=()),
        )


def test_partition_rejects_wildcard_overlap(tmp_path: Path) -> None:
    generated = _generated(_record())
    wildcard = SkippedMutation(
        task_id="HumanEval/0",
        operator_family="*",
        seed=None,
        reason="canonical source is malformed",
    )

    with pytest.raises(DatasetValidationError, match="task-wide"):
        publish_dataset(
            output_dir=tmp_path / "dataset",
            generated=replace(generated, skipped=(wildcard,)),
        )


def test_publication_rejects_oversized_input_suite(tmp_path: Path) -> None:
    generated = _generated(
        _record(input_reprs=("(1,)", "(2,)")),
    )
    suite = resolve_canonical_suite(
        task_ids=generated.config.task_ids,
        max_inputs=1,
        source=HumanEvalSource.SNAPSHOT,
    )
    config = generated.config.model_copy(
        update={
            "max_inputs_per_mutant": 1,
            "canonical_suite_digest": canonical_suite_digest(suite),
        }
    )

    with pytest.raises(DatasetValidationError, match="exceeds"):
        publish_dataset(
            output_dir=tmp_path / "dataset",
            generated=replace(generated, config=config),
        )


def test_publication_rejects_cross_record_canonical_disagreement(
    tmp_path: Path,
) -> None:
    generated = _generated(
        _record(seed=0),
        _record(seed=1, prompt="def changed(x):\n"),
    )

    with pytest.raises(DatasetValidationError, match="canonical task"):
        publish_dataset(
            output_dir=tmp_path / "dataset",
            generated=generated,
        )


def test_stale_dataset_identity_rejects_changed_skip_log(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "dataset"
    artifacts = publish_dataset(
        output_dir=destination,
        generated=_generated(_record(seed=1)),
    )
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    manifest["skipped"][0]["reason"] = "changed reason"
    artifacts.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetValidationError, match="dataset identity"):
        load_dataset(
            destination,
            expected_dataset_identity=artifacts.manifest.dataset_identity,
            max_manifest_bytes=_MANIFEST_BYTE_CEILING,
            max_records_bytes=_RECORDS_BYTE_CEILING,
        )


def test_snapshot_dataset_load_is_offline_after_publication(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "dataset"
    artifacts = publish_dataset(
        output_dir=destination,
        generated=_generated(_record()),
    )
    monkeypatch.setattr(
        provenance_module,
        "load_humaneval_plus",
        lambda source: (_ for _ in ()).throw(
            AssertionError("source must not be reacquired")
        ),
    )

    loaded = load_dataset(
        destination,
        expected_dataset_identity=artifacts.manifest.dataset_identity,
        max_manifest_bytes=_MANIFEST_BYTE_CEILING,
        max_records_bytes=_RECORDS_BYTE_CEILING,
    )

    assert loaded.records == (_record(),)


def test_hf_dataset_uses_hf_for_publication_and_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task = HumanEvalPlusTask(
        task_id="HumanEval/0",
        prompt="def f(x):\n",
        canonical_solution="    return x < 1\n",
        entry_point="f",
        test=_TEST,
    )
    hf_calls = 0

    def load_hf() -> list[HumanEvalPlusTask]:
        nonlocal hf_calls
        hf_calls += 1
        return [task]

    def reject_snapshot() -> bytes:
        raise AssertionError("packaged snapshot must not be consulted")

    monkeypatch.setattr(
        provenance_module,
        "load_humaneval_plus",
        humaneval_loader_module.load_humaneval_plus,
    )
    monkeypatch.setattr(
        humaneval_loader_module,
        "_load_from_hf",
        load_hf,
    )
    monkeypatch.setattr(
        humaneval_loader_module,
        "packaged_snapshot_bytes",
        reject_snapshot,
    )
    generated = generate_mutants(
        families=(OperatorFamily.COMPARISON_FLIP,),
        seeds=1,
        max_inputs_per_mutant=1,
        timeout_seconds=5.0,
        task_ids=("HumanEval/0",),
        dataset_source=HumanEvalSource.HF,
    )
    monkeypatch.setattr(
        humaneval_loader_module,
        "_load_from_hf",
        lambda: (_ for _ in ()).throw(
            AssertionError("HF must not be reacquired")
        ),
    )

    artifacts = publish_dataset(
        output_dir=tmp_path / "dataset",
        generated=generated,
    )
    loaded = load_dataset(
        tmp_path / "dataset",
        expected_dataset_identity=artifacts.manifest.dataset_identity,
        max_manifest_bytes=_MANIFEST_BYTE_CEILING,
        max_records_bytes=_RECORDS_BYTE_CEILING,
    )

    assert loaded.manifest.config.dataset_source is HumanEvalSource.HF
    assert hf_calls == 1


def test_publication_rejects_arbitrary_canonical_program(
    tmp_path: Path,
) -> None:
    generated = _generated(_record(prompt="def arbitrary(x):\n"))

    with pytest.raises(DatasetValidationError, match="canonical task"):
        publish_dataset(
            output_dir=tmp_path / "dataset",
            generated=generated,
        )


@pytest.mark.parametrize("malformed_source", [True, False])
def test_valid_failed_task_wildcard_round_trips(
    tmp_path: Path,
    monkeypatch,
    malformed_source: bool,
) -> None:
    task = _malformed_task(malformed_source=malformed_source)
    monkeypatch.setattr(
        provenance_module,
        "load_humaneval_plus",
        lambda source: [task],
    )
    reason = (
        "canonical source is malformed"
        if malformed_source
        else "canonical test is malformed"
    )
    generated = _failed_generated(
        task=task,
        skipped=(
            SkippedMutation(
                task_id=task.task_id,
                operator_family="*",
                seed=None,
                reason=reason,
            ),
        ),
    )

    artifacts = publish_dataset(
        output_dir=tmp_path / "dataset",
        generated=generated,
    )
    loaded = load_dataset(
        tmp_path / "dataset",
        expected_dataset_identity=artifacts.manifest.dataset_identity,
        max_manifest_bytes=_MANIFEST_BYTE_CEILING,
        max_records_bytes=_RECORDS_BYTE_CEILING,
    )

    assert loaded.records == ()
    assert loaded.manifest.skipped == generated.skipped


def test_failed_source_rejects_concrete_skip(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task = _malformed_task(malformed_source=True)
    monkeypatch.setattr(
        provenance_module,
        "load_humaneval_plus",
        lambda source: [task],
    )
    generated = _failed_generated(
        task=task,
        skipped=(
            SkippedMutation(
                task_id=task.task_id,
                operator_family=OperatorFamily.COMPARISON_FLIP,
                seed=0,
                reason="concrete",
            ),
        ),
    )

    with pytest.raises(DatasetValidationError, match="task-wide"):
        publish_dataset(
            output_dir=tmp_path / "dataset",
            generated=generated,
        )


def test_failed_test_rejects_concrete_record(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task = _malformed_task(malformed_source=False)
    monkeypatch.setattr(
        provenance_module,
        "load_humaneval_plus",
        lambda source: [task],
    )
    generated = _failed_generated(task=task, records=(_record(),))

    with pytest.raises(DatasetValidationError, match="task-wide"):
        publish_dataset(
            output_dir=tmp_path / "dataset",
            generated=generated,
        )


@pytest.mark.parametrize(
    "skipped",
    [
        (),
        (
            SkippedMutation(
                task_id="HumanEval/0",
                operator_family="*",
                seed=None,
                reason="wrong failure",
            ),
        ),
    ],
)
def test_failed_task_rejects_missing_or_wrong_wildcard(
    tmp_path: Path,
    monkeypatch,
    skipped,
) -> None:
    task = _malformed_task(malformed_source=True)
    monkeypatch.setattr(
        provenance_module,
        "load_humaneval_plus",
        lambda source: [task],
    )
    generated = _failed_generated(task=task, skipped=skipped)

    message = "exactly one" if not skipped else "preparation failure"
    with pytest.raises(DatasetValidationError, match=message):
        publish_dataset(
            output_dir=tmp_path / "dataset",
            generated=generated,
        )


def _rewrite_manifest_identity(manifest: dict[str, object]) -> None:
    manifest["dataset_identity"] = identity_hash_for(
        schema="dr_code.mutants.dataset",
        payload={
            "accepted_by_family": manifest["accepted_by_family"],
            "accepted_count": manifest["accepted_count"],
            "config_identity": manifest["config_identity"],
            "records_sha256": manifest["records_sha256"],
            "skipped": manifest["skipped"],
        },
    )


@pytest.mark.timeout(1)
def test_loader_validates_huge_coordinate_space_without_enumeration(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "dataset"
    artifacts = publish_dataset(
        output_dir=destination,
        generated=_generated(_record()),
    )
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    manifest["config"]["seeds"] = 10**100
    config = GenerationConfig.model_validate_json(
        json.dumps(manifest["config"])
    )
    manifest["config_identity"] = config.identity_hash()
    _rewrite_manifest_identity(manifest)
    artifacts.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetValidationError, match="incomplete"):
        load_dataset(
            destination,
            expected_dataset_identity=manifest["dataset_identity"],
            max_manifest_bytes=_MANIFEST_BYTE_CEILING,
            max_records_bytes=_RECORDS_BYTE_CEILING,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seeds", "1"),
        ("seeds", True),
        ("max_inputs_per_mutant", "10"),
        ("timeout_seconds", 5),
        ("timeout_seconds", True),
        ("dataset_id", 7),
    ],
)
def test_loader_rejects_coerced_config_scalars(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    destination = tmp_path / field
    artifacts = publish_dataset(
        output_dir=destination,
        generated=_generated(_record()),
    )
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    manifest["config"][field] = value
    artifacts.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        DatasetValidationError, match="invalid mutant manifest"
    ):
        load_dataset(
            destination,
            expected_dataset_identity=artifacts.manifest.dataset_identity,
            max_manifest_bytes=_MANIFEST_BYTE_CEILING,
            max_records_bytes=_RECORDS_BYTE_CEILING,
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("accepted_count",), "1"),
        (("accepted_by_family", 0, "count"), True),
        (("skipped", 0, "seed"), "0"),
    ],
)
def test_loader_rejects_coerced_nested_manifest_scalars(
    tmp_path: Path,
    path: tuple[object, ...],
    value: object,
) -> None:
    destination = tmp_path / "dataset"
    artifacts = publish_dataset(
        output_dir=destination,
        generated=_generated(_record(seed=1)),
    )
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    target = manifest
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    _rewrite_manifest_identity(manifest)
    artifacts.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        DatasetValidationError, match="invalid mutant manifest"
    ):
        load_dataset(
            destination,
            expected_dataset_identity=artifacts.manifest.dataset_identity,
            max_manifest_bytes=_MANIFEST_BYTE_CEILING,
            max_records_bytes=_RECORDS_BYTE_CEILING,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", "0"),
        ("site_node_path", True),
        ("task_id", 0),
        ("operator_family", 1),
    ],
)
def test_loader_rejects_coerced_nested_record_scalars(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    destination = tmp_path / field
    artifacts = publish_dataset(
        output_dir=destination,
        generated=_generated(_record()),
    )
    record = json.loads(artifacts.records_path.read_text(encoding="utf-8"))
    record[field] = value
    records_bytes = (
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    artifacts.records_path.write_bytes(records_bytes)

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    manifest["records_sha256"] = hashlib.sha256(records_bytes).hexdigest()
    _rewrite_manifest_identity(manifest)
    artifacts.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        DatasetValidationError,
        match="invalid mutants.jsonl line",
    ):
        load_dataset(
            destination,
            expected_dataset_identity=manifest["dataset_identity"],
            max_manifest_bytes=_MANIFEST_BYTE_CEILING,
            max_records_bytes=_RECORDS_BYTE_CEILING,
        )


def test_loader_stops_directory_scan_at_third_entry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "dataset"
    artifacts = publish_dataset(
        output_dir=destination,
        generated=_generated(_record()),
    )
    entries_seen = 0

    def iterdir(path: Path):
        nonlocal entries_seen
        assert path == destination
        for entry in (artifacts.manifest_path, artifacts.records_path):
            entries_seen += 1
            yield entry
        while True:
            entries_seen += 1
            yield destination / f"unexpected-{entries_seen}"

    monkeypatch.setattr(Path, "iterdir", iterdir)

    with pytest.raises(DatasetValidationError, match="exactly"):
        load_dataset(
            destination,
            expected_dataset_identity=artifacts.manifest.dataset_identity,
            max_manifest_bytes=_MANIFEST_BYTE_CEILING,
            max_records_bytes=_RECORDS_BYTE_CEILING,
        )

    assert entries_seen == 3


def test_loader_rejects_legacy_outcome_shape_in_jsonl(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "dataset"
    artifacts = publish_dataset(
        output_dir=destination,
        generated=_generated(_record()),
    )
    record = json.loads(artifacts.records_path.read_text(encoding="utf-8"))
    record["canonical_expected"][0] = {
        "kind": "value",
        "output_repr": "legacy",
    }
    records_bytes = (
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    artifacts.records_path.write_bytes(records_bytes)
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    manifest["records_sha256"] = hashlib.sha256(records_bytes).hexdigest()
    _rewrite_manifest_identity(manifest)
    artifacts.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        DatasetValidationError,
        match="invalid mutants.jsonl line",
    ):
        load_dataset(
            destination,
            expected_dataset_identity=manifest["dataset_identity"],
            max_manifest_bytes=_MANIFEST_BYTE_CEILING,
            max_records_bytes=_RECORDS_BYTE_CEILING,
        )
