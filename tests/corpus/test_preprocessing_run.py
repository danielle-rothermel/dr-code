from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dr_code.corpus.preprocessing_contract import (
    PREPROCESSING_MANIFEST_SCHEMA_VERSION,
)
from dr_code.corpus import (
    CorpusRunError,
    run_preprocessing_corpus,
)
from dr_code.preprocessing.definitions import (
    HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
)


def _write_input(
    path: Path,
    outputs: list[str | None],
    *,
    row_group_size: int = 1,
) -> Path:
    schema = pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("decoder_output", pa.string(), nullable=True),
        ]
    )
    table = pa.Table.from_arrays(
        [
            pa.array([f"sample-{index}" for index in range(len(outputs))]),
            pa.array(outputs),
        ],
        schema=schema,
    )
    pq.write_table(table, path, row_group_size=row_group_size)
    return path


def _manifest(path: Path) -> dict[str, object]:
    return json.loads((path / "manifest.json").read_text())


def _unreadable_source_evidence() -> str:
    raise ValueError("unreadable")


def test_row_group_resume_publishes_validated_schema_four_run(
    tmp_path: Path,
) -> None:
    source = _write_input(
        tmp_path / "input.parquet",
        [None, "", "def f():\n    return 1\n"],
    )
    root = tmp_path / "runs"
    partial = run_preprocessing_corpus(
        input_path=source,
        output_root=root,
        run_id="resume",
        max_row_groups=2,
        batch_size=1,
    )
    assert partial == root / "resume.partial"
    assert _manifest(partial)["completed_row_groups"] == [0, 1]

    complete = run_preprocessing_corpus(
        input_path=source,
        output_root=root,
        run_id="resume",
        batch_size=1,
    )
    manifest = _manifest(complete)
    assert manifest["schema_version"] == PREPROCESSING_MANIFEST_SCHEMA_VERSION
    assert len(manifest["installed_environment"]["identity"]) == 64
    assert manifest["complete"] is True
    assert set(manifest["relation_sha256"]) == {
        "results",
        "candidates",
        "step_facts",
        "rejections",
    }
    results = pq.read_table(complete / "results.parquet").to_pylist()
    assert [row["decoder_output_presence"] for row in results] == [
        "missing",
        "present",
        "present",
    ]


def test_resume_rejects_coordinate_and_checkpoint_corruption(
    tmp_path: Path,
) -> None:
    source = _write_input(
        tmp_path / "input.parquet",
        ["def f():\n    return 1\n", None],
    )
    root = tmp_path / "runs"
    partial = run_preprocessing_corpus(
        input_path=source,
        output_root=root,
        run_id="corrupt",
        batch_size=1,
        max_row_groups=1,
    )
    with pytest.raises(CorpusRunError, match="batch_size"):
        run_preprocessing_corpus(
            input_path=source,
            output_root=root,
            run_id="corrupt",
            batch_size=2,
        )

    part_manifest = partial / "parts" / "row_group_00000000" / "manifest.json"
    payload = json.loads(part_manifest.read_text())
    payload["relations"]["results"]["sha256"] = "0" * 64
    part_manifest.write_text(json.dumps(payload))
    with pytest.raises(CorpusRunError, match="hash mismatch"):
        run_preprocessing_corpus(
            input_path=source,
            output_root=root,
            run_id="corrupt",
            batch_size=1,
        )


def test_deep_lambda_row_group_survives_interrupted_resume(
    tmp_path: Path,
) -> None:
    deep_lambda = "lambda value: " + " + ".join(["value"] * 500)
    source = _write_input(
        tmp_path / "deep.parquet",
        ["def f():\n    return 1\n", deep_lambda],
    )
    root = tmp_path / "runs"
    run_preprocessing_corpus(
        input_path=source,
        output_root=root,
        run_id="deep",
        max_row_groups=1,
    )
    complete = run_preprocessing_corpus(
        input_path=source,
        output_root=root,
        run_id="deep",
    )
    assert _manifest(complete)["complete"] is True
    assert pq.read_table(complete / "results.parquet").num_rows == 2


def test_nul_decoder_output_publishes_one_stable_invalid_result(
    tmp_path: Path,
) -> None:
    decoder_output = "def f():\n    return '\x00'\n"
    source = _write_input(tmp_path / "nul.parquet", [decoder_output])

    complete = run_preprocessing_corpus(
        input_path=source,
        output_root=tmp_path / "runs",
        run_id="nul",
    )

    results = pq.read_table(complete / "results.parquet").to_pylist()
    assert len(results) == 1
    assert results[0] == {
        "sample_id": "sample-0",
        "decoder_output_presence": "present",
        "raw_output_sha256": hashlib.sha256(
            decoder_output.encode()
        ).hexdigest(),
        "outcome": "decoder_output_invalid",
        "outcome_code": None,
        "failure_code": "decoder_output_invalid",
        "failed_step": "validate_decoder_output",
        "cause": "decoder output contains unsupported characters",
        "propagated_through": [
            step.instance_name
            for step in HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION.steps
        ],
        "final_candidate_count": 0,
    }
    assert pq.read_table(complete / "step_facts.parquet").to_pylist() == [
        {
            "sample_id": "sample-0",
            "step_name": "validate_decoder_output",
            "facts_json": (
                '{"contains_nul":true,"contains_surrogate":false,'
                '"text_character_count":27}'
            ),
        }
    ]
    assert pq.read_table(complete / "candidates.parquet").num_rows == 0
    assert pq.read_table(complete / "rejections.parquet").num_rows == 0


def test_complete_partial_after_final_rename_interruption_is_publishable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dr_code.corpus import preprocessing_run

    source = _write_input(tmp_path / "input.parquet", [None])
    root = tmp_path / "runs"
    destination = root / "publish"
    original = preprocessing_run.os.replace

    def fail_final(source_path: Path | str, target: Path | str) -> None:
        if Path(target) == destination:
            raise OSError("publish interrupted")
        original(source_path, target)

    monkeypatch.setattr(preprocessing_run.os, "replace", fail_final)
    with pytest.raises(OSError, match="publish interrupted"):
        run_preprocessing_corpus(
            input_path=source,
            output_root=root,
            run_id="publish",
        )
    assert _manifest(root / "publish.partial")["complete"] is True

    monkeypatch.setattr(preprocessing_run.os, "replace", original)
    assert (
        run_preprocessing_corpus(
            input_path=source,
            output_root=root,
            run_id="publish",
        )
        == destination
    )


def test_final_run_rename_flushes_complete_directory_then_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dr_code.corpus import preprocessing_run

    source = _write_input(tmp_path / "input.parquet", [None])
    root = tmp_path / "runs"
    completed = root / "durable"
    events: list[str] = []
    original_directory = preprocessing_run.fsync_directory
    original_replace = preprocessing_run.os.replace

    def fsync_directory(path: Path) -> None:
        events.append(f"directory:{path.name}")
        original_directory(path)

    def replace(source_path: Path | str, target: Path | str) -> None:
        if Path(target) == completed:
            events.append("final-rename")
        original_replace(source_path, target)

    monkeypatch.setattr(
        preprocessing_run,
        "fsync_directory",
        fsync_directory,
    )
    monkeypatch.setattr(preprocessing_run.os, "replace", replace)

    run_preprocessing_corpus(
        input_path=source,
        output_root=root,
        run_id="durable",
    )

    rename = events.index("final-rename")
    assert "directory:durable.partial" in events[:rename]
    assert events[rename + 1] == "directory:runs"


def test_concurrent_same_run_writer_cannot_delete_live_temporary_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dr_code.corpus import preprocessing_run

    source = _write_input(
        tmp_path / "input.parquet",
        ["def f():\n    return 1\n"],
    )
    root = tmp_path / "runs"
    entered_projection = threading.Event()
    release_projection = threading.Event()
    original = preprocessing_run._project_batch

    def blocking_projection(
        batch: pa.RecordBatch,
        runner: preprocessing_run.BoundPreprocessingRunner,
    ):
        entered_projection.set()
        assert release_projection.wait(timeout=10)
        return original(batch, runner)

    monkeypatch.setattr(
        preprocessing_run, "_project_batch", blocking_projection
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        active = executor.submit(
            run_preprocessing_corpus,
            input_path=source,
            output_root=root,
            run_id="contended",
        )
        assert entered_projection.wait(timeout=10)
        temporary_parts = list(
            (root / "contended.partial" / "parts").glob(
                ".row_group_00000000.*.tmp"
            )
        )
        assert len(temporary_parts) == 1
        temporary_part = temporary_parts[0]
        inode = temporary_part.stat().st_ino

        with pytest.raises(CorpusRunError, match="live writer"):
            run_preprocessing_corpus(
                input_path=source,
                output_root=root,
                run_id="contended",
            )

        assert temporary_part.stat().st_ino == inode
        release_projection.set()
        assert active.result(timeout=10) == root / "contended"


def test_source_coordinates_authenticate_installed_package_bytes() -> None:
    from dr_code.corpus import preprocessing_run
    from dr_code.implementation_identity import package_source_digest

    assert (
        preprocessing_run._source_coordinates()[
            "dr_code_python_package_sha256"
        ]
        == package_source_digest()
    )


@pytest.mark.parametrize(
    ("source_evidence", "message"),
    [
        (lambda: "", "invalid"),
        (_unreadable_source_evidence, "unavailable"),
    ],
)
def test_source_evidence_failure_rejects_before_output_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_evidence: Callable[[], str],
    message: str,
) -> None:
    from dr_code.corpus import preprocessing_run

    source = _write_input(tmp_path / "input.parquet", [None])
    root = tmp_path / "runs"
    monkeypatch.setattr(
        preprocessing_run,
        "package_source_digest",
        source_evidence,
    )

    with pytest.raises(CorpusRunError, match=message):
        run_preprocessing_corpus(
            input_path=source,
            output_root=root,
            run_id="untrusted",
        )

    assert not root.exists()


def test_dependency_drift_rejects_resume_before_output_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dr_code.corpus import preprocessing_run

    source = _write_input(tmp_path / "input.parquet", [None, None])
    root = tmp_path / "runs"
    partial = run_preprocessing_corpus(
        input_path=source,
        output_root=root,
        run_id="environment",
        max_row_groups=1,
    )
    before = {
        path.relative_to(root): (path.stat().st_mtime_ns, path.read_bytes())
        for path in root.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        preprocessing_run,
        "installed_environment_provenance",
        lambda: {
            "distributions": [{"name": "drift", "version": "1"}],
            "identity": "0" * 64,
        },
    )

    with pytest.raises(CorpusRunError, match="installed_environment"):
        run_preprocessing_corpus(
            input_path=source,
            output_root=root,
            run_id="environment",
        )

    after = {
        path.relative_to(root): (path.stat().st_mtime_ns, path.read_bytes())
        for path in root.rglob("*")
        if path.is_file()
    }
    assert partial.is_dir()
    assert after == before


def test_input_mutation_after_stable_capture_does_not_change_processing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dr_code.corpus import preprocessing_run

    source = _write_input(
        tmp_path / "input.parquet",
        ["def f():\n    return 1\n"],
    )
    original = preprocessing_run._validate_input_schema

    def mutate_live_source(schema: pa.Schema) -> None:
        source.write_bytes(b"corrupt")
        original(schema)

    monkeypatch.setattr(
        preprocessing_run,
        "_validate_input_schema",
        mutate_live_source,
    )
    completed = run_preprocessing_corpus(
        input_path=source,
        output_root=tmp_path / "runs",
        run_id="stable",
    )

    assert pq.read_table(completed / "results.parquet").num_rows == 1


def test_dangling_output_root_symlink_rejects_before_mutation(
    tmp_path: Path,
) -> None:
    source = _write_input(tmp_path / "input.parquet", [None])
    target = tmp_path / "missing-target"
    output_root = tmp_path / "runs"
    output_root.symlink_to(target, target_is_directory=True)
    source_before = source.read_bytes()

    with pytest.raises(CorpusRunError, match="symlink"):
        run_preprocessing_corpus(
            input_path=source,
            output_root=output_root,
            run_id="dangling",
        )

    assert output_root.is_symlink()
    assert not target.exists()
    assert source.read_bytes() == source_before


@pytest.mark.parametrize(
    "run_id",
    [
        ".victim.partial.tmp",
        "victim.partial",
    ],
)
def test_public_api_rejects_reserved_run_id_before_output_mutation(
    tmp_path: Path,
    run_id: str,
) -> None:
    source = _write_input(tmp_path / "input.parquet", [None])
    root = tmp_path / "runs"
    root.mkdir()
    marker = root / "existing"
    marker.write_bytes(b"unchanged")

    with pytest.raises(CorpusRunError, match="reserved"):
        run_preprocessing_corpus(
            input_path=source,
            output_root=root,
            run_id=run_id,
        )

    assert list(root.iterdir()) == [marker]
    assert marker.read_bytes() == b"unchanged"


@pytest.mark.parametrize(
    "reserved_relative",
    [
        "reserved",
        "reserved.partial",
        ".reserved.lock",
        ".reserved.partial.tmp",
    ],
)
def test_reserved_preprocessing_path_symlink_rejects_before_mutation(
    tmp_path: Path,
    reserved_relative: str,
) -> None:
    source = _write_input(tmp_path / "input.parquet", [None])
    root = tmp_path / "runs"
    root.mkdir()
    external = tmp_path / f"external-{reserved_relative.replace('.', '-')}"
    external.mkdir()
    marker = external / "marker"
    marker.write_bytes(b"unchanged")
    (root / reserved_relative).symlink_to(
        external,
        target_is_directory=True,
    )

    with pytest.raises(CorpusRunError, match="symlink"):
        run_preprocessing_corpus(
            input_path=source,
            output_root=root,
            run_id="reserved",
        )

    assert marker.read_bytes() == b"unchanged"


def test_preprocessing_output_parent_symlink_alias_rejects_before_mutation(
    tmp_path: Path,
) -> None:
    source = _write_input(tmp_path / "input.parquet", [None])
    external = tmp_path / "external"
    external.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(external, target_is_directory=True)

    with pytest.raises(CorpusRunError, match="symlink"):
        run_preprocessing_corpus(
            input_path=source,
            output_root=alias / "runs",
            run_id="aliased",
        )

    assert not (external / "runs").exists()
