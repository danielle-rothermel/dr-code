#!/usr/bin/env python3

"""Build a validated corpus from the archived generation pool dump."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

from drc_generation_corpus import (
    BuildManifest,
    CorpusPopulation,
    DatasetName,
    build_generation_corpus,
)
from drc_generation_corpus.adapters import (
    BigCodeBenchLiteProCodeCompAdapter,
    ClassEvalCodeCompAdapter,
    HumanEvalAdapter,
    HumanEvalProCodeCompAdapter,
    MbppProCodeCompAdapter,
    NlLatentsAdapter,
)
from drc_generation_corpus.adapters.base import CorpusAdapter
from drc_generation_corpus.tasks import HumanEvalTaskAdapter

_EXPECTED_POPULATIONS: Final = {
    DatasetName.HUMAN_EVAL: CorpusPopulation(
        generations=203_407,
        source_records=630_089,
        encoder_artifacts=221_084,
        requests=203_407,
        tasks=164,
    ),
    DatasetName.MBPP_PRO: CorpusPopulation(
        generations=22_639,
        source_records=143_655,
        encoder_artifacts=111_631,
        requests=22_639,
        tasks=375,
    ),
    DatasetName.HUMANEVAL_PRO: CorpusPopulation(
        generations=9_848,
        source_records=62_543,
        encoder_artifacts=48_607,
        requests=9_848,
        tasks=163,
    ),
    DatasetName.CLASS_EVAL: CorpusPopulation(
        generations=5_934,
        source_records=37_564,
        encoder_artifacts=29_196,
        requests=5_934,
        tasks=196,
    ),
    DatasetName.BIGCODEBENCH_LITE_PRO: CorpusPopulation(
        generations=3_262,
        source_records=20_712,
        encoder_artifacts=16_082,
        requests=3_262,
        tasks=108,
    ),
    DatasetName.NL_LATENTS: CorpusPopulation(
        generations=191_462,
        source_records=192_333,
        encoder_artifacts=526,
        requests=191_462,
        tasks=294,
    ),
}


def _existing_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"path does not exist: {path}")
    return path


def _output_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _adapter(dataset: DatasetName, task_source: Path) -> CorpusAdapter:
    if dataset is DatasetName.HUMAN_EVAL:
        if not task_source.is_file():
            raise ValueError(
                "human_eval task source must be a snapshot JSON file"
            )
        return HumanEvalAdapter(HumanEvalTaskAdapter(task_source))
    if not task_source.is_dir():
        raise ValueError(f"{dataset.value} task source must be a directory")
    if dataset is DatasetName.MBPP_PRO:
        return MbppProCodeCompAdapter(task_source)
    if dataset is DatasetName.HUMANEVAL_PRO:
        return HumanEvalProCodeCompAdapter(task_source)
    if dataset is DatasetName.CLASS_EVAL:
        return ClassEvalCodeCompAdapter(task_source)
    if dataset is DatasetName.BIGCODEBENCH_LITE_PRO:
        return BigCodeBenchLiteProCodeCompAdapter(task_source)
    if dataset is DatasetName.NL_LATENTS:
        return NlLatentsAdapter(task_source)
    raise AssertionError(f"unhandled dataset {dataset!r}")


def _print_summary(destination: Path, manifest: BuildManifest) -> None:
    print(f"Published corpus: {destination}")
    print(f"Adapter: {manifest.adapter_name}@{manifest.adapter_version}")
    for name in (
        "generations",
        "source_records",
        "encoder_artifacts",
        "requests",
        "tasks",
    ):
        artifact = getattr(manifest, name)
        print(f"{name}: {artifact.rows:,} rows ({artifact.sha256})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build one validated generation corpus without executing any "
            "candidate code."
        )
    )
    parser.add_argument(
        "dataset",
        type=DatasetName,
        choices=tuple(DatasetName),
        help="dataset population to extract",
    )
    parser.add_argument(
        "--dump-directory",
        type=_existing_path,
        required=True,
        help="pool-dump directory containing manifest.json and gzip tables",
    )
    parser.add_argument(
        "--task-source",
        type=_existing_path,
        required=True,
        help="pinned snapshot file or archived task/cache directory",
    )
    parser.add_argument(
        "--output-directory",
        type=_output_path,
        required=True,
        help="new or empty corpus destination",
    )
    arguments = parser.parse_args()
    try:
        adapter = _adapter(arguments.dataset, arguments.task_source)
        manifest = build_generation_corpus(
            dump_directory=arguments.dump_directory,
            destination=arguments.output_directory,
            adapter=adapter,
            expected_population=_EXPECTED_POPULATIONS[arguments.dataset],
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    _print_summary(arguments.output_directory, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
