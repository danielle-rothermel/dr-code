from __future__ import annotations

from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).parents[1]
OPERATIONAL_EVALUATION_PATHS = (
    Path(".github/workflows/ci.yml"),
    Path("README.md"),
    Path("docs/humaneval-plus-subprocess.md"),
    Path("docs/preprocessing-extraction-validation-redesign-plan.md"),
    Path("scripts/evaluate_preprocessing_candidates.py"),
    Path("src/dr_code/corpus/candidate_evaluation.py"),
    Path("src/dr_code/humaneval/subprocess_runner.py"),
)
CONTAINER_REQUIREMENTS = (
    "docker",
    "podman",
    "DR_CODE_SANDBOX_IMAGE",
    "DR_CODE_SANDBOX_RUNTIME",
)


@pytest.mark.parametrize("relative_path", OPERATIONAL_EVALUATION_PATHS)
def test_operational_evaluation_paths_do_not_require_containers(
    relative_path: Path,
) -> None:
    source = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")

    for requirement in CONTAINER_REQUIREMENTS:
        assert requirement.casefold() not in source.casefold()


def test_obsolete_container_assets_are_absent() -> None:
    obsolete_paths = (
        Path("docker/humaneval-plus/Dockerfile"),
        Path("docs/humaneval-plus-sandbox.md"),
        Path("src/dr_code/humaneval/sandbox.py"),
    )

    assert all(
        not (REPOSITORY_ROOT / relative_path).exists()
        for relative_path in obsolete_paths
    )
