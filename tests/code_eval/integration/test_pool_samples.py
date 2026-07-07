"""Smoke-test validate() on real dr-llm pool decoder outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import code_eval

POOL_SAMPLES_PATH = Path(__file__).resolve().parents[1] / "corpus" / "pool_samples.jsonl"

pytestmark = pytest.mark.integration


def _load_pool_samples() -> list[dict]:
    if not POOL_SAMPLES_PATH.exists():
        pytest.skip(f"pool samples not found at {POOL_SAMPLES_PATH}")
    return [json.loads(line) for line in POOL_SAMPLES_PATH.read_text().splitlines() if line.strip()]


@pytest.fixture(scope="module")
def pool_samples() -> list[dict]:
    return _load_pool_samples()


@pytest.fixture(scope="module")
def extraction_validator() -> code_eval.LLMCodeValidator:
    return code_eval.LLMCodeValidator(config=code_eval.EXTRACTION_CONFIG)


def test_pool_samples_no_crashes(
    pool_samples: list[dict],
    extraction_validator: code_eval.LLMCodeValidator,
) -> None:
    for sample in pool_samples:
        result = extraction_validator.validate(
            sample["raw_output"],
            task_id=sample["task_id"],
        )
        assert isinstance(result, code_eval.ValidationResult)


def test_pool_samples_expect_success(
    pool_samples: list[dict],
    extraction_validator: code_eval.LLMCodeValidator,
) -> None:
    for sample in pool_samples:
        result = extraction_validator.validate(
            sample["raw_output"],
            task_id=sample["task_id"],
        )
        assert (
            result.recovery.overall_success == sample["expect_success"]
        ), f"task={sample['task_id']} pattern={sample['pattern']}"


def test_pool_samples_best_source_when_success(
    pool_samples: list[dict],
    extraction_validator: code_eval.LLMCodeValidator,
) -> None:
    for sample in pool_samples:
        if not sample["expect_success"]:
            continue
        result = extraction_validator.validate(
            sample["raw_output"],
            task_id=sample["task_id"],
        )
        source = result.recovery.selected_source()
        assert source is not None
        assert "def " in source or "import " in source
