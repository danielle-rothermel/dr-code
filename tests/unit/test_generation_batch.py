"""Unit tests for fresh generation batch runner."""

from __future__ import annotations

from unittest.mock import patch

from dr_providers import LlmResponse

from dr_code.generation.batch import generate_attempts, select_tasks
from dr_code.generation.profiles import default_profiles_path
from dr_code.generation.run_config import GenerationRunConfig
from dr_code.models.attempts import AttemptSource


def test_select_tasks_filters_by_task_ids() -> None:
    config = GenerationRunConfig(
        run_id="test-run",
        profile_id="openrouter/google/gemini-3.1-flash-lite/off/v1",
        profiles_path=default_profiles_path(),
        task_ids=["HumanEval/0", "HumanEval/1"],
    )
    tasks = select_tasks(config)
    assert [task.task_id for task in tasks] == ["HumanEval/0", "HumanEval/1"]


def test_select_tasks_applies_limit() -> None:
    config = GenerationRunConfig(
        run_id="test-run",
        profile_id="openrouter/google/gemini-3.1-flash-lite/off/v1",
        profiles_path=default_profiles_path(),
        task_ids=["HumanEval/0", "HumanEval/1", "HumanEval/2"],
        limit=2,
    )
    tasks = select_tasks(config)
    assert len(tasks) == 2


@patch("dr_code.generation.batch.query_from_prompt")
@patch("dr_code.generation.batch.OpenRouterProvider")
def test_generate_attempts_builds_fresh_stub_records(
    mock_provider_cls: object,
    mock_query_from_prompt: object,
) -> None:
    mock_provider_cls.return_value.__enter__.return_value = object()
    mock_provider_cls.return_value.__exit__.return_value = None
    mock_query_from_prompt.return_value = LlmResponse(
        raw_json={},
        provider="openrouter",
        model="google/gemini-3.1-flash-lite",
        latency_ms=42,
        text="def has_close_elements(numbers, threshold):\n    return False\n",
        finish_reason="stop",
    )
    profile_id = "openrouter/google/gemini-3.1-flash-lite/off/v1"
    config = GenerationRunConfig(
        run_id="test-run",
        profile_id=profile_id,
        profiles_path=default_profiles_path(),
        task_ids=["HumanEval/0"],
    )
    records = generate_attempts(config)
    assert len(records) == 1
    record = records[0]
    assert record.run_id == "test-run"
    assert record.task_id == "HumanEval/0"
    assert record.provenance.source == AttemptSource.FRESH_STUB
    assert record.provenance.model == "google/gemini-3.1-flash-lite"
    assert record.provenance.dec_llm_config_id == profile_id
    assert record.decoder_input
    assert record.raw_output.startswith("def has_close_elements")

    mock_query_from_prompt.assert_called_once()
    kwargs = mock_query_from_prompt.call_args.kwargs
    assert kwargs["model"] == "google/gemini-3.1-flash-lite"
    assert "Write functional code in Python" in kwargs["prompt"]
