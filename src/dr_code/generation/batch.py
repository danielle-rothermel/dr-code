"""Batch fresh decoder generation for stage 1b."""

from __future__ import annotations

from dr_providers import OpenRouterProvider, ProviderName
from dr_providers.query.from_prompt import query_from_prompt

from dr_code.datasets.humaneval_loader import load_humaneval_plus
from dr_code.generation.profiles import resolve_profile
from dr_code.generation.prompts import (
    build_decoder_prompt,
    decoder_input_from_task,
)
from dr_code.generation.run_config import GenerationRunConfig
from dr_code.models.attempts import AttemptRecord
from dr_code.models.humaneval import HumanEvalPlusTask


def select_tasks(config: GenerationRunConfig) -> list[HumanEvalPlusTask]:
    """Select HumanEval+ tasks for a generation run."""
    all_tasks = load_humaneval_plus(prefer_snapshot=config.prefer_snapshot)
    if config.task_ids is None:
        selected = list(all_tasks)
    else:
        index = {task.task_id: task for task in all_tasks}
        selected: list[HumanEvalPlusTask] = []
        for task_id in config.task_ids:
            if task_id not in index:
                msg = f"Unknown HumanEval+ task_id: {task_id}"
                raise KeyError(msg)
            selected.append(index[task_id])
    if config.limit is not None:
        return selected[: config.limit]
    return selected


def generate_attempts(config: GenerationRunConfig) -> list[AttemptRecord]:
    """Generate fresh_stub AttemptRecord rows via dr-providers."""
    profile = resolve_profile(
        config.profile_id,
        profiles_path=config.profiles_path,
    )
    tasks = select_tasks(config)
    records: list[AttemptRecord] = []
    with OpenRouterProvider() as provider:
        for task in tasks:
            prompt = build_decoder_prompt(task)
            response = query_from_prompt(
                provider=provider,
                provider_name=ProviderName.OPENROUTER,
                model=profile.model,
                prompt=prompt,
                max_tokens=config.max_tokens,
                reasoning=profile.reasoning,
                sampling=profile.sampling,
                metadata={"run_id": config.run_id, "task_id": task.task_id},
            )
            records.append(
                AttemptRecord.stub_for_fresh(
                    task,
                    decoder_input=decoder_input_from_task(task),
                    raw_output=response.text,
                    run_id=config.run_id,
                    model=profile.model,
                    profile_id=profile.profile_id,
                )
            )
    return records
