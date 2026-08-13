from __future__ import annotations

import asyncio

import pytest

from _humaneval_builders import _task
from drc_humaneval.metric_operator import CodeTest, CodeTestSettings
from drc_humaneval.task import HumanEvalTask
from dr_code.metrics import (
    MetricName,
    MetricQuestion,
    MetricsDefinition,
    OperatorFailureRecord,
    extract_metrics,
)
from dr_code.trace import CodeArtifact, JsonArtifact, external_trace


def test_code_test_requires_the_eval_batch_execution_outcome() -> None:
    code = CodeArtifact(source="def add_one(x):\n    return x + 1\n")
    trace = external_trace(
        {
            "input": code,
            "output": code,
            "task": JsonArtifact(payload=_task().model_dump(mode="json")),
        }
    )
    definition = MetricsDefinition(
        definition_id="code-test",
        version="0",
        questions=(
            MetricQuestion(
                metric=MetricName.CODE_TEST,
                on="output",
                settings=CodeTestSettings(),
            ),
        ),
    )

    (record,) = asyncio.run(extract_metrics(definition, trace))

    assert isinstance(record, OperatorFailureRecord)
    assert record.failure.failure_type == "RuntimeError"
    assert record.failure.failure_message == (
        "code_test has no candidate execution outcome"
    )


def test_code_test_reuses_validated_task_for_equal_frozen_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drc_humaneval import metric_operator

    validation_count = 0
    validate_task_payload = metric_operator._validate_task_payload

    def counting_validate_task_payload(
        artifact: JsonArtifact,
    ) -> HumanEvalTask:
        nonlocal validation_count
        validation_count += 1
        return validate_task_payload(artifact)

    monkeypatch.setattr(
        metric_operator,
        "_validate_task_payload",
        counting_validate_task_payload,
    )
    operator = CodeTest(CodeTestSettings())
    payload = _task().model_dump(mode="json")

    operator.validate_auxiliary({"task": JsonArtifact(payload=payload)})
    operator.validate_auxiliary({"task": JsonArtifact(payload=payload)})

    assert validation_count == 1
