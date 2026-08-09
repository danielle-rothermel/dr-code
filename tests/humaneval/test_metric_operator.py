from __future__ import annotations

import asyncio

from _humaneval_builders import _task
from dr_code.humaneval.metric_operator import CodeTestSettings
from dr_code.metrics import (
    MetricName,
    MetricQuestion,
    MetricsDefinition,
    OperatorFailureRecord,
    extract_metrics,
)
from dr_code.trace import CodeArtifact, JsonArtifact, external_trace


def test_code_test_requires_the_evaluation_batch_execution_outcome() -> None:
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
