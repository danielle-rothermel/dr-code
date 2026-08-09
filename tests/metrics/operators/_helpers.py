from __future__ import annotations

import asyncio

from dr_code.trace import CodeArtifact, TextArtifact, external_trace


SAMPLE_TEXT = (
    "Here is some text with a `code` fence:\n"
    "```python\ndef foo(x):\n    return x + 1\n```\n"
    "It has keywords like def and return, plus + - * operators.\n"
)


SAMPLE_CODE = (
    "def add(a, b):\n"
    '    """Sum two numbers."""\n'
    "    total = a + b\n"
    "    if total > 0:\n"
    "        return total\n"
    "    return 0\n"
)


def _definition(questions) -> object:
    from dr_code.metrics import MetricsDefinition

    return MetricsDefinition(
        definition_id="parity", version="1", questions=tuple(questions)
    )


def _question(metric_name: str, on: str = "input", **settings) -> object:
    from dr_code.metrics import MetricName, MetricQuestion

    return MetricQuestion(
        metric=MetricName(metric_name), on=on, settings=settings
    )


def _text_trace(text: str):
    return external_trace(
        {"input": TextArtifact(text=text), "output": TextArtifact(text=text)}
    )


def _code_trace(source: str):
    return external_trace(
        {
            "input": CodeArtifact(source=source),
            "output": CodeArtifact(source=source),
        }
    )


def _extract(definition, trace, **kwargs):
    from dr_code.metrics import extract_metrics

    return asyncio.run(extract_metrics(definition, trace, **kwargs))


def _facts(record):
    assert record.status.value == "measured", record
    return {fact.name: fact.value for fact in record.facts}


def _value(record, key):
    return _facts(record)[key]
