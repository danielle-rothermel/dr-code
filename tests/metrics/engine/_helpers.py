from __future__ import annotations

import asyncio


def _definition(questions) -> object:
    from dr_code.metrics import MetricsDefinition

    return MetricsDefinition(
        definition_id="def", version="1", questions=tuple(questions)
    )


def _q(metric_name: str, on: str = "input", **settings) -> object:
    from dr_code.metrics import MetricName, MetricQuestion

    return MetricQuestion(
        metric=MetricName(metric_name), on=on, settings=settings
    )


def _facts(record):
    assert record.status.value == "measured", record
    return {fact.name: fact.value for fact in record.facts}


def _extract(definition, trace, **kwargs):
    from dr_code.metrics import extract_metrics

    return asyncio.run(extract_metrics(definition, trace, **kwargs))


def _extract_batch(definition, traces, **kwargs):
    from dr_code.metrics import extract_metrics_batch

    return asyncio.run(extract_metrics_batch(definition, traces, **kwargs))
