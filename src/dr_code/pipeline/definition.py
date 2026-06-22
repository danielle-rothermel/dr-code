"""Pipeline definition for dr-code eval."""

from __future__ import annotations

from dr_queues import (
    HandlerRegistry,
    Pipeline,
    PipelineDefinition,
    PipelineLane,
    PipelineStep,
)

PIPELINE_ID = "dr_code_eval"
_PARSE_HANDLER = "parse_attempt"
_TEST_HANDLER = "run_tests"


def build_eval_pipeline(registry: HandlerRegistry) -> Pipeline:
    """Build the parse → test eval pipeline."""
    definition = PipelineDefinition(
        id=PIPELINE_ID,
        lanes=[PipelineLane(id="default")],
        steps=[
            PipelineStep(name="parse", handler_key=_PARSE_HANDLER),
            PipelineStep(name="test", handler_key=_TEST_HANDLER),
        ],
    )
    return Pipeline(definition, registry)
