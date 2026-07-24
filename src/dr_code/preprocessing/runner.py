"""Bind-time wiring + single-fold runner over a preprocessing definition.

Mirrors ``synthetic.dataset_builder.apply_recipe``: a single mechanical
fold over bound steps. Bind-time wiring failures raise ``WiringError``
before any input is processed — incompatible definitions are wiring bugs,
not data. Runtime data failures (``StepFailedError``) become ``Absent``
with the cause, and the pipeline always completes with a full trace.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

from dr_code.trace import (
    Absent,
    Artifact,
    ArtifactKind,
    CodeArtifact,
    CodeCandidateSetArtifact,
    JsonArtifact,
    TextArtifact,
    Trace,
    TraceProducer,
    WiringError,
    is_absent,
)
from dr_code.preprocessing.definition import (
    PreprocessingDefinition,
    preprocessing_definition_hash,
)
from dr_code.preprocessing.registry import REGISTRY
from dr_code.preprocessing.steps.base import Step, StepFailedError

#: ArtifactKind -> the concrete artifact model a TraceValue may be.
_KIND_TYPES = {
    ArtifactKind.TEXT: TextArtifact,
    ArtifactKind.CODE: CodeArtifact,
    ArtifactKind.CODE_CANDIDATE_SET: CodeCandidateSetArtifact,
    ArtifactKind.JSON: JsonArtifact,
}


@dataclass(frozen=True, slots=True)
class BoundStep:
    """A resolved step instance bound to validated settings."""

    instance_name: str
    step: Step


def bind_definition(
    definition: PreprocessingDefinition,
) -> tuple[BoundStep, ...]:
    """Resolve each ``StepSpec``, validate settings, and chain kinds.

    Any mismatch (unknown step, bad settings, incompatible INPUT/OUTPUT
    kind chain) raises ``WiringError`` before any input is processed.
    """
    bound: list[BoundStep] = []
    seen_names: set[str] = set()
    expected_input: ArtifactKind | None = None

    for spec in definition.steps:
        instance_name = spec.instance_name

        if instance_name in seen_names:
            raise WiringError(f"duplicate instance name: {instance_name!r}")
        seen_names.add(instance_name)

        step_cls = REGISTRY.get(spec.step.value)
        if step_cls is None:
            raise WiringError(f"unknown step: {spec.step.value!r}")

        try:
            settings = step_cls.Settings.model_validate(spec.settings)
        except ValidationError as exc:
            raise WiringError(
                f"invalid settings for step {spec.step.value!r}: {exc}"
            ) from exc

        step = step_cls(settings)

        if expected_input is not None and step.INPUT != expected_input:
            raise WiringError(
                f"broken INPUT/OUTPUT kind chain: step "
                f"{instance_name!r} expects {step.INPUT.value!r}, "
                f"previous step outputs {expected_input.value!r}"
            )
        expected_input = step.OUTPUT

        bound.append(BoundStep(instance_name=instance_name, step=step))

    return tuple(bound)


def run_preprocessing(
    definition: PreprocessingDefinition,
    input_value: Artifact,
) -> Trace:
    """Run a definition as a single mechanical fold over its steps.

      value = input_value
      for bound in bind_definition(definition):
          value or Absent -> run step / skip-and-propagate
          record value under bound.instance_name; merge facts

    ``StepFailedError`` -> ``Absent`` (failed_step=instance_name, cause);
    downstream steps record the same ``Absent`` with
    ``propagated_through`` extended. Always completes: the trace has
    ``input``, one value per instance name, and ``output``.
    """
    bound_steps = bind_definition(definition)

    if bound_steps:
        first_input_kind = bound_steps[0].step.INPUT
        expected_type = _KIND_TYPES[first_input_kind]
        if not isinstance(input_value, expected_type):
            raise WiringError(
                f"input artifact kind {type(input_value).__name__!r} "
                f"does not match first step input "
                f"{first_input_kind.value!r}"
            )

    values: dict[str, Artifact | Absent] = {"input": input_value}
    step_facts: dict[str, dict[str, str]] = {}

    current: Artifact | Absent = input_value
    for bound in bound_steps:
        if is_absent(current):
            current = Absent(
                failed_step=current.failed_step,
                cause=current.cause,
                propagated_through=(
                    *current.propagated_through,
                    bound.instance_name,
                ),
            )
        else:
            try:
                output = bound.step.apply(current)
            except StepFailedError as exc:
                current = Absent(
                    failed_step=bound.instance_name,
                    cause=exc.cause,
                )
            else:
                current = output.value
                if output.facts:
                    step_facts[bound.instance_name] = dict(output.facts)
        values[bound.instance_name] = current

    values["output"] = current

    producer = TraceProducer(
        producer_id=definition.definition_id,
        version=definition.version,
        definition_hash=preprocessing_definition_hash(definition),
    )
    return Trace(values=values, producer=producer, step_facts=step_facts)


__all__ = ["BoundStep", "bind_definition", "run_preprocessing"]
