"""Bind-time wiring + single-fold runner over a preprocessing definition.

Mirrors ``synthetic.dataset_builder.apply_recipe``: a single mechanical
fold over bound steps. Bind-time wiring failures raise ``WiringError``
before any input is processed — incompatible definitions are wiring bugs,
not data. Runtime data failures (``StepFailedError``) become ``Absent``
with the cause, and the pipeline always completes with a full trace.

Two entry points stamp provenance on the resulting trace.
``run_preprocessing`` resolves the canonical registered definition for the
caller's ``(definition_id, version)``, rejects a caller-built object that
claims a registered coordinate without matching it, and stamps
``PreprocessingTraceProducer``. ``run_external_preprocessing`` accepts an
unregistered definition and stamps ``ExternalPreprocessingTraceProducer``.
Traces assembled outside the component system carry
``ExternalTraceProducer``.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

from dr_code.trace import (
    INPUT_KEY,
    OUTPUT_KEY,
    Absent,
    Artifact,
    ArtifactKind,
    CodeArtifact,
    CodeCandidateSetArtifact,
    ComponentCoordinate,
    ExternalPreprocessingTraceProducer,
    InspectedCodeCandidateSetArtifact,
    JsonArtifact,
    JsonFactValue,
    PreprocessingDefinitionCoordinate,
    PreprocessingTraceProducer,
    StepCoordinate,
    TextArtifact,
    Trace,
    WiringError,
    coordinate_settings,
    is_absent,
)
from dr_code.preprocessing.definition import (
    PreprocessingDefinition,
)
from dr_code.preprocessing.definitions import resolve_preprocessing_definition
from dr_code.preprocessing.registry import REGISTRY
from dr_code.preprocessing.steps.base import (
    Step,
    StepFailedError,
)

#: ArtifactKind -> the concrete artifact model a TraceValue may be.
_KIND_TYPES = {
    ArtifactKind.TEXT: TextArtifact,
    ArtifactKind.CODE: CodeArtifact,
    ArtifactKind.CODE_CANDIDATE_SET: CodeCandidateSetArtifact,
    ArtifactKind.INSPECTED_CODE_CANDIDATE_SET: (
        InspectedCodeCandidateSetArtifact
    ),
    ArtifactKind.JSON: JsonArtifact,
}


@dataclass(frozen=True, slots=True)
class BoundStep:
    """A resolved step instance bound to validated settings."""

    instance_name: str
    step: Step
    coordinate: StepCoordinate


def bind_definition(
    definition: PreprocessingDefinition,
) -> tuple[BoundStep, ...]:
    """Resolve each ``StepSpec``, validate settings, and chain kinds.

    Any mismatch (unknown step, bad settings, incompatible INPUT/OUTPUT
    kind chain) raises ``WiringError`` before any input is processed.
    """
    bound: list[BoundStep] = []
    expected_input: ArtifactKind | None = None

    for spec in definition.steps:
        instance_name = spec.instance_name

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

        bound.append(
            BoundStep(
                instance_name=instance_name,
                step=step,
                coordinate=StepCoordinate(
                    instance_name=instance_name,
                    component=ComponentCoordinate(
                        registered_name=step_cls.NAME.value,
                        version=step_cls.VERSION,
                        settings=coordinate_settings(settings),
                    ),
                ),
            )
        )

    return tuple(bound)


def run_preprocessing(
    definition: PreprocessingDefinition,
    input_value: Artifact,
) -> Trace:
    """Run one exact registered definition as a mechanical fold."""
    registered = resolve_preprocessing_definition(
        definition_id=definition.definition_id,
        version=definition.version,
    )
    if definition != registered:
        raise ValueError(
            "preprocessing definition does not match its registered "
            f"coordinate: {definition.definition_id}@{definition.version}"
        )
    return _run_definition(registered, input_value, registered=True)


def run_external_preprocessing(
    definition: PreprocessingDefinition,
    input_value: Artifact,
) -> Trace:
    """Run an explicitly unregistered definition with external provenance."""
    return _run_definition(definition, input_value, registered=False)


def _run_definition(
    definition: PreprocessingDefinition,
    input_value: Artifact,
    *,
    registered: bool,
) -> Trace:
    """Execute a validated definition as a single mechanical fold.

      value = input_value
      for bound in bind_definition(definition):
          value or Absent -> run step / skip-and-propagate
          record value under bound.instance_name; merge facts

    ``StepFailedError`` -> ``Absent`` (failed_step=instance_name, plus the
    step's failure code and cause); downstream steps record the same
    ``Absent`` with ``propagated_through`` extended. Always completes: the
    trace has ``input``, one value per instance name, and ``output``.
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

    values: dict[str, Artifact | Absent] = {INPUT_KEY: input_value}
    step_facts: dict[str, dict[str, JsonFactValue]] = {}

    current: Artifact | Absent = input_value
    for bound in bound_steps:
        if is_absent(current):
            current = Absent(
                failed_step=current.failed_step,
                failure_code=current.failure_code,
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
                    failure_code=exc.code,
                    cause=exc.cause,
                )
            else:
                current = output.value
                if output.facts:
                    step_facts[bound.instance_name] = dict(output.facts)
        values[bound.instance_name] = current

    values[OUTPUT_KEY] = current

    coordinate = PreprocessingDefinitionCoordinate(
        definition_id=definition.definition_id,
        version=definition.version,
        steps=tuple(bound.coordinate for bound in bound_steps),
    )
    producer = (
        PreprocessingTraceProducer(definition=coordinate)
        if registered
        else ExternalPreprocessingTraceProducer(definition=coordinate)
    )
    return Trace(values=values, producer=producer, step_facts=step_facts)


__all__ = [
    "BoundStep",
    "bind_definition",
    "run_external_preprocessing",
    "run_preprocessing",
]
