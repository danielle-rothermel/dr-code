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
    TraceProducer,
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
class _BoundStep:
    instance_name: str
    step: Step
    coordinate: StepCoordinate


def _bind_steps(
    definition: PreprocessingDefinition,
) -> tuple[_BoundStep, ...]:
    """Validate settings and artifact-kind wiring once at bind time."""

    bound: list[_BoundStep] = []
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
            _BoundStep(
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


@dataclass(frozen=True, slots=True)
class BoundPreprocessingRunner:
    """Bound wiring reusable across inputs."""

    definition: PreprocessingDefinition
    steps: tuple[_BoundStep, ...]
    producer: TraceProducer

    def run(self, input_value: Artifact) -> Trace:
        """Convert StepFailedError to Absent; propagate unexpected defects."""

        if self.steps:
            first_input_kind = self.steps[0].step.INPUT
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
        for bound in self.steps:
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
                        failure_code=exc.code.value,
                        cause=exc.cause,
                    )
                    if exc.evidence:
                        step_facts[bound.instance_name] = dict(exc.evidence)
                else:
                    current = output.value
                    if output.facts:
                        step_facts[bound.instance_name] = dict(output.facts)
            values[bound.instance_name] = current

        values[OUTPUT_KEY] = current

        return Trace(
            values=values, producer=self.producer, step_facts=step_facts
        )


def _bind(
    definition: PreprocessingDefinition,
    *,
    registered: bool,
) -> BoundPreprocessingRunner:
    steps = _bind_steps(definition)
    coordinate = PreprocessingDefinitionCoordinate(
        definition_id=definition.definition_id,
        version=definition.version,
        steps=tuple(bound.coordinate for bound in steps),
    )
    producer = (
        PreprocessingTraceProducer(definition=coordinate)
        if registered
        else ExternalPreprocessingTraceProducer(definition=coordinate)
    )
    return BoundPreprocessingRunner(
        definition=definition, steps=steps, producer=producer
    )


def bind_preprocessing(
    definition: PreprocessingDefinition,
) -> BoundPreprocessingRunner:
    registered = resolve_preprocessing_definition(
        definition_id=definition.definition_id,
        version=definition.version,
    )
    if definition != registered:
        raise ValueError(
            "preprocessing definition does not match its registered "
            f"coordinate: {definition.definition_id}@{definition.version}"
        )
    return _bind(registered, registered=True)


def bind_external_preprocessing(
    definition: PreprocessingDefinition,
) -> BoundPreprocessingRunner:
    return _bind(definition, registered=False)


def run_preprocessing(
    definition: PreprocessingDefinition,
    input_value: Artifact,
) -> Trace:
    return bind_preprocessing(definition).run(input_value)


def run_external_preprocessing(
    definition: PreprocessingDefinition,
    input_value: Artifact,
) -> Trace:
    return bind_external_preprocessing(definition).run(input_value)


__all__ = [
    "BoundPreprocessingRunner",
    "bind_external_preprocessing",
    "bind_preprocessing",
    "run_external_preprocessing",
    "run_preprocessing",
]
