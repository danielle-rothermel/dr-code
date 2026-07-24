"""Bind-time wiring + single-fold runner over a preprocessing definition.

Mirrors ``synthetic.dataset_builder.apply_recipe``: a single mechanical
fold over bound steps. Bind-time wiring failures raise ``WiringError``
before any input is processed — incompatible definitions are wiring bugs,
not data. Runtime data failures (``StepFailedError``) become ``Absent``
with the cause, and the pipeline always completes with a full trace.

Binding a definition stamps the provenance every trace it produces
carries. ``bind_preprocessing`` resolves the canonical registered
definition for the caller's ``(definition_id, version)``, rejects a
caller-built object that claims a registered coordinate without matching
it, and stamps ``PreprocessingTraceProducer``.
``bind_external_preprocessing`` accepts an unregistered definition and
stamps ``ExternalPreprocessingTraceProducer``. Traces assembled outside the
component system carry ``ExternalTraceProducer``.

``run_preprocessing`` and ``run_external_preprocessing`` are the one-shot
form of the same two paths; callers preprocessing many inputs under one
definition bind once and reuse the returned runner.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import JsonValue, ValidationError

from dr_code.preprocessing.decoder_output import normalize_decoder_output
from dr_code.preprocessing.definition import PreprocessingDefinition
from dr_code.preprocessing.definitions import resolve_preprocessing_definition
from dr_code.preprocessing.failures import PreprocessingFailureCode
from dr_code.preprocessing.registry import REGISTRY
from dr_code.preprocessing.steps.base import Step, StepFailedError
from dr_code.trace import (
    Absent,
    Artifact,
    ArtifactKind,
    CodeArtifact,
    CodeCandidateSetArtifact,
    ComponentCoordinate,
    ExternalPreprocessingTraceProducer,
    IdentifiedCandidateSetArtifact,
    JsonArtifact,
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

#: Instance name recording decoder-output validation, ahead of every step.
DECODER_VALIDATION_STEP = "validate_decoder_output"

#: ArtifactKind -> the concrete artifact model a TraceValue may be.
_KIND_TYPES = {
    ArtifactKind.TEXT: TextArtifact,
    ArtifactKind.CODE: CodeArtifact,
    ArtifactKind.CODE_CANDIDATE_SET: CodeCandidateSetArtifact,
    ArtifactKind.IDENTIFIED_CANDIDATE_SET: IdentifiedCandidateSetArtifact,
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


@dataclass(frozen=True, slots=True)
class BoundPreprocessingRunner:
    """One definition bound once and reusable across many inputs.

    Binding resolves every step and its producer coordinate, so callers that
    preprocess many inputs under the same definition pay that cost once. The
    fold and the stamped producer are identical to the one-shot entry points
    — this is reuse of the same implementation, not a second one.
    """

    bound_steps: tuple[BoundStep, ...]
    producer: TraceProducer

    def run(self, input_value: Artifact) -> Trace:
        """Run the bound steps over one input and return its full trace.

          value = input_value
          for bound in self.bound_steps:
              value or Absent -> run step / skip-and-propagate
              record value under bound.instance_name; merge facts

        ``StepFailedError`` -> ``Absent`` (failed_step=instance_name, cause,
        failure_code); downstream steps record the same ``Absent`` with
        ``propagated_through`` extended. Always completes: the trace has
        ``input``, one value per instance name, and ``output``.

        Decoder text is normalized before the fold so NUL and lone surrogate
        code points are visible and JSON-safe. Text that carried them never
        reaches a step: the fold starts from an ``Absent`` recorded under
        ``validate_decoder_output``.
        """
        if self.bound_steps:
            first_input_kind = self.bound_steps[0].step.INPUT
            expected_type = _KIND_TYPES[first_input_kind]
            if not isinstance(input_value, expected_type):
                raise WiringError(
                    f"input artifact kind {type(input_value).__name__!r} "
                    f"does not match first step input "
                    f"{first_input_kind.value!r}"
                )

        step_facts: dict[str, dict[str, JsonValue]] = {}
        current: Artifact | Absent = input_value

        if isinstance(input_value, TextArtifact):
            normalized = normalize_decoder_output(input_value.text)
            input_value = TextArtifact(text=normalized.text)
            current = input_value
            if not normalized.is_valid:
                step_facts[DECODER_VALIDATION_STEP] = {
                    "text_character_count": len(input_value.text),
                    **normalized.facts,
                }
                current = Absent(
                    failed_step=DECODER_VALIDATION_STEP,
                    cause="decoder output contains unsupported characters",
                    failure_code=(
                        PreprocessingFailureCode.DECODER_OUTPUT_INVALID.value
                    ),
                )

        values: dict[str, Artifact | Absent] = {"input": input_value}

        for bound in self.bound_steps:
            if is_absent(current):
                current = Absent(
                    failed_step=current.failed_step,
                    cause=current.cause,
                    failure_code=current.failure_code,
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
                        failure_code=exc.failure_code.value,
                    )
                    if exc.facts:
                        step_facts[bound.instance_name] = dict(exc.facts)
                else:
                    current = output.value
                    if output.facts:
                        step_facts[bound.instance_name] = dict(output.facts)
            values[bound.instance_name] = current

        values["output"] = current
        return Trace(
            values=values, producer=self.producer, step_facts=step_facts
        )


def bind_preprocessing(
    definition: PreprocessingDefinition,
) -> BoundPreprocessingRunner:
    """Bind one exact registered definition once for repeated runs."""
    return _bind(_registered_definition(definition), registered=True)


def bind_external_preprocessing(
    definition: PreprocessingDefinition,
) -> BoundPreprocessingRunner:
    """Bind an explicitly unregistered definition with external provenance."""
    return _bind(definition, registered=False)


def run_preprocessing(
    definition: PreprocessingDefinition,
    input_value: Artifact,
) -> Trace:
    """Run one exact registered definition as a mechanical fold."""
    return bind_preprocessing(definition).run(input_value)


def run_external_preprocessing(
    definition: PreprocessingDefinition,
    input_value: Artifact,
) -> Trace:
    """Run an explicitly unregistered definition with external provenance."""
    return bind_external_preprocessing(definition).run(input_value)


def _registered_definition(
    definition: PreprocessingDefinition,
) -> PreprocessingDefinition:
    """Resolve the canonical definition, rejecting an impersonator.

    A caller-built object that claims a registered coordinate without
    matching it never runs: the registry, not the caller, decides what a
    registered coordinate means.
    """
    registered = resolve_preprocessing_definition(
        definition_id=definition.definition_id,
        version=definition.version,
    )
    if definition != registered:
        raise ValueError(
            "preprocessing definition does not match its registered "
            f"coordinate: {definition.definition_id}@{definition.version}"
        )
    return registered


def _bind(
    definition: PreprocessingDefinition,
    *,
    registered: bool,
) -> BoundPreprocessingRunner:
    bound_steps = bind_definition(definition)
    coordinate = PreprocessingDefinitionCoordinate(
        definition_id=definition.definition_id,
        version=definition.version,
        steps=tuple(bound.coordinate for bound in bound_steps),
    )
    producer: TraceProducer = (
        PreprocessingTraceProducer(definition=coordinate)
        if registered
        else ExternalPreprocessingTraceProducer(definition=coordinate)
    )
    return BoundPreprocessingRunner(bound_steps=bound_steps, producer=producer)


__all__ = [
    "DECODER_VALIDATION_STEP",
    "BoundPreprocessingRunner",
    "BoundStep",
    "bind_definition",
    "bind_external_preprocessing",
    "bind_preprocessing",
    "run_external_preprocessing",
    "run_preprocessing",
]
