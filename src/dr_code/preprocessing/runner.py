"""Bind-time wiring + single-fold runner over a preprocessing definition.

Mirrors ``synthetic.dataset_builder.apply_recipe``: a single mechanical
fold over bound steps. Bind-time wiring failures raise ``WiringError``
before any input is processed — incompatible definitions are wiring bugs,
not data. Runtime data failures (``StepFailedError``) become ``Absent``
with the cause, and the pipeline always completes with a full trace.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import JsonValue

from dr_code.eval.lifecycle import PreprocessingConfig
from dr_code.eval.tasks import SampleIdentity
from dr_code.preprocessing.decoder_output import normalize_decoder_output
from dr_code.preprocessing.failures import PreprocessingFailureCode
from dr_code.trace import (
    Absent,
    Artifact,
    ArtifactKind,
    CodeArtifact,
    CodeCandidateSetArtifact,
    IdentifiedCandidateSetArtifact,
    JsonArtifact,
    TextArtifact,
    Trace,
    TraceProducer,
    WiringError,
    is_absent,
)
from dr_code.preprocessing.registry import REGISTRY
from dr_code.preprocessing.steps.base import Step, StepFailedError

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


@dataclass(frozen=True, slots=True)
class BoundPreprocessingRunner:
    """A preprocessing definition resolved once and reusable for many inputs."""

    bound_steps: tuple[BoundStep, ...]
    input_kind: ArtifactKind | None
    producer: TraceProducer

    def run(
        self,
        input_value: Artifact,
        *,
        sample_identity: SampleIdentity | None = None,
    ) -> Trace:
        """Run the bound steps over one input and return its full trace."""
        if self.input_kind is not None:
            expected_type = _KIND_TYPES[self.input_kind]
            if not isinstance(input_value, expected_type):
                raise WiringError(
                    f"input artifact kind {type(input_value).__name__!r} "
                    f"does not match first step input "
                    f"{self.input_kind.value!r}"
                )

        validation_facts: dict[str, JsonValue] | None = None
        if isinstance(input_value, TextArtifact):
            normalized = normalize_decoder_output(input_value.text)
            input_value = TextArtifact(text=normalized.text)
            if not normalized.is_valid:
                validation_facts = {
                    "text_character_count": len(input_value.text),
                    **normalized.facts,
                }

        values: dict[str, Artifact | Absent] = {"input": input_value}
        step_facts: dict[str, dict[str, JsonValue]] = {}

        if validation_facts is None:
            current: Artifact | Absent = input_value
        else:
            current = Absent(
                failed_step="validate_decoder_output",
                cause="decoder output contains unsupported characters",
                failure_code=PreprocessingFailureCode.DECODER_OUTPUT_INVALID.value,
            )
            step_facts["validate_decoder_output"] = validation_facts
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
            values=values,
            producer=self.producer,
            step_facts=step_facts,
            sample_identity=sample_identity,
        )


def bind_definition(
    config: PreprocessingConfig,
) -> tuple[BoundStep, ...]:
    """Bind one self-authenticated concrete preprocessing config.

    Definitions are materialized before this boundary, so every variable has
    already been substituted and every setting is fully validated.
    """
    config = PreprocessingConfig.model_validate(
        config.model_dump(mode="python")
    )
    bound: list[BoundStep] = []
    seen_names: set[str] = set()
    expected_input: ArtifactKind | None = None

    for spec in config.steps:
        instance_name = spec.instance_name

        if instance_name in seen_names:
            raise WiringError(f"duplicate instance name: {instance_name!r}")
        seen_names.add(instance_name)

        step_cls = REGISTRY.get(spec.step)
        if step_cls is None:
            raise WiringError(f"unknown step: {spec.step!r}")

        settings = step_cls.Settings.model_validate_json(
            json.dumps(dict(spec.settings), allow_nan=False),
            strict=True,
        )
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


def bind_preprocessing(
    config: PreprocessingConfig,
) -> BoundPreprocessingRunner:
    """Bind one concrete config once for repeated preprocessing calls."""
    config = PreprocessingConfig.model_validate(
        config.model_dump(mode="python")
    )
    bound_steps = bind_definition(config)
    input_kind = bound_steps[0].step.INPUT if bound_steps else None
    producer = TraceProducer(
        producer_id=config.definition_ref.definition_id,
        version=config.definition_ref.version,
        definition_hash=config.definition_ref.identity_hash,
        preprocessing_config_hash=config.config_identity_hash,
        implementation_hash=config.implementation_hash,
    )
    return BoundPreprocessingRunner(
        bound_steps=bound_steps,
        input_kind=input_kind,
        producer=producer,
    )


def run_preprocessing(
    config: PreprocessingConfig,
    input_value: Artifact,
    *,
    sample_identity: SampleIdentity | None = None,
) -> Trace:
    """Run one concrete preprocessing config once."""
    return bind_preprocessing(config).run(
        input_value,
        sample_identity=sample_identity,
    )


__all__ = [
    "BoundPreprocessingRunner",
    "BoundStep",
    "bind_definition",
    "bind_preprocessing",
    "run_preprocessing",
]
