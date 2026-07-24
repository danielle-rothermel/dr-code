"""Typed loader for the standalone preprocessing hard-example fixture."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "hard_examples.json"
FIXTURE_SHA256 = (
    "ed96b9819724fc5b450d37495dd6ca0bbf9fdd3a1304f258631bee34764dc26a"
)
PARTITION_ALGORITHM = "sha256-prefix-mod-5-v1"
ANNOTATION_EXPORT_CHECKPOINT_SHA256 = (
    "c9ebe01e398bfe589fe67b69260553dc37647a8d9a662463cda5c169eb75a441"
)
AUTHORITATIVE_CORPUS_SHA256 = (
    "a58acf1b1ed0ad54dc91d12bcca80398f3f3850b559f8051f52af2e4d4f1c4f5"
)
POST_HOLDOUT_TRANSITIONS_SHA256 = (
    "9a6e4e88f3b1f672616b14cf9490604beeff7413a3508984e2bd10b2ada3b7b6"
)
POST_HOLDOUT_BASELINE_CANDIDATES_SHA256 = (
    "64d3effc33089e1fa36aa1db9ce0377e55cf3b324e1e8ab41105c0d99106e560"
)
POST_HOLDOUT_BASELINE_RUN_ID = "generation-corpus-functions-v1-20260719"


class AnnotationSource(BaseModel):
    """Portable viewer annotation retained without changing its verdict."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["annotation"]
    corpus_sha256: str
    sample_id: str
    decoder_output_sha256: str
    verdict: Literal["should_be_parseable", "expected_no_code"]
    note: str
    tags: tuple[str, ...]


class EstablishedFixtureSource(BaseModel):
    """Existing named regression fixture consolidated into this suite."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["established_fixture"]
    name: str


class CorpusSpotCheckSource(BaseModel):
    """Named corpus regression promoted independently of viewer annotation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["corpus_spot_check"]
    sample_id: str
    decoder_output_sha256: str


class FullCorpusRegressionSource(BaseModel):
    """Post-holdout regression discovered by authoritative corpus comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["full_corpus_regression"]
    corpus_sha256: str
    sample_id: str
    decoder_output_sha256: str
    baseline_run_id: Literal["generation-corpus-functions-v1-20260719"]


HardExampleSource = Annotated[
    AnnotationSource
    | EstablishedFixtureSource
    | CorpusSpotCheckSource
    | FullCorpusRegressionSource,
    Field(discriminator="kind"),
]


class OriginOperationExpectation(BaseModel):
    """One ordered provenance operation with exact stable details."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str
    details: dict[str, JsonValue]


class HardExample(BaseModel):
    """One unique decoder output and its adjudicated regression oracle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    decoder_output: str
    decoder_output_sha256: str
    partition: Literal["development", "holdout"]
    cohort: Literal[
        "sealed_hard_suite", "post_holdout_full_corpus_regression"
    ] = "sealed_hard_suite"
    categories: tuple[str, ...]
    adjudication: Literal[
        "annotation_verdict",
        "intrinsic_invalid",
        "contract_conflict",
        "established_contract",
        "target_contract",
        "production_regression",
    ]
    expected_outcome: Literal["candidates", "absent"]
    sources: tuple[HardExampleSource, ...]
    exact_candidates: tuple[str, ...] = ()
    expected_top_level_function_names: tuple[str, ...] = ()
    required_origin_paths: tuple[
        tuple[OriginOperationExpectation, ...], ...
    ] = ()
    required_origin_operation_kinds: tuple[str, ...] = ()
    forbidden_origin_operation_kinds: tuple[str, ...] = ()
    failure_code: str | None = None
    failed_step: str | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> HardExample:
        digest = hashlib.sha256(
            self.decoder_output.encode("utf-8")
        ).hexdigest()
        if digest != self.decoder_output_sha256:
            raise ValueError(f"{self.id}: decoder output digest mismatch")
        if (
            self.cohort == "sealed_hard_suite"
            and self.partition != partition_for_digest(digest)
        ):
            raise ValueError(f"{self.id}: partition does not match digest")
        if not self.sources:
            raise ValueError(f"{self.id}: at least one source is required")
        if tuple(sorted(set(self.categories))) != self.categories:
            raise ValueError(
                f"{self.id}: categories must be sorted and unique"
            )
        for source in self.sources:
            if isinstance(source, AnnotationSource):
                if source.decoder_output_sha256 != digest:
                    raise ValueError(
                        f"{self.id}: annotation output digest mismatch"
                    )
                if tuple(sorted(source.tags)) != source.tags:
                    raise ValueError(
                        f"{self.id}: annotation tags must be sorted"
                    )
            elif isinstance(source, CorpusSpotCheckSource):
                if source.decoder_output_sha256 != digest:
                    raise ValueError(
                        f"{self.id}: spot-check output digest mismatch"
                    )
            elif isinstance(source, FullCorpusRegressionSource):
                if source.decoder_output_sha256 != digest:
                    raise ValueError(
                        f"{self.id}: regression output digest mismatch"
                    )
        regression_sources = tuple(
            source
            for source in self.sources
            if isinstance(source, FullCorpusRegressionSource)
        )
        if self.cohort == "post_holdout_full_corpus_regression":
            if (
                self.partition != "development"
                or self.adjudication != "production_regression"
                or len(regression_sources) != 1
                or self.categories
                != (
                    "full_corpus_regression",
                    "post_holdout",
                    "return_salvage",
                )
            ):
                raise ValueError(
                    f"{self.id}: invalid post-holdout regression contract"
                )
        elif regression_sources:
            raise ValueError(
                f"{self.id}: regression source requires post-holdout cohort"
            )
        if self.expected_outcome == "absent" and self.exact_candidates:
            raise ValueError(f"{self.id}: absent cases cannot name candidates")
        if self.expected_outcome == "absent" and (
            self.expected_top_level_function_names
            or self.required_origin_paths
            or self.required_origin_operation_kinds
            or self.forbidden_origin_operation_kinds
        ):
            raise ValueError(
                f"{self.id}: absent cases cannot constrain success"
            )
        if self.expected_outcome == "candidates" and (
            self.failure_code is not None or self.failed_step is not None
        ):
            raise ValueError(f"{self.id}: success cannot name a failure")
        if self.expected_outcome == "candidates" and (
            not self.exact_candidates
            or not self.expected_top_level_function_names
        ):
            raise ValueError(
                f"{self.id}: success requires exact source and function names"
            )
        if tuple(sorted(set(self.expected_top_level_function_names))) != (
            self.expected_top_level_function_names
        ):
            raise ValueError(
                f"{self.id}: expected function names must be sorted and unique"
            )
        if tuple(sorted(set(self.forbidden_origin_operation_kinds))) != (
            self.forbidden_origin_operation_kinds
        ):
            raise ValueError(
                f"{self.id}: forbidden operations must be sorted and unique"
            )
        if tuple(sorted(set(self.required_origin_operation_kinds))) != (
            self.required_origin_operation_kinds
        ):
            raise ValueError(
                f"{self.id}: required operations must be sorted and unique"
            )
        if any(not path for path in self.required_origin_paths):
            raise ValueError(
                f"{self.id}: required origin paths cannot be empty"
            )
        return self

    @property
    def annotations(self) -> tuple[AnnotationSource, ...]:
        return tuple(
            source
            for source in self.sources
            if isinstance(source, AnnotationSource)
        )


class HardExampleFixture(BaseModel):
    """Versioned root document for deterministic loading and validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    partition_algorithm: Literal["sha256-prefix-mod-5-v1"]
    annotation_export_checkpoint_sha256: str
    authoritative_corpus_sha256: str
    annotation_records_sha256: str
    post_holdout_transitions_sha256: str
    post_holdout_baseline_candidates_sha256: str
    cases: tuple[HardExample, ...]

    @model_validator(mode="after")
    def validate_unique_cases(self) -> HardExampleFixture:
        ids = [case.id for case in self.cases]
        digests = [case.decoder_output_sha256 for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("hard-example case IDs must be unique")
        if len(digests) != len(set(digests)):
            raise ValueError("hard-example decoder outputs must be unique")
        if self.annotation_export_checkpoint_sha256 != (
            ANNOTATION_EXPORT_CHECKPOINT_SHA256
        ):
            raise ValueError("annotation export checkpoint digest mismatch")
        if self.authoritative_corpus_sha256 != AUTHORITATIVE_CORPUS_SHA256:
            raise ValueError("authoritative corpus digest mismatch")
        if self.post_holdout_transitions_sha256 != (
            POST_HOLDOUT_TRANSITIONS_SHA256
        ):
            raise ValueError(
                "post-holdout transition artifact digest mismatch"
            )
        if self.post_holdout_baseline_candidates_sha256 != (
            POST_HOLDOUT_BASELINE_CANDIDATES_SHA256
        ):
            raise ValueError(
                "post-holdout baseline candidates digest mismatch"
            )
        annotation_records = sorted(
            (
                source.model_dump(mode="json", exclude={"kind"})
                for case in self.cases
                for source in case.sources
                if isinstance(source, AnnotationSource)
            ),
            key=lambda record: (
                record["corpus_sha256"],
                record["sample_id"],
                record["decoder_output_sha256"],
            ),
        )
        canonical = json.dumps(
            annotation_records,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        if hashlib.sha256(canonical).hexdigest() != (
            self.annotation_records_sha256
        ):
            raise ValueError("embedded annotation records digest mismatch")
        if {
            source.corpus_sha256
            for case in self.cases
            for source in case.sources
            if isinstance(
                source, AnnotationSource | FullCorpusRegressionSource
            )
        } != {self.authoritative_corpus_sha256}:
            raise ValueError("corpus-backed records reference another corpus")
        regression_sources = [
            source
            for case in self.cases
            for source in case.sources
            if isinstance(source, FullCorpusRegressionSource)
        ]
        if len(regression_sources) != 18:
            raise ValueError(
                "fixture requires exactly 18 post-holdout regressions"
            )
        if len({source.sample_id for source in regression_sources}) != 18:
            raise ValueError(
                "post-holdout regression sample IDs must be unique"
            )
        return self


def partition_for_digest(digest: str) -> Literal["development", "holdout"]:
    """Assign a stable 20-percent holdout without corpus-dependent state."""
    if int(digest[:8], 16) % 5 == 0:
        return "holdout"
    return "development"


def load_hard_examples() -> HardExampleFixture:
    """Load the committed fixture without consulting DuckDB or corpus files."""
    payload = FIXTURE_PATH.read_bytes()
    if hashlib.sha256(payload).hexdigest() != FIXTURE_SHA256:
        raise ValueError("hard-example fixture digest mismatch")
    return HardExampleFixture.model_validate_json(payload)


__all__ = [
    "AnnotationSource",
    "FullCorpusRegressionSource",
    "FIXTURE_SHA256",
    "HardExample",
    "HardExampleFixture",
    "load_hard_examples",
    "partition_for_digest",
]
