"""Curated façade for the trace boundary contract package."""

from dr_code.trace.absent import Absent, is_absent
from dr_code.trace.archive import (
    ArchivedAbsentV2,
    ArchivedSerializedTraceV2,
    ArchivedTraceProducerV2,
    LoadedSerializedTrace,
    load_serialized_trace,
)
from dr_code.trace.artifacts import (
    Artifact,
    ArtifactKind,
    CandidateInspection,
    CandidateLineage,
    CandidateOrigin,
    CodeArtifact,
    CodeCandidateSetArtifact,
    ExtractionOperation,
    IdentifiedCandidate,
    IdentifiedCandidateSetArtifact,
    JsonArtifact,
    TextArtifact,
    parsed_module,
)
from dr_code.trace.identity import stable_hash
from dr_code.trace.observation import SampleIdentity, sample_identity_hash
from dr_code.trace.provenance import (
    EXTERNAL_PRODUCER_ID,
    ExternalSource,
    TraceProducer,
)
from dr_code.trace.serialization import (
    TRACE_SCHEMA_VERSION,
    SerializedTrace,
    deserialize_trace,
    serialize_trace,
)
from dr_code.trace.trace import (
    INPUT_KEY,
    OUTPUT_KEY,
    RESERVED_KEYS,
    Trace,
    TraceValue,
    WiringError,
    external_trace,
)

__all__ = (
    "Absent",
    "ArchivedAbsentV2",
    "ArchivedSerializedTraceV2",
    "ArchivedTraceProducerV2",
    "Artifact",
    "ArtifactKind",
    "CandidateInspection",
    "CandidateLineage",
    "CandidateOrigin",
    "CodeArtifact",
    "CodeCandidateSetArtifact",
    "EXTERNAL_PRODUCER_ID",
    "ExternalSource",
    "ExtractionOperation",
    "INPUT_KEY",
    "IdentifiedCandidate",
    "IdentifiedCandidateSetArtifact",
    "JsonArtifact",
    "LoadedSerializedTrace",
    "OUTPUT_KEY",
    "RESERVED_KEYS",
    "SerializedTrace",
    "SampleIdentity",
    "TRACE_SCHEMA_VERSION",
    "TextArtifact",
    "Trace",
    "TraceProducer",
    "TraceValue",
    "WiringError",
    "deserialize_trace",
    "external_trace",
    "is_absent",
    "load_serialized_trace",
    "parsed_module",
    "serialize_trace",
    "sample_identity_hash",
    "stable_hash",
)
