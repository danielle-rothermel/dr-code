"""Curated façade for the trace boundary contract package."""

from dr_code.trace.absent import Absent, is_absent
from dr_code.trace.artifacts import (
    Artifact,
    ArtifactKind,
    CodeArtifact,
    CodeCandidateSetArtifact,
    JsonArtifact,
    TextArtifact,
    parsed_module,
)
from dr_code.trace.identity import stable_hash
from dr_code.trace.provenance import (
    EXTERNAL_PRODUCER,
    EXTERNAL_PRODUCER_ID,
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
    "Artifact",
    "ArtifactKind",
    "CodeArtifact",
    "CodeCandidateSetArtifact",
    "EXTERNAL_PRODUCER",
    "EXTERNAL_PRODUCER_ID",
    "INPUT_KEY",
    "JsonArtifact",
    "OUTPUT_KEY",
    "RESERVED_KEYS",
    "SerializedTrace",
    "TRACE_SCHEMA_VERSION",
    "TextArtifact",
    "Trace",
    "TraceProducer",
    "TraceValue",
    "WiringError",
    "deserialize_trace",
    "external_trace",
    "is_absent",
    "parsed_module",
    "serialize_trace",
    "stable_hash",
)
