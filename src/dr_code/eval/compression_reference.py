"""Generic compression references: typed key -> immutable byte artifact.

The generic layer knows nothing about dataset fields (e.g. Whetstone's
``task.gt_code_wo_comments``). A :class:`CompressionReferenceKey` is a
stable typed key; a :class:`CompressionReferenceArtifact` is the immutable
bytes it resolves to. The denominator behavior when the reference is
empty is **explicit**: :func:`compression_ratio` returns the caller's
declared zero-denominator sentinel rather than silently dividing by zero.
"""

from __future__ import annotations

import base64
from typing import Self

from pydantic import field_serializer, field_validator

from dr_code.eval.identity import (
    SCHEMA_COMPRESSION_REFERENCE_ARTIFACT,
    SCHEMA_COMPRESSION_REFERENCE_KEY,
    identity_hash_for,
)
from dr_code.models import FrozenModel


class CompressionReferenceKey(FrozenModel):
    """Stable typed key naming one Compression Reference Artifact.

    The key is a namespaced opaque identifier. The owning Definition
    decides whether the key (and the resolved artifact identity) are
    identity-bearing; this generic type carries no dataset knowledge.
    """

    namespace: str
    name: str

    @field_validator("namespace", "name")
    @classmethod
    def _reject_empty(cls, value: str) -> str:
        if value == "":
            raise ValueError("compression reference key parts must be non-empty")
        return value

    def identity_payload(self) -> dict[str, str]:
        return {"namespace": self.namespace, "name": self.name}

    def identity_hash(self) -> str:
        return identity_hash_for(
            schema=SCHEMA_COMPRESSION_REFERENCE_KEY,
            payload=self.identity_payload(),
        )


class CompressionReferenceArtifact(FrozenModel):
    """Immutable bytes used as a compression denominator.

    Byte content is the whole artifact; its identity is the hash of those
    bytes. Independent of any dataset field or experiment selection rule.
    """

    content: bytes

    @field_serializer("content")
    def _serialize_content(self, content: bytes) -> str:
        return base64.b64encode(content).decode("ascii")

    @field_validator("content", mode="before")
    @classmethod
    def _accept_b64(cls, value: object) -> object:
        if isinstance(value, str):
            return base64.b64decode(value.encode("ascii"))
        return value

    @property
    def byte_length(self) -> int:
        return len(self.content)

    def identity_payload(self) -> dict[str, str | int]:
        return {
            "byte_length": self.byte_length,
            "content_b64": base64.b64encode(self.content).decode("ascii"),
        }

    def identity_hash(self) -> str:
        return identity_hash_for(
            schema=SCHEMA_COMPRESSION_REFERENCE_ARTIFACT,
            payload=self.identity_payload(),
        )


class ReferenceResolutionError(KeyError):
    """A key had no bound artifact in the resolver."""


class CompressionReferenceResolver(FrozenModel):
    """Resolves keys to immutable artifacts; unknown keys raise explicitly.

    Bindings are ``(namespace, name) -> artifact``. The resolver does not
    invent artifacts or fall back to a default: an unbound key is an
    explicit :class:`ReferenceResolutionError`, distinct from a resolved
    but empty artifact (zero denominator).
    """

    bindings: tuple[tuple[CompressionReferenceKey, CompressionReferenceArtifact], ...] = ()

    @classmethod
    def from_mapping(
        cls,
        mapping: dict[CompressionReferenceKey, CompressionReferenceArtifact],
    ) -> Self:
        return cls(bindings=tuple(mapping.items()))

    def resolve(
        self,
        key: CompressionReferenceKey,
    ) -> CompressionReferenceArtifact:
        for bound_key, artifact in self.bindings:
            if bound_key == key:
                return artifact
        raise ReferenceResolutionError(
            f"no compression reference bound for {key.namespace}/{key.name}"
        )


# Sentinel for an explicit zero-denominator outcome. Callers choose to
# surface it as not-applicable / invalid; the generic layer never coerces
# a zero denominator to 0.0 or 1.0.
ZERO_DENOMINATOR: None = None


def compression_ratio(
    *,
    numerator_bytes: int,
    reference: CompressionReferenceArtifact,
) -> float | None:
    """Return numerator / reference length, or ``None`` when the
    denominator is zero.

    The zero-denominator case is explicit and returned as ``None`` (the
    :data:`ZERO_DENOMINATOR` sentinel); it is never silently coerced.
    """

    denominator = reference.byte_length
    if denominator == 0:
        return ZERO_DENOMINATOR
    return numerator_bytes / denominator


__all__ = [
    "ZERO_DENOMINATOR",
    "CompressionReferenceArtifact",
    "CompressionReferenceKey",
    "CompressionReferenceResolver",
    "ReferenceResolutionError",
    "compression_ratio",
]
