"""TraceProducer; EXTERNAL producer."""

from __future__ import annotations

from typing import Final

from dr_code.models import FrozenModel

EXTERNAL_PRODUCER_ID: Final = "external"


class TraceProducer(FrozenModel):
    """Identifies the producing definition, or `external` (eval-flow L2)."""

    # preprocessing definition_id, or "external"
    producer_id: str
    version: str | None = None
    # stable_hash of the full frozen definition
    definition_hash: str | None = None


EXTERNAL_PRODUCER: Final = TraceProducer(producer_id=EXTERNAL_PRODUCER_ID)
