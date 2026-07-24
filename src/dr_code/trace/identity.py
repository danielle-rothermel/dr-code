"""stable_hash for frozen definitions."""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel


def stable_hash(model: BaseModel) -> str:
    """Content hash for frozen definitions: json.dumps(model_dump(
    mode="json"), sort_keys=True) -> BLAKE2b hex. sort_keys makes the
    hash field-order-proof; same hashing family as
    synthetic.dataset_builder._seed_for.
    """
    blob = json.dumps(
        model.model_dump(mode="json"),
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.blake2b(blob.encode()).hexdigest()
