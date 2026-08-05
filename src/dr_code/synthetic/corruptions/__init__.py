"""Corruptions.

Each corruption is a pure function of (source, rng) that returns a
`CorruptedSample`.

Corruptions live in individual files (one per file) and are aggregated in
`dr_code.synthetic.corruptions.registry`.
"""

from dr_code.synthetic.corruptions.base import Corruption
from dr_code.synthetic.corruptions.registry import REGISTRY

__all__ = [
    "REGISTRY",
    "Corruption",
]
