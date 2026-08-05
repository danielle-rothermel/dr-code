"""Metrics engine internals.

The engine is split across ``engine`` (extraction orchestration),
``execution`` (sandbox request batching and caching), and ``views``
(derived-artifact caching). Consumers import from those modules
directly.
"""
