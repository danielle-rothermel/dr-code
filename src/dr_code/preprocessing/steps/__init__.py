"""Preprocessing step implementations.

Each step is a pure function of (settings, input) that returns a
``StepOutput``. Steps wrap existing core functions; they never
reimplement them. Steps live in individual files and are aggregated in
``dr_code.preprocessing.registry``.
"""
