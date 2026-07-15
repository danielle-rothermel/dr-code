"""The `dr_code` top level is a namespace, not a public surface.

Nothing is re-exported here on purpose: import from the submodules
(`dr_code.humaneval`, `dr_code.synthetic`) and the top-level analysis
modules (`dr_code.code_analysis`, `dr_code.text_analysis`, ...). Each
subpackage curates its own public API via its `__init__` and `__all__`;
consumers depend on those, not on names hoisted to this package root.
"""
