"""The `dr_code` top level is a namespace, not a public surface.

Nothing is re-exported here on purpose: import from the functional packages
(`dr_code.evaluation`, `dr_code.humaneval`, `dr_code.metrics`,
`dr_code.preprocessing`, `dr_code.synthetic`, `dr_code.trace`) or the shared
`dr_code.core` foundation. Each functional package curates its own public API
via its `__init__` and `__all__`; consumers depend on those, not on names
hoisted to this package root.
"""
