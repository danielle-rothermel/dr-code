# Transform-library exports are wide, with enumeration split from policy

`dr_code.code_transforms`, `dr_code.text_transforms`, and
`dr_code.code_analysis` are a general-purpose library surface, consumed
directly from notebooks and sibling repos to do common code and
text-with-code work without reimplementation. Export policy errs wide: any
helper that implements a coherent operation is a public export with contract
tests, not a private implementation detail. Implementation techniques (e.g.
`ast.NodeTransformer` subclasses) stay private behind exported functions —
wide surface means many deliberate contracts, not exposed internals.

Transforms decompose into enumeration and policy. The enumeration half —
finding every annotation site, discovering a function's local bindings — is
exported on its own so consumers can identify, collect, inspect, and
selectively modify; each opinionated transform (`strip_type_annotations`,
`alpha_rename_locals`) is a thin composition of enumeration plus one policy.
Site values carry what inspection needs without re-walking the tree: the
node, its unparsed source string, and its source location.

`dr_code.code_analysis` holds predicates and inspection over parseable
Python (`equivalent`, `validate_python`, `module_level_names`, the
enumeration functions); `code_transforms` holds transforms strictly and may
import from `code_analysis`, never the reverse. Analysis functions share the
parseable-input contract — raise `SyntaxError` on unparseable input — unless
documented total, as `equivalent` is.
