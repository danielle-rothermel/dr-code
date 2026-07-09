# Transform-library exports are wide, with enumeration split from policy

`dr_code.code_transforms`, `dr_code.text_transforms`,
`dr_code.code_analysis`, and `dr_code.text_analysis` are a general-purpose
library surface, consumed
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

Transforms and analysis are separate modules on both sides of the ADR 0007
code/text boundary, giving a four-module grid: transform modules return
modified code or text; analysis modules return facts about it (predicates,
enumeration, inspection). `dr_code.code_analysis` covers parseable Python
(`equivalent`, `validate_python`, `module_level_names`, the enumeration
functions); `dr_code.text_analysis` covers text that probably contains code
(line classifiers such as `is_code_like_line`, `fence_marker`, block
segmentation). Each analysis module inherits its side's input contract —
code-side raises `SyntaxError` unless documented total, as `equivalent` is;
text-side is total and never raises. On each side, the transform module may
import its analysis sibling, never the reverse.
