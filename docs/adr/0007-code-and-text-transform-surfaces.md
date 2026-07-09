# Code transforms and text transforms are distinct library modules

dr-code's library surface includes two transform modules split by input
contract. `dr_code.code_transforms` operates on Python source *as code*:
every function assumes parseable input and raises `SyntaxError` otherwise —
rejecting unparseable input is a feature, not a failure mode.
`dr_code.text_transforms` operates on text that only *probably* contains
code (raw LLM submissions, markdown, prose-wrapped snippets): every
function is total, never raises, and passes unrepairable input through
best-effort. The boundary is the caller's domain assumption, not the
implementation technique — a regex-based docstring stripper would still be
a code transform. Both modules hold transforms strictly;
predicates, enumeration, and inspection live in the analysis siblings
`dr_code.code_analysis` and `dr_code.text_analysis` (ADR 0008).

Placement tiebreaker: when a job on valid code already belongs to a
formatter (trailing whitespace, line endings, indentation width), the
hand-rolled string version exists only for repair-before-parse and lives on
the text side.

Each behavior has exactly one implementation. The parser's extraction
pipeline and the synthetic corruption generator call through these modules
rather than carrying private copies, and the parser's trace layer wraps the
pure functions to record before/after text — the transforms themselves
never learn about traces. Corruptions (`dr_code.synthetic.corruptions`) are
the inverse direction and stay in the synthetic package; where a corruption
and a transform share mechanics (fence wrapping, the smart-quote table),
the shared piece lives in the transform module.
