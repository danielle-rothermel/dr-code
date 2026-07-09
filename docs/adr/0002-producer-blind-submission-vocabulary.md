# dr-code is producer-blind and speaks "submission"

dr-code's domain is task + submission + profile → outcomes, metrics, and
explanations. Nothing in its language or contracts may reference how a
submission was produced: no producer frameworks, no experiment framing, no
graph vocabulary. Those concepts belong to the producing repos (dr-graph
owns graph shapes; consumer repos own experiments and orchestration), which
vary producers freely — so any producer assumption baked into the evaluator
would be wrong tomorrow and would mislead agents reading it.

Concretely: the input to scoring is a **submission** — plain text
(`str`), unwrapped by the producer before the call. Callers passing
anything else get a type error at the boundary, never a scored failure.
"Candidate" is not a synonym for submission: it names the parser-internal
blocks the extraction ladder walks and ranks, and the explain facade's
vocabulary is built on it. Task-side names keep "HumanEval": the extension
seam for future task families is profile-ID namespacing (`humaneval@v1` →
`mbpp@v1`), not a task-generic core.
