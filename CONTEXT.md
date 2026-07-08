# dr-code

Producer-blind evaluation context: given a piece of candidate Python text
and a HumanEval+ task, extract, execute, and score it under versioned
profiles — and explain how. The language here names evaluation concepts
only. How submissions are produced (models, tools, computation graphs,
experiment designs) is deliberately outside this context; that vocabulary
belongs to the producing repos (dr-graph, whetstone-ai).

## Language

**Submission**:
The terminal output text handed to the evaluator for one HumanEval+ task.
dr-code never knows or records how it was produced.
_Avoid_: Generation, decoder attempt, completion, sample

**HumanEval+ task**:
The canonical programming task being evaluated, identified by task id and
entry point and paired with HumanEval+ tests.
_Avoid_: Problem, exercise

**Task family**:
A benchmark's namespace of profiles and machinery. `humaneval` is the only
family today; a new family arrives as new profile namespaces (e.g.
`mbpp@v1`), never by generalizing existing ones.
_Avoid_: Benchmark type, dataset kind

**Candidate**:
A code block the parser considers during extraction — walked, ranked,
rejected, or selected. Parser-internal; never a synonym for submission.
_Avoid_: Submission, snippet

**Parser profile**:
A versioned identity key for extraction behavior (`humaneval-best-effort@v1`,
`humaneval-field-marker@v1`). An ID freezes once experiment outcomes are
recorded against it; until then, behavior under a working ID may change.
_Avoid_: Parser config, extraction settings

**Scoring profile**:
The versioned bundle (`humaneval@v1`) of parser profile, subprocess timeout,
and metrics profile that turns one submission into a scored result.
Resolution is closed: unknown profile ids hard-fail.
_Avoid_: Eval config, run settings

**Submission outcome**:
The overall classification of one scored submission (extraction failure,
tested pass/fail, error, timeout). Only exists when the harness completed;
it never describes harness trouble.
_Avoid_: Verdict, result, parse/test outcome

**Harness failure**:
A failure of the evaluation machinery itself (runner breakage, protocol
errors) while scoring a submission. A separate channel from submission
outcomes — the two are structurally impossible to confuse, because a
harness mistake recorded as a model result silently poisons experiments.
_Avoid_: Infra error outcome, internal error

**Evaluation case**:
One HumanEval+ test case executed against a submission's extracted code,
with its own status (pass, fail, error, timeout).
_Avoid_: Test result, assertion

**Extraction trace**:
The parser's own record of its candidate walk, emitted as data during
extraction: a candidate lineage tree whose transform steps carry
before/after text and whose checks carry verdicts and reasons, topped by
the winner rationale. The explain facade renders this record; it never
re-derives it.
_Avoid_: Explanation, debug output, replay

**Corruption recipe**:
A named, composable sequence of text corruptions (fences, prose wrappers,
broken indentation, noise) applied to a known-good solution. Recipes are
plain data; a recipe plus a seed deterministically reproduces a sample.
_Avoid_: Mutation, augmentation

**Corruption corpus**:
A generated dataset of recipe-corrupted solutions used to exercise and
hand-tune the parser. Any number can be generated, at any size, from
recipes and seeds; no particular corpus is canonical.
_Avoid_: Regression suite, benchmark run, golden data
