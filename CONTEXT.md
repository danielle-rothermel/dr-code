# dr-code

Research context for evaluating whether compressed natural-language
descriptions can be decoded into working HumanEval Python. The language here
names the evaluation concepts and their artifacts, independent of the Python
modules that implement them. Since the composable migration (PR #9) the repo
is a dependency-clean nucleus: versioned parsing/scoring profiles, offline
batch CLIs, and a localhost explain facade — no queue- or Mongo-backed
orchestration.

## Language

**Compression-correctness question**:
The research question asking how much a function description can be compressed while still allowing a decoder to reconstruct correct code.
_Avoid_: Benchmark, coding challenge

**Function description**:
Natural-language input that describes the target HumanEval function to a decoder.
_Avoid_: Prompt, spec, problem statement

**Decoder**:
The model or process that turns a function description into candidate Python code.
_Avoid_: Generator, solver, assistant

**Decoder attempt**:
One decoder output for one HumanEval task, including the function description, raw output, task identity, and provenance.
_Avoid_: Completion, sample, row

**Attempt provenance**:
Metadata describing where a decoder attempt came from, such as pool replay or fresh generation, plus model and experiment identifiers.
_Avoid_: Metadata, source info

**Pool replay**:
Evaluation of historical decoder attempts from the dr-llm HumanEval pool.
_Avoid_: Backfill, import, historical run

**Fresh generation**:
Evaluation of newly produced decoder attempts. Generation itself happens outside this repo (the nucleus has no provider dependency); dr-code consumes the resulting attempts.
_Avoid_: Live run, new samples

**HumanEval+ task**:
The canonical programming task being evaluated, identified by task id and entry point and paired with HumanEval+ tests.
_Avoid_: Problem, exercise

**Parse outcome**:
The result of extracting candidate Python code from a decoder attempt's raw output.
_Avoid_: Extraction result, validation result

**Test outcome**:
The result of running HumanEval+ tests against extracted code, including skipped and infrastructure-failure states.
_Avoid_: Test result, verdict

**Parser profile**:
A versioned identity key for extraction behavior (`humaneval-best-effort@v1`, `humaneval-field-marker@v1`). Recorded outcomes reference these IDs, so behavior changes require a new version, never an edit to an existing one.
_Avoid_: Parser config, extraction settings

**Scoring profile**:
The versioned bundle (`humaneval@v1`) of parser profile, subprocess timeout, and metrics profile that turns one decoder attempt into a scored result. Resolution is closed: unknown profile ids hard-fail.
_Avoid_: Eval config, run settings

**Golden fixture**:
The whetstone Stage 0 parser/scoring fixture that pins the ported humaneval code to its pre-migration behavior. Doctrine: fix the port, never the fixture.
_Avoid_: Snapshot test, expected output

**Corpus baseline**:
Pinned per-recipe aggregates of the v1 best-effort parser over the 4,100-sample synthetic corruption corpus; the breadth check on parser identity.
_Avoid_: Regression suite, benchmark run

**Extraction explanation**:
A stage-by-stage replay of a parser profile's candidate walk — every candidate, why it was rejected, and the winner rationale — served by the explain facade for the parser playground.
_Avoid_: Debug output, trace

**Evaluation run**:
A bounded batch of decoder attempts taken through the offline CLIs (import, parse, test, analyze), identified by its exported artifacts (attempts Parquet, parse/test JSONL, analysis outputs).
_Avoid_: Batch, job, experiment

**Analysis slice**:
A grouping used to compare outcomes across dimensions such as source, model, task, or compression range.
_Avoid_: Segment, cohort, bucket
