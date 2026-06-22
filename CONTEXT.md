# dr-code

Research context for evaluating whether compressed natural-language descriptions can be decoded into working HumanEval Python. The language here names the evaluation pipeline and its artifacts, independent of the Python modules that implement them.

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
Evaluation of newly requested decoder attempts produced for this harness.
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

**Evaluation run**:
A bounded execution of the pipeline over a selected set of decoder attempts, identified by a run id and producing exported artifacts.
_Avoid_: Batch, job, experiment

**Evaluation run lifecycle**:
The progression of an Evaluation run from declaration, through seeded Decoder attempts and Pipeline stage execution, to exported artifacts for analysis.
_Avoid_: Driver, orchestration flow

**Pipeline stage**:
A named phase in the evaluation flow: generation dataset, parsing, testing, or analysis.
_Avoid_: Step, phase

**Analysis slice**:
A grouping used to compare outcomes across dimensions such as source, model, task, or compression range.
_Avoid_: Segment, cohort, bucket
