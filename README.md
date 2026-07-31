# dr-code

## Python execution

Untrusted Python runs through the **dr-exec** package: a hermetic
`interpreter -I -c <source>` child with caller-declared budgets (wall clock,
output, input), an explicit environment grant, a per-run scratch working
directory, and race-safe process-group teardown. Outcomes are data — every
spawned run returns a `RunResult` carrying a raw returncode, captured streams,
truncation marks, and exactly one attribution — so consumers branch on the
attribution rather than on exception types.

HumanEval runs one batch per candidate function through dr-exec's batch driver
kit: each case is delivered as an incremental NDJSON result the moment it is
produced, so a wall-clock deadline, an output overflow, or a late child death
costs only the unfinished tail — completed cases already survive. `dr-code`
declares its execution budgets and the `OPENBLAS_NUM_THREADS=1` grant at its
call sites, maps dr-exec attributions onto HumanEval case verdicts (for
example a candidate-process crash to a per-case error), and owns the case
schemas, scoring, and caching.

The `process_boundary_only` containment profile provides no operating-system
containment. Candidate code runs as the invoking user with full filesystem,
credential, process, and network reach; a payload writing directly to a file
descriptor can still reach the protocol channel. Run evaluations only on
disposable workers whose permissions, network access, resources, and lifetime
are constrained externally.
