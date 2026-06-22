# Flexible Worker Pools

## Target Features

1. **Stage-selective execution**
   - Run only parse workers.
   - Run only test workers.
   - Run parse and test workers together.
   - Status: partially supported.
   - Blocker location: mostly `dr-code`. `dr-queues` can run one named stage
     from an existing manifest, but `dr-code` does not expose first-class
     parse-only, test-only, or both-stage eval commands.

2. **Reusable and resumable run IDs**
   - Start or continue work under an existing `run_id`.
   - Do not require parse, test, seeding, and reporting to happen in one
     process lifetime.
   - Preserve queue and result state across separate invocations.
   - Status: partially supported.
   - Blocker location: shared. `dr-queues` persists manifests and durable
     queues, but its completion waiting is not resume-aware. `dr-code`
     currently recreates run setup, seeds, waits, exports, and reports as one
     monolithic flow.

3. **Decoupled seeding from worker startup**
   - Allow workers to start before input jobs exist.
   - Support starting test workers while the parsed/test input queue is empty.
   - Later seed parse input jobs for the same `run_id`.
   - Status: partially supported.
   - Blocker location: mostly `dr-code`. `dr-queues` has separate setup,
     stage-worker, and seed primitives, and workers can idle on empty queues.
     `dr-code` does not yet have separate commands for initialize, start
     workers, and seed attempts.

4. **Blocking idle worker behavior**
   - Workers wait on empty queues instead of treating emptiness as completion
     or failure.
   - This especially matters for test workers waiting for parse outputs.
   - Status: supported.
   - Blocker location: none known. `dr-queues` workers poll RabbitMQ until
     stopped, so empty queues are idle state rather than terminal state.

5. **Independent stage scaling**
   - Configure parse worker count independently from test worker count.
   - Support asymmetric ratios such as `parse=100,test=10`.
   - Status: supported for initial worker counts.
   - Blocker location: none for initial launch. `dr-code` already accepts
     worker specs such as `parse=8,test=2`, and `dr-queues` stores per-stage
     defaults in the run manifest.

6. **Stage-specific lifecycle control**
   - Stop parse workers after the parse backlog is done.
   - Let test workers continue processing downstream work.
   - Add more test workers later for the same `run_id`.
   - Status: partially supported.
   - Blocker location: shared. `dr-queues` has stage worker replacement and PID
     files, and `dr-code` has test-worker tuning plus parse-stop logic. The
     missing piece is a general lifecycle interface for starting, stopping,
     replacing, and observing each stage intentionally.

7. **Incremental continuation**
   - Allow multiple commands or processes to participate in the same run over
     time.
   - Let later invocations attach to existing queues, manifests, and results
     for the same `run_id`.
   - Avoid recreating incompatible run state when continuing.
   - Status: partially supported.
   - Blocker location: mostly `dr-code`. The lower-level manifest and queue
     state can be reused, but `dr-code` lacks a continuation-oriented command
     model that treats init, seed, worker management, waiting, export, and
     reporting as separate operations.

8. **Partial completion awareness**
   - Report when parse is done separately from when test is done.
   - Support waiting for one stage or for the full pipeline.
   - Status: partially supported.
   - Blocker location: shared. `dr-code` can count Mongo stage-output events
     for parse and terminal events for test, but there is no first-class
     status/wait command. `dr-queues` also lacks generic stage-completion and
     queue-depth helpers.

9. **Pipeline-shaped queue topology**
   - Parse workers consume seeded attempts and emit parsed jobs.
   - Test workers consume parse outputs and emit terminal test outcomes.
   - Keep parse and test semantics unchanged while making orchestration more
     flexible.
   - Status: supported.
   - Blocker location: none known. `dr-queues` already chains each stage output
     queue into the next stage input queue, and `dr-code` handlers already keep
     parse and test as separate pipeline stages.

10. **Dynamic worker scaling**
    - Start additional workers for an already-running `run_id`.
    - Scale up test workers after parse has produced backlog without reseeding
      or restarting the run.
    - Status: partially supported.
    - Blocker location: shared. `dr-code` can hot-swap test workers through the
      tuning script. `dr-queues` supports replacement by stage, but PID tracking
      is one process per stage and does not model additive multi-process scale
      out cleanly.

## Design Goal

The current eval path treats `parse -> test` as one orchestration unit. Flexible
worker pools should keep the same pipeline state and stage semantics, but allow
independently managed stage workers to drive that state over time.

## Useful `dr-queues` Changes

These changes would make the queue runtime better at supporting flexible worker
pools for any pipeline, not just `dr-code`.

1. **Resume-aware completion waiting**
   - Add a wait primitive that can initialize progress from persisted event
     history instead of only counting terminal messages observed after the wait
     process starts.
   - Support waiting for either a specific stage or the final terminal count.

2. **Stage status and queue introspection**
   - Expose queue depths for each stage input and output queue.
   - Report per-stage started, completed, terminal, and in-flight counts.
   - Make "stage done" computable from expected jobs, stage output events, and
     queue state.

3. **Stage lifecycle management**
   - Add generic commands/helpers to start, stop, replace, and list workers by
     `run_id` and stage.
   - Distinguish replace-style scaling from additive scale-out.
   - Track multiple worker processes per stage instead of a single PID file.

4. **Attach-to-existing-run helpers**
   - Provide a safe way to load an existing manifest, validate it matches the
     requested pipeline definition, and reuse its queues without rewriting
     incompatible state.
   - Make seeding an explicit operation that can run independently after queue
     setup.

5. **Reusable operational CLI**
   - Extend the stage-worker CLI with status, wait, stop, and scale commands
     that can be reused by downstream projects.
   - Keep handler modules project-specific, but make worker orchestration
     project-neutral.

## `dr-code` Changes on Top

These changes would use the `dr-queues` primitives to make flexible worker pools
the normal eval workflow.

1. **Split the eval driver into separate operations**
   - `init`: build the eval pipeline manifest and declare/reuse queues for a
     `run_id`.
   - `seed`: load pool attempts and publish parse-stage jobs for that `run_id`.
   - `workers`: start parse workers, test workers, or both for that `run_id`.
   - `wait`: wait for parse completion, test completion, or full pipeline
     completion.
   - `export`: write attempts, parse JSONL, test JSONL, manifest, and reports
     from current persisted state.

2. **Add stage-selective CLI flags**
   - Accept a stage selector such as `--stages parse`, `--stages test`, or
     `--stages parse,test`.
   - Keep worker counts stage-specific with the existing `parse=N,test=M`
     shape.

3. **Make continuation explicit**
   - Add a `--run-id` continuation path that loads an existing manifest instead
     of recreating it by default.
   - Validate expected job count, task indices, dump source, and pipeline ID
     before attaching to an existing run.
   - Refuse to reseed duplicates unless the user explicitly opts in.

4. **Expose eval-specific status**
   - Report seeded attempts, parse completions, terminal test outcomes,
     missing parse/test counts, worker PIDs, and queue depths.
   - Keep Mongo event counts as the source of truth for stage progress.
   - Surface parse-only completion separately from full eval completion.

5. **Generalize the existing test tuning path**
   - Keep `tune_test_workers.py` behavior, but build it on top of the general
     worker lifecycle commands.
   - Allow scaling parse or test workers intentionally, not only hot-swapping
     test workers during the tuning workflow.

6. **Support partial exports and final reports**
   - Export parse outcomes from parse-stage output events before terminal test
     events exist.
   - Export test outcomes from terminal events as they become available.
   - Generate a partial status report for in-flight runs and a proof report once
     terminal count reaches expected jobs.

7. **Document operational recipes**
   - Start test workers before seeding.
   - Seed parse input later under the same `run_id`.
   - Run many parse workers and fewer test workers.
   - Stop parse workers when parse is done.
   - Scale up test workers while preserving the same run state.
