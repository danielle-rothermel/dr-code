# Flexible Worker Pools

## Target Features

1. **Stage-selective execution**
   - Run only parse workers.
   - Run only test workers.
   - Run parse and test workers together.

2. **Reusable and resumable run IDs**
   - Start or continue work under an existing `run_id`.
   - Do not require parse, test, seeding, and reporting to happen in one
     process lifetime.
   - Preserve queue and result state across separate invocations.

3. **Decoupled seeding from worker startup**
   - Allow workers to start before input jobs exist.
   - Support starting test workers while the parsed/test input queue is empty.
   - Later seed parse input jobs for the same `run_id`.

4. **Blocking idle worker behavior**
   - Workers wait on empty queues instead of treating emptiness as completion
     or failure.
   - This especially matters for test workers waiting for parse outputs.

5. **Independent stage scaling**
   - Configure parse worker count independently from test worker count.
   - Support asymmetric ratios such as `parse=100,test=10`.

6. **Stage-specific lifecycle control**
   - Stop parse workers after the parse backlog is done.
   - Let test workers continue processing downstream work.
   - Add more test workers later for the same `run_id`.

7. **Incremental continuation**
   - Allow multiple commands or processes to participate in the same run over
     time.
   - Let later invocations attach to existing queues, manifests, and results
     for the same `run_id`.
   - Avoid recreating incompatible run state when continuing.

8. **Partial completion awareness**
   - Report when parse is done separately from when test is done.
   - Support waiting for one stage or for the full pipeline.

9. **Pipeline-shaped queue topology**
   - Parse workers consume seeded attempts and emit parsed jobs.
   - Test workers consume parse outputs and emit terminal test outcomes.
   - Keep parse and test semantics unchanged while making orchestration more
     flexible.

10. **Dynamic worker scaling**
    - Start additional workers for an already-running `run_id`.
    - Scale up test workers after parse has produced backlog without reseeding
      or restarting the run.

## Design Goal

The current eval path treats `parse -> test` as one orchestration unit. Flexible
worker pools should keep the same pipeline state and stage semantics, but allow
independently managed stage workers to drive that state over time.
