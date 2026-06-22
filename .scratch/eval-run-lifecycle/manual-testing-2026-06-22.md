# Manual Evaluation Run Lifecycle Testing - 2026-06-22

## Environment
- Branch/status: `## context`
- HEAD: `48b6312`
- Notes: Manual testing for the new `scripts/eval_run.py` lifecycle. Generated run artifacts live under `.scratch/eval-run-lifecycle/manual-smoke/`.

## Test 1: Prepare tiny attempts input
- What tested: Generated a tiny `AttemptRecord` parquet input from the existing pool fixture.
- Why tested: Provides a deterministic two-record workload for exercising real parse/test lifecycle behavior without depending on the large local pool dump.
- Commands:
  - `uv run python - <<'PY' ... load_pool_parquet(... )[:2]; write_attempts(... ) ... PY`
- Expected: `.scratch/eval-run-lifecycle/manual-smoke/attempts.parquet` exists with 2 records.
- What happened: Wrote 2 records: `f6b1a5385226a9b7` and `0beb04181ae53759`, both for `HumanEval/0` / `has_close_elements`.
- Actions/fixes: None.
- Retest result: Not needed.
- Status: pass

## Test 2: CLI help surface
- What tested: Top-level CLI help and command-specific help for `preflight`, `init`, `seed`, `start`, `stop`, `wait`, `status`, `export`, and `run`.
- Why tested: Catches import-time failures, Typer signature errors, and missing command registration before creating lifecycle state.
- Commands:
  - `uv run scripts/eval_run.py --help`
  - `uv run scripts/eval_run.py <command> --help` for each lifecycle subcommand.
- Expected: All help commands exit 0 and show the expected options.
- What happened: All help commands exited 0. Top-level command list included all expected lifecycle commands.
- Actions/fixes: None.
- Retest result: Not needed.
- Status: pass

## Test 3: Infrastructure preflight
- What tested: RabbitMQ/Mongo reachability and default pool dump artifact presence.
- Why tested: The lifecycle depends on RabbitMQ, Mongo-backed `dr-queues` state, and documented dump-backed seed inputs.
- Commands:
  - `uv run scripts/eval_run.py preflight --no-require-dump`
  - `uv run scripts/eval_run.py preflight`
- Expected: Both commands exit 0; the first validates services, the second also validates dump files for indices `[0, 1, 2, 3, 4]`.
- What happened: Both commands exited 0. RabbitMQ was reachable at `localhost:5672`, MongoDB at `localhost:27017`, and dump artifacts were present for `[0, 1, 2, 3, 4]`.
- Actions/fixes: None.
- Retest result: Not needed.
- Status: pass

## Test 4: One-shot in-process run
- What tested: `eval_run.py run` in `in-process` mode using the two-record attempts parquet, then status, export files, line counts, and stage 4 analysis.
- Why tested: Validates the simplest end-to-end lifecycle path: seeding, in-process parse/test execution, terminal counting, export writing, proof summary, and analysis join compatibility.
- Commands:
  - `uv run scripts/eval_run.py run --mode in-process --attempts .scratch/eval-run-lifecycle/manual-smoke/attempts.parquet --workers parse=1,test=1 --run-id manual-inproc-20260622 --output-root .scratch/eval-run-lifecycle/manual-smoke/exports --completion-timeout 600 --overwrite`
  - `uv run scripts/eval_run.py status --run-id manual-inproc-20260622 --json`
  - `find .scratch/eval-run-lifecycle/manual-smoke/exports/manual-inproc-20260622 -maxdepth 2 -type f -print`
  - `wc -l .../parse.jsonl .../test.jsonl`
  - `uv run scripts/analyze_eval_run.py --attempts .../attempts.parquet --parse .../parse.jsonl --test .../test.jsonl --output-dir .../analysis`
- Expected: Run exits 0, status reports `terminal_jobs == expected_jobs == 2`, all core export files exist, parse/test JSONL each have 2 rows, and analysis reports `missing_test: 0`.
- What happened: Run exited 0 with `terminals=2/2`, `outcome_kind_counts: tested=2`, and exports under `.scratch/eval-run-lifecycle/manual-smoke/exports/manual-inproc-20260622`. Status JSON showed `terminal_jobs: 2` and `expected_jobs: 2`. `parse.jsonl` and `test.jsonl` each had 2 lines. Analysis completed with `missing_test: 0`.
- Actions/fixes: None.
- Retest result: Not needed.
- Status: pass

## Test 5: Detached lifecycle commands
- What tested: Separate `init`, `seed`, `status`, `start`, `wait`, `export`, `stop`, final status, export files, line counts, and analysis for a detached run.
- Why tested: Validates the decomposed Mongo-backed lifecycle commands independently of the one-shot `run` wrapper.
- Commands:
  - `uv run scripts/eval_run.py init --run-id manual-lifecycle-20260622 --workers parse=1,test=1 --overwrite`
  - `uv run scripts/eval_run.py seed --run-id manual-lifecycle-20260622 --attempts .scratch/eval-run-lifecycle/manual-smoke/attempts.parquet`
  - `uv run scripts/eval_run.py status --run-id manual-lifecycle-20260622`
  - `uv run scripts/eval_run.py start --run-id manual-lifecycle-20260622 --workers parse=1,test=1`
  - `uv run scripts/eval_run.py wait --run-id manual-lifecycle-20260622 --target terminal --timeout 600 --poll-interval 0.5`
  - `uv run scripts/eval_run.py status --run-id manual-lifecycle-20260622 --json`
  - `uv run scripts/eval_run.py export --run-id manual-lifecycle-20260622 --output-root .scratch/eval-run-lifecycle/manual-smoke/exports`
  - `uv run scripts/eval_run.py stop --run-id manual-lifecycle-20260622`
  - `uv run scripts/eval_run.py status --run-id manual-lifecycle-20260622 --json`
  - Export file, line count, and `analyze_eval_run.py` checks.
- Expected: Seeded status starts at `terminals=0/2`, wait reaches `terminals=2/2`, exported parse/test JSONL each have 2 rows, analysis reports `missing_test: 0`, and workers become stopped after `stop`.
- What happened: Init and seed exited 0. Initial status showed parse `ready=2`. Start launched pids `29629,29663`. Wait reached `terminals=2/2 complete=True`. Export wrote `.scratch/eval-run-lifecycle/manual-smoke/exports/manual-lifecycle-20260622`. After `stop` and a short wait, both detached workers were recorded with `status: stopped`. Export files existed, parse/test JSONL each had 2 rows, and analysis reported `missing_test: 0`.
- Actions/fixes: None.
- Retest result: Not needed.
- Status: pass

## Test 6: One-shot detached run
- What tested: `eval_run.py run` in `detached` mode using the two-record attempts parquet, then status, worker cleanup, export files, line counts, and analysis.
- Why tested: Validates the README-facing detached one-shot path and the `finally` cleanup behavior in the wrapper.
- Commands:
  - `uv run scripts/eval_run.py run --mode detached --attempts .scratch/eval-run-lifecycle/manual-smoke/attempts.parquet --workers parse=1,test=1 --run-id manual-detached-20260622 --output-root .scratch/eval-run-lifecycle/manual-smoke/exports --completion-timeout 600 --overwrite`
  - `uv run scripts/eval_run.py status --run-id manual-detached-20260622 --json`
  - Export file, line count, and `analyze_eval_run.py` checks.
- Expected: Run exits 0 with `terminals=2/2`; status shows terminal count 2 and both detached workers stopped; exports and analysis match the in-process smoke expectations.
- What happened: Run exited 0 with `terminals=2/2`, `outcome_kind_counts: tested=2`, and exports under `.scratch/eval-run-lifecycle/manual-smoke/exports/manual-detached-20260622`. Status JSON showed `terminal_jobs: 2`, `expected_jobs: 2`, and both detached workers with `status: stopped`. Export files existed, parse/test JSONL each had 2 rows, and analysis reported `missing_test: 0`.
- Actions/fixes: None.
- Retest result: Not needed.
- Status: pass

## Test 7: Dump-backed seed path
- What tested: `eval_run.py run` in `in-process` mode using the default dump-backed seed path with `--task-indices 0 --limit-per-task 1`.
- Why tested: Verifies the documented pool replay path, including dump preflight and `load_proof_attempts`, rather than only the attempts-file import path.
- Commands:
  - `uv run scripts/eval_run.py run --mode in-process --task-indices 0 --limit-per-task 1 --workers parse=1,test=1 --run-id manual-dump-20260622 --output-root .scratch/eval-run-lifecycle/manual-smoke/exports --completion-timeout 600 --overwrite`
  - `uv run scripts/eval_run.py status --run-id manual-dump-20260622 --json`
  - Export file, line count, and `analyze_eval_run.py` checks.
- Expected: Run exits 0 with `terminals=1/1`; core export files exist; parse/test JSONL each have 1 row; analysis reports `missing_test: 0`.
- What happened: Run exited 0 with `terminals=1/1`, `outcome_kind_counts: tested=1`, and exports under `.scratch/eval-run-lifecycle/manual-smoke/exports/manual-dump-20260622`. Status JSON showed `terminal_jobs: 1`, `expected_jobs: 1`, and stopped in-process workers. Export files existed, parse/test JSONL each had 1 row, and analysis reported `missing_test: 0`.
- Actions/fixes: None.
- Retest result: Not needed.
- Status: pass

## Final State
- Manual testing complete: Yes. CLI help, service preflight, dump preflight, one-shot in-process, decomposed detached lifecycle, one-shot detached, export reconstruction, worker cleanup, and analysis joins all passed.
- Remaining issues: None found during this staged smoke pass.
- Code changes made: None. No implementation fixes were needed.
- Verification: Manual lifecycle checks passed with `manual-inproc-20260622` at `2/2`, `manual-lifecycle-20260622` at `2/2`, `manual-detached-20260622` at `2/2`, and `manual-dump-20260622` at `1/1`. Analysis reported `missing_test: 0` for each exported run.
