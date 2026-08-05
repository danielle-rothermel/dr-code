# dr-exec adoption

This branch is the home of dr-exec adoption in dr-code: swapping subprocess
execution onto the pinned `dr-exec` package and retiring the OCI sandbox.
This document is the standing requirements list for that work.

## Requirements dr-exec must satisfy before adoption (F64-1..5)

Open executor-lifecycle and typed-boundary requirements. Adoption is not
complete until each is either satisfied by dr-exec or explicitly re-scoped,
and the migrated callers are validated — deferral is not resolution.

- **F64-1 — post-reap process-group signaling**: signaling a process group
  after the leader is reaped must be correct (no signals to recycled pids).
- **F64-2 — post-Popen cleanup and fallback reaping**: descendants must be
  cleaned up when `Popen` itself fails partway, with a fallback reaping
  path.
- **F64-3 — platform support**: process-tree cleanup is POSIX-only today;
  supported platforms must be declared and enforced.
- **F64-4 — IPC-worker failure handling**: worker-channel failures must be
  attributed as infrastructure, never as candidate outcomes.
- **F64-5 — typed input/timeout validation**: inputs and timeouts validated
  at a typed boundary, not by convention.

Also to re-evaluate during the caller migration, not assumed solved by it:
deterministic interpreter/hash policy; failure-detail namespace reuse and
deadline accounting in the HumanEval callers.

## Conformance contracts

dr-exec ships a shared conformance suite run against every implementation,
pinning: bounded input and output, wall-clock deadlines, process-tree
cleanup, protocol integrity, and typed failure attribution. Structural
conformance alone does not establish semantic conformance.

## Adoption work in dr-code

1. **Engine runner seam**: the metrics engine's runner injection
   (`SandboxRunner`, `run_python_in_sandbox`) becomes the dr-exec runner
   type, and the engine's fail-closed infrastructure exception tuple
   becomes a dr-exec exception type. This makes
   `dr_code.metrics.engine.engine` a direct dr-exec importer — a deliberate
   boundary crossing, decided here and nowhere else.
2. **Kill-returncode convention changes exactly once, here**:
   `CANDIDATE_KILL_RETURNCODES` is `{137, 139}` (container exit codes).
   Under native execution it becomes negative `Popen.returncode` signal
   values. This changes the meaning of the persisted
   `ExecutionOutcome.returncode` and invalidates the execution cache; tests
   assert against the exported contract, never shell-normalized
   `128 + signal` values.
3. **OCI sandbox retirement**: delete `humaneval/sandbox.py`,
   `tests/humaneval/test_sandbox.py`, `tests/humaneval/test_sandbox_unit.py`;
   drop the image pull, `DR_CODE_SANDBOX_IMAGE`, and
   `DR_CODE_RUN_SANDBOX_TESTS` from CI; rewrite the README execution
   sections with the trust posture below. The security ledger below moves
   into a dated `CHANGELOG.md` entry in the same change.
4. **Runner protocol integrity (optional, dr-exec-adjacent)**: the HumanEval
   runner script's results channel contains accidental stdout collisions
   only; an adversarial candidate can forge its own task's results through
   the runner's module globals or file descriptor 1. An authenticated
   result channel is the fix if single-task integrity against adversarial
   candidates ever becomes a requirement.

## Security ledger (moves to CHANGELOG at sandbox retirement)

The sandbox test suites pin guarantees that subprocess execution does not
replace: container image pinning by digest; credential, filesystem,
network, process, and memory isolation; container lifecycle termination;
runtime discovery, image inspection, runtime allowlisting, and runtime
environment construction. Under dr-exec, submitted programs are trusted at
the execution layer: the subprocess boundary retains the invoking worker's
permissions, external worker isolation is the deployment boundary, and
evaluations run only on disposable workers.
