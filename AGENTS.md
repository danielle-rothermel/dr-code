# Agent Instructions

## Verification

- **Cursor agents:** Do not run `scripts/pre-check.sh` unless the user
  explicitly asks you to run checks or verification.
- **Other agents:** Run `scripts/pre-check.sh` after making code changes.
- Skip `scripts/pre-check.sh` after documentation-only or configuration-only
  changes (non-Cursor agents).
- When running checks, treat every issue reported by `scripts/pre-check.sh` as
  in scope to fix, even when the issue is outside the files you touched.
- Read failed check output from `.cache/pre-check/`.

## Tests

- Do not add tests unless the user explicitly asks for tests.

## Commits

- After completing a requested sequence of changes, commit the result unless
  the user specifically asks not to commit at the end.
- Use a single-line commit message.
- If your own changes are easy to isolate, commit only your own changes.
- If your own changes are not cleanly extractable, commit all changes in the
  files you touched.
- Before committing code changes, run `scripts/pre-check.sh` (non-Cursor agents,
  or when the user explicitly asked you to run checks).
- Before committing documentation-only or configuration-only changes, do not run
  `scripts/pre-check.sh`.

## Multi-Player Workflow

- Assume other agents and users may be operating in this repo at the same time.
- Never delete, revert, reset, or otherwise undo unexpected changes in the repo.
- Assume unexpected changes were made intentionally by the user or another
  process.
- If unexpected changes appear suspect or make the requested work difficult,
  ask how to proceed and describe the concern.
- Do not undo suspect changes while waiting for guidance.
