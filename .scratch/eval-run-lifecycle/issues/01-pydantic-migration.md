# Pydantic migration

Status: ready-for-agent

## Parent

.scratch/eval-run-lifecycle/PRD.md

## What to build

Convert dr-code result dataclasses to Pydantic models so the upcoming Evaluation run lifecycle interface uses the same model style as the rest of the domain. Result objects should be frozen; builder code should accumulate in local variables and return final models.

## Acceptance criteria

- [ ] No dr-code source file imports `dataclass` or uses `@dataclass`.
- [ ] Current dataclass result shapes remain available with equivalent field names and values.
- [ ] Mutable builder flows are rewritten to return frozen Pydantic models without changing user-visible behavior.
- [ ] Existing checks pass.

## Blocked by

None - can start immediately
