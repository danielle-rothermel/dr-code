# PR #90 — HumanEval evaluation contracts

<https://github.com/danielle-rothermel/dr-code/pull/90>

- [x] In-container stdout hardening under `python -I` with the bounded
  reader. **Confirmed green**: CI run 30995648448 executed the full
  sandbox suite (`DR_CODE_RUN_SANDBOX_TESTS=1`, 1081 passed, 5 xfailed) —
  all 14 locally-skipped probes ran, including output-limit and
  timeout/container-kill.
- [ ] `EvaluationTaskResult`'s serialization-mode JSON Schema still
  advertises the five excluded keys as required. Nothing consumes
  serialization-mode schema today; latent drift for future codegen.
- [ ] `HumanEvalTask` has the sibling round-trip defect: its two computed
  fields fail `model_validate(model_dump())` — pre-existing, worked around
  by `code_test._validate_task_payload`. Decide whether task computed
  fields should serialize at all.
- [ ] Authenticated result channel (making single-task forgery impossible
  rather than documented) deliberately not built — new scope, likely
  dr-exec-adjacent (see `docs/dr-exec-adoption.md`).
