# PR #86 — Exhaustive candidate preprocessing

<https://github.com/danielle-rothermel/dr-code/pull/86>

- [ ] **Largest open decision**: does the top-level-function filter stay in
  preprocessing (current), or move into HumanEval's acceptance policy so
  lambda/class solutions score as test failures rather than extraction
  failures? The divergence changes measured pass rates.
- [ ] Fate of `SubmissionOutcome.NO_TOP_LEVEL_FUNCTIONS` — preempted at
  extraction for pipeline-extracted candidates.
- [ ] Confirm the review-round judgment call: the JSON-string and
  escaped-recovery readings were routed through additive fenced+unfenced
  segments (their docstrings claim "re-read as segments"); revert
  per-reading if strict pre-existing semantics were intended.
- [ ] `representation.py` uses a process-global `lru_cache(512)` keyed on
  source text for module-shape classification (authorized by the recorded
  plan decision; `CandidateInspection` carries no filter-specific fields).
