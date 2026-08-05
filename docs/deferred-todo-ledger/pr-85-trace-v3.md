# PR #85 — Trace schema v3

<https://github.com/danielle-rothermel/dr-code/pull/85>

- [ ] Degenerate values accepted by design-silence:
  `ExtractionOperation(operation_name="")` and `Absent(failure_code="")`
  validate. Decide whether to add `min_length` validators.
  (`CandidateOrigin(input_location=-5)` is rejected via `Field(ge=0)`, and
  `CandidateInspection` rejects impossible parse/compile combinations —
  both added during review.)
- [ ] `Trace.values`/`step_facts` are plain dicts after snapshotting — a
  holder can mutate the trace's own containers in place. The docstring
  claims only caller-mapping independence (true). Decide whether
  `MappingProxyType` hardening is wanted.
- [x] Keep-or-revert `21574d0` (deep-copy of `JsonArtifact.payload` at
  trace construction, making the snapshot claim uniform at the cost of one
  deepcopy per JSON artifact per trace). **Kept, by decision at merge.**
