# PR #88 — Evaluation package

<https://github.com/danielle-rothermel/dr-code/pull/88>

- [ ] Aggregation result models accept incoherent counts on
  deserialization (`AggregationOk(counted=-3, ...)` validates);
  `aggregate` itself never produces them. Bounds validators = new scope.
- [ ] Empty-string ids validate uniformly across coordinates
  (`plan_id=""`, `dataset_id=""`, `Score.name=""`); tightening is a
  package-wide decision.
- [ ] The `not math.isfinite` branch in `_reduce` is defense-in-depth —
  only the overflow paths are reachable through the public API.
- [ ] An overflowing int under `PROPORTION`/`COUNT` counts as an ordinary
  truthy value (an unrepresentably large value is nonzero); revisit if
  proportion-over-overflowing-ints becomes a real case.
- [ ] Importing `dr_code.evaluation` eagerly loads `preprocessing.registry`,
  `metrics.registry`, the metrics engine, and `humaneval.sandbox` via the
  facade imports in `plan.py`. Runtime guarantees hold (`aggregate`
  consults no registry, pinned by `test_aggregate_needs_no_registry`);
  importing from leaf modules would cut import cost and cycle risk —
  package-wide import-hygiene scope.
