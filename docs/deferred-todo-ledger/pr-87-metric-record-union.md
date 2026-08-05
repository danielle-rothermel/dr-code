# PR #87 — Metric record discriminated union

<https://github.com/danielle-rothermel/dr-code/pull/87>

- [ ] The loadability guarantee covers settings and
  operator-implementation/version churn, NOT metric-name churn:
  `MetricQuestionCoordinate.metric` is the closed `MetricName` enum, so
  records naming a deleted metric do not load. Widening to `str` is a real
  design tradeoff — decide if metric families will ever be retired with
  archives kept.
- [ ] `MetricQuestionCoordinate.settings` accepts duplicate setting names
  (unreachable via `question_settings()`; reachable via hand-built
  payloads). One-line validator if wanted.
- [ ] `record_rows` emits `question_settings` as one column holding a tuple
  of dicts — self-describing but not directly groupable; per-setting
  columns would be new scope.
- [ ] `MetricFactUnit.TEXT` exists solely for `parse_outcome.parse_error`
  (free-form prose, not a measurement); `Score` (#88) rejects it.
- [ ] `schema_version` is optional on load: `METRIC_RECORD_ADAPTER` treats
  an omitted key as v1, so a missing/corrupted version marker is
  indistinguishable from an explicit v1 record (explicit `schema_version=2`
  is rejected). Decide whether to require the field at the load boundary
  before any records are persisted.
