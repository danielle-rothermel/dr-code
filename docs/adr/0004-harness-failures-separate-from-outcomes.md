# Harness failures are structurally separate from submission outcomes

The failure mode that most reliably ruins experiments on this setup is a
harness mistake recorded as a model result — thousands of rows that look
like "the model produced garbage" when the actual cause was plumbing.
dr-code's contracts make that structurally impossible, at three layers:

1. **Contract violations raise.** Wrong input types, unknown tasks, and
   unknown profile IDs are typed errors at the boundary; they never become
   scores.
2. **Mid-evaluation harness trouble returns a separate channel.** Scoring
   returns a discriminated union: either a completed score carrying a
   submission outcome, or a harness-failure record (kind, cause detail,
   retryable hint). Consumers must branch; no enum value is shared between
   the two, so analysis cannot silently lump them.
3. **Failures carry their cause.** Extraction failures carry the candidate
   walk; case results carry exception type/message and elapsed time;
   timeouts carry the budget in force — enough to distinguish "genuinely
   slow submission" from machine flakes in analysis.

The explain facade renders the same trace the parser actually produced
(single trace source) — tooling that could drift from real behavior would
lie during hand-tuning, which is the same failure category.
