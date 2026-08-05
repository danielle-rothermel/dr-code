# PR #89 — Extraction recall fixes and hard-examples corpus

<https://github.com/danielle-rothermel/dr-code/pull/89>

- [ ] The 5 strict xfails in `tests/preprocessing/test_hard_examples.py`
  are open behavior decisions: lambda-only solutions (2), EOF-truncated
  JSON envelopes (2), singleton string container (1). Deciding any flips
  its xfail into a feature request.
- [ ] Salvage boundary tracking is indentation-depth, not function scope
  (documented): a `return` inside an indented non-function block yields a
  non-compiling salvage that the compilability filter absorbs. Real scope
  tracking deliberately not built.
- [ ] `return 1;` on an otherwise-complete candidate contributes a
  redundant salvage differing only by trailing newline — cosmetic,
  additive, not fixed.
- [ ] `split_by_fences` discards fence tags, so no reading can distinguish
  a ```` ```json ```` fence from an untagged one; moot under the
  content-shape predicate, but a future tag-aware representation needs a
  `text_analysis` change.
- [ ] Corpus blind spot: of 130 cases, 9 carry field markers and 10 carry
  a fenced JSON envelope, and the sets are disjoint. The marker/envelope
  interaction is pinned by unit tests only; recorded outputs mixing the
  two would strengthen the corpus.
