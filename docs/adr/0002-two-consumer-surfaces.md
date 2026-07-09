# dr-code has exactly two consumer surfaces

dr-code ships one library surface — the curated `dr_code.humaneval` public
API together with the `dr_code.code_transforms` and
`dr_code.text_transforms` modules (ADR 0007), consumed by whetstone-ai as a
git-pinned dependency — and one HTTP surface, the localhost-only serve
facade (`/explain`, `/profiles`, `/health`) consumed by viewers through a
generated OpenAPI client. The
React component package (see ADR 0006) is a rendering companion to these
surfaces, typed against the facade's schema — not a third data surface.
There is deliberately no third surface: no batch CLIs, no file-artifact
contracts, no queue or database integration. Batch orchestration,
persistence, and experiment analysis belong to consumer repos; durable or
analytical data never flows through the facade (viewers read it from the
consumer's database via their own read layer). New facade endpoints are
added only when a concrete playground interaction needs them.
