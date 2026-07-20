# Preprocessing corpus analysis

## Scope

This report is derived from the authoritative preprocessing Parquets. It validates the corpus-to-results join by unique `sample_id`; detailed cross-tabs and failure/origin tables are in `tables/`.

## Denominators

| denominator | count |
| --- | ---: |
| all | 365216 |
| present | 307870 |
| nonblank | 307761 |

## Funnel

| stage | unit | count | metric | rate |
| --- | --- | ---: | --- | ---: |
| source samples | sample | 365216 | share of all samples | 1.0 |
| decoder output present | sample | 307870 | share of all samples | 0.84298059 |
| decoder output nonblank | sample | 307761 | share of all samples | 0.84268214 |
| function candidates extracted | sample | 300184 | share of all samples | 0.82193551 |
| final candidates | candidate_row | 325769 | candidates per extracted sample | 1.08523106 |
| final candidates with converged origins | candidate_row | 292726 | share of final candidate rows | 0.89856923 |

## Outcomes

| outcome | all count (rate) | present count (rate) | nonblank count (rate) |
| --- | ---: | ---: | ---: |
| decoder_output_blank | 109 (0.00029845) | 109 (0.00035405) | 0 (0.0) |
| decoder_output_missing | 57346 (0.15701941) | 0 (0.0) | 0 (0.0) |
| function_candidates_extracted | 300184 (0.82193551) | 300184 (0.97503492) | 300184 (0.97538025) |
| no_code_candidates | 363 (0.00099393) | 363 (0.00117907) | 363 (0.00117949) |
| no_compilable_candidate | 5050 (0.01382743) | 5050 (0.01640303) | 5050 (0.01640884) |
| no_nonblank_cleaned_candidate | 1 (2.74e-06) | 1 (3.25e-06) | 1 (3.25e-06) |
| no_top_level_function_candidate | 1965 (0.00538038) | 1965 (0.00638256) | 1965 (0.00638482) |
| plain_literal_only | 198 (0.00054214) | 198 (0.00064313) | 198 (0.00064336) |

## Candidate evaluation funnel

| stage | count | extracted-candidate rate |
| --- | ---: | ---: |
| extracted final candidates | 325769 | 1.0 |
| evaluation attempted candidates | 325769 | 1.0 |
| tested candidates | 325769 | 1.0 |
| candidate passed | 141975 | 0.43581495 |
| candidate failed | 43548 | 0.13367754 |
| candidate timed_out | 140246 | 0.43050751 |
| candidate infrastructure_failure | 0 | 0.0 |

### Evaluation provenance

- `candidate_evaluation_manifest_sha256`: `660bd54df2d74499c2a20e9f8c95fbeb5952c05539dd54b7d3a3a65dd60ea615`
- `metrics_profile`: `humaneval-metrics@v1`
- `operator`: `code_test@1`
- `snapshot_sha256`: `141cdc0e5f035fae5b614640f866f8d5c544462d7f51fe59174c87235b8b4fe2`
- `runner_identity`: `oci:docker:sha256:b0ea75e88e3047e718fd301b7e375894567ee0b75be093ef688b67fccef4fe45`
- `sandbox_image`: `sha256:b0ea75e88e3047e718fd301b7e375894567ee0b75be093ef688b67fccef4fe45`
- `execution_fingerprint`: `b6c08244f829e6a0f9b98a19c61a08afa28f4f6a78538338ca35efa25d5c053a`
- `metrics_definition_hash`: `79d1065c6f64464bc56dc345b5346d5670f2b65c05610e472109e240f13a48e79e280b40ef4d09cbc054ad00ef46b974be8cc91aca81fc3f7986f5c22ed20455`
- `trusted_source_sha256`: `{"dr_code.corpus.candidate_evaluation":"33b923de9d02cf9f15fb68bf13664917633b58777edf7388a52764a207ae9d3a","dr_code.humaneval.batch_runner":"37583731301d936946502e16999b952c4d1d39d478085080d94f590e8396bebb","dr_code.humaneval.parsed_code":"c3ffe1d07eaf0a928447e68c6002c65bda837ae8905010e104ea19b4598246af","dr_code.humaneval.parsed_tests":"94eb76cd71e8c436082e81b44ca8c16e902f7ff7ddfc9efda5321dffa6e7faa8","dr_code.humaneval.sandbox":"f22c13521d854d7e778e67117c414ca238bbe8cd8477f175f2a2c9214dee4819","dr_code.humaneval.task":"240ebef036c1d5373585f9e56f109a1b2b3ac05a8e21413138f21984c5b60878","dr_code.metrics.engine.execution":"6be139124a50556fedaf1f4c3cd5f4a74183fbb252ab49dfae30a2680936ba7c","dr_code.metrics.operators.code_test":"5034885f9bb8e6f3b88edd447ae5d7f10a8ec1eb4c998a1e9978daf89eb67cdf","runner_script":"e7c2c15f441002afca9d024f341611d4d4202df7cf4c60eba70531bdee9adcc2"}`
- `operator_settings`: `{"task_key":"task","timeout_seconds":2.0}`
- `python`: `3.13.2`
- `python_implementation`: `CPython`
- Limitation: The candidate evaluation manifest does not contain membership/results file hashes; same-run linkage is validated through row counts, full relational joins, singleton coordinates, and deterministic evaluation keys.

## Conclusions

- 97.50% of present decoder outputs produced at least one final top-level-function candidate.
- The most frequent candidate rejection was `not_compilable` at `filter_compilable` (13670 rejection rows across 12261 samples).
- 24796 samples (6.79% of all samples) retained multiple final candidates; candidate rows and sample outcomes are therefore reported separately.
- `normalized_raw_response` / `fenced_blocks` supplied 285953 of 726150 final-origin attributions; its recovery rate is 94.40% from extracted candidates.
- `code_eval_source_line` is the largest source kind with 177700 samples (48.66% of all samples).
- 135773 extracted samples (45.23%) had at least one passing candidate.

## Viewer data

`viewer-data.json` contains 12 preprocessing examples and 12 candidate-test examples. Selection is deterministic and raw text is intentionally bounded; the authoritative Parquets retain complete sources.
`failure-examples/` contains 7577 nonblank, zero-final-candidate examples across 5 terminal failure groups. Its indexes and bounded details are loaded lazily by the viewer.

## Limitations

- Preprocessing metrics alone do not claim task correctness or execution success.
- Viewer examples cap raw decoder and candidate text at 1200 and 1200 characters.
- Candidate test rates describe HumanEval+ execution under the supplied evaluation profile; they do not generalize to non-HumanEval tasks.
- The candidate evaluation manifest does not contain membership/results file hashes; same-run linkage is validated through row counts, full relational joins, singleton coordinates, and deterministic evaluation keys.
