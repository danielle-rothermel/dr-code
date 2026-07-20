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

## Conclusions

- 97.50% of present decoder outputs produced at least one final top-level-function candidate.
- The most frequent candidate rejection was `not_compilable` at `filter_compilable` (13670 rejection rows across 12261 samples).
- 24796 samples (6.79% of all samples) retained multiple final candidates; candidate rows and sample outcomes are therefore reported separately.
- `normalized_raw_response` / `fenced_blocks` supplied 285953 of 726150 final-origin attributions; its recovery rate is 94.40% from extracted candidates.
- `code_eval_source_line` is the largest source kind with 177700 samples (48.66% of all samples).

## Viewer data

`viewer-data.json` contains 12 deterministic, bounded examples. Raw text is intentionally truncated; the preprocessing run is authoritative for complete sources.

## Limitations

- This is a preprocessing-only analysis; it does not claim task correctness or execution success.
- Candidate-evaluation membership/results may be recorded as optional provenance, but are not joined until their contract is defined.
- Viewer examples cap raw decoder and candidate text at 1200 and 1200 characters.
