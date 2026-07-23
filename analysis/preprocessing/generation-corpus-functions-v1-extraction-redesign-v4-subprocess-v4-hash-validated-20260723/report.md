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
| function candidates extracted | sample | 305048 | share of all samples | 0.83525366 |
| final candidates | candidate_row | 433412 | candidates per extracted sample | 1.42079935 |
| final candidates with converged origins | candidate_row | 237792 | share of final candidate rows | 0.54865117 |

## Outcomes

| outcome | all count (rate) | present count (rate) | nonblank count (rate) |
| --- | ---: | ---: | ---: |
| decoder_output_blank | 109 (0.00029845) | 109 (0.00035405) | 0 (0.0) |
| decoder_output_missing | 57346 (0.15701941) | 0 (0.0) | 0 (0.0) |
| function_candidates_extracted | 305048 (0.83525366) | 305048 (0.99083379) | 305048 (0.99118472) |
| no_code_candidates | 313 (0.00085703) | 313 (0.00101666) | 313 (0.00101702) |
| no_compilable_candidate | 749 (0.00205084) | 749 (0.00243285) | 749 (0.00243371) |
| no_top_level_function_candidate | 1635 (0.0044768) | 1635 (0.00531068) | 1635 (0.00531256) |
| plain_literal_only | 16 (4.381e-05) | 16 (5.197e-05) | 16 (5.199e-05) |

## Candidate evaluation funnel

| stage | count | extracted-candidate rate |
| --- | ---: | ---: |
| extracted final candidates | 433412 | 1.0 |
| evaluation attempted candidates | 433412 | 1.0 |
| tested candidates | 433412 | 1.0 |
| candidate passed | 305622 | 0.70515353 |
| candidate failed | 117856 | 0.27192602 |
| candidate timed_out | 9934 | 0.02292045 |
| candidate infrastructure_failure | 0 | 0.0 |

### Evaluation provenance

- `candidate_evaluation_manifest_sha256`: `a5cc78b2c66e3b17a1c59123d4bfe22d9dd4da35cbbee788eff07a68c33a9592`
- `metrics_profile`: `humaneval-metrics@v1`
- `operator`: `code_test@1`
- `snapshot_sha256`: `141cdc0e5f035fae5b614640f866f8d5c544462d7f51fe59174c87235b8b4fe2`
- `runner_identity`: `subprocess:python-isolated@v1`
- `sandbox_image`: `null`
- `execution_fingerprint`: `1ccd695ab2c431f4d798979db14cf0b5a58df356ed3016cfa0ffda3093d7b6e5`
- `metrics_definition_hash`: `79d1065c6f64464bc56dc345b5346d5670f2b65c05610e472109e240f13a48e79e280b40ef4d09cbc054ad00ef46b974be8cc91aca81fc3f7986f5c22ed20455`
- `trusted_source_sha256`: `{"dr_code.corpus.candidate_evaluation":"b2ca40605043460521c590d24ebd23d941ccf9af2f904c02e3818da0c272051f","dr_code.humaneval.batch_runner":"749f62159d298093e90582c059987e258e87c1008666d9a62a88c587b407e311","dr_code.humaneval.parsed_code":"c3ffe1d07eaf0a928447e68c6002c65bda837ae8905010e104ea19b4598246af","dr_code.humaneval.parsed_tests":"94eb76cd71e8c436082e81b44ca8c16e902f7ff7ddfc9efda5321dffa6e7faa8","dr_code.humaneval.subprocess_runner":"feb49d6c895c7b190cf76ebdf48b6dbe067fefb44374be652b18718307a529a1","dr_code.humaneval.task":"29ea6aa5ad94b85ca1b3ac1411635fe639e2c54a65a46b3c9b16942c4a3a6033","dr_code.metrics.engine.execution":"4aa172ca1dbaa5a30f68393a23098d318351c89fff6f456e00d957c8b7ccbcfa","dr_code.metrics.operators.code_test":"50dd9a1e9a904a11d87e82125561625d9ccd52c8ed56e5cab74aa5a22b732671","runner_script":"3db0a24492cc6782beb310471cbf377375d0b74ce86dfcbe348235e7d23eee4a"}`
- `operator_settings`: `{"task_key":"task","timeout_seconds":2.0}`
- `python`: `3.13.2`
- `python_implementation`: `CPython`

## Conclusions

- 99.08% of present decoder outputs produced at least one final top-level-function candidate.
- The most frequent candidate rejection was `not_compilable` at `filter_compilable` (94027 rejection rows across 59013 samples).
- 113798 samples (31.16% of all samples) retained multiple final candidates; candidate rows and sample outcomes are therefore reported separately.
- `response_representation({"name":"normalized_raw_response"}) -> unfenced_segment({"index":0}) -> anchored_python_block({"index":0})` supplied 129997 of 798135 final-origin attributions; its recovery rate is 94.53% from extracted candidates.
- `code_eval_source_line` is the largest source kind with 177700 samples (48.66% of all samples).
- 230814 extracted samples (75.66%) had at least one passing candidate.

## Viewer data

`viewer-data.json` contains 11 preprocessing examples and 12 candidate-test examples. Selection is deterministic and raw text is intentionally bounded; the authoritative Parquets retain complete sources.
`failure-examples/` contains 2713 nonblank, zero-final-candidate examples across 4 terminal failure groups. Its indexes and bounded details are loaded lazily by the viewer.

## Limitations

- Preprocessing metrics alone do not claim task correctness or execution success.
- Viewer examples cap raw decoder and candidate text at 1200 and 1200 characters.
- Candidate test rates describe HumanEval+ execution under the supplied evaluation profile; they do not generalize to non-HumanEval tasks.
