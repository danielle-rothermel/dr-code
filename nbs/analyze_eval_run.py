import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


with app.setup:
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import pandas as pd
    import pyarrow.parquet as pq


@app.cell
def _():
    mo.md(
        """
        # Stage 4 — Eval Run Analysis

        Loads artifacts written by `scripts/analyze_eval_run.py`.
        Metrics source of truth is the CLI export, not this notebook.
        """
    )
    return


@app.cell
def _():
    OUTPUT_DIR = Path("exports/demo/analysis")
    return (OUTPUT_DIR,)


@app.cell
def _(OUTPUT_DIR):
    enriched_path = OUTPUT_DIR / "enriched.parquet"
    summary_path = OUTPUT_DIR / "summary.json"
    by_source_path = OUTPUT_DIR / "aggregates" / "by_source.parquet"
    by_task_path = OUTPUT_DIR / "aggregates" / "by_task.parquet"

    enriched_df = pd.DataFrame(
        pq.read_table(enriched_path).to_pydict()
        if enriched_path.is_file()
        else {}
    )
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else {}
    )
    by_source_df = pd.DataFrame(
        pq.read_table(by_source_path).to_pydict()
        if by_source_path.is_file()
        else {}
    )
    by_task_df = pd.DataFrame(
        pq.read_table(by_task_path).to_pydict()
        if by_task_path.is_file()
        else {}
    )
    return by_source_df, by_task_df, enriched_df, summary


@app.cell
def _(summary):
    if not summary:
        mo.md("Run the analysis CLI first to populate exports.")
    else:
        outcome_counts = summary.get("outcome_kind_counts", {})
        pass_rate = summary.get("correctness_pass_rate")
        weighted_pass_rate = summary.get("correctness_pass_rate_weighted")
        join_failures = summary.get("join_failures", {})
        mo.vstack(
            [
                mo.md("## Headline"),
                mo.hstack(
                    [
                        mo.stat(
                            label="Pass rate",
                            value=f"{pass_rate:.1%}" if pass_rate is not None else "n/a",
                        ),
                        mo.stat(
                            label="Weighted pass rate",
                            value=(
                                f"{weighted_pass_rate:.1%}"
                                if weighted_pass_rate is not None
                                else "n/a"
                            ),
                        ),
                        mo.stat(
                            label="Missing test joins",
                            value=str(join_failures.get("missing_test_count", 0)),
                        ),
                    ]
                ),
                mo.md(f"**Outcome kinds:** `{outcome_counts}`"),
            ]
        )
    return


@app.cell
def _(enriched_df):
    mo.md("## Compression vs Pass")
    return


@app.cell
def _(enriched_df):
    if enriched_df.empty:
        compression_chart = mo.md("No enriched rows loaded.")
    else:
        chart_df = enriched_df.copy()
        chart_df["passed"] = chart_df["all_tests_passed"].map(
            {True: "pass", False: "fail"}
        ).fillna("not_tested")
        compression_chart = (
            alt.Chart(chart_df)
            .mark_circle(size=80, opacity=0.8)
            .encode(
                x=alt.X(
                    "decoder_input_len_zstd22:Q",
                    title="decoder_input zstd22 bytes",
                ),
                y=alt.Y("test_pass_rate:Q", title="test pass rate"),
                color=alt.Color(
                    "provenance_source:N",
                    title="source",
                ),
                tooltip=[
                    "sample_id",
                    "task_id",
                    "provenance_model",
                    "passed",
                    "outcome_kind",
                    "decoder_input_len_zstd22",
                    "test_pass_rate",
                ],
            )
            .properties(width=640, height=360)
        )
    compression_chart
    return


@app.cell
def _(summary):
    mo.md("## Parse Funnel")
    return


@app.cell
def _(summary):
    funnel = summary.get("parse_funnel", {})
    if not funnel:
        funnel_chart = mo.md("No funnel data loaded.")
    else:
        funnel_df = pd.DataFrame(
            {
                "stage": ["raw", "parse_success", "tested", "all_tests_passed"],
                "count": [
                    funnel.get("raw", 0),
                    funnel.get("parse_success", 0),
                    funnel.get("tested", 0),
                    funnel.get("all_tests_passed", 0),
                ],
            }
        )
        funnel_chart = (
            alt.Chart(funnel_df)
            .mark_bar()
            .encode(
                x=alt.X("stage:N", sort="-y", title="stage"),
                y=alt.Y("count:Q", title="rows"),
                tooltip=["stage", "count"],
            )
            .properties(width=520, height=280)
        )
    funnel_chart
    return


@app.cell
def _(by_source_df):
    mo.md("## Source Comparison")
    return


@app.cell
def _(by_source_df):
    if by_source_df.empty:
        source_chart = mo.md("No source aggregates loaded.")
    else:
        source_chart = (
            alt.Chart(by_source_df)
            .mark_bar()
            .encode(
                x=alt.X("slice_key:N", title="source"),
                y=alt.Y("pass_rate:Q", title="pass rate"),
                tooltip=[
                    "slice_key",
                    "pass_rate",
                    "weighted_pass_rate",
                    "row_count",
                    "weighted_count",
                ],
            )
            .properties(width=420, height=280)
        )
    source_chart
    return


@app.cell
def _(by_task_df):
    mo.md("## Task Hardness")
    return


@app.cell
def _(by_task_df):
    if by_task_df.empty:
        task_table = mo.md("No task aggregates loaded.")
    else:
        task_table = mo.ui.table(
            by_task_df.sort_values("pass_rate", ascending=True),
            selection=None,
        )
    task_table
    return


if __name__ == "__main__":
    app.run()
