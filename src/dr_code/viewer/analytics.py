"""Named DuckDB analytical operations over registered external Parquets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import date, datetime
from decimal import Decimal
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Final, cast

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from dr_code.corpus.preprocessing_run import (
    CorpusRunError,
    validate_preprocessing_relations,
)
from dr_code.corpus.run_descriptor import file_sha256, normalize_origins
from dr_code.viewer.database import ViewerDatabase
from dr_code.viewer.domain import (
    Annotation,
    ComparisonStage,
    ExampleDetail,
    ExampleSummary,
    FailureGroup,
    Failures,
    IncompatibleRunsError,
    InvalidQueryError,
    OutcomeTransition,
    Page,
    ReviewPage,
    RunComparison,
    RunDescriptor,
    RunNotFoundError,
    RunSummary,
    Tag,
    TaskAnnotation,
    TaskAnnotationProvenance,
    TaskNotFoundError,
    MachineTaskAnnotationWriteResult,
    Verdict,
    Waterfall,
    WaterfallStage,
    validate_sha256,
    validate_task_identity,
)


@dataclass(frozen=True, slots=True)
class _StageDefinition:
    stage_id: str
    label: str
    predicate: str


_PREPROCESSING_STAGES: Final = (
    _StageDefinition("source", "Corpus rows", "TRUE"),
    _StageDefinition(
        "output_present",
        "Decoder output present",
        "r.decoder_output_presence = 'present'",
    ),
    _StageDefinition(
        "output_nonblank",
        "Decoder output nonblank",
        """
        EXISTS (
            SELECT 1 FROM facts AS sf
            WHERE sf.sample_id = r.sample_id
              AND sf.step_name = 'require_nonblank_text'
              AND cast(json_extract(sf.facts_json, '$.is_nonblank') AS BOOLEAN)
        )
        """,
    ),
    _StageDefinition(
        "has_extracted_candidate",
        "Candidates extracted",
        """
        EXISTS (
            SELECT 1 FROM facts AS sf
            WHERE sf.sample_id = r.sample_id
              AND sf.step_name = 'extract_candidates'
              AND cast(json_extract(sf.facts_json, '$.candidate_count') AS BIGINT) > 0
        )
        """,
    ),
    _StageDefinition(
        "has_compilable_candidate",
        "Compilable candidates",
        """
        EXISTS (
            SELECT 1 FROM facts AS sf
            WHERE sf.sample_id = r.sample_id
              AND sf.step_name = 'filter_compilable'
              AND cast(json_extract(sf.facts_json, '$.survivor_candidate_count') AS BIGINT) > 0
        )
        """,
    ),
    _StageDefinition(
        "has_top_level_candidate",
        "Top-level functions",
        """
        EXISTS (
            SELECT 1 FROM facts AS sf
            WHERE sf.sample_id = r.sample_id
              AND sf.step_name = 'filter_has_top_level_function'
              AND cast(json_extract(sf.facts_json, '$.survivor_candidate_count') AS BIGINT) > 0
        )
        """,
    ),
)

_EVALUATION_STAGES: Final = (
    _StageDefinition(
        "has_tested_candidate",
        "Tested",
        """
        EXISTS (
            SELECT 1
            FROM membership AS em
            JOIN evaluation_results AS er USING (evaluation_key)
            WHERE em.sample_id = r.sample_id AND er.record_status = 'measured'
        )
        """,
    ),
    _StageDefinition(
        "has_passing_candidate",
        "Passing",
        """
        EXISTS (
            SELECT 1
            FROM membership AS em
            JOIN evaluation_results AS er USING (evaluation_key)
            WHERE em.sample_id = r.sample_id AND er.outcome = 'passed'
        )
        """,
    ),
)

_PREPROCESSING_COMPARISON_STAGE_IDS: Final = (
    "has_extracted_candidate",
    "has_compilable_candidate",
    "has_top_level_candidate",
)
_CORPUS_CONTEXT_COLUMN: Final = "__corpus_context"
_EMPTY_CONTEXT_FIELD: Final = "__empty"


class ViewerAnalytics:
    """Query registered immutable artifacts and persisted local review state."""

    def __init__(
        self,
        database: ViewerDatabase,
        descriptors: Iterable[RunDescriptor],
    ) -> None:
        self._database = database
        self._snapshot_directory = tempfile.TemporaryDirectory(
            prefix="dr-code-viewer-artifacts-"
        )
        values = tuple(
            self._pin_descriptor(descriptor) for descriptor in descriptors
        )
        self._runs = {descriptor.run_id: descriptor for descriptor in values}
        if len(self._runs) != len(values):
            raise InvalidQueryError("registered run IDs must be unique")
        for descriptor in values:
            self._validate_relational_integrity(descriptor)
        database.register_runs(values)

    def _pin_descriptor(self, descriptor: RunDescriptor) -> RunDescriptor:
        root = Path(self._snapshot_directory.name)

        def pin(path: Path, sha256: str) -> Path:
            destination = root / sha256
            if destination.is_file():
                return destination
            temporary = root / f".{sha256}.tmp"
            try:
                shutil.copyfile(path, temporary)
                if file_sha256(temporary) != sha256:
                    raise InvalidQueryError(
                        f"run {descriptor.run_id!r} changed while registering artifacts"
                    )
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            return destination

        artifact_paths = {
            name: pin(
                getattr(descriptor, f"{name}_path"),
                descriptor.artifact_sha256[name],
            )
            for name in ("results", "candidates", "step_facts", "rejections")
        }
        values: dict[str, object] = {
            "corpus_path": pin(
                descriptor.corpus_path, descriptor.corpus_sha256
            ),
            "preprocessing_manifest_path": pin(
                descriptor.preprocessing_manifest_path,
                descriptor.preprocessing_manifest_sha256,
            ),
            **{f"{name}_path": path for name, path in artifact_paths.items()},
        }
        if descriptor.evaluation_manifest_path is not None:
            assert descriptor.candidate_membership_path is not None
            assert descriptor.candidate_results_path is not None
            values.update(
                {
                    "candidate_membership_path": pin(
                        descriptor.candidate_membership_path,
                        descriptor.artifact_sha256["candidate_membership"],
                    ),
                    "candidate_results_path": pin(
                        descriptor.candidate_results_path,
                        descriptor.artifact_sha256["candidate_results"],
                    ),
                }
            )
        return replace(descriptor, **values)

    def list_runs(self) -> tuple[RunSummary, ...]:
        return tuple(
            self._summary(descriptor) for descriptor in self._runs.values()
        )

    def waterfall(self, run_id: str) -> Waterfall:
        descriptor = self._run(run_id)
        definitions = self._stage_definitions(descriptor)
        counts = self._stage_counts(descriptor, definitions)
        stages: list[WaterfallStage] = []
        for order, (definition, count) in enumerate(zip(definitions, counts)):
            denominator = counts[order - 1] if order else counts[0]
            stages.append(
                WaterfallStage(
                    stage_id=definition.stage_id,
                    label=definition.label,
                    unit="sample",
                    order=order,
                    count=count,
                    denominator=denominator,
                    rate=_rate(count, denominator),
                )
            )
        return Waterfall(run=self._summary(descriptor), stages=tuple(stages))

    def failures(self, run_id: str) -> Failures:
        descriptor = self._run(run_id)
        rows = self._connection.execute(
            """
            WITH facts AS (SELECT * FROM read_parquet(?))
            SELECT
                r.failure_code,
                r.failed_step,
                r.cause,
                count(*) AS sample_count
            FROM read_parquet(?) AS r
            WHERE EXISTS (
                    SELECT 1 FROM facts AS sf
                    WHERE sf.sample_id = r.sample_id
                      AND sf.step_name = 'require_nonblank_text'
                      AND cast(json_extract(sf.facts_json, '$.is_nonblank') AS BOOLEAN)
                  )
              AND r.final_candidate_count = 0
              AND r.failure_code IS NOT NULL
              AND r.failed_step IS NOT NULL
            GROUP BY r.failure_code, r.failed_step, r.cause
            ORDER BY
                sample_count DESC,
                r.failure_code,
                r.failed_step,
                r.cause NULLS FIRST
            """,
            [str(descriptor.step_facts_path), str(descriptor.results_path)],
        ).fetchall()
        groups = tuple(
            FailureGroup(
                failure_code=row[0],
                failed_step=row[1],
                cause=row[2],
                count=row[3],
            )
            for row in rows
        )
        return Failures(
            run=self._summary(descriptor),
            groups=groups,
            total_count=sum(group.count for group in groups),
        )

    def examples(
        self,
        run_id: str,
        *,
        stage_id: str | None = None,
        failure_code: str | None = None,
        failed_step: str | None = None,
        cause: str | None = None,
        cause_is_null: bool = False,
        compare_run_id: str | None = None,
        baseline_outcome: str | None = None,
        candidate_outcome: str | None = None,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Page:
        """Return a stable page for one named selector.

        ``stage_id="lost:<stage_id>"`` selects samples retained by the
        preceding waterfall stage but not by the named stage.
        """
        descriptor = self._run(run_id)
        _validate_page(limit, offset)
        has_task_id = self._corpus_has_column(descriptor, "task_id")
        task_id_expression = "c.task_id" if has_task_id else "NULL::VARCHAR"
        selectors = sum(
            value is not None
            for value in (stage_id, failure_code, compare_run_id)
        )
        if selectors > 1:
            raise InvalidQueryError(
                "select one of stage_id, failure_code, or compare_run_id"
            )
        if (failure_code is None) != (failed_step is None):
            raise InvalidQueryError(
                "failure examples require both failure_code and failed_step"
            )
        if cause is not None and cause_is_null:
            raise InvalidQueryError(
                "cause and cause_is_null cannot be selected together"
            )
        if (cause is not None or cause_is_null) and failure_code is None:
            raise InvalidQueryError("cause selectors require a failure group")
        if compare_run_id is None and (
            baseline_outcome is not None or candidate_outcome is not None
        ):
            raise InvalidQueryError(
                "transition outcomes require compare_run_id"
            )
        if compare_run_id is not None and (
            baseline_outcome is None or candidate_outcome is None
        ):
            raise InvalidQueryError(
                "transition examples require both baseline and candidate outcomes"
            )

        if compare_run_id is not None:
            candidate = self._run(compare_run_id)
            self._validate_compatible(descriptor, candidate)
            where = "baseline.outcome = ? AND candidate.outcome = ?"
            params: list[object] = [
                str(descriptor.corpus_path),
                str(descriptor.results_path),
                str(candidate.results_path),
                baseline_outcome,
                candidate_outcome,
            ]
            from_sql = """
                read_parquet(?) AS c
                JOIN read_parquet(?) AS baseline USING (sample_id)
                JOIN read_parquet(?) AS candidate USING (sample_id)
                LEFT JOIN annotations AS a
                  ON a.corpus_sha256 = ?
                 AND a.sample_id = baseline.sample_id
                 AND a.decoder_output_sha256 = baseline.raw_output_sha256
            """
            params.insert(3, descriptor.corpus_sha256)
            result_alias = "baseline"
        else:
            from_sql = """
                read_parquet(?) AS c
                JOIN read_parquet(?) AS r USING (sample_id)
                LEFT JOIN annotations AS a
                  ON a.corpus_sha256 = ?
                 AND a.sample_id = r.sample_id
                 AND a.decoder_output_sha256 = r.raw_output_sha256
            """
            params = [
                str(descriptor.corpus_path),
                str(descriptor.results_path),
                descriptor.corpus_sha256,
            ]
            result_alias = "r"
            predicates: list[str] = []
            if stage_id is not None:
                definition = self._stage_definition(descriptor, stage_id)
                where = definition.predicate
                if "FROM facts" in where:
                    # Named CTE preserves the exact aggregate predicate.
                    from_sql = """
                        read_parquet(?) AS c
                        JOIN read_parquet(?) AS r USING (sample_id)
                        LEFT JOIN annotations AS a
                          ON a.corpus_sha256 = ?
                         AND a.sample_id = r.sample_id
                         AND a.decoder_output_sha256 = r.raw_output_sha256
                    """
            else:
                where = "TRUE"
            if failure_code is not None:
                assert failed_step is not None
                predicates.append(
                    "EXISTS (SELECT 1 FROM facts AS sf "
                    "WHERE sf.sample_id = r.sample_id "
                    "AND sf.step_name = 'require_nonblank_text' "
                    "AND cast(json_extract(sf.facts_json, '$.is_nonblank') "
                    "AS BOOLEAN)) AND "
                    "r.final_candidate_count = 0 AND r.failure_code = ? "
                    "AND r.failed_step = ?"
                )
                params.extend((failure_code, failed_step))
                if cause is not None or cause_is_null:
                    predicates.append("r.cause IS NOT DISTINCT FROM ?")
                    params.append(cause)
            if predicates:
                where = f"({where}) AND " + " AND ".join(predicates)

        if search is not None:
            normalized_search = search.strip()
            if normalized_search:
                where += (
                    " AND (c.sample_id ILIKE ? ESCAPE '\\'"
                    f" OR coalesce({task_id_expression}, '') ILIKE ? ESCAPE '\\'"
                    " OR coalesce(c.decoder_output, '') ILIKE ? ESCAPE '\\')"
                )
                pattern = _like_pattern(normalized_search)
                params.extend((pattern, pattern, pattern))

        named_ctes: list[str] = []
        cte_params: list[object] = []
        if compare_run_id is None and (
            stage_id is not None or failure_code is not None
        ):
            needs_facts = failure_code is not None
            definition = None
            if stage_id is not None:
                definition = self._stage_definition(descriptor, stage_id)
                needs_facts = "FROM facts" in definition.predicate
            if needs_facts:
                named_ctes.append("facts AS (SELECT * FROM read_parquet(?))")
                cte_params.append(str(descriptor.step_facts_path))
            if (
                definition is not None
                and "FROM membership" in definition.predicate
            ):
                assert descriptor.candidate_membership_path is not None
                assert descriptor.candidate_results_path is not None
                named_ctes.extend(
                    (
                        "membership AS (SELECT * FROM read_parquet(?))",
                        "evaluation_results AS (SELECT * FROM read_parquet(?))",
                    )
                )
                cte_params.extend(
                    (
                        str(descriptor.candidate_membership_path),
                        str(descriptor.candidate_results_path),
                    )
                )
        cte_prefix = ", ".join(named_ctes)
        if cte_prefix:
            cte_prefix += ","
        params = [*cte_params, *params]

        query = f"""
            WITH {cte_prefix}
            selected AS (
                SELECT
                    c.sample_id,
                    {task_id_expression} AS task_id,
                    {result_alias}.raw_output_sha256,
                    {result_alias}.outcome,
                    {result_alias}.failure_code,
                    {result_alias}.failed_step,
                    c.decoder_output,
                    a.verdict
                FROM {from_sql}
                WHERE {where}
            ),
            counted AS (
                SELECT *, count(*) OVER () AS total
                FROM selected
            )
            SELECT * FROM counted
            ORDER BY sample_id
            LIMIT ? OFFSET ?
        """
        rows = self._connection.execute(
            query, [*params, limit, offset]
        ).fetchall()
        total = (
            rows[0][8]
            if rows
            else self._count_examples(
                cte_prefix=cte_prefix,
                from_sql=from_sql,
                where=where,
                params=params,
            )
        )
        return Page(
            items=tuple(
                ExampleSummary(
                    sample_id=row[0],
                    task_id=row[1],
                    decoder_output_sha256=row[2],
                    outcome=row[3],
                    failure_code=row[4],
                    failed_step=row[5],
                    decoder_output=row[6],
                    annotation_verdict=(
                        Verdict(row[7]) if row[7] is not None else None
                    ),
                )
                for row in rows
            ),
            total=total,
            limit=limit,
            offset=offset,
        )

    def example(self, run_id: str, sample_id: str) -> ExampleDetail:
        descriptor = self._run(run_id)
        if not sample_id:
            raise InvalidQueryError("sample_id must not be blank")
        cursor = self._connection.execute(
            f"""
            SELECT {self._corpus_projection(descriptor)},
                   r.* EXCLUDE (sample_id)
            FROM read_parquet(?) AS c
            JOIN read_parquet(?) AS r USING (sample_id)
            WHERE c.sample_id = ?
            """,
            [
                str(descriptor.corpus_path),
                str(descriptor.results_path),
                sample_id,
            ],
        )
        columns = [item[0] for item in cursor.description]
        row = cursor.fetchone()
        if row is None:
            raise InvalidQueryError(
                f"sample {sample_id!r} is not present in run {run_id!r}"
            )
        joined = dict(zip(columns, row, strict=True))
        candidates = tuple(
            _normalize_candidate_row(row, descriptor)
            for row in self._relation_rows(
                descriptor.candidates_path,
                sample_id,
                (
                    "candidate_index",
                    "candidate_id",
                    "cleaned_source",
                    "origins",
                    "compile_warnings",
                    "top_level_function_names",
                ),
                "candidate_index, candidate_id",
            )
        )
        facts = self._relation_rows(
            descriptor.step_facts_path,
            sample_id,
            ("step_name", "facts_json"),
            "step_name",
        )
        rejections = self._relation_rows(
            descriptor.rejections_path,
            sample_id,
            ("step_name", "reason_code", "details_json"),
            "step_name, input_index, candidate_id",
        )
        decoder_output_sha256 = cast(str | None, joined["raw_output_sha256"])
        annotation = (
            self._database.get_annotation(
                descriptor.corpus_sha256, sample_id, decoder_output_sha256
            )
            if decoder_output_sha256 is not None
            else None
        )
        return self._example_detail(
            descriptor,
            sample_id,
            joined,
            candidates=candidates,
            facts=facts,
            rejections=rejections,
            annotation=annotation,
        )

    def review_examples(
        self,
        run_id: str,
        *,
        failure_code: str,
        failed_step: str,
        cause: str | None = None,
        cause_is_null: bool = False,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> ReviewPage:
        """Return complete details for one stable failure-group page."""
        if (cause is None and not cause_is_null) or (
            cause is not None and cause_is_null
        ):
            raise InvalidQueryError(
                "review examples require exactly one of cause or "
                "cause_is_null=true"
            )
        page = self.examples(
            run_id,
            failure_code=failure_code,
            failed_step=failed_step,
            cause=cause,
            cause_is_null=cause_is_null,
            search=search,
            limit=limit,
            offset=offset,
        )
        sample_ids = tuple(item.sample_id for item in page.items)
        if not sample_ids:
            return ReviewPage(
                items=(),
                total=page.total,
                limit=page.limit,
                offset=page.offset,
            )
        descriptor = self._run(run_id)
        joined_by_sample = self._joined_rows(descriptor, sample_ids)
        candidates_by_sample = self._relation_rows_by_sample(
            descriptor.candidates_path,
            sample_ids,
            (
                "candidate_index",
                "candidate_id",
                "cleaned_source",
                "origins",
                "compile_warnings",
                "top_level_function_names",
            ),
            "candidate_index, candidate_id",
        )
        candidates_by_sample = {
            sample_id: tuple(
                _normalize_candidate_row(row, descriptor) for row in rows
            )
            for sample_id, rows in candidates_by_sample.items()
        }
        facts_by_sample = self._relation_rows_by_sample(
            descriptor.step_facts_path,
            sample_ids,
            ("step_name", "facts_json"),
            "step_name",
        )
        rejections_by_sample = self._relation_rows_by_sample(
            descriptor.rejections_path,
            sample_ids,
            ("step_name", "reason_code", "details_json"),
            "step_name, input_index, candidate_id",
        )
        output_by_sample = {
            item.sample_id: item.decoder_output_sha256 for item in page.items
        }
        annotation_identities = tuple(
            (sample_id, output_sha256)
            for sample_id, output_sha256 in output_by_sample.items()
            if output_sha256 is not None
        )
        annotations = self._database.get_annotations(
            descriptor.corpus_sha256, annotation_identities
        )
        return ReviewPage(
            items=tuple(
                self._example_detail(
                    descriptor,
                    sample_id,
                    joined_by_sample[sample_id],
                    candidates_by_sample[sample_id],
                    facts_by_sample[sample_id],
                    rejections_by_sample[sample_id],
                    annotations.get((sample_id, output_by_sample[sample_id])),
                )
                for sample_id in sample_ids
            ),
            total=page.total,
            limit=page.limit,
            offset=page.offset,
        )

    def compare(
        self, baseline_run_id: str, candidate_run_id: str
    ) -> RunComparison:
        baseline = self._run(baseline_run_id)
        candidate = self._run(candidate_run_id)
        self._validate_compatible(baseline, candidate)
        baseline_waterfall = self.waterfall(baseline_run_id)
        candidate_waterfall = self.waterfall(candidate_run_id)
        for before, after in zip(
            baseline_waterfall.stages, candidate_waterfall.stages, strict=True
        ):
            if before.stage_id != after.stage_id:
                raise IncompatibleRunsError(
                    "runs expose different waterfall stage mappings"
                )

        baseline_stages = {
            stage.stage_id: stage for stage in baseline_waterfall.stages
        }
        candidate_stages = {
            stage.stage_id: stage for stage in candidate_waterfall.stages
        }
        baseline_corpus_rows = baseline_stages["source"].count
        candidate_corpus_rows = candidate_stages["source"].count
        baseline_nonblank_rows = baseline_stages["output_nonblank"].count
        candidate_nonblank_rows = candidate_stages["output_nonblank"].count
        comparison_stage_ids = _PREPROCESSING_COMPARISON_STAGE_IDS + (
            ("has_passing_candidate",) if baseline.has_evaluation else ()
        )
        comparison_values = [
            (
                "lost:output_present",
                "Decoder output missing",
                baseline_corpus_rows - baseline_stages["output_present"].count,
                candidate_corpus_rows
                - candidate_stages["output_present"].count,
                baseline_corpus_rows,
                candidate_corpus_rows,
            ),
            (
                "lost:output_nonblank",
                "Decoder output empty",
                baseline_stages["output_present"].count
                - baseline_nonblank_rows,
                candidate_stages["output_present"].count
                - candidate_nonblank_rows,
                baseline_corpus_rows,
                candidate_corpus_rows,
            ),
            *(
                (
                    stage_id,
                    baseline_stages[stage_id].label,
                    baseline_stages[stage_id].count,
                    candidate_stages[stage_id].count,
                    baseline_nonblank_rows,
                    candidate_nonblank_rows,
                )
                for stage_id in comparison_stage_ids
            ),
        ]
        stages = tuple(
            ComparisonStage(
                stage_id=stage_id,
                label=label,
                unit="sample",
                baseline_count=baseline_count,
                baseline_denominator_count=baseline_denominator,
                candidate_count=candidate_count,
                candidate_denominator_count=candidate_denominator,
                count_delta=candidate_count - baseline_count,
                baseline_rate=_rate(baseline_count, baseline_denominator),
                candidate_rate=_rate(candidate_count, candidate_denominator),
                rate_delta=_difference(
                    _rate(candidate_count, candidate_denominator),
                    _rate(baseline_count, baseline_denominator),
                ),
            )
            for (
                stage_id,
                label,
                baseline_count,
                candidate_count,
                baseline_denominator,
                candidate_denominator,
            ) in comparison_values
        )
        rows = self._connection.execute(
            """
            SELECT
                baseline.outcome,
                candidate.outcome,
                count(*) AS sample_count
            FROM read_parquet(?) AS baseline
            JOIN read_parquet(?) AS candidate USING (sample_id)
            GROUP BY baseline.outcome, candidate.outcome
            ORDER BY baseline.outcome, candidate.outcome
            """,
            [str(baseline.results_path), str(candidate.results_path)],
        ).fetchall()
        return RunComparison(
            baseline=self._summary(baseline),
            candidate=self._summary(candidate),
            stages=stages,
            transitions=tuple(
                OutcomeTransition(
                    baseline_outcome=row[0],
                    candidate_outcome=row[1],
                    count=row[2],
                )
                for row in rows
            ),
        )

    def list_tags(self) -> tuple[Tag, ...]:
        return self._database.list_tags()

    def create_tag(self, name: str) -> Tag:
        return self._database.create_tag(name)

    def put_annotation(
        self,
        corpus_sha256: str,
        sample_id: str,
        decoder_output_sha256: str,
        *,
        verdict: Verdict | str | None,
        note: str | None = None,
        tag_ids: Iterable[str] = (),
    ) -> Annotation:
        self._validate_annotation_target(
            corpus_sha256, sample_id, decoder_output_sha256
        )
        return self._database.put_annotation(
            corpus_sha256,
            sample_id,
            decoder_output_sha256,
            verdict=verdict,
            note=note,
            tag_ids=tag_ids,
        )

    def delete_annotation(
        self,
        corpus_sha256: str,
        sample_id: str,
        decoder_output_sha256: str,
    ) -> bool:
        return self._database.delete_annotation(
            corpus_sha256, sample_id, decoder_output_sha256
        )

    def export_annotations(self) -> list[dict[str, object]]:
        return self._database.export_annotations()

    def get_task_annotation(
        self,
        dataset_id: str,
        task_id: str,
        task_identity: str,
    ) -> TaskAnnotation | None:
        self._validate_task_target(dataset_id, task_id, task_identity)
        return self._database.get_task_annotation(
            dataset_id,
            task_id,
            task_identity,
        )

    def put_task_annotation(
        self,
        dataset_id: str,
        task_id: str,
        task_identity: str,
        *,
        category: str | None = None,
        note: str | None = None,
        tag_ids: Iterable[str] = (),
    ) -> TaskAnnotation:
        """Persist browser input as human state with cleared provenance."""
        self._validate_task_target(dataset_id, task_id, task_identity)
        return self._database.put_task_annotation(
            dataset_id,
            task_id,
            task_identity,
            category=category,
            note=note,
            tag_ids=tag_ids,
        )

    def put_machine_task_annotation(
        self,
        dataset_id: str,
        task_id: str,
        task_identity: str,
        *,
        category: str | None,
        note: str | None = None,
        tag_ids: Iterable[str] = (),
        provenance: TaskAnnotationProvenance,
    ) -> MachineTaskAnnotationWriteResult:
        """Expose the protected atomic path for internal machine producers."""
        self._validate_task_target(dataset_id, task_id, task_identity)
        return self._database.put_machine_task_annotation(
            dataset_id,
            task_id,
            task_identity,
            category=category,
            note=note,
            tag_ids=tag_ids,
            provenance=provenance,
        )

    def delete_task_annotation(
        self,
        dataset_id: str,
        task_id: str,
        task_identity: str,
    ) -> bool:
        self._validate_task_target(dataset_id, task_id, task_identity)
        return self._database.delete_task_annotation(
            dataset_id,
            task_id,
            task_identity,
        )

    def export_task_annotations(self) -> list[dict[str, object]]:
        return self._database.export_task_annotations()

    @property
    def _connection(self) -> duckdb.DuckDBPyConnection:
        return self._database.connection

    def _run(self, run_id: str) -> RunDescriptor:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise RunNotFoundError(f"unknown run ID: {run_id}") from exc

    def _summary(self, descriptor: RunDescriptor) -> RunSummary:
        return RunSummary(
            run_id=descriptor.run_id,
            label=descriptor.label,
            dataset_id=descriptor.dataset_id,
            manifest_sha256=descriptor.preprocessing_manifest_sha256,
            corpus_sha256=descriptor.corpus_sha256,
            definition_id=descriptor.definition_id,
            definition_version=descriptor.definition_version,
            has_evaluation=descriptor.has_evaluation,
            definition_identity=descriptor.definition_identity,
        )

    def _stage_definitions(
        self, descriptor: RunDescriptor
    ) -> tuple[_StageDefinition, ...]:
        return _PREPROCESSING_STAGES + (
            _EVALUATION_STAGES if descriptor.has_evaluation else ()
        )

    def _stage_definition(
        self, descriptor: RunDescriptor, stage_id: str
    ) -> _StageDefinition:
        definitions = self._stage_definitions(descriptor)
        if stage_id.startswith("lost:"):
            current_id = stage_id.removeprefix("lost:")
            for index, definition in enumerate(definitions):
                if definition.stage_id != current_id:
                    continue
                if index == 0:
                    raise InvalidQueryError(
                        "the source stage has no preceding-stage loss"
                    )
                previous = definitions[index - 1]
                return _StageDefinition(
                    stage_id=stage_id,
                    label=f"Lost before {definition.label}",
                    predicate=(
                        f"({previous.predicate}) AND NOT "
                        f"({definition.predicate})"
                    ),
                )
            raise InvalidQueryError(
                f"unknown waterfall loss stage for run "
                f"{descriptor.run_id!r}: {current_id}"
            )
        for definition in definitions:
            if definition.stage_id == stage_id:
                return definition
        raise InvalidQueryError(
            f"unknown waterfall stage for run {descriptor.run_id!r}: {stage_id}"
        )

    def _stage_counts(
        self,
        descriptor: RunDescriptor,
        definitions: tuple[_StageDefinition, ...],
    ) -> tuple[int, ...]:
        ctes = [
            "facts AS (SELECT * FROM read_parquet(?))",
            "corpus AS (SELECT * FROM read_parquet(?))",
            "results AS (SELECT * FROM read_parquet(?))",
        ]
        params: list[object] = [
            str(descriptor.step_facts_path),
            str(descriptor.corpus_path),
            str(descriptor.results_path),
        ]
        if descriptor.has_evaluation:
            assert descriptor.candidate_membership_path is not None
            assert descriptor.candidate_results_path is not None
            ctes.extend(
                (
                    "membership AS (SELECT * FROM read_parquet(?))",
                    "evaluation_results AS (SELECT * FROM read_parquet(?))",
                )
            )
            params.extend(
                (
                    str(descriptor.candidate_membership_path),
                    str(descriptor.candidate_results_path),
                )
            )
        expressions = ",\n".join(
            f"count(*) FILTER (WHERE {stage.predicate})"
            for stage in definitions
        )
        row = self._connection.execute(
            f"""
            WITH {", ".join(ctes)}
            SELECT {expressions}
            FROM corpus AS c
            JOIN results AS r USING (sample_id)
            """,
            params,
        ).fetchone()
        assert row is not None
        return tuple(row)

    def _count_examples(
        self,
        *,
        cte_prefix: str,
        from_sql: str,
        where: str,
        params: list[object],
    ) -> int:
        row = self._connection.execute(
            f"WITH {cte_prefix} selected AS ("
            f"SELECT 1 FROM {from_sql} WHERE {where}"
            ") SELECT count(*) FROM selected",
            params,
        ).fetchone()
        assert row is not None
        return row[0]

    def _relation_rows(
        self,
        path: Path,
        sample_id: str,
        columns: tuple[str, ...],
        order_by: str,
    ) -> tuple[dict[str, object], ...]:
        return self._relation_rows_by_sample(
            path, (sample_id,), columns, order_by
        )[sample_id]

    def _joined_rows(
        self,
        descriptor: RunDescriptor,
        sample_ids: tuple[str, ...],
    ) -> dict[str, dict[str, object]]:
        cursor = self._connection.execute(
            f"""
            SELECT {self._corpus_projection(descriptor)},
                   r.* EXCLUDE (sample_id)
            FROM read_parquet(?) AS c
            JOIN read_parquet(?) AS r USING (sample_id)
            WHERE c.sample_id IN (SELECT unnest(?))
            ORDER BY c.sample_id
            """,
            [
                str(descriptor.corpus_path),
                str(descriptor.results_path),
                list(sample_ids),
            ],
        )
        columns = [item[0] for item in cursor.description]
        sample_id_index = columns.index("sample_id")
        return {
            row[sample_id_index]: dict(zip(columns, row, strict=True))
            for row in cursor.fetchall()
        }

    def _relation_rows_by_sample(
        self,
        path: Path,
        sample_ids: tuple[str, ...],
        columns: tuple[str, ...],
        order_by: str,
    ) -> dict[str, tuple[dict[str, object], ...]]:
        projection = ", ".join(columns)
        cursor = self._connection.execute(
            f"""
            SELECT sample_id, {projection}
            FROM read_parquet(?)
            WHERE sample_id IN (SELECT unnest(?))
            ORDER BY sample_id, {order_by}
            """,
            [str(path), list(sample_ids)],
        )
        grouped: dict[str, list[dict[str, object]]] = {
            sample_id: [] for sample_id in sample_ids
        }
        for row in cursor.fetchall():
            grouped[row[0]].append(
                {
                    key: _json_ready(value)
                    for key, value in zip(columns, row[1:], strict=True)
                }
            )
        return {key: tuple(rows) for key, rows in grouped.items()}

    def _example_detail(
        self,
        descriptor: RunDescriptor,
        sample_id: str,
        joined: dict[str, object],
        candidates: tuple[dict[str, object], ...],
        facts: tuple[dict[str, object], ...],
        rejections: tuple[dict[str, object], ...],
        annotation: Annotation | None,
    ) -> ExampleDetail:
        decoder_output = cast(str | None, joined["decoder_output"])
        decoder_output_sha256 = cast(str | None, joined["raw_output_sha256"])
        raw_context = joined[_CORPUS_CONTEXT_COLUMN]
        context_value = _json_ready(raw_context)
        if not isinstance(context_value, dict):
            raise InvalidQueryError("corpus context projection is invalid")
        has_context_fields = any(
            name not in {"sample_id", "decoder_output"}
            for name in pq.ParquetFile(
                descriptor.corpus_path
            ).schema_arrow.names
        )
        context: dict[str, object] = (
            {str(key): value for key, value in context_value.items()}
            if has_context_fields
            else {}
        )
        task_id = context.get("task_id")
        task_identity = (
            self._database.task_identity_for_run(
                descriptor.run_id,
                descriptor.dataset_id,
                task_id,
            )
            if descriptor.evaluation_coordinates is not None
            and isinstance(task_id, str)
            else None
        )
        return ExampleDetail(
            sample_id=sample_id,
            dataset_id=(
                descriptor.dataset_id
                if descriptor.evaluation_coordinates is not None
                else None
            ),
            task_identity=task_identity,
            corpus_sha256=descriptor.corpus_sha256,
            decoder_output_sha256=decoder_output_sha256,
            context=context,
            outcome=cast(str, joined["outcome"]),
            failure_code=cast(str | None, joined["failure_code"]),
            failed_step=cast(str | None, joined["failed_step"]),
            cause=cast(str | None, joined["cause"]),
            raw_decoder_output=decoder_output,
            candidates=candidates,
            facts=facts,
            rejections=rejections,
            annotation=annotation,
        )

    def _corpus_projection(self, descriptor: RunDescriptor) -> str:
        context_fields: list[str] = []
        for field in pq.ParquetFile(descriptor.corpus_path).schema_arrow:
            if field.name in {"sample_id", "decoder_output"}:
                continue
            identifier = _quoted_identifier(field.name)
            if pa.types.is_temporal(field.type) or pa.types.is_duration(
                field.type
            ):
                expression = f"cast(c.{identifier} AS VARCHAR)"
            else:
                expression = f"c.{identifier}"
            context_fields.append(f"{identifier} := {expression}")
        if not context_fields:
            context_fields.append(
                f"{_quoted_identifier(_EMPTY_CONTEXT_FIELD)} := NULL::BOOLEAN"
            )
        return (
            "c.sample_id, c.decoder_output, struct_pack("
            + ", ".join(context_fields)
            + f") AS {_quoted_identifier(_CORPUS_CONTEXT_COLUMN)}"
        )

    def _corpus_has_column(
        self, descriptor: RunDescriptor, column: str
    ) -> bool:
        return (
            column in pq.ParquetFile(descriptor.corpus_path).schema_arrow.names
        )

    def _validate_relational_integrity(
        self, descriptor: RunDescriptor
    ) -> None:
        try:
            validate_preprocessing_relations(
                input_parquet=pq.ParquetFile(descriptor.corpus_path),
                results_path=descriptor.results_path,
                candidates_path=descriptor.candidates_path,
                step_facts_path=descriptor.step_facts_path,
                rejections_path=descriptor.rejections_path,
            )
        except CorpusRunError as exc:
            raise InvalidQueryError(
                f"run {descriptor.run_id!r} failed relational validation: "
                + str(exc)
            ) from exc

    def _validate_compatible(
        self, baseline: RunDescriptor, candidate: RunDescriptor
    ) -> None:
        if baseline.corpus_sha256 != candidate.corpus_sha256:
            raise IncompatibleRunsError(
                "runs use different corpus fingerprints"
            )
        if baseline.dataset_id != candidate.dataset_id:
            raise IncompatibleRunsError(
                "runs use different dataset identities"
            )
        if baseline.has_evaluation != candidate.has_evaluation:
            raise IncompatibleRunsError(
                "runs expose different evaluation waterfall stages"
            )
        if baseline.has_evaluation:
            baseline_coordinates = _semantic_coordinates(baseline)
            candidate_coordinates = _semantic_coordinates(candidate)
            if baseline_coordinates != candidate_coordinates:
                raise IncompatibleRunsError(
                    "runs use different candidate-evaluation semantic coordinates"
                )
        difference = self._connection.execute(
            """
            SELECT count(*) FROM (
                SELECT coalesce(b.sample_id, c.sample_id)
                FROM read_parquet(?) AS b
                FULL OUTER JOIN read_parquet(?) AS c USING (sample_id)
                WHERE b.sample_id IS NULL OR c.sample_id IS NULL
            )
            """,
            [str(baseline.results_path), str(candidate.results_path)],
        ).fetchone()
        assert difference is not None
        if difference[0]:
            raise IncompatibleRunsError(
                "runs have different result sample membership"
            )

    def _validate_annotation_target(
        self,
        corpus_sha256: str,
        sample_id: str,
        decoder_output_sha256: str,
    ) -> None:
        validate_sha256(corpus_sha256, "corpus_sha256")
        validate_sha256(decoder_output_sha256, "decoder_output_sha256")
        matching = [
            descriptor
            for descriptor in self._runs.values()
            if descriptor.corpus_sha256 == corpus_sha256
        ]
        if not matching:
            raise InvalidQueryError("annotation corpus is not registered")
        for descriptor in matching:
            found = self._connection.execute(
                """
                SELECT 1 FROM read_parquet(?)
                WHERE sample_id = ? AND raw_output_sha256 = ?
                LIMIT 1
                """,
                [
                    str(descriptor.results_path),
                    sample_id,
                    decoder_output_sha256,
                ],
            ).fetchone()
            if found is not None:
                return
        raise InvalidQueryError(
            "annotation sample and decoder output are not present in a "
            "registered run for this corpus"
        )

    def _validate_task_target(
        self,
        dataset_id: str,
        task_id: str,
        task_identity: str,
    ) -> None:
        identity = validate_task_identity(
            dataset_id,
            task_id,
            task_identity,
        )
        if self._database.task_is_registered(identity):
            return
        raise TaskNotFoundError(
            "task identity is not present in any registered candidate "
            "evaluation"
        )


def _rate(count: int, denominator: int) -> float | None:
    return count / denominator if denominator else None


def _difference(
    candidate: float | None, baseline: float | None
) -> float | None:
    if candidate is None or baseline is None:
        return None
    return candidate - baseline


def _validate_page(limit: int, offset: int) -> None:
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= 200
    ):
        raise InvalidQueryError("limit must be an integer between 1 and 200")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise InvalidQueryError("offset must be a nonnegative integer")


def _like_pattern(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    )
    return f"%{escaped}%"


def _json_ready(value: object) -> object:
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return _json_ready(float(value))
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, bytes):
        return value.hex()
    return value


def _normalize_candidate_row(
    row: dict[str, object], descriptor: RunDescriptor
) -> dict[str, object]:
    return {
        **row,
        "origins": normalize_origins(row.get("origins")),
    }


def _semantic_coordinates(descriptor: RunDescriptor) -> dict[str, object]:
    assert descriptor.evaluation_coordinates is not None
    return {
        key: value
        for key, value in descriptor.evaluation_coordinates.items()
        if key
        not in {
            "corpus_sha256",
            "preprocessing_run",
            "evaluation_identity",
        }
    }


def _quoted_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


__all__ = ("ViewerAnalytics",)
