"""Translate public viewer read models into classifier inputs."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

from dr_code.classifier.prompt import render_parse_prompt, render_test_prompt
from dr_code.classifier.taxonomy import FailureFamily
from dr_code.viewer.analytics import ViewerAnalytics

EXTRACTION_VERSION = "failure-extraction-v2"
EXTRACTION_BATCH_SIZE: Final = 128


@dataclass(frozen=True, slots=True)
class FailureItem:
    family: FailureFamily
    sample_id: str
    candidate_id: str | None
    evaluation_key: str | None
    dataset_id: str
    task_id: str | None
    task_identity: str | None
    rendered_input: str


@dataclass(frozen=True, slots=True)
class ExtractedFailures:
    items: tuple[FailureItem, ...]
    parse_total: int
    test_total: int


@dataclass(frozen=True, slots=True)
class FailureStream:
    items: Iterator[FailureItem]
    parse_total: int
    test_total: int


def extract_parse_failures(
    analytics: ViewerAnalytics,
    run_id: str,
    *,
    limit: int | None = None,
) -> tuple[tuple[FailureItem, ...], int]:
    items, total = _stream_parse_failures(
        analytics,
        run_id,
        limit=limit,
    )
    return tuple(items), total


def extract_test_failures(
    analytics: ViewerAnalytics,
    run_id: str,
    *,
    limit: int | None = None,
) -> tuple[tuple[FailureItem, ...], int]:
    items, total = _stream_test_failures(
        analytics,
        run_id,
        limit=limit,
    )
    return tuple(items), total


def extract_failures(
    analytics: ViewerAnalytics,
    run_id: str,
    *,
    parse_limit: int | None,
    test_limit: int | None,
) -> ExtractedFailures:
    extracted = stream_failures(
        analytics,
        run_id,
        parse_limit=parse_limit,
        test_limit=test_limit,
    )
    return ExtractedFailures(
        items=tuple(extracted.items),
        parse_total=extracted.parse_total,
        test_total=extracted.test_total,
    )


def stream_failures(
    analytics: ViewerAnalytics,
    run_id: str,
    *,
    parse_limit: int | None,
    test_limit: int | None,
) -> FailureStream:
    """Stream stable classifier inputs in bounded database pages."""
    parse_items, parse_total = _stream_parse_failures(
        analytics,
        run_id,
        limit=parse_limit,
    )
    test_items, test_total = _stream_test_failures(
        analytics,
        run_id,
        limit=test_limit,
    )

    def items() -> Iterator[FailureItem]:
        yield from parse_items
        yield from test_items

    return FailureStream(
        items=items(),
        parse_total=parse_total,
        test_total=test_total,
    )


def _stream_parse_failures(
    analytics: ViewerAnalytics,
    run_id: str,
    *,
    limit: int | None,
) -> tuple[Iterator[FailureItem], int]:
    first_limit = (
        EXTRACTION_BATCH_SIZE
        if limit is None
        else min(limit, EXTRACTION_BATCH_SIZE)
    )
    first = analytics.parse_failures_for_classification(
        run_id,
        limit=first_limit,
    )
    selected = first.total if limit is None else min(limit, first.total)

    def items() -> Iterator[FailureItem]:
        offset = 0
        page = first
        while offset < selected:
            if not page.items:
                raise RuntimeError(
                    "parse classification pagination ended before total"
                )
            for item in page.items:
                yield FailureItem(
                    family=FailureFamily.PARSE,
                    sample_id=item.sample_id,
                    candidate_id=None,
                    evaluation_key=None,
                    dataset_id=_required_dataset_id(item.dataset_id),
                    task_id=_authorized_task_id(
                        item.task_id,
                        item.task_identity,
                    ),
                    task_identity=item.task_identity,
                    rendered_input=render_parse_prompt(
                        decoder_output=item.decoder_output,
                        failure_code=item.failure_code,
                        failed_step=item.failed_step,
                        cause=item.cause,
                        task_context=item.task_context,
                    ),
                )
            offset += len(page.items)
            if offset >= selected:
                break
            page = analytics.parse_failures_for_classification(
                run_id,
                limit=min(EXTRACTION_BATCH_SIZE, selected - offset),
                offset=offset,
            )

    return items(), first.total


def _stream_test_failures(
    analytics: ViewerAnalytics,
    run_id: str,
    *,
    limit: int | None,
) -> tuple[Iterator[FailureItem], int]:
    first_limit = (
        EXTRACTION_BATCH_SIZE
        if limit is None
        else min(limit, EXTRACTION_BATCH_SIZE)
    )
    first = analytics.candidate_test_failures_for_classification(
        run_id,
        limit=first_limit,
    )
    selected = first.total if limit is None else min(limit, first.total)

    def items() -> Iterator[FailureItem]:
        offset = 0
        page = first
        while offset < selected:
            if not page.items:
                raise RuntimeError(
                    "test classification pagination ended before total"
                )
            for item in page.items:
                yield FailureItem(
                    family=FailureFamily.TEST,
                    sample_id=item.sample_id,
                    candidate_id=item.candidate_id,
                    evaluation_key=item.evaluation_key,
                    dataset_id=_required_dataset_id(item.dataset_id),
                    task_id=_authorized_task_id(
                        item.task_id,
                        item.task_identity,
                    ),
                    task_identity=item.task_identity,
                    rendered_input=render_test_prompt(
                        cleaned_source=item.cleaned_source,
                        outcome=item.outcome,
                        function_count=item.function_count,
                        best_function_name=item.best_function_name,
                        total_cases=item.total_cases,
                        passed_count=item.passed_count,
                        failed_count=item.failed_count,
                        error_count=item.error_count,
                        timeout_count=item.timeout_count,
                        coverage_complete=item.coverage_complete,
                        task_context=item.task_context,
                    ),
                )
            offset += len(page.items)
            if offset >= selected:
                break
            page = analytics.candidate_test_failures_for_classification(
                run_id,
                limit=min(EXTRACTION_BATCH_SIZE, selected - offset),
                offset=offset,
            )

    return items(), first.total


def _required_dataset_id(value: str | None) -> str:
    if value is None:
        raise ValueError(
            "classification input requires a nonnull dataset identity"
        )
    return value


def _authorized_task_id(
    task_id: str | None,
    task_identity: str | None,
) -> str | None:
    if task_identity is None:
        return None
    if task_id is None:
        raise ValueError("classification task_identity requires a task_id")
    return task_id


__all__ = (
    "EXTRACTION_BATCH_SIZE",
    "EXTRACTION_VERSION",
    "ExtractedFailures",
    "FailureItem",
    "FailureStream",
    "extract_failures",
    "extract_parse_failures",
    "extract_test_failures",
    "stream_failures",
)
