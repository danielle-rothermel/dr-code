"""Unit tests for pipeline steps 2-6 wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from code_eval.candidate_recovery import run_candidate_recovery
from code_eval.config import DEFAULT_CONFIG, ValidatorConfig
from code_eval.extraction import run_extraction, text_normalize
from code_eval.models.candidate import Candidate
from code_eval.models.extracted_candidate import ExtractedCandidate
from code_eval.models.normalized_form import NormalizedForm
from code_eval.models.tool_versions import ToolVersions
from code_eval.names import ExtractorName, NormalizerName, RepairName
from code_eval.pipeline.normalize_step import run_normalize
from code_eval.pipeline.steps import backfill_extraction_log
from code_eval.subprocess_runner import SubprocessRunner


def test_text_normalize_expands_tabs() -> None:
    normalized = text_normalize("x\t= 1\n", tab_width=DEFAULT_CONFIG.tab_width)
    assert "\t" not in normalized


def test_extract_fenced_code_produces_candidate() -> None:
    raw = "```python\ndef foo():\n    return 1\n```"
    extraction = run_extraction(raw, DEFAULT_CONFIG)
    assert extraction.candidates
    assert any("def foo" in c.source for c in extraction.candidates)
    assert extraction.extraction_log


def test_candidate_recovery_marks_valid_python() -> None:
    raw = "```python\ndef foo():\n    return 1\n```"
    extraction = run_extraction(raw, DEFAULT_CONFIG)
    recovery = run_candidate_recovery(extraction.candidates, DEFAULT_CONFIG)
    assert recovery.candidates
    assert recovery.valid_candidates
    assert recovery.attempts
    assert recovery.overall_success
    assert all(candidate.is_valid for candidate in recovery.valid_candidates)
    assert recovery.selection.best_candidate_id is not None
    assert recovery.selection.best_attempt_id is not None


def test_candidate_recovery_records_repair_attempts() -> None:
    extracted = (
        ExtractedCandidate(
            source="def foo():\n    return “ok”\n",
            extractor=ExtractorName.FENCES,
            extractor_path=(ExtractorName.FENCES.value,),
        ),
    )

    recovery = run_candidate_recovery(extracted, DEFAULT_CONFIG)

    assert len(recovery.attempts) > 1
    repaired = [
        attempt
        for attempt in recovery.attempts
        if RepairName.SMART_QUOTES.value in attempt.repairs_applied
    ]
    assert repaired
    assert any(attempt.is_valid for attempt in repaired)


def test_candidate_recovery_records_deduped_attempts_with_canonical_candidate() -> None:
    duplicate = ExtractedCandidate(
        source="def foo():\n    return 1\n",
        extractor=ExtractorName.FENCES,
        extractor_path=(ExtractorName.FENCES.value,),
    )

    recovery = run_candidate_recovery((duplicate, duplicate), DEFAULT_CONFIG)

    assert len(recovery.candidates) == 1
    assert len(recovery.attempts) == 2
    deduped = recovery.attempts[1]
    assert deduped.deduped is True
    assert deduped.canonical_candidate_id == recovery.candidates[0].candidate_id
    assert deduped.validation == recovery.candidates[0].validation


def test_backfill_extraction_log_marks_valid_extractors() -> None:
    raw = "```python\ndef foo():\n    return 1\n```"
    extraction = run_extraction(raw, DEFAULT_CONFIG)
    recovery = run_candidate_recovery(extraction.candidates, DEFAULT_CONFIG)
    backfilled = backfill_extraction_log(extraction.extraction_log, recovery.valid_candidates)
    valid_paths = {
        name for candidate in recovery.valid_candidates for name in candidate.extractor_path
    }
    for step in backfilled:
        assert step.yielded_valid_candidate == (step.extractor.value in valid_paths)


def test_normalize_memo_skips_failed_subprocess_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Failed normalizations must not poison later candidates with the same source."""
    import code_eval.normalizers as normalizers_pkg
    import code_eval.pipeline.normalize_step as normalize_step

    calls = {"count": 0}

    class FlakyNormalizer:
        NAME = NormalizerName.L2_RUFF_FORMAT

        def __init__(self, runner: SubprocessRunner | None = None) -> None:
            del runner

        def normalize(self, source: str) -> NormalizedForm:
            calls["count"] += 1
            if calls["count"] == 1:
                return NormalizedForm(
                    normalizer=NormalizerName.L2_RUFF_FORMAT,
                    source=source,
                    transformations_applied=(),
                    success=False,
                )
            return NormalizedForm(
                normalizer=NormalizerName.L2_RUFF_FORMAT,
                source=source + "# ok\n",
                transformations_applied=(NormalizerName.L2_RUFF_FORMAT.value,),
                success=True,
            )

    monkeypatch.setitem(normalizers_pkg.NORMALIZERS, NormalizerName.L2_RUFF_FORMAT, FlakyNormalizer)
    monkeypatch.setattr(
        normalize_step,
        "_SUBPROCESS_NORMALIZERS",
        frozenset({NormalizerName.L2_RUFF_FORMAT}),
    )

    source = "x = 1\n"
    cand_a = Candidate(
        candidate_id="a",
        source=source,
        extractor=ExtractorName.FENCES,
        extractor_path=(ExtractorName.FENCES.value,),
    )
    cand_b = Candidate(
        candidate_id="b",
        source=source,
        extractor=ExtractorName.FENCES,
        extractor_path=(ExtractorName.FENCES.value,),
    )
    config = ValidatorConfig(
        normalizers=(NormalizerName.L2_RUFF_FORMAT,),
        cache_dir=tmp_path,
    )
    versions = ToolVersions(python="3.12.0", code_eval="0.1.0", ruff="0.8.4")

    out = run_normalize((cand_a, cand_b), config, versions, SubprocessRunner())
    assert calls["count"] == 2
    assert not out["a"][NormalizerName.L2_RUFF_FORMAT.value].success
    assert out["b"][NormalizerName.L2_RUFF_FORMAT.value].success


def test_normalize_memo_reuses_successful_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Successful normalizations are memoized within one validate call."""
    import code_eval.normalizers as normalizers_pkg
    import code_eval.pipeline.normalize_step as normalize_step

    calls = {"count": 0}

    class CountingNormalizer:
        NAME = NormalizerName.L0_CANONICAL_AST

        def normalize(self, source: str) -> NormalizedForm:
            calls["count"] += 1
            return NormalizedForm(
                normalizer=NormalizerName.L0_CANONICAL_AST,
                source=source,
                transformations_applied=(NormalizerName.L0_CANONICAL_AST.value,),
                success=True,
            )

    monkeypatch.setitem(
        normalizers_pkg.NORMALIZERS, NormalizerName.L0_CANONICAL_AST, CountingNormalizer
    )
    monkeypatch.setattr(normalize_step, "_SUBPROCESS_NORMALIZERS", frozenset())

    source = "def foo():\n    return 1\n"
    cand_a = Candidate(
        candidate_id="a",
        source=source,
        extractor=ExtractorName.DIRECT_PARSE,
        extractor_path=(ExtractorName.DIRECT_PARSE.value,),
    )
    cand_b = Candidate(
        candidate_id="b",
        source=source,
        extractor=ExtractorName.DIRECT_PARSE,
        extractor_path=(ExtractorName.DIRECT_PARSE.value,),
    )
    config = ValidatorConfig(
        normalizers=(NormalizerName.L0_CANONICAL_AST,),
        cache_dir=tmp_path,
    )
    versions = ToolVersions(python="3.12.0", code_eval="0.1.0", ruff="0.8.4")

    run_normalize((cand_a, cand_b), config, versions, SubprocessRunner())
    assert calls["count"] == 1
