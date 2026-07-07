"""Candidate Recovery - repair, validate, dedupe, and select code candidates."""

from __future__ import annotations

import hashlib
from typing import Final

from code_eval.config import ValidatorConfig
from code_eval.models.candidate import Candidate
from code_eval.models.candidate_rank import CandidateRank
from code_eval.models.candidate_recovery_attempt import CandidateRecoveryAttempt
from code_eval.models.candidate_recovery_result import CandidateRecoveryResult
from code_eval.models.candidate_selection import CandidateSelection
from code_eval.models.extracted_candidate import ExtractedCandidate
from code_eval.models.validation_outcome import ValidationOutcome
from code_eval.names import ExtractorName, RepairName, ValidatorName
from code_eval.repairs import REPAIRS
from code_eval.validators import VALIDATORS

ATTEMPT_ID_PREFIX: Final[str] = "e"
CANDIDATE_ATTEMPT_PREFIX: Final[str] = "a"
HASH_ENCODING: Final[str] = "utf-8"
HASH_SIZE_BYTES: Final[int] = 6

REPAIR_ORDER: Final[tuple[type, ...]] = REPAIRS

type RepairAttempt = tuple[str, tuple[str, ...]]


def run_candidate_recovery(
    extracted: tuple[ExtractedCandidate, ...],
    config: ValidatorConfig,
) -> CandidateRecoveryResult:
    """Run repair attempts, validation, dedupe, and selection for extracted code."""
    candidates: list[Candidate] = []
    attempts: list[CandidateRecoveryAttempt] = []
    candidate_by_id: dict[str, Candidate] = {}
    first_attempt_id_by_candidate_id: dict[str, str] = {}

    for extracted_index, extracted_candidate in enumerate(extracted):
        for attempt_index, (source, repairs_applied) in enumerate(
            _repair_attempts_for(extracted_candidate)
        ):
            attempt_id = _attempt_id(extracted_index, attempt_index)
            candidate = _make_candidate(
                extracted_candidate,
                repairs_applied,
                source,
                _candidate_attempt_suffix(attempt_index),
            )
            existing = candidate_by_id.get(candidate.candidate_id)
            if existing is not None:
                attempts.append(
                    _make_attempt(
                        attempt_id=attempt_id,
                        extracted_index=extracted_index,
                        attempt_index=attempt_index,
                        extracted_candidate=extracted_candidate,
                        candidate=existing,
                        source=source,
                        repairs_applied=repairs_applied,
                        deduped=True,
                        canonical_candidate_id=existing.candidate_id,
                    )
                )
                continue

            validation, is_valid = _validate_source(
                source,
                config.validators,
                config.enable_import_resolve_validator,
            )
            candidate = candidate.model_copy(
                update={"validation": validation, "is_valid": is_valid}
            )
            candidates.append(candidate)
            candidate_by_id[candidate.candidate_id] = candidate
            first_attempt_id_by_candidate_id[candidate.candidate_id] = attempt_id
            attempts.append(
                _make_attempt(
                    attempt_id=attempt_id,
                    extracted_index=extracted_index,
                    attempt_index=attempt_index,
                    extracted_candidate=extracted_candidate,
                    candidate=candidate,
                    source=source,
                    repairs_applied=repairs_applied,
                    deduped=False,
                    canonical_candidate_id=None,
                )
            )

    valid_candidates = tuple(candidate for candidate in candidates if candidate.is_valid)
    selection = _select_candidate(valid_candidates, first_attempt_id_by_candidate_id)
    return CandidateRecoveryResult(
        candidates=tuple(candidates),
        valid_candidates=valid_candidates,
        attempts=tuple(attempts),
        selection=selection,
    )


def _short_hash(text: str) -> str:
    return hashlib.blake2b(
        text.encode(HASH_ENCODING),
        digest_size=HASH_SIZE_BYTES,
    ).hexdigest()


def _repair_attempts_for(extracted_candidate: ExtractedCandidate) -> tuple[RepairAttempt, ...]:
    attempts: list[RepairAttempt] = [(extracted_candidate.source, ())]

    for repair_cls in REPAIR_ORDER:
        repair = repair_cls()
        result = repair.apply(extracted_candidate.source)
        if result.changed:
            attempts.append((result.source, result.applied_tags))

    chained_source = extracted_candidate.source
    chained_tags: list[str] = []
    for repair_cls in REPAIR_ORDER:
        repair = repair_cls()
        result = repair.apply(chained_source)
        if result.changed:
            chained_source = result.source
            chained_tags.extend(result.applied_tags)
    chained_attempt = (chained_source, tuple(chained_tags))
    if chained_tags and chained_attempt not in attempts:
        attempts.append(chained_attempt)

    return tuple(attempts)


def _make_candidate(
    extracted: ExtractedCandidate,
    repairs_applied: tuple[str, ...],
    final_source: str,
    suffix: str,
) -> Candidate:
    extractor_str = "+".join(extracted.extractor_path) or extracted.extractor.value
    candidate_id = f"{extractor_str}:{_short_hash(final_source)}:{suffix}"
    return Candidate(
        candidate_id=candidate_id,
        source=final_source,
        extractor=extracted.extractor,
        extractor_path=extracted.extractor_path,
        repairs_applied=repairs_applied,
    )


def _make_attempt(
    *,
    attempt_id: str,
    extracted_index: int,
    attempt_index: int,
    extracted_candidate: ExtractedCandidate,
    candidate: Candidate,
    source: str,
    repairs_applied: tuple[str, ...],
    deduped: bool,
    canonical_candidate_id: str | None,
) -> CandidateRecoveryAttempt:
    return CandidateRecoveryAttempt(
        attempt_id=attempt_id,
        extracted_index=extracted_index,
        attempt_index=attempt_index,
        candidate_id=candidate.candidate_id,
        canonical_candidate_id=canonical_candidate_id,
        source_before=extracted_candidate.source,
        source_after=source,
        repairs_applied=repairs_applied,
        changed=source != extracted_candidate.source,
        deduped=deduped,
        validation=candidate.validation,
        is_valid=candidate.is_valid,
    )


def _validate_source(
    source: str,
    validators: tuple[ValidatorName, ...],
    enable_import_resolve: bool,
) -> tuple[tuple[ValidationOutcome, ...], bool]:
    outcomes: list[ValidationOutcome] = []
    for validator_name in validators:
        validator = VALIDATORS[validator_name]()
        outcomes.append(validator.validate(source))
    if enable_import_resolve:
        validator = VALIDATORS[ValidatorName.IMPORT_RESOLVE]()
        outcomes.append(validator.validate(source))
    return tuple(outcomes), all(outcome.passed for outcome in outcomes)


def _attempt_id(extracted_index: int, attempt_index: int) -> str:
    return f"{ATTEMPT_ID_PREFIX}{extracted_index:03d}:{_candidate_attempt_suffix(attempt_index)}"


def _candidate_attempt_suffix(attempt_index: int) -> str:
    return f"{CANDIDATE_ATTEMPT_PREFIX}{attempt_index}"


def _select_candidate(
    valid_candidates: tuple[Candidate, ...],
    attempt_id_by_candidate_id: dict[str, str],
) -> CandidateSelection:
    ranked = tuple(
        sorted(
            (
                _rank_candidate(candidate, attempt_id_by_candidate_id[candidate.candidate_id])
                for candidate in valid_candidates
            ),
            key=lambda rank: rank.rank_key,
        )
    )
    best = ranked[0] if ranked else None
    return CandidateSelection(
        best_candidate_id=best.candidate_id if best is not None else None,
        best_attempt_id=best.attempt_id if best is not None else None,
        ranked_valid_candidates=ranked,
    )


def _rank_candidate(candidate: Candidate, attempt_id: str) -> CandidateRank:
    uses_text_normalize = ExtractorName.TEXT_NORMALIZE.value in candidate.extractor_path
    rank_key = (
        len(candidate.repairs_applied),
        len(candidate.extractor_path),
        1 if uses_text_normalize else 0,
        candidate.candidate_id,
    )
    return CandidateRank(
        candidate_id=candidate.candidate_id,
        attempt_id=attempt_id,
        rank_key=rank_key,
        repair_count=len(candidate.repairs_applied),
        extractor_path_length=len(candidate.extractor_path),
        uses_text_normalize=uses_text_normalize,
    )


# Suppress unused-name lint for RepairName which is part of the public
# attribution surface even though it is not referenced directly here.
_ = RepairName
