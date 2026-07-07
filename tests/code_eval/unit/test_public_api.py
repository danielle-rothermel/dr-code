"""Public API smoke tests — what `import code_eval` is allowed to expose."""

from __future__ import annotations

import code_eval


def test_public_api_surface() -> None:
    assert hasattr(code_eval, "LLMCodeValidator")
    assert hasattr(code_eval, "ValidatorConfig")
    assert hasattr(code_eval, "ValidationResult")
    assert hasattr(code_eval, "Candidate")
    assert hasattr(code_eval, "CandidateRecoveryResult")
    assert hasattr(code_eval, "CandidateRecoveryAttempt")
    assert hasattr(code_eval, "CandidateSelection")
    assert hasattr(code_eval, "CandidateRank")
    assert hasattr(code_eval, "ExtractionResult")
    assert hasattr(code_eval, "DEFAULT_CONFIG")
    assert hasattr(code_eval, "EXTRACTION_CONFIG")
    assert hasattr(code_eval, "NormalizerName")
    assert hasattr(code_eval, "ValidatorName")
    assert hasattr(code_eval, "__version__")


def test_validator_validates_fenced_snippet_with_default_config() -> None:
    v = code_eval.LLMCodeValidator()
    raw = "```python\ndef hello():\n    return 1\n```"
    result = v.validate(raw, task_id="HumanEval/0")
    assert result.recovery.overall_success
    assert result.recovery.valid_candidates
    assert result.normalizations
    best = result.recovery.selected_candidate()
    assert best is not None
    assert result.recovery.selected_source() == best.source
    assert best.candidate_id in result.normalizations


def test_extraction_config_skips_normalization() -> None:
    v = code_eval.LLMCodeValidator(config=code_eval.EXTRACTION_CONFIG)
    raw = "```python\ndef hello():\n    return 1\n```"
    result = v.validate(raw, task_id="HumanEval/0")
    assert result.recovery.overall_success
    assert result.recovery.valid_candidates
    assert result.normalizations == {}
    source = result.recovery.selected_source()
    assert source is not None
    assert "def hello" in source


def test_extraction_log_yielded_valid_backfilled() -> None:
    v = code_eval.LLMCodeValidator(config=code_eval.EXTRACTION_CONFIG)
    raw = "```python\ndef hello():\n    return 1\n```"
    result = v.validate(raw, task_id="HumanEval/0")
    assert result.recovery.overall_success
    valid_paths = {
        name for candidate in result.recovery.valid_candidates for name in candidate.extractor_path
    }
    for step in result.extraction.extraction_log:
        assert step.yielded_valid_candidate == (step.extractor.value in valid_paths)


def test_extraction_log_all_false_when_parse_fails() -> None:
    v = code_eval.LLMCodeValidator(config=code_eval.EXTRACTION_CONFIG)
    result = v.validate("This is prose, not Python.", task_id="HumanEval/0")
    assert not result.recovery.overall_success
    assert all(not step.yielded_valid_candidate for step in result.extraction.extraction_log)
