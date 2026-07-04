"""Render a static raw-vs-normalized first-stage extraction ladder.

Run::

    uv run --with pyarrow python scripts/render_extraction_ladder.py
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from code_eval import EXTRACTION_CONFIG
from code_eval.extraction import EXTRACTION_CATALOG, run_extraction
from code_eval.models.extracted_candidate import ExtractedCandidate
from code_eval.names import ValidatorName
from code_eval.validators import VALIDATORS

ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DEFAULT_ATTEMPTS: Final[Path] = (
    ROOT.parent / "dr-code" / "exports" / "runs" / "proof-20840125" / "attempts.parquet"
)
DEFAULT_OUTPUT: Final[Path] = ROOT / "extraction_ladder.html"
TEMPLATE_PATH: Final[Path] = Path(__file__).with_name("extraction_ladder_template.html")
JSON_PLACEHOLDER: Final[str] = "__EXTRACTION_LADDER_JSON__"
VALIDATION_SOURCE: Final[str] = "pre_repair"
VALIDATION_EXCEPTION_PREFIX: Final[str] = "validator raised"
JSONL_SUFFIX: Final[str] = ".jsonl"
PARQUET_SUFFIX: Final[str] = ".parquet"


def _read_attempt_rows(path: Path) -> list[dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix == JSONL_SUFFIX:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if suffix == PARQUET_SUFFIX:
        try:
            import pyarrow.parquet as pq  # type: ignore[import-not-found]
        except ImportError as exc:
            msg = "Reading parquet requires pyarrow; run with `uv run --with pyarrow ...`."
            raise SystemExit(msg) from exc
        return [dict(row) for row in pq.read_table(path).to_pylist()]
    raise ValueError(f"Unsupported attempts format: {suffix}")


def _field(row: dict[str, object], name: str) -> str:
    value = row.get(name)
    return "" if value is None else str(value)


def _optional_field(row: dict[str, object], name: str) -> str | None:
    value = row.get(name)
    return None if value is None else str(value)


def _int_field(row: dict[str, object], name: str, default: int) -> int:
    value = row.get(name)
    return default if value is None else int(value)


def _run_validator(validator_name: ValidatorName, source: str) -> dict[str, object]:
    try:
        outcome = VALIDATORS[validator_name]().validate(source)
    except Exception as exc:
        return {
            "validator": validator_name.value,
            "passed": False,
            "detail": f"{VALIDATION_EXCEPTION_PREFIX}: {type(exc).__name__}: {exc}",
            "ast_shape": None,
        }
    return outcome.model_dump(mode="json")


def _validate_before_repair(source: str) -> tuple[list[dict[str, object]], bool]:
    outcomes = [
        _run_validator(validator_name, source) for validator_name in EXTRACTION_CONFIG.validators
    ]
    if EXTRACTION_CONFIG.enable_import_resolve_validator:
        outcomes.append(_run_validator(ValidatorName.IMPORT_RESOLVE, source))
    return outcomes, all(bool(outcome["passed"]) for outcome in outcomes)


def _candidate_to_json(candidate: ExtractedCandidate, *, local_id: str) -> dict[str, object]:
    outcomes, is_valid = _validate_before_repair(candidate.source)
    return {
        "id": local_id,
        "source": candidate.source,
        "char_count": len(candidate.source),
        "extractor": candidate.extractor.value,
        "extractor_path": list(candidate.extractor_path),
        "notes": candidate.notes,
        "validation_source": VALIDATION_SOURCE,
        "validation": outcomes,
        "is_valid_before_repair": is_valid,
    }


def _candidate_count(rows: list[dict[str, object]], field: str) -> int:
    return sum(len(row[field]) for row in rows if isinstance(row[field], list))


def _valid_count(rows: list[dict[str, object]], field: str) -> int:
    count = 0
    for row in rows:
        candidates = row[field]
        if not isinstance(candidates, list):
            continue
        count += sum(
            1
            for candidate in candidates
            if isinstance(candidate, dict) and bool(candidate.get("is_valid_before_repair"))
        )
    return count


def sample_to_json(row: dict[str, object], *, index: int) -> dict[str, object]:
    raw_output = _field(row, "raw_output")
    extraction = run_extraction(raw_output, EXTRACTION_CONFIG)
    normalized_output = extraction.normalized_output
    passes = [
        {
            "extractor": extraction_pass.extractor.value,
            "raw_error": "",
            "normalized_error": "",
            "raw_candidates": [
                _candidate_to_json(
                    candidate, local_id=f"{extraction_pass.extractor.value}:raw:{index}"
                )
                for index, candidate in enumerate(extraction_pass.raw_candidates)
            ],
            "normalized_candidates": [
                _candidate_to_json(
                    candidate,
                    local_id=f"{extraction_pass.extractor.value}:normalized:{index}",
                )
                for index, candidate in enumerate(extraction_pass.normalized_candidates)
            ],
        }
        for extraction_pass in extraction.passes
    ]
    raw_count = _candidate_count(passes, "raw_candidates")
    normalized_count = _candidate_count(passes, "normalized_candidates")
    valid_raw_count = _valid_count(passes, "raw_candidates")
    valid_normalized_count = _valid_count(passes, "normalized_candidates")
    return {
        "index": index,
        "sample_id": _field(row, "sample_id"),
        "run_id": _optional_field(row, "run_id"),
        "task_id": _field(row, "task_id"),
        "entry_point": _field(row, "entry_point"),
        "occurrence_count": _int_field(row, "provenance_occurrence_count", 1),
        "source": _field(row, "provenance_source"),
        "model": _optional_field(row, "provenance_model"),
        "pool_name": _optional_field(row, "provenance_pool_name"),
        "pool_attempt_id": _optional_field(row, "provenance_pool_attempt_id"),
        "input": {
            "raw_output": raw_output,
            "normalized_output": normalized_output,
            "normalized_changed": normalized_output != raw_output,
        },
        "passes": passes,
        "summary": {
            "raw_candidate_count": raw_count,
            "normalized_candidate_count": normalized_count,
            "total_candidate_count": raw_count + normalized_count,
            "valid_before_repair_raw_count": valid_raw_count,
            "valid_before_repair_normalized_count": valid_normalized_count,
            "valid_before_repair_total_count": valid_raw_count + valid_normalized_count,
            "empty_extractor_count": sum(
                1
                for extractor_row in passes
                if not extractor_row["raw_candidates"]
                and not extractor_row["normalized_candidates"]
            ),
        },
    }


def build_ladder_data(
    rows: list[dict[str, object]],
    *,
    attempts_path: Path,
    start_index: int,
) -> dict[str, object]:
    return {
        "meta": {
            "source_attempts_path": str(attempts_path),
            "generated_at": datetime.now(UTC).isoformat(),
            "row_count": len(rows),
            "start_index": start_index,
            "extractor_order": [name.value for name, _ in EXTRACTION_CATALOG],
        },
        "samples": [
            sample_to_json(row, index=start_index + offset) for offset, row in enumerate(rows)
        ],
    }


def _select_rows(
    rows: list[dict[str, object]],
    *,
    task_id: str | None,
    start_index: int,
    limit: int | None,
) -> list[dict[str, object]]:
    selected = rows
    if task_id is not None:
        selected = [row for row in selected if _field(row, "task_id") == task_id]
    if start_index < 0 or start_index > len(selected):
        raise ValueError(f"--start-index {start_index} out of range for {len(selected)} row(s)")
    selected = selected[start_index:]
    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be non-negative")
        selected = selected[:limit]
    return selected


def render_html(data: dict[str, object]) -> str:
    template = TEMPLATE_PATH.read_text()
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return template.replace(JSON_PLACEHOLDER, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a static first-stage extraction ladder.")
    parser.add_argument("--attempts", type=Path, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--start-index", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.attempts.is_file():
        raise SystemExit(f"attempts export not found: {args.attempts}")
    rows = _select_rows(
        _read_attempt_rows(args.attempts),
        task_id=args.task_id,
        start_index=args.start_index,
        limit=args.limit,
    )
    data = build_ladder_data(rows, attempts_path=args.attempts, start_index=args.start_index)
    args.out.write_text(render_html(data))
    print(f"wrote {args.out} ({len(rows)} samples)")


if __name__ == "__main__":
    main()
