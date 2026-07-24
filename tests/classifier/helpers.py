from pathlib import Path

from dr_code.classifier import classify as classify_module
from dr_code.classifier.records import ExperimentHeaderRecord, ItemRecord


def capture_artifact_for_test(
    path: Path,
) -> tuple[ExperimentHeaderRecord, tuple[ItemRecord, ...]]:
    """Inspect an artifact through the bounded production reader and spool."""
    with classify_module._RecordSpool(path.parent) as spool:
        captured = classify_module._load_artifact_capture(
            path,
            on_record=spool.add_existing,
        )
        if captured is None:
            raise FileNotFoundError(path)
        spool.select_all()
        return captured[0], tuple(spool.iter_selected())
