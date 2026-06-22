"""Pre-flight checks for pipeline runs."""

from __future__ import annotations

import os
import shutil
import socket
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field

from dr_code.models.base import FrozenModel
from dr_code.pipeline.seed import DEFAULT_DUMP_DIR, DEFAULT_PROOF_INDICES


class PreflightReport(FrozenModel):
    """Result of infrastructure pre-flight checks."""

    ok: bool
    checks: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def raise_if_failed(self) -> None:
        if not self.ok:
            msg = "Pre-flight failed:\n" + "\n".join(
                f"  - {e}" for e in self.errors
            )
            raise RuntimeError(msg)


def run_preflight(
    *,
    dump_dir: Path | str = DEFAULT_DUMP_DIR,
    task_indices: list[int] | tuple[int, ...] = DEFAULT_PROOF_INDICES,
    require_docker: bool = False,
    require_dump: bool = True,
) -> PreflightReport:
    """Verify RabbitMQ, Mongo, and dump artifacts."""
    checks: list[str] = []
    errors: list[str] = []
    _check_tcp(
        "RabbitMQ",
        os.environ.get("AMQP_URL", "amqp://guest:guest@localhost:5672/"),
        5672,
        checks,
        errors,
    )
    _check_tcp(
        "MongoDB",
        os.environ.get("MONGODB_URL", "mongodb://localhost:27017/dr_queues"),
        27017,
        checks,
        errors,
    )
    if require_docker:
        _check_docker(checks, errors)
    if require_dump:
        _check_dump(Path(dump_dir), task_indices, checks, errors)
    return PreflightReport(
        ok=not errors,
        checks=checks,
        errors=errors,
    )


def _check_tcp(
    label: str,
    url: str,
    default_port: int,
    checks: list[str],
    errors: list[str],
) -> None:
    parsed = urlparse(url.replace("amqp://", "http://"))
    host = parsed.hostname or "localhost"
    port = parsed.port or default_port
    try:
        with socket.create_connection((host, port), timeout=3):
            checks.append(f"{label} reachable at {host}:{port}")
    except OSError as exc:
        errors.append(f"{label} not reachable at {host}:{port}: {exc}")


def _check_docker(checks: list[str], errors: list[str]) -> None:
    if shutil.which("docker") is None:
        errors.append("docker CLI not found on PATH")
        return
    import subprocess

    result = subprocess.run(
        ["docker", "info"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        errors.append("docker daemon not available (docker info failed)")
    else:
        checks.append("Docker daemon available")


def _check_dump(
    dump_dir: Path,
    task_indices: list[int] | tuple[int, ...],
    checks: list[str],
    errors: list[str],
) -> None:
    if not dump_dir.is_dir():
        errors.append(f"Dump directory missing: {dump_dir}")
        return
    per_elem = dump_dir / "per_elem"
    for index in task_indices:
        dedup = per_elem / f"human_eval-{index}-decode-dedup.jsonl"
        parquet = per_elem / f"human_eval-{index}-decode.parquet"
        if not dedup.is_file():
            errors.append(f"Missing dedup file: {dedup}")
        if not parquet.is_file():
            errors.append(f"Missing parquet file: {parquet}")
    if not errors:
        checks.append(
            f"Dump artifacts present for task indices {list(task_indices)}",
        )
