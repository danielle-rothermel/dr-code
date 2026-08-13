"""Caller provisioning for the production dr-exec process executor."""

from __future__ import annotations

from pathlib import Path

from dr_exec import (
    DirectoryRunStore,
    IsolatedHostPythonRuntime,
    ProcessExecutor,
)


def host_process_executor(
    record_root: Path,
    *,
    runtime_executable: Path,
) -> ProcessExecutor:
    """Provision a process executor for a caller-selected Python runtime."""

    return ProcessExecutor(
        runtime=IsolatedHostPythonRuntime(runtime_executable),
        run_store=DirectoryRunStore(root=record_root),
    )


__all__ = ["host_process_executor"]
