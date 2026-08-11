from __future__ import annotations

from pathlib import Path
import sys

from dr_exec import ProcessExecutor

from dr_code.core.execution.executor import host_process_executor


def test_host_process_executor_is_caller_provisioning_only(
    tmp_path: Path,
) -> None:
    executor = host_process_executor(
        tmp_path / "records",
        runtime_executable=Path(sys.executable),
    )

    assert isinstance(executor, ProcessExecutor)
