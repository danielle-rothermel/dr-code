"""Single entry point for invoking external tools.

All subprocess work in `code_eval` goes through `SubprocessRunner`. This
ensures:

- A consistent timeout policy.
- stdin/stdout piping (no temp files).
- Tool versions captured once at construction time.
- Failures become structured `Diagnostic`s, not exceptions.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

from code_eval.models.base import FrozenModel
from code_eval.names import DEFAULT_SUBPROCESS_TIMEOUT_S


class SubprocessResult(FrozenModel):
    """Outcome of one subprocess invocation."""

    tool: str
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    tool_found: bool
    duration_s: float

    @property
    def ok(self) -> bool:
        return self.tool_found and not self.timed_out and self.returncode == 0


class SubprocessRunner(FrozenModel):
    """Pinned-tool subprocess invoker."""

    timeout_s: float = DEFAULT_SUBPROCESS_TIMEOUT_S

    def run(
        self,
        tool: str,
        args: tuple[str, ...],
        stdin_text: str | None = None,
    ) -> SubprocessResult:
        """Invoke `tool` with `args`, optionally piping `stdin_text`."""
        import time

        exe = shutil.which(tool)
        if exe is None:
            return SubprocessResult(
                tool=tool,
                args=args,
                returncode=-1,
                stdout="",
                stderr=f"tool not found on PATH: {tool}",
                timed_out=False,
                tool_found=False,
                duration_s=0.0,
            )

        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [exe, *args],
                input=stdin_text,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return SubprocessResult(
                tool=tool,
                args=args,
                returncode=-1,
                stdout=exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                stderr=(
                    exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
                ),
                timed_out=True,
                tool_found=True,
                duration_s=time.perf_counter() - start,
            )

        return SubprocessResult(
            tool=tool,
            args=args,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            timed_out=False,
            tool_found=True,
            duration_s=time.perf_counter() - start,
        )


def discover_version(
    tool: str,
    version_arg: str = "--version",
    timeout_s: float = DEFAULT_SUBPROCESS_TIMEOUT_S,
) -> str | None:
    """Return the version string for `tool`, or None if not installed."""
    result = SubprocessRunner(timeout_s=timeout_s).run(tool, (version_arg,))
    if not result.tool_found or result.timed_out:
        return None
    out = (result.stdout + result.stderr).strip().splitlines()
    return out[0] if out else None


def python_version() -> str:
    """Return the running Python's version string."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
