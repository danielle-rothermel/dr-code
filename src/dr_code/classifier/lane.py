"""Provider-neutral classification lane and command adapter."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import shlex
import shutil
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Final, Protocol

from dr_exec import (
    PROCESS_BOUNDARY_ONLY,
    Attribution,
    BudgetAxis,
    Budgets,
    EnvironmentGrant,
    OutputBudget,
    OverflowPolicy,
    Records,
    RunResult,
    run_untrusted_command,
)
from pydantic import BaseModel, ConfigDict, StrictStr, ValidationError

from dr_code.classifier.taxonomy import FailureFamily, is_valid_label
from dr_code.eval.identity import identity_hash_for

SUBSCRIPTION_LANE_ADAPTER_VERSION = "pi-subscription-lane-v4"
LANE_POLICY_IDENTITY_SCHEMA = "dr_code.failure_classifier.lane_policy"
MAX_RESPONSE_ERROR_CHARS = 512
MAX_IMPLEMENTATION_CLOSURE_FILES = 10_000
MAX_IMPLEMENTATION_CLOSURE_BYTES = 512 * 1024 * 1024

MAX_SUBSCRIPTION_OUTPUT_BYTES: Final[int] = 1024 * 1024
"""The lane's own transport output bound: a subscription response over this
is a runaway provider, scored as a transport failure, not a model answer."""

MAX_SUBSCRIPTION_INPUT_BYTES: Final[int] = 4 * 1024 * 1024
"""The lane's own transport input bound on the prompt delivered over stdin."""

_BEHAVIOR_ENVIRONMENT_NAMES = frozenset(
    {
        "AWS_PROFILE",
        "AWS_REGION",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_BASE_URL",
        "AZURE_OPENAI_DEPLOYMENT_NAME_MAP",
        "AZURE_OPENAI_RESOURCE_NAME",
        "CLOUDFLARE_ACCOUNT_ID",
        "CLOUDFLARE_GATEWAY_ID",
        "GOOGLE_CLOUD_LOCATION",
        "GOOGLE_CLOUD_PROJECT",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "NODE_EXTRA_CA_CERTS",
        "NODE_OPTIONS",
        "PATH",
        "PI_CODING_AGENT_DIR",
        "PI_OFFLINE",
        "PI_PACKAGE_DIR",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TMPDIR",
    }
)
_SECRET_ENVIRONMENT_NAMES = frozenset(
    {
        "ALL_PROXY",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_SECRET_ACCESS_KEY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
    }
)
_SECRET_ENVIRONMENT_SUFFIXES = (
    "_ACCESS_KEY_ID",
    "_API_KEY",
    "_AUTH",
    "_BEARER_TOKEN",
    "_COOKIE",
    "_CREDENTIAL",
    "_CREDENTIALS",
    "_PASSWORD",
    "_SECRET",
    "_SESSION",
    "_TOKEN",
)

LanePolicyScalar = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class LanePolicy:
    """Immutable, canonical transport and generation policy coordinates."""

    adapter: str
    transport: tuple[tuple[str, LanePolicyScalar], ...] = ()
    generation: tuple[tuple[str, LanePolicyScalar], ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.adapter, str)
            or not self.adapter
            or self.adapter.strip() != self.adapter
        ):
            raise ValueError(
                "lane policy adapter must be nonblank and trimmed"
            )
        object.__setattr__(
            self,
            "transport",
            _canonical_policy_settings(self.transport, "transport"),
        )
        object.__setattr__(
            self,
            "generation",
            _canonical_policy_settings(self.generation, "generation"),
        )


class Lane(Protocol):
    """A completion boundary with explicit resumable policy identity."""

    provider: str
    model: str
    policy: LanePolicy

    def complete(self, prompt: str) -> str:
        """Return one raw response or raise ``LaneTransportError``."""


class TransportFailureKind(StrEnum):
    TIMEOUT = "timeout"
    OUTPUT_LIMIT = "output_limit"
    NONZERO_EXIT = "nonzero_exit"
    MISSING_EXECUTABLE = "missing_executable"
    OPERATING_SYSTEM = "operating_system"


class LaneTransportError(RuntimeError):
    """A typed failure before a model response was available."""

    def __init__(self, kind: TransportFailureKind, detail: str) -> None:
        super().__init__(detail)
        self.kind = kind
        self.detail = detail


@dataclass(frozen=True, slots=True)
class SubscriptionLane:
    """Invoke a subscription-capable command with explicit coordinates."""

    provider: str
    model: str
    timeout_seconds: float
    executable: str = "pi"
    _executable_sha256: str = field(init=False, repr=False, compare=False)
    _environment: EnvironmentGrant = field(
        init=False,
        repr=False,
        compare=False,
    )
    _environment_identity: str = field(
        init=False,
        repr=False,
        compare=False,
    )
    _implementation_identity: str = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for value, name in (
            (self.provider, "provider"),
            (self.model, "model"),
            (self.executable, "executable"),
        ):
            if (
                not isinstance(value, str)
                or not value
                or value.strip() != value
            ):
                raise ValueError(f"{name} must be a nonblank trimmed string")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError(
                "timeout_seconds must be a finite positive number"
            )
        resolved = shutil.which(self.executable)
        if resolved is None:
            raise ValueError(f"executable not found: {self.executable}")
        executable_path = Path(resolved).resolve()
        if not executable_path.is_file() or not os.access(
            executable_path, os.X_OK
        ):
            raise ValueError(
                f"executable is not an executable file: {executable_path}"
            )
        environment, environment_identity = _capture_environment()
        implementation_identity = _implementation_identity_for(
            executable_path,
            path_environment=dict(environment.resolved).get("PATH"),
        )
        object.__setattr__(self, "executable", str(executable_path))
        object.__setattr__(
            self,
            "_executable_sha256",
            _file_sha256(executable_path),
        )
        object.__setattr__(self, "_environment", environment)
        object.__setattr__(
            self,
            "_environment_identity",
            environment_identity,
        )
        object.__setattr__(
            self,
            "_implementation_identity",
            implementation_identity,
        )

    def complete(self, prompt: str) -> str:
        self._verify_implementation()
        command = [
            self.executable,
            "-p",
            "--provider",
            self.provider,
            "--model",
            self.model,
            "--no-approve",
            "--no-context-files",
            "--no-extensions",
            "--no-prompt-templates",
            "--no-skills",
            "--no-themes",
            "--no-tools",
            "--no-session",
            "--mode",
            "text",
            "--thinking",
            "off",
        ]
        result = run_untrusted_command(
            command,
            profile=PROCESS_BOUNDARY_ONLY,
            budgets=Budgets(
                wall_clock=self.timeout_seconds,
                output=OutputBudget(
                    limit_bytes=MAX_SUBSCRIPTION_OUTPUT_BYTES,
                    overflow_policy=OverflowPolicy.FAIL,
                ),
                input=MAX_SUBSCRIPTION_INPUT_BYTES,
            ),
            records=Records.none(),
            input_text=prompt,
            environment=self._environment,
        )
        return self._response_from(result)

    def _response_from(self, result: RunResult) -> str:
        """Map a spawned run's attribution onto a response or typed failure.

        Outcomes are data: the mapping keys on ``result.outcome.attribution``,
        never on an exception class. Absence is a missing executable; a
        machine failure preserves its errno so EACCES stays distinguishable;
        a nonzero payload exit delivers its returncode and bounded stderr; a
        budget outcome is typed by the axis it violated.
        """
        outcome = result.outcome
        attribution = outcome.attribution
        if attribution is Attribution.PAYLOAD:
            if result.returncode:
                detail = result.stderr.strip()[:500]
                raise LaneTransportError(
                    TransportFailureKind.NONZERO_EXIT,
                    f"command exited {result.returncode}"
                    + (f": {detail}" if detail else ""),
                )
            return result.stdout
        if attribution is Attribution.ABSENCE:
            raise LaneTransportError(
                TransportFailureKind.MISSING_EXECUTABLE,
                f"executable not found: {self.executable}",
            )
        if attribution is Attribution.BUDGET:
            if outcome.violated_axis is BudgetAxis.WALL_CLOCK:
                raise LaneTransportError(
                    TransportFailureKind.TIMEOUT,
                    "command timed out after "
                    f"{self.timeout_seconds:g} seconds",
                )
            if outcome.violated_axis is BudgetAxis.OUTPUT:
                raise LaneTransportError(
                    TransportFailureKind.OUTPUT_LIMIT,
                    "command output exceeded limit",
                )
            raise LaneTransportError(
                TransportFailureKind.OPERATING_SYSTEM,
                "command exceeded its "
                f"{_violated_axis_name(outcome.violated_axis)} budget",
            )
        if (
            attribution is Attribution.MACHINE
            and outcome.spawn_errno == errno.EACCES
        ):
            raise LaneTransportError(
                TransportFailureKind.OPERATING_SYSTEM,
                "permission denied starting the command",
            )
        raise LaneTransportError(
            TransportFailureKind.OPERATING_SYSTEM,
            f"command did not complete: {attribution.value} failure",
        )

    def _verify_implementation(self) -> None:
        try:
            current_identity = _implementation_identity_for(
                Path(self.executable),
                path_environment=dict(self._environment.resolved).get("PATH"),
            )
        except FileNotFoundError as exc:
            raise LaneTransportError(
                TransportFailureKind.MISSING_EXECUTABLE,
                f"executable not found: {self.executable}",
            ) from exc
        except (OSError, ValueError) as exc:
            raise LaneTransportError(
                TransportFailureKind.OPERATING_SYSTEM,
                f"could not authenticate implementation closure: {exc}",
            ) from exc
        if current_identity != self._implementation_identity:
            raise LaneTransportError(
                TransportFailureKind.OPERATING_SYSTEM,
                "implementation closure changed after policy capture",
            )

    @property
    def policy(self) -> LanePolicy:
        """Return every fixed transport and generation policy coordinate."""
        self._verify_implementation()
        return LanePolicy(
            adapter=SUBSCRIPTION_LANE_ADAPTER_VERSION,
            transport=(
                ("executable", self.executable),
                ("executable_sha256", self._executable_sha256),
                ("environment_identity", self._environment_identity),
                ("environment_policy", "pi-allowlisted-environment-v1"),
                ("implementation_identity", self._implementation_identity),
                (
                    "implementation_policy",
                    "resolved-package-runtime-closure-v1",
                ),
                ("prompt_delivery", "stdin"),
                ("timeout_seconds", float(self.timeout_seconds)),
            ),
            generation=(
                ("approval", "disabled"),
                ("context_files_enabled", False),
                ("extensions_enabled", False),
                ("mode", "text"),
                ("prompt_templates_enabled", False),
                ("session_enabled", False),
                ("skills_enabled", False),
                ("themes_enabled", False),
                ("thinking", "off"),
                ("tools_enabled", False),
            ),
        )


def _violated_axis_name(axis: BudgetAxis | None) -> str:
    return axis.value if axis is not None else "resource"


def _capture_environment() -> tuple[EnvironmentGrant, str]:
    """Freeze the granted process settings without persisting secret values.

    The grant is a deny-by-default ``named`` snapshot resolved at
    construction, so every later run receives exactly this environment.
    ``environment_identity`` is derived from the same allowlist the grant is
    built from: secret names contribute only their presence, never a value,
    so a secret rotation leaves the identity fixed while a changed non-secret
    setting moves it.
    """
    allowlist = tuple(
        sorted(
            name
            for name in os.environ
            if name in _BEHAVIOR_ENVIRONMENT_NAMES
            or _is_secret_environment_name(name)
        )
    )
    grant = EnvironmentGrant.named(allowlist)
    identity_payload = {
        "credentials_present": sorted(
            name for name in allowlist if _is_secret_environment_name(name)
        ),
        "settings": {
            name: os.environ[name]
            for name in allowlist
            if not _is_secret_environment_name(name)
        },
    }
    return (
        grant,
        identity_hash_for(
            schema="dr_code.failure_classifier.lane_environment",
            payload=identity_payload,
        ),
    )


def _is_secret_environment_name(name: str) -> bool:
    upper = name.upper()
    return upper in _SECRET_ENVIRONMENT_NAMES or upper.endswith(
        _SECRET_ENVIRONMENT_SUFFIXES
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_identity_for(
    executable: Path,
    *,
    path_environment: str | None,
) -> str:
    executable = executable.resolve(strict=True)
    package_root = _nearest_package_root(executable)
    closure_root = package_root if package_root is not None else executable
    files = (
        tuple(_implementation_files(package_root))
        if package_root is not None
        else (executable,)
    )
    coordinates: list[dict[str, object]] = []
    total_bytes = 0
    for index, path in enumerate(files, start=1):
        if index > MAX_IMPLEMENTATION_CLOSURE_FILES:
            raise ValueError(
                "implementation closure exceeds the file-count limit"
            )
        sha256, size = _stable_file_coordinate(path)
        total_bytes += size
        if total_bytes > MAX_IMPLEMENTATION_CLOSURE_BYTES:
            raise ValueError("implementation closure exceeds the byte limit")
        coordinates.append(
            {
                "path": (
                    path.relative_to(closure_root).as_posix()
                    if package_root is not None
                    else executable.name
                ),
                "sha256": sha256,
                "size": size,
            }
        )
    runtime = _resolved_shebang_runtime(
        executable,
        path_environment=path_environment,
    )
    runtime_coordinate: dict[str, object] | None = None
    if runtime is not None:
        runtime_sha256, runtime_size = _stable_file_coordinate(runtime)
        total_bytes += runtime_size
        if total_bytes > MAX_IMPLEMENTATION_CLOSURE_BYTES:
            raise ValueError("implementation closure exceeds the byte limit")
        runtime_coordinate = {
            "path": str(runtime),
            "sha256": runtime_sha256,
            "size": runtime_size,
        }
    return identity_hash_for(
        schema="dr_code.failure_classifier.implementation_closure",
        payload={
            "executable": str(executable),
            "files": coordinates,
            "package_root": (
                str(package_root) if package_root is not None else None
            ),
            "runtime": runtime_coordinate,
        },
    )


def _nearest_package_root(executable: Path) -> Path | None:
    for parent in (executable.parent, *executable.parents):
        if (parent / "package.json").is_file():
            return parent
    return None


def _implementation_files(package_root: Path) -> Iterator[Path]:
    paths: list[Path] = []
    for raw_root, directory_names, file_names in os.walk(package_root):
        directory_names[:] = sorted(
            name
            for name in directory_names
            if name not in {".git", "node_modules"}
        )
        root = Path(raw_root)
        for name in sorted(file_names):
            path = root / name
            if path.is_file() and not path.is_symlink():
                paths.append(path)
    dependency_lock = package_root / "node_modules" / ".package-lock.json"
    if dependency_lock.is_file() and not dependency_lock.is_symlink():
        paths.append(dependency_lock)
    yield from sorted(
        paths,
        key=lambda path: path.relative_to(package_root).as_posix(),
    )


def _resolved_shebang_runtime(
    executable: Path,
    *,
    path_environment: str | None,
) -> Path | None:
    with executable.open("rb") as stream:
        first_line = stream.readline(4_096)
    if not first_line.startswith(b"#!"):
        return None
    try:
        command = shlex.split(first_line[2:].decode("utf-8").strip())
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("executable has an invalid shebang") from exc
    if not command:
        raise ValueError("executable has an empty shebang")
    runtime = Path(command[0])
    if runtime.name == "env":
        names = tuple(
            value for value in command[1:] if not value.startswith("-")
        )
        if not names:
            raise ValueError("env shebang does not name a runtime")
        resolved = shutil.which(names[0], path=path_environment)
        if resolved is None:
            raise ValueError(f"shebang runtime not found: {names[0]}")
        runtime = Path(resolved)
    if not runtime.is_absolute():
        resolved = shutil.which(str(runtime), path=path_environment)
        if resolved is None:
            raise ValueError(f"shebang runtime not found: {runtime}")
        runtime = Path(resolved)
    runtime = runtime.resolve(strict=True)
    if not runtime.is_file() or not os.access(runtime, os.X_OK):
        raise ValueError(f"shebang runtime is not executable: {runtime}")
    return runtime


def _stable_file_coordinate(path: Path) -> tuple[str, int]:
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        digest = hashlib.sha256()
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    coordinates = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    if coordinates != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError(f"implementation file changed while hashing: {path}")
    return digest.hexdigest(), before.st_size


def lane_policy(lane: Lane) -> LanePolicy:
    """Require the lane's complete immutable resumability policy."""
    try:
        policy = lane.policy
    except AttributeError as exc:
        lane_type = type(lane)
        raise ValueError(
            "classification lane must expose a typed canonical policy; "
            f"{lane_type.__module__}.{lane_type.__qualname__} has none"
        ) from exc
    if not isinstance(policy, LanePolicy):
        raise ValueError(
            "classification lane policy must be a LanePolicy instance"
        )
    return policy


def lane_policy_identity(policy: LanePolicy) -> str:
    """Hash the complete canonical lane policy."""
    return identity_hash_for(
        schema=LANE_POLICY_IDENTITY_SCHEMA,
        payload={
            "adapter": policy.adapter,
            "transport": dict(policy.transport),
            "generation": dict(policy.generation),
        },
    )


def _canonical_policy_settings(
    settings: tuple[tuple[str, LanePolicyScalar], ...],
    name: str,
) -> tuple[tuple[str, LanePolicyScalar], ...]:
    if not isinstance(settings, tuple):
        raise ValueError(f"lane policy {name} settings must be a tuple")
    values: dict[str, LanePolicyScalar] = {}
    for item in settings:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(
                f"lane policy {name} settings must contain key/value tuples"
            )
        key, value = item
        if not isinstance(key, str) or not key or key.strip() != key:
            raise ValueError(
                f"lane policy {name} setting names must be nonblank and trimmed"
            )
        if key in values:
            raise ValueError(f"duplicate lane policy {name} setting: {key}")
        if not (value is None or isinstance(value, (str, int, float, bool))):
            raise ValueError(
                f"lane policy {name} setting {key!r} is not JSON-safe"
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(
                f"lane policy {name} setting {key!r} must be finite"
            )
        values[key] = value
    return tuple(sorted(values.items()))


class LabelResponse(BaseModel):
    """Validated model output at the JSON response boundary."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    label: StrictStr
    rationale: StrictStr


def parse_label_response(raw: str, family: FailureFamily) -> LabelResponse:
    """Parse one complete strict JSON object for a specific taxonomy."""
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise _response_error(f"response is not strict JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise _response_error("response must be a JSON object")
    try:
        response = LabelResponse.model_validate(payload)
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_context=False)
        safe_errors = [
            {
                "location": [str(part) for part in error["loc"]],
                "type": error["type"],
            }
            for error in errors[:8]
        ]
        omitted = len(errors) - len(safe_errors)
        raise _response_error(
            "response schema is invalid: "
            + json.dumps(
                {"errors": safe_errors, "omitted": omitted},
                separators=(",", ":"),
                sort_keys=True,
            )
        ) from exc
    if not is_valid_label(family, response.label):
        raise _response_error(f"label is outside the {family.value} taxonomy")
    rationale = response.rationale
    if (
        not rationale
        or rationale.strip() != rationale
        or "\n" in rationale
        or "\r" in rationale
    ):
        raise _response_error(
            "rationale must be a nonblank, trimmed, single-line string"
        )
    if len(rationale) > 280:
        raise _response_error("rationale must be at most 280 characters")
    return response


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object key")
        value[key] = item
    return value


def _reject_constant(value: str) -> object:
    raise ValueError(f"invalid JSON number: {value}")


def _response_error(detail: str) -> ValueError:
    marker = "...[error truncated]"
    if len(detail) > MAX_RESPONSE_ERROR_CHARS:
        detail = detail[: MAX_RESPONSE_ERROR_CHARS - len(marker)] + marker
    return ValueError(detail)


__all__ = (
    "LabelResponse",
    "Lane",
    "LanePolicy",
    "LaneTransportError",
    "MAX_RESPONSE_ERROR_CHARS",
    "MAX_SUBSCRIPTION_INPUT_BYTES",
    "MAX_SUBSCRIPTION_OUTPUT_BYTES",
    "SubscriptionLane",
    "TransportFailureKind",
    "lane_policy",
    "lane_policy_identity",
    "parse_label_response",
)
