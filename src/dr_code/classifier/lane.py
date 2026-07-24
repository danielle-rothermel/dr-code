"""Provider-neutral classification lane and command adapter."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shlex
import shutil
import stat
import sys
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Protocol, cast

from pydantic import BaseModel, ConfigDict, StrictStr, ValidationError

from dr_code.classifier.taxonomy import FailureFamily, is_valid_label
from dr_code.eval.identity import identity_hash_for
from dr_code.execution import (
    SubprocessError,
    SubprocessOutputLimitError,
    SubprocessStartError,
    SubprocessTimeoutError,
    run_subprocess,
)

SUBSCRIPTION_LANE_ADAPTER_VERSION = "pi-subscription-lane-v5"
LANE_POLICY_IDENTITY_SCHEMA = "dr_code.failure_classifier.lane_policy"
MAX_RESPONSE_ERROR_CHARS = 512
MAX_IMPLEMENTATION_CLOSURE_ENTRIES = 50_000
MAX_IMPLEMENTATION_CLOSURE_BYTES = 512 * 1024 * 1024
_COPY_CHUNK_BYTES = 1024 * 1024

# Persisted implementation-identity contract. Tests pin the schema, layout,
# logical paths, and payload keys so stored experiment identity cannot drift
# with an internal field rename.
_IMPLEMENTATION_IDENTITY_SCHEMA = (
    "dr_code.failure_classifier.implementation_closure"
)
_IMPLEMENTATION_SNAPSHOT_LAYOUT = "captured-node-package-v1"
_PACKAGE_SNAPSHOT_DIRECTORY = "package"
_RUNTIME_SNAPSHOT_DIRECTORY = "runtime"
_SUPPORTED_LOCK_NAMES = (
    "npm-shrinkwrap.json",
    "package-lock.json",
    "node_modules/.package-lock.json",
)

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
        "NODE_PATH",
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


class _ImplementationEntryKind(StrEnum):
    DIRECTORY = "directory"
    REGULAR = "regular"
    SYMLINK = "symlink"


@dataclass(frozen=True, slots=True)
class _StableStatCoordinate:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True, slots=True)
class _CapturedImplementationEntry:
    source_path: str
    snapshot_path: str
    kind: _ImplementationEntryKind
    mode: int
    link_target: str | None
    size: int
    sha256: str | None


@dataclass(frozen=True, slots=True)
class _ResolvedRuntime:
    source: Path
    command_name: str


@dataclass(frozen=True, slots=True)
class _ImplementationSnapshot:
    source_executable: Path
    source_package_root: Path | None
    root: Path
    entrypoint: Path
    runtime: Path | None
    entries: tuple[_CapturedImplementationEntry, ...]
    entrypoint_sha256: str
    identity: str
    _directory: tempfile.TemporaryDirectory[str] = field(
        repr=False,
        compare=False,
    )

    @property
    def command_prefix(self) -> tuple[str, ...]:
        if self.runtime is None:
            return (str(self.entrypoint),)
        return (str(self.runtime), str(self.entrypoint))


@dataclass(slots=True)
class _CaptureBudget:
    entries: int = 0
    bytes: int = 0

    def claim_entry(self) -> None:
        self.entries += 1
        if self.entries > MAX_IMPLEMENTATION_CLOSURE_ENTRIES:
            raise ValueError(
                "implementation closure exceeds the entry-count limit"
            )

    def claim_bytes(self, declared_size: int) -> None:
        if declared_size < 0:
            raise ValueError(
                "implementation entry has a negative declared size"
            )
        if self.bytes + declared_size > MAX_IMPLEMENTATION_CLOSURE_BYTES:
            raise ValueError("implementation closure exceeds the byte limit")
        self.bytes += declared_size


@dataclass(frozen=True, slots=True)
class _ScanDirectory:
    source: Path
    destination: Path
    source_relative: PurePosixPath
    snapshot_relative: PurePosixPath
    expected: _StableStatCoordinate


@dataclass(frozen=True, slots=True)
class _ValidateDirectory:
    source: Path
    expected: _StableStatCoordinate


@dataclass(frozen=True, slots=True)
class SubscriptionLane:
    """Invoke a subscription-capable command with explicit coordinates."""

    provider: str
    model: str
    timeout_seconds: float
    executable: str = "pi"
    _executable_sha256: str = field(init=False, repr=False, compare=False)
    _environment: tuple[tuple[str, str], ...] = field(
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
    _implementation_snapshot: _ImplementationSnapshot = field(
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
        implementation_snapshot = _capture_implementation_snapshot(
            executable_path,
            path_environment=dict(environment).get("PATH"),
        )
        object.__setattr__(self, "executable", str(executable_path))
        object.__setattr__(
            self,
            "_executable_sha256",
            implementation_snapshot.entrypoint_sha256,
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
            implementation_snapshot.identity,
        )
        object.__setattr__(
            self, "_implementation_snapshot", implementation_snapshot
        )

    def complete(self, prompt: str) -> str:
        snapshot = self._implementation_snapshot
        command = [
            *snapshot.command_prefix,
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
        try:
            completed = run_subprocess(
                command=command,
                input_text=prompt,
                timeout_seconds=self.timeout_seconds,
                environment=_snapshot_environment(
                    snapshot,
                    self._environment,
                ),
            )
        except SubprocessTimeoutError as exc:
            raise LaneTransportError(
                TransportFailureKind.TIMEOUT,
                f"command timed out after {self.timeout_seconds:g} seconds",
            ) from exc
        except SubprocessOutputLimitError as exc:
            raise LaneTransportError(
                TransportFailureKind.OUTPUT_LIMIT,
                "command output exceeded limit",
            ) from exc
        except SubprocessStartError as exc:
            if isinstance(exc.__cause__, FileNotFoundError):
                raise LaneTransportError(
                    TransportFailureKind.MISSING_EXECUTABLE,
                    f"executable not found: {self.executable}",
                ) from exc
            raise LaneTransportError(
                TransportFailureKind.OPERATING_SYSTEM,
                str(exc),
            ) from exc
        except SubprocessError as exc:
            raise LaneTransportError(
                TransportFailureKind.OPERATING_SYSTEM,
                str(exc),
            ) from exc
        if completed.returncode:
            detail = completed.stderr.strip()[:500]
            raise LaneTransportError(
                TransportFailureKind.NONZERO_EXIT,
                f"command exited {completed.returncode}"
                + (f": {detail}" if detail else ""),
            )
        return completed.stdout

    @property
    def policy(self) -> LanePolicy:
        """Return every fixed transport and generation policy coordinate."""
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
                    "installed-node-runtime-closure-v2",
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


def _capture_environment() -> tuple[tuple[tuple[str, str], ...], str]:
    """Freeze required process settings without persisting secret values."""
    for name in ("NODE_OPTIONS", "NODE_PATH"):
        if os.environ.get(name):
            raise ValueError(
                f"{name} must be empty for authenticated provider execution"
            )
    inherited = {
        name: value
        for name, value in os.environ.items()
        if name in _BEHAVIOR_ENVIRONMENT_NAMES
        or _is_secret_environment_name(name)
    }
    identity_payload = {
        "credentials_present": sorted(
            name for name in inherited if _is_secret_environment_name(name)
        ),
        "settings": {
            name: value
            for name, value in inherited.items()
            if not _is_secret_environment_name(name)
        },
    }
    return (
        tuple(sorted(inherited.items())),
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


def _snapshot_environment(
    snapshot: _ImplementationSnapshot,
    captured: tuple[tuple[str, str], ...],
) -> dict[str, str]:
    environment = dict(captured)
    path_prefixes: list[str] = []
    if snapshot.runtime is not None:
        path_prefixes.append(str(snapshot.runtime.parent))
    if snapshot.source_package_root is not None:
        path_prefixes.append(
            str(
                snapshot.root
                / _PACKAGE_SNAPSHOT_DIRECTORY
                / "node_modules"
                / ".bin"
            )
        )
    original_path = environment.get("PATH")
    if original_path:
        path_prefixes.append(original_path)
    environment["PATH"] = os.pathsep.join(path_prefixes)
    return environment


def _capture_implementation_snapshot(
    executable: Path,
    *,
    path_environment: str | None,
) -> _ImplementationSnapshot:
    executable = executable.resolve(strict=True)
    package_root = _nearest_package_root(executable)
    directory = tempfile.TemporaryDirectory(
        prefix="dr-code-provider-snapshot-"
    )
    snapshot_root = Path(directory.name)
    package_destination = snapshot_root / _PACKAGE_SNAPSHOT_DIRECTORY
    budget = _CaptureBudget()
    captured_entries: list[_CapturedImplementationEntry] = []
    manifest_bytes: dict[str, bytes] = {}
    installed_package_roots: set[str] = set()
    package_directory_mode = 0o755
    try:
        if package_root is None:
            package_destination.mkdir(mode=0o700)
            observed = _stat_coordinate(
                executable,
                follow_symlinks=False,
            )
            if not stat.S_ISREG(observed.mode):
                raise ValueError(
                    f"executable is not a regular file: {executable}"
                )
            budget.claim_entry()
            source_relative = PurePosixPath(executable.name)
            snapshot_relative = (
                PurePosixPath(_PACKAGE_SNAPSHOT_DIRECTORY) / source_relative
            )
            captured_entry, _ = _copy_stable_regular_file(
                source=executable,
                destination=package_destination / executable.name,
                source_path=source_relative.as_posix(),
                snapshot_path=snapshot_relative.as_posix(),
                expected=observed,
                budget=budget,
                capture_bytes=False,
            )
            captured_entries.append(captured_entry)
            entrypoint_relative = source_relative
        else:
            try:
                entrypoint_relative = PurePosixPath(
                    executable.relative_to(package_root).as_posix()
                )
            except ValueError as exc:
                raise ValueError(
                    "resolved executable is outside its package root"
                ) from exc
            package_observed = _stat_coordinate(
                package_root,
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(package_observed.mode):
                raise ValueError(
                    f"implementation package root is not a directory: "
                    f"{package_root}"
                )
            package_directory_mode = stat.S_IMODE(package_observed.mode)
            package_destination.mkdir(mode=0o700)
            (
                package_entries,
                manifest_bytes,
                installed_package_roots,
            ) = _copy_package_tree(
                source_root=package_root,
                destination_root=package_destination,
                root_expected=package_observed,
                budget=budget,
            )
            captured_entries.extend(package_entries)
            _validate_installed_package_coordinates(
                manifests=manifest_bytes,
                installed_package_roots=installed_package_roots,
            )

        entrypoint_snapshot_relative = (
            PurePosixPath(_PACKAGE_SNAPSHOT_DIRECTORY) / entrypoint_relative
        )
        entrypoint = snapshot_root / entrypoint_snapshot_relative
        entrypoint_entry = next(
            (
                entry
                for entry in captured_entries
                if entry.snapshot_path
                == entrypoint_snapshot_relative.as_posix()
            ),
            None,
        )
        if (
            entrypoint_entry is None
            or entrypoint_entry.kind is not _ImplementationEntryKind.REGULAR
            or entrypoint_entry.sha256 is None
        ):
            raise ValueError(
                "captured implementation entrypoint is not a regular file"
            )

        resolved_runtime = _resolved_shebang_runtime(
            entrypoint,
            path_environment=path_environment,
        )
        captured_runtime: Path | None = None
        runtime_source: Path | None = None
        if (
            resolved_runtime is not None
            and resolved_runtime.command_name == "node"
        ):
            runtime_source = resolved_runtime.source
            runtime_destination_directory = (
                snapshot_root / _RUNTIME_SNAPSHOT_DIRECTORY
            )
            runtime_destination_directory.mkdir(mode=0o700)
            captured_runtime = (
                runtime_destination_directory / resolved_runtime.command_name
            )
            runtime_observed = _stat_coordinate(
                runtime_source,
                follow_symlinks=False,
            )
            if not stat.S_ISREG(runtime_observed.mode):
                raise ValueError(
                    f"shebang runtime is not a regular file: {runtime_source}"
                )
            budget.claim_entry()
            runtime_entry, _ = _copy_stable_regular_file(
                source=runtime_source,
                destination=captured_runtime,
                source_path=str(runtime_source),
                snapshot_path=(
                    PurePosixPath(_RUNTIME_SNAPSHOT_DIRECTORY)
                    / resolved_runtime.command_name
                ).as_posix(),
                expected=runtime_observed,
                budget=budget,
                capture_bytes=False,
            )
            captured_entries.append(runtime_entry)

        entries = tuple(
            sorted(captured_entries, key=lambda entry: entry.snapshot_path)
        )
        _validate_snapshot_symlinks(
            snapshot_root=snapshot_root,
            package_destination=package_destination,
            entries=entries,
        )
        _finalize_snapshot_permissions(
            snapshot_root=snapshot_root,
            package_directory_mode=package_directory_mode,
            entries=entries,
            has_runtime=captured_runtime is not None,
        )
        identity = _implementation_identity_for(
            executable=executable,
            package_root=package_root,
            entrypoint_path=entrypoint_snapshot_relative.as_posix(),
            runtime_source=runtime_source,
            runtime_path=(
                captured_runtime.relative_to(snapshot_root).as_posix()
                if captured_runtime is not None
                else None
            ),
            entries=entries,
        )
        return _ImplementationSnapshot(
            source_executable=executable,
            source_package_root=package_root,
            root=snapshot_root,
            entrypoint=entrypoint,
            runtime=captured_runtime,
            entries=entries,
            entrypoint_sha256=entrypoint_entry.sha256,
            identity=identity,
            _directory=directory,
        )
    except BaseException:
        directory.cleanup()
        raise


def _nearest_package_root(executable: Path) -> Path | None:
    for parent in (executable.parent, *executable.parents):
        package_manifest = parent / "package.json"
        try:
            observed = os.lstat(package_manifest)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError(
                f"implementation manifest is unreadable: {package_manifest}"
            ) from exc
        if not stat.S_ISREG(observed.st_mode):
            raise ValueError(
                "implementation root package manifest must be a regular file: "
                f"{package_manifest}"
            )
        return parent
    return None


def _copy_package_tree(
    *,
    source_root: Path,
    destination_root: Path,
    root_expected: _StableStatCoordinate,
    budget: _CaptureBudget,
) -> tuple[
    list[_CapturedImplementationEntry],
    dict[str, bytes],
    set[str],
]:
    captured_entries: list[_CapturedImplementationEntry] = []
    manifests: dict[str, bytes] = {}
    installed_package_roots: set[str] = set()
    work: list[_ScanDirectory | _ValidateDirectory] = [
        _ScanDirectory(
            source=source_root,
            destination=destination_root,
            source_relative=PurePosixPath(),
            snapshot_relative=PurePosixPath(_PACKAGE_SNAPSHOT_DIRECTORY),
            expected=root_expected,
        )
    ]
    while work:
        item = work.pop()
        if isinstance(item, _ValidateDirectory):
            _require_stable_directory(item.source, expected=item.expected)
            continue
        _require_stable_directory(item.source, expected=item.expected)
        try:
            with os.scandir(item.source) as stream:
                child_directories = _copy_directory_entries(
                    entries=stream,
                    directory=item,
                    source_root=source_root,
                    budget=budget,
                    captured_entries=captured_entries,
                    manifests=manifests,
                    installed_package_roots=installed_package_roots,
                )
        except OSError as exc:
            raise ValueError(
                f"implementation directory is unreadable: {item.source}"
            ) from exc
        work.append(
            _ValidateDirectory(source=item.source, expected=item.expected)
        )
        work.extend(reversed(child_directories))
    return captured_entries, manifests, installed_package_roots


def _copy_directory_entries(
    *,
    entries: Iterable[os.DirEntry[str]],
    directory: _ScanDirectory,
    source_root: Path,
    budget: _CaptureBudget,
    captured_entries: list[_CapturedImplementationEntry],
    manifests: dict[str, bytes],
    installed_package_roots: set[str],
) -> list[_ScanDirectory]:
    child_directories: list[_ScanDirectory] = []
    for entry in entries:
        source = directory.source / entry.name
        destination = directory.destination / entry.name
        source_relative = directory.source_relative / entry.name
        snapshot_relative = directory.snapshot_relative / entry.name
        observed = _stat_coordinate(source, follow_symlinks=False)
        budget.claim_entry()
        if stat.S_ISDIR(observed.mode):
            destination.mkdir(mode=0o700)
            captured_entries.append(
                _CapturedImplementationEntry(
                    source_path=source_relative.as_posix(),
                    snapshot_path=snapshot_relative.as_posix(),
                    kind=_ImplementationEntryKind.DIRECTORY,
                    mode=stat.S_IMODE(observed.mode),
                    link_target=None,
                    size=0,
                    sha256=None,
                )
            )
            package_coordinate = _installed_package_coordinate_for_directory(
                source_relative
            )
            if package_coordinate is not None:
                installed_package_roots.add(package_coordinate)
            child_directories.append(
                _ScanDirectory(
                    source=source,
                    destination=destination,
                    source_relative=source_relative,
                    snapshot_relative=snapshot_relative,
                    expected=observed,
                )
            )
            continue
        if stat.S_ISREG(observed.mode):
            capture_bytes = _is_captured_manifest(source_relative)
            captured_entry, content = _copy_stable_regular_file(
                source=source,
                destination=destination,
                source_path=source_relative.as_posix(),
                snapshot_path=snapshot_relative.as_posix(),
                expected=observed,
                budget=budget,
                capture_bytes=capture_bytes,
            )
            captured_entries.append(captured_entry)
            if content is not None:
                manifests[source_relative.as_posix()] = content
            continue
        if stat.S_ISLNK(observed.mode):
            captured_entries.append(
                _copy_stable_symlink(
                    source=source,
                    destination=destination,
                    source_root=source_root,
                    source_relative=source_relative,
                    snapshot_relative=snapshot_relative,
                    expected=observed,
                    budget=budget,
                )
            )
            continue
        raise ValueError(
            f"implementation closure contains a special file: {source}"
        )
    return child_directories


def _validate_installed_package_coordinates(
    *,
    manifests: Mapping[str, bytes],
    installed_package_roots: set[str],
) -> None:
    package_bytes = manifests.get("package.json")
    if package_bytes is None:
        raise ValueError("captured root package manifest is missing")
    selected_lock_name = next(
        (name for name in _SUPPORTED_LOCK_NAMES if name in manifests),
        None,
    )
    if selected_lock_name is None:
        raise ValueError(
            "implementation package has no supported captured npm lock"
        )
    package = _load_json_object_bytes(
        package_bytes,
        description="root package manifest",
    )
    lock = _load_json_object_bytes(
        manifests[selected_lock_name],
        description=f"implementation lock {selected_lock_name}",
    )
    lockfile_version = lock.get("lockfileVersion")
    if (
        isinstance(lockfile_version, bool)
        or not isinstance(lockfile_version, int)
        or lockfile_version not in (2, 3)
    ):
        raise ValueError(
            "implementation lock has an unsupported lockfileVersion"
        )
    packages = lock.get("packages")
    if not isinstance(packages, dict):
        raise ValueError("implementation lock packages must be an object")

    root_name, root_version = _package_name_version(
        package,
        description="root package",
    )
    if selected_lock_name != "node_modules/.package-lock.json":
        _require_matching_package_values(
            locked=lock,
            expected_name=root_name,
            expected_version=root_version,
            description="root package lock",
            require_name=True,
        )
        root_lock = packages.get("")
        if not isinstance(root_lock, dict):
            raise ValueError(
                "implementation lock root package entry is missing or malformed"
            )
        _require_matching_package_values(
            locked=cast(Mapping[str, object], root_lock),
            expected_name=root_name,
            expected_version=root_version,
            description="root package lock entry",
            require_name=True,
        )

    locked_package_roots: set[str] = set()
    for coordinate, value in sorted(packages.items()):
        if not isinstance(coordinate, str):
            raise ValueError(
                "implementation lock package coordinates must be strings"
            )
        if not isinstance(value, dict):
            raise ValueError(
                "implementation lock package entries must be objects"
            )
        if coordinate == "":
            continue
        locked_values = cast(Mapping[str, object], value)
        expected_name = _package_name_for_coordinate(coordinate)
        locked_package_roots.add(coordinate)
        locked_name = locked_values.get("name")
        if locked_name is not None and locked_name != expected_name:
            raise ValueError(
                f"locked package {coordinate} name does not match its "
                "package coordinate"
            )
        current_platform = _lock_package_matches_current_platform(
            locked_values
        )
        optional = locked_values.get("optional")
        if optional is not None and not isinstance(optional, bool):
            raise ValueError(
                f"locked package {coordinate} optional marker is malformed"
            )
        if coordinate not in installed_package_roots:
            if optional is True and not current_platform:
                continue
            raise ValueError(
                f"required locked package is absent: {coordinate}"
            )
        manifest_path = f"{coordinate}/package.json"
        installed_bytes = manifests.get(manifest_path)
        if installed_bytes is None:
            raise ValueError(
                f"installed package manifest is missing: {coordinate}"
            )
        installed = _load_json_object_bytes(
            installed_bytes,
            description=f"installed package {coordinate}",
        )
        installed_name, installed_version = _package_name_version(
            installed,
            description=f"installed package {coordinate}",
        )
        if installed_name != expected_name:
            raise ValueError(
                f"installed package {coordinate} name does not match its "
                "package coordinate"
            )
        _require_matching_package_values(
            locked=locked_values,
            expected_name=installed_name,
            expected_version=installed_version,
            description=f"installed package {coordinate}",
        )

    unlocked = sorted(installed_package_roots - locked_package_roots)
    if unlocked:
        raise ValueError(
            f"installed package is absent from the lock: {unlocked[0]}"
        )


def _load_json_object_bytes(
    content: bytes,
    *,
    description: str,
) -> dict[str, object]:
    try:
        value = json.loads(
            content,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"{description} is not strict JSON") from exc
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise ValueError(f"{description} is not an object")
    return value


def _package_name_version(
    package: Mapping[str, object],
    *,
    description: str,
) -> tuple[str, str]:
    values: list[str] = []
    for field_name in ("name", "version"):
        value = package.get(field_name)
        if not isinstance(value, str) or not value or value.strip() != value:
            raise ValueError(
                f"{description} {field_name} must be a nonblank trimmed string"
            )
        values.append(value)
    return values[0], values[1]


def _require_matching_package_values(
    *,
    locked: Mapping[str, object],
    expected_name: str,
    expected_version: str,
    description: str,
    require_name: bool = False,
) -> None:
    locked_name = locked.get("name")
    if require_name and not isinstance(locked_name, str):
        raise ValueError(
            f"{description} name does not match captured package metadata"
        )
    if locked_name is not None and locked_name != expected_name:
        raise ValueError(
            f"{description} name does not match captured package metadata"
        )
    locked_version = locked.get("version")
    if (
        not isinstance(locked_version, str)
        or locked_version != expected_version
    ):
        raise ValueError(
            f"{description} version does not match captured package metadata"
        )


def _package_name_for_coordinate(coordinate: str) -> str:
    if (
        not coordinate
        or "\\" in coordinate
        or "\0" in coordinate
        or coordinate.endswith("/")
    ):
        raise ValueError(f"malformed lock package coordinate: {coordinate!r}")
    path = PurePosixPath(coordinate)
    if (
        path.is_absolute()
        or path.as_posix() != coordinate
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise ValueError(f"malformed lock package coordinate: {coordinate!r}")
    parts = path.parts
    if len(parts) >= 2 and parts[-2] == "node_modules":
        name = parts[-1]
        if not name or name.startswith((".", "@")):
            raise ValueError(
                f"malformed lock package coordinate: {coordinate!r}"
            )
        return name
    if len(parts) >= 3 and parts[-3] == "node_modules":
        scope, name = parts[-2:]
        if (
            len(scope) < 2
            or not scope.startswith("@")
            or name.startswith((".", "@"))
        ):
            raise ValueError(
                f"malformed lock package coordinate: {coordinate!r}"
            )
        return f"{scope}/{name}"
    raise ValueError(f"malformed lock package coordinate: {coordinate!r}")


def _installed_package_coordinate_for_directory(
    relative: PurePosixPath,
) -> str | None:
    parts = relative.parts
    if len(parts) >= 2 and parts[-2] == "node_modules":
        name = parts[-1]
        if name.startswith((".", "@")):
            return None
        return relative.as_posix()
    if (
        len(parts) >= 3
        and parts[-3] == "node_modules"
        and parts[-2].startswith("@")
        and not parts[-1].startswith((".", "@"))
    ):
        return relative.as_posix()
    return None


def _installed_package_coordinate_for_manifest(
    relative: PurePosixPath,
) -> str | None:
    if relative.name != "package.json":
        return None
    return _installed_package_coordinate_for_directory(relative.parent)


def _is_captured_manifest(relative: PurePosixPath) -> bool:
    relative_path = relative.as_posix()
    return (
        relative_path == "package.json"
        or relative_path in _SUPPORTED_LOCK_NAMES
        or _installed_package_coordinate_for_manifest(relative) is not None
    )


def _lock_package_matches_current_platform(
    locked: Mapping[str, object],
) -> bool:
    npm_os = {"darwin": "darwin", "linux": "linux", "win32": "win32"}.get(
        sys.platform,
        sys.platform,
    )
    machine = platform.machine().lower()
    npm_cpu = {
        "aarch64": "arm64",
        "amd64": "x64",
        "arm64": "arm64",
        "i386": "ia32",
        "i686": "ia32",
        "x86_64": "x64",
    }.get(machine, machine)
    return _npm_constraint_allows(locked.get("os"), npm_os, "os") and (
        _npm_constraint_allows(locked.get("cpu"), npm_cpu, "cpu")
    )


def _npm_constraint_allows(
    raw: object,
    current: str,
    description: str,
) -> bool:
    if raw is None:
        return True
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            f"locked package {description} constraint is malformed"
        )
    values: list[str] = []
    for value in raw:
        if (
            not isinstance(value, str)
            or not value
            or value.strip() != value
            or value == "!"
        ):
            raise ValueError(
                f"locked package {description} constraint is malformed"
            )
        values.append(value)
    denied = {value[1:] for value in values if value.startswith("!")}
    allowed = {value for value in values if not value.startswith("!")}
    return current not in denied and (not allowed or current in allowed)


def _resolved_shebang_runtime(
    executable: Path,
    *,
    path_environment: str | None,
) -> _ResolvedRuntime | None:
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
    command_name = runtime.name
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
        command_name = names[0]
    if not runtime.is_absolute():
        resolved = shutil.which(str(runtime), path=path_environment)
        if resolved is None:
            raise ValueError(f"shebang runtime not found: {runtime}")
        runtime = Path(resolved)
    runtime = runtime.resolve(strict=True)
    runtime_observed = _stat_coordinate(runtime, follow_symlinks=False)
    if not stat.S_ISREG(runtime_observed.mode) or not (
        stat.S_IMODE(runtime_observed.mode) & 0o111
    ):
        raise ValueError(f"shebang runtime is not executable: {runtime}")
    if (
        not command_name
        or command_name in (".", "..")
        or Path(command_name).name != command_name
    ):
        raise ValueError("shebang runtime has an invalid command name")
    return _ResolvedRuntime(source=runtime, command_name=command_name)


def _copy_stable_regular_file(
    *,
    source: Path,
    destination: Path,
    source_path: str,
    snapshot_path: str,
    expected: _StableStatCoordinate,
    budget: _CaptureBudget,
    capture_bytes: bool,
) -> tuple[_CapturedImplementationEntry, bytes | None]:
    budget.claim_bytes(expected.size)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise ValueError(
            "host does not support no-follow implementation capture"
        )
    flags = os.O_RDONLY | no_follow
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise ValueError(
            f"implementation file is unreadable: {source}"
        ) from exc
    try:
        stream = os.fdopen(descriptor, "rb", closefd=True)
    except BaseException:
        os.close(descriptor)
        raise
    captured = bytearray() if capture_bytes else None
    with stream:
        before = _stat_coordinate_for_result(os.fstat(stream.fileno()))
        if before != expected or not stat.S_ISREG(before.mode):
            raise ValueError(
                f"implementation file changed before copying: {source}"
            )
        digest = hashlib.sha256()
        remaining = before.size
        try:
            with destination.open("xb") as output:
                while remaining:
                    chunk = stream.read(min(_COPY_CHUNK_BYTES, remaining))
                    if not chunk:
                        raise ValueError(
                            "implementation file changed while copying: "
                            f"{source}"
                        )
                    remaining -= len(chunk)
                    digest.update(chunk)
                    output.write(chunk)
                    if captured is not None:
                        captured.extend(chunk)
                if stream.read(1):
                    raise ValueError(
                        f"implementation file changed while copying: {source}"
                    )
                after = _stat_coordinate_for_result(os.fstat(stream.fileno()))
        except OSError as exc:
            raise ValueError(
                f"could not create captured implementation file: {destination}"
            ) from exc
    if before != after:
        raise ValueError(
            f"implementation file changed while copying: {source}"
        )
    return (
        _CapturedImplementationEntry(
            source_path=source_path,
            snapshot_path=snapshot_path,
            kind=_ImplementationEntryKind.REGULAR,
            mode=stat.S_IMODE(before.mode),
            link_target=None,
            size=before.size,
            sha256=digest.hexdigest(),
        ),
        bytes(captured) if captured is not None else None,
    )


def _copy_stable_symlink(
    *,
    source: Path,
    destination: Path,
    source_root: Path,
    source_relative: PurePosixPath,
    snapshot_relative: PurePosixPath,
    expected: _StableStatCoordinate,
    budget: _CaptureBudget,
) -> _CapturedImplementationEntry:
    budget.claim_bytes(expected.size)
    if not stat.S_ISLNK(expected.mode):
        raise ValueError(f"implementation entry is not a symlink: {source}")
    try:
        link_target = os.readlink(source)
    except OSError as exc:
        raise ValueError(
            f"implementation symlink is unreadable: {source}"
        ) from exc
    after = _stat_coordinate(source, follow_symlinks=False)
    encoded_target = os.fsencode(link_target)
    if after != expected or len(encoded_target) != expected.size:
        raise ValueError(
            f"implementation symlink changed while copying: {source}"
        )
    _validate_source_symlink(
        source=source,
        source_root=source_root,
        source_relative=source_relative,
        link_target=link_target,
    )
    try:
        destination.symlink_to(link_target)
    except OSError as exc:
        raise ValueError(
            f"could not create captured implementation symlink: {destination}"
        ) from exc
    return _CapturedImplementationEntry(
        source_path=source_relative.as_posix(),
        snapshot_path=snapshot_relative.as_posix(),
        kind=_ImplementationEntryKind.SYMLINK,
        mode=stat.S_IMODE(expected.mode),
        link_target=link_target,
        size=expected.size,
        sha256=hashlib.sha256(encoded_target).hexdigest(),
    )


def _validate_source_symlink(
    *,
    source: Path,
    source_root: Path,
    source_relative: PurePosixPath,
    link_target: str,
) -> None:
    if PurePosixPath(link_target).is_absolute() or os.path.isabs(link_target):
        raise ValueError(
            f"implementation symlink has an absolute target: {source}"
        )
    _logical_link_target(source_relative, link_target)
    try:
        resolved_target = source.resolve(strict=True)
        resolved_target.relative_to(source_root)
        target_stat = resolved_target.stat()
    except FileNotFoundError as exc:
        raise ValueError(
            f"implementation symlink has a dangling target: {source}"
        ) from exc
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"implementation symlink escapes its package root: {source}"
        ) from exc
    if not (
        stat.S_ISREG(target_stat.st_mode) or stat.S_ISDIR(target_stat.st_mode)
    ):
        raise ValueError(
            f"implementation symlink targets a special file: {source}"
        )


def _logical_link_target(
    source_relative: PurePosixPath,
    link_target: str,
) -> PurePosixPath:
    parts = list(source_relative.parent.parts)
    for part in PurePosixPath(link_target).parts:
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                raise ValueError(
                    "implementation symlink escapes its package root: "
                    f"{source_relative.as_posix()}"
                )
            parts.pop()
            continue
        parts.append(part)
    return PurePosixPath(*parts)


def _validate_snapshot_symlinks(
    *,
    snapshot_root: Path,
    package_destination: Path,
    entries: tuple[_CapturedImplementationEntry, ...],
) -> None:
    resolved_package_destination = package_destination.resolve(strict=True)
    for entry in entries:
        if entry.kind is not _ImplementationEntryKind.SYMLINK:
            continue
        path = snapshot_root / entry.snapshot_path
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(resolved_package_destination)
            observed = resolved.stat()
        except FileNotFoundError as exc:
            raise ValueError(
                f"captured implementation symlink is dangling: {path}"
            ) from exc
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"captured implementation symlink escapes its snapshot: {path}"
            ) from exc
        if not (
            stat.S_ISREG(observed.st_mode) or stat.S_ISDIR(observed.st_mode)
        ):
            raise ValueError(
                f"captured implementation symlink targets a special file: {path}"
            )


def _require_stable_directory(
    path: Path,
    *,
    expected: _StableStatCoordinate,
) -> None:
    observed = _stat_coordinate(path, follow_symlinks=False)
    if observed != expected or not stat.S_ISDIR(observed.mode):
        raise ValueError(
            f"implementation directory changed while copying: {path}"
        )


def _finalize_snapshot_permissions(
    *,
    snapshot_root: Path,
    package_directory_mode: int,
    entries: tuple[_CapturedImplementationEntry, ...],
    has_runtime: bool,
) -> None:
    try:
        for entry in entries:
            if entry.kind is _ImplementationEntryKind.REGULAR:
                (snapshot_root / entry.snapshot_path).chmod(
                    entry.mode & ~0o222
                )
        directories = sorted(
            (
                entry
                for entry in entries
                if entry.kind is _ImplementationEntryKind.DIRECTORY
            ),
            key=lambda entry: len(PurePosixPath(entry.snapshot_path).parts),
            reverse=True,
        )
        for entry in directories:
            (snapshot_root / entry.snapshot_path).chmod(entry.mode & ~0o222)
        (snapshot_root / _PACKAGE_SNAPSHOT_DIRECTORY).chmod(
            package_directory_mode & ~0o222
        )
        if has_runtime:
            (snapshot_root / _RUNTIME_SNAPSHOT_DIRECTORY).chmod(0o555)
        snapshot_root.chmod(0o500)
    except OSError as exc:
        raise ValueError(
            "could not finalize captured implementation permissions"
        ) from exc


def _implementation_identity_for(
    *,
    executable: Path,
    package_root: Path | None,
    entrypoint_path: str,
    runtime_source: Path | None,
    runtime_path: str | None,
    entries: tuple[_CapturedImplementationEntry, ...],
) -> str:
    return identity_hash_for(
        schema=_IMPLEMENTATION_IDENTITY_SCHEMA,
        payload={
            "entries": [
                {
                    "kind": entry.kind.value,
                    "link_target": entry.link_target,
                    "mode": entry.mode,
                    "sha256": entry.sha256,
                    "size": entry.size,
                    "snapshot_path": entry.snapshot_path,
                    "source_path": entry.source_path,
                }
                for entry in entries
            ],
            "entrypoint_path": entrypoint_path,
            "layout": _IMPLEMENTATION_SNAPSHOT_LAYOUT,
            "source_executable": str(executable),
            "source_package_root": (
                str(package_root) if package_root is not None else None
            ),
            "source_runtime": (
                str(runtime_source) if runtime_source is not None else None
            ),
            "runtime_path": runtime_path,
        },
    )


def _stat_coordinate(
    path: Path,
    *,
    follow_symlinks: bool,
) -> _StableStatCoordinate:
    try:
        observed = path.stat(follow_symlinks=follow_symlinks)
    except OSError as exc:
        raise ValueError(
            f"implementation entry is unreadable: {path}"
        ) from exc
    return _stat_coordinate_for_result(observed)


def _stat_coordinate_for_result(
    observed: os.stat_result,
) -> _StableStatCoordinate:
    return _StableStatCoordinate(
        device=observed.st_dev,
        inode=observed.st_ino,
        mode=observed.st_mode,
        size=observed.st_size,
        mtime_ns=observed.st_mtime_ns,
        ctime_ns=observed.st_ctime_ns,
    )


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
    "SubscriptionLane",
    "TransportFailureKind",
    "lane_policy",
    "lane_policy_identity",
    "parse_label_response",
)
