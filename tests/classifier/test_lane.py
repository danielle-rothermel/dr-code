from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import sys
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import pytest

import dr_code.classifier.lane as lane_module
from dr_code.classifier.classify import classify_one_repeat
from dr_code.classifier.lane import (
    LaneTransportError,
    MAX_RESPONSE_ERROR_CHARS,
    SubscriptionLane,
    TransportFailureKind,
    parse_label_response,
)
from dr_code.classifier.taxonomy import FailureFamily
from dr_code.execution import (
    SubprocessCompletedProcess,
    SubprocessError,
    SubprocessInfrastructureError,
    SubprocessOutputLimitError,
    SubprocessTimeoutError,
    run_subprocess,
)


@pytest.mark.parametrize(
    "raw",
    [
        '```json\n{"label":"prose-no-code","rationale":"x"}\n```',
        'answer: {"label":"prose-no-code","rationale":"x"}',
        '{"label":"prose-no-code","rationale":"x"} trailing',
        '{"label":"prose-no-code","label":"other","rationale":"x"}',
        '{"label":"prose-no-code","rationale":"x","confidence":1}',
        '{"label":"prose-no-code"}',
        '{"rationale":"x"}',
        '["prose-no-code","x"]',
        '{"label":"prose-no-code","rationale":NaN}',
        '{"label":"prose-no-code","rationale":""}',
        '{"label":"prose-no-code","rationale":" x"}',
        '{"label":"prose-no-code","rationale":"x\\ny"}',
        '{"label":"invented","rationale":"x"}',
        '{"label":1,"rationale":"x"}',
    ],
)
def test_strict_response_rejects_every_noncanonical_shape(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_label_response(raw, FailureFamily.PARSE)


def test_strict_response_accepts_exact_taxonomy_object() -> None:
    response = parse_label_response(
        '{"label":"prose-no-code","rationale":"Only prose is present."}',
        FailureFamily.PARSE,
    )
    assert response.label == "prose-no-code"


class _ScriptedLane:
    provider = "provider"
    model = "model"

    def __init__(self, replies: list[str | Exception]) -> None:
        self.replies = replies
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        value = self.replies.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def test_invalid_response_gets_exactly_one_correction_retry() -> None:
    lane = _ScriptedLane(
        [
            '{"label":"invented","rationale":"x"}',
            '{"label":"prose-no-code","rationale":"valid"}',
        ]
    )
    outcome = classify_one_repeat(lane, FailureFamily.PARSE, "prompt")
    assert outcome.label == "prose-no-code"
    assert outcome.phase == "correction"
    assert outcome.attempt == 2
    assert outcome.corrected
    assert outcome.primary_validation_failure is not None
    assert len(lane.prompts) == 2
    assert "Correct it once" in lane.prompts[1]

    broken = _ScriptedLane(["bad", "still bad"])
    outcome = classify_one_repeat(broken, FailureFamily.PARSE, "prompt")
    assert outcome.failure is not None
    assert outcome.failure.kind == "invalid_response"
    assert outcome.phase == "correction"
    assert outcome.attempt == 2
    assert not outcome.corrected
    assert outcome.primary_validation_failure is not None
    assert len(broken.prompts) == 2


def test_transport_failure_is_not_retried() -> None:
    lane = _ScriptedLane(
        [LaneTransportError(TransportFailureKind.TIMEOUT, "deadline exceeded")]
    )
    outcome = classify_one_repeat(lane, FailureFamily.PARSE, "prompt")
    assert outcome.failure is not None
    assert outcome.failure.kind == "transport"
    assert outcome.failure.detail == "timeout"
    assert outcome.phase == "primary"
    assert outcome.attempt == 1
    assert len(lane.prompts) == 1


def test_correction_transport_is_audited_without_transport_detail() -> None:
    lane = _ScriptedLane(
        [
            '{"label":"invented","rationale":"x"}',
            LaneTransportError(
                TransportFailureKind.NONZERO_EXIT,
                "provider-secret-value",
            ),
        ]
    )

    outcome = classify_one_repeat(lane, FailureFamily.PARSE, "prompt")

    assert outcome.failure is not None
    assert outcome.failure.kind == "transport"
    assert outcome.failure.detail == "nonzero_exit"
    assert "provider-secret-value" not in repr(outcome)
    assert outcome.phase == "correction"
    assert outcome.attempt == 2
    assert not outcome.corrected
    assert outcome.primary_validation_failure == (
        "label is outside the parse taxonomy"
    )


def test_subscription_lane_uses_shared_runner_with_captured_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def complete(**kwargs: Any) -> SubprocessCompletedProcess:
        captured.update(kwargs)
        return SubprocessCompletedProcess(
            returncode=0,
            stdout="response",
            stderr="",
        )

    monkeypatch.setattr(lane_module, "run_subprocess", complete)
    command = _write_provider_command(tmp_path, "print('unused')\n")
    lane = SubscriptionLane(
        provider="provider",
        model="model",
        timeout_seconds=7.5,
        executable=str(command),
    )
    assert lane.complete("prompt") == "response"
    environment = cast(dict[str, str], captured.pop("environment"))
    assert environment["PATH"] == os.environ["PATH"]
    assert captured == {
        "command": [
            str(lane._implementation_snapshot.entrypoint),
            "-p",
            "--provider",
            "provider",
            "--model",
            "model",
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
        ],
        "input_text": "prompt",
        "timeout_seconds": 7.5,
    }


@pytest.mark.parametrize(
    ("error", "kind", "detail"),
    [
        (
            SubprocessTimeoutError("private timeout detail"),
            TransportFailureKind.TIMEOUT,
            "command timed out after 1 seconds",
        ),
        (
            SubprocessOutputLimitError("private output-limit detail"),
            TransportFailureKind.OUTPUT_LIMIT,
            "command output exceeded limit",
        ),
        (
            SubprocessInfrastructureError("ipc failed"),
            TransportFailureKind.OPERATING_SYSTEM,
            "ipc failed",
        ),
        (
            SubprocessError("unexpected execution failure"),
            TransportFailureKind.OPERATING_SYSTEM,
            "unexpected execution failure",
        ),
    ],
)
def test_subscription_lane_types_all_execution_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: SubprocessError,
    kind: TransportFailureKind,
    detail: str,
) -> None:
    def fail(**_: Any) -> SubprocessCompletedProcess:
        raise error

    monkeypatch.setattr(lane_module, "run_subprocess", fail)
    command = _write_provider_command(tmp_path, "print('unused')\n")
    lane = SubscriptionLane(
        provider="provider",
        model="model",
        timeout_seconds=1,
        executable=str(command),
    )
    with pytest.raises(LaneTransportError) as raised:
        lane.complete("prompt")
    assert raised.value.kind is kind
    assert raised.value.detail == detail
    assert raised.value.__cause__ is error


def test_subscription_lane_inherits_environment_and_resolves_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _write_provider_command(
        tmp_path,
        (
            "import json, os, sys\n"
            "print(json.dumps({\n"
            "    'arguments': sys.argv[1:],\n"
            "    'credential': os.environ['DR_CODE_TEST_CREDENTIAL'],\n"
            "    'stdin': sys.stdin.read(),\n"
            "}))\n"
        ),
    )
    monkeypatch.setenv("DR_CODE_TEST_CREDENTIAL", "inherited")
    monkeypatch.setenv(
        "PATH",
        f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}",
    )
    lane = SubscriptionLane(
        provider="provider",
        model="model",
        timeout_seconds=2,
        executable=command.name,
    )

    response = json.loads(lane.complete("prompt"))

    assert response == {
        "arguments": [
            "-p",
            "--provider",
            "provider",
            "--model",
            "model",
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
        ],
        "credential": "inherited",
        "stdin": "prompt",
    }


def test_subscription_lane_types_missing_executable(tmp_path: Path) -> None:
    executable = tmp_path / "does-not-exist"
    with pytest.raises(ValueError, match="executable not found"):
        SubscriptionLane(
            provider="provider",
            model="model",
            timeout_seconds=1,
            executable=str(executable),
        )


def test_subscription_lane_types_other_start_failure(tmp_path: Path) -> None:
    command = tmp_path / "not-executable"
    command.write_text("provider")
    with pytest.raises(ValueError, match="executable not found"):
        SubscriptionLane(
            provider="provider",
            model="model",
            timeout_seconds=1,
            executable=str(command),
        )


def test_subscription_lane_policy_binds_resolved_executable_and_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    first_command = _write_provider_command(first, "print('first')\n")
    second_command = _write_provider_command(second, "print('second')\n")
    monkeypatch.setenv("PATH", str(first))
    monkeypatch.setenv("OPENAI_API_KEY", "first-secret")
    lane = SubscriptionLane(
        provider="provider",
        model="model",
        timeout_seconds=2,
        executable=first_command.name,
    )
    first_policy = dict(lane.policy.transport)
    monkeypatch.setenv("OPENAI_API_KEY", "second-secret")
    secret_rotated = SubscriptionLane(
        provider="provider",
        model="model",
        timeout_seconds=2,
        executable=first_command.name,
    )
    assert (
        dict(secret_rotated.policy.transport)["environment_identity"]
        == first_policy["environment_identity"]
    )

    monkeypatch.setenv("PATH", str(second))

    assert lane.complete("prompt").strip() == "first"
    assert lane.executable == str(first_command.resolve())
    assert (
        first_policy["executable_sha256"]
        == hashlib.sha256(first_command.read_bytes()).hexdigest()
    )
    assert "first-secret" not in repr(lane)
    changed_environment = SubscriptionLane(
        provider="provider",
        model="model",
        timeout_seconds=2,
        executable=str(second_command),
    )
    second_policy = dict(changed_environment.policy.transport)
    assert (
        first_policy["environment_identity"]
        != second_policy["environment_identity"]
    )
    first_command.write_text(
        f"#!{sys.executable}\nprint('changed')\n",
    )
    assert lane.complete("prompt").strip() == "first"
    assert dict(lane.policy.transport) == first_policy


def test_subscription_lane_executes_captured_imported_sibling_after_mutation(
    tmp_path: Path,
) -> None:
    _write_node_fixture(tmp_path)
    helper = tmp_path / "provider_helper.py"
    helper.write_text("RESPONSE = 'first'\n")
    command = _write_provider_command(
        tmp_path,
        "from provider_helper import RESPONSE\nprint(RESPONSE)\n",
    )
    lane = SubscriptionLane(
        provider="provider",
        model="model",
        timeout_seconds=2,
        executable=str(command),
    )

    helper.write_text("RESPONSE = 'changed'\n")

    assert lane.complete("prompt").strip() == "first"


def test_subscription_lane_wrapper_race_executes_captured_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependency = _write_node_fixture(tmp_path)
    command = _write_provider_command(
        tmp_path,
        (
            "from pathlib import Path\n"
            "print((Path(__file__).parent / 'node_modules' / 'dependency' / "
            "'index.js').read_text())\n"
        ),
    )
    lane = SubscriptionLane(
        provider="provider",
        model="model",
        timeout_seconds=2,
        executable=str(command),
    )
    shared_run_subprocess = lane_module.run_subprocess

    def mutate_then_run(**kwargs: Any) -> SubprocessCompletedProcess:
        dependency.write_text("export default 'changed';\n")
        return shared_run_subprocess(**kwargs)

    monkeypatch.setattr(lane_module, "run_subprocess", mutate_then_run)

    assert lane.complete("prompt").strip() == "export default 'original';"


def test_subscription_lane_validates_only_captured_manifest_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependency = _write_node_fixture(tmp_path)
    command = _write_provider_command(tmp_path, "print('ok')\n")
    copy_package_tree = lane_module._copy_package_tree

    def copy_then_mutate(**kwargs: Any) -> Any:
        captured = copy_package_tree(**kwargs)
        (tmp_path / "package.json").write_text("not json")
        (tmp_path / "npm-shrinkwrap.json").write_text("not json")
        (dependency.parent / "package.json").write_text("not json")
        return captured

    monkeypatch.setattr(
        lane_module,
        "_copy_package_tree",
        copy_then_mutate,
    )

    lane = SubscriptionLane(
        provider="provider",
        model="model",
        timeout_seconds=2,
        executable=str(command),
    )

    assert lane.complete("prompt").strip() == "ok"


def test_subscription_lane_executes_captured_node_after_runtime_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_directory = tmp_path / "runtime"
    runtime_directory.mkdir()
    node = runtime_directory / "node"
    node.write_text('#!/bin/sh\nexec /bin/sh "$@"\n')
    node.chmod(0o755)
    command_directory = tmp_path / "provider"
    command_directory.mkdir()
    command = command_directory / "provider"
    command.write_text("#!/usr/bin/env node\nprintf 'original\\n'\n")
    command.chmod(0o755)
    monkeypatch.setenv(
        "PATH",
        f"{runtime_directory}{os.pathsep}{os.environ.get('PATH', '')}",
    )
    lane = SubscriptionLane(
        provider="provider",
        model="model",
        timeout_seconds=2,
        executable=str(command),
    )
    snapshot = lane._implementation_snapshot

    node.write_text("#!/bin/sh\nprintf 'changed\\n'\n")

    assert lane.complete("prompt").strip() == "original"
    assert snapshot.runtime is not None
    assert snapshot.command_prefix == (
        str(snapshot.runtime),
        str(snapshot.entrypoint),
    )
    assert str(node.resolve()) not in snapshot.command_prefix
    assert str(command.resolve()) not in snapshot.command_prefix
    environment = lane_module._snapshot_environment(
        snapshot,
        lane._environment,
    )
    assert environment["PATH"].split(os.pathsep)[0] == str(
        snapshot.runtime.parent
    )


@pytest.mark.parametrize("action", ["add", "remove"])
def test_subscription_lane_reuses_capture_after_source_membership_changes(
    tmp_path: Path,
    action: str,
) -> None:
    dependency = _write_node_fixture(tmp_path)
    command = _write_provider_command(tmp_path, "print('ok')\n")
    lane = SubscriptionLane(
        provider="provider",
        model="model",
        timeout_seconds=2,
        executable=str(command),
    )

    if action == "add":
        (dependency.parent / "added.wasm").write_bytes(b"wasm")
    else:
        dependency.unlink()

    assert lane.complete("prompt").strip() == "ok"


def test_subscription_lane_executes_captured_in_root_bin_symlink(
    tmp_path: Path,
) -> None:
    _write_node_fixture(tmp_path)
    executable = tmp_path / "node_modules" / "dependency" / "cli"
    executable.write_text("#!/bin/sh\nprintf 'captured-bin\\n'\n")
    executable.chmod(0o755)
    bin_directory = tmp_path / "node_modules" / ".bin"
    bin_directory.mkdir()
    (bin_directory / "dependency").symlink_to("../dependency/cli")
    command = _write_provider_command(
        tmp_path,
        (
            "import subprocess\n"
            "print(subprocess.check_output([\n"
            "    'dependency'\n"
            "], text=True).strip())\n"
        ),
    )

    lane = SubscriptionLane(
        provider="provider",
        model="model",
        timeout_seconds=2,
        executable=str(command),
    )

    assert lane.complete("prompt").strip() == "captured-bin"
    assert dict(lane.policy.transport)["implementation_policy"] == (
        "installed-node-runtime-closure-v2"
    )


def test_subscription_lane_rejects_absolute_symlink(tmp_path: Path) -> None:
    dependency = _write_node_fixture(tmp_path)
    (tmp_path / "node_modules" / "absolute.js").symlink_to(
        dependency.resolve()
    )
    command = _write_provider_command(tmp_path, "print('ok')\n")

    with pytest.raises(ValueError, match="absolute target"):
        SubscriptionLane(
            provider="provider",
            model="model",
            timeout_seconds=2,
            executable=str(command),
        )


def test_subscription_lane_rejects_escaping_symlink(tmp_path: Path) -> None:
    _write_node_fixture(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.js"
    outside.write_text("outside")
    link = tmp_path / "node_modules" / "escape.js"
    link.symlink_to(os.path.relpath(outside, start=link.parent))
    command = _write_provider_command(tmp_path, "print('ok')\n")

    with pytest.raises(ValueError, match="symlink escapes"):
        SubscriptionLane(
            provider="provider",
            model="model",
            timeout_seconds=2,
            executable=str(command),
        )


def test_subscription_lane_rejects_dangling_symlink(tmp_path: Path) -> None:
    _write_node_fixture(tmp_path)
    (tmp_path / "node_modules" / "dangling.js").symlink_to("missing.js")
    command = _write_provider_command(tmp_path, "print('ok')\n")

    with pytest.raises(ValueError, match="dangling target"):
        SubscriptionLane(
            provider="provider",
            model="model",
            timeout_seconds=2,
            executable=str(command),
        )


def test_subscription_lane_rejects_special_implementation_entry(
    tmp_path: Path,
) -> None:
    _write_node_fixture(tmp_path)
    os.mkfifo(tmp_path / "node_modules" / "dependency" / "special")
    command = _write_provider_command(tmp_path, "print('ok')\n")

    with pytest.raises(ValueError, match="special file"):
        SubscriptionLane(
            provider="provider",
            model="model",
            timeout_seconds=2,
            executable=str(command),
        )


def test_subscription_lane_rejects_symlink_to_special_entry(
    tmp_path: Path,
) -> None:
    _write_node_fixture(tmp_path)
    dependency_directory = tmp_path / "node_modules" / "dependency"
    special = dependency_directory / "z-special"
    os.mkfifo(special)
    (dependency_directory / "a-link").symlink_to("z-special")
    command = _write_provider_command(tmp_path, "print('ok')\n")

    with pytest.raises(ValueError, match="targets a special file"):
        SubscriptionLane(
            provider="provider",
            model="model",
            timeout_seconds=2,
            executable=str(command),
        )


def test_subscription_lane_enforces_entry_cap_during_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_node_fixture(tmp_path)
    command = _write_provider_command(tmp_path, "print('ok')\n")
    scanned_source_directories: list[Path] = []
    scan_directory = lane_module.os.scandir

    def record_scan(path: Any) -> Any:
        if isinstance(path, (str, os.PathLike)):
            candidate = Path(path)
            if candidate == tmp_path or tmp_path in candidate.parents:
                scanned_source_directories.append(candidate)
        return scan_directory(path)

    monkeypatch.setattr(
        lane_module,
        "MAX_IMPLEMENTATION_CLOSURE_ENTRIES",
        1,
    )
    monkeypatch.setattr(lane_module.os, "scandir", record_scan)

    with pytest.raises(ValueError, match="entry-count limit"):
        SubscriptionLane(
            provider="provider",
            model="model",
            timeout_seconds=2,
            executable=str(command),
        )
    assert scanned_source_directories == [tmp_path]


def test_subscription_lane_rejects_oversized_declared_file_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _write_provider_command(tmp_path, "print('ok')\n")
    source_was_opened = False
    open_file = lane_module.os.open

    def record_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal source_was_opened
        if Path(path) == command.resolve():
            source_was_opened = True
        return open_file(path, flags, *args, **kwargs)

    monkeypatch.setattr(
        lane_module,
        "MAX_IMPLEMENTATION_CLOSURE_BYTES",
        command.stat().st_size - 1,
    )
    monkeypatch.setattr(lane_module.os, "open", record_open)

    with pytest.raises(ValueError, match="byte limit"):
        SubscriptionLane(
            provider="provider",
            model="model",
            timeout_seconds=2,
            executable=str(command),
        )
    assert not source_was_opened


def test_subscription_lane_counts_node_before_opening_runtime_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_directory = tmp_path / "runtime"
    runtime_directory.mkdir()
    node = runtime_directory / "node"
    node.write_text("#!/bin/sh\n" + "#" * 1_000)
    node.chmod(0o755)
    command_directory = tmp_path / "provider"
    command_directory.mkdir()
    command = command_directory / "provider"
    command.write_text("#!/usr/bin/env node\nprintf ok\n")
    command.chmod(0o755)
    monkeypatch.setenv("PATH", str(runtime_directory))
    runtime_was_opened = False
    open_file = lane_module.os.open

    def record_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal runtime_was_opened
        if Path(path) == node.resolve():
            runtime_was_opened = True
        return open_file(path, flags, *args, **kwargs)

    monkeypatch.setattr(
        lane_module,
        "MAX_IMPLEMENTATION_CLOSURE_BYTES",
        command.stat().st_size,
    )
    monkeypatch.setattr(lane_module.os, "open", record_open)

    with pytest.raises(ValueError, match="byte limit"):
        SubscriptionLane(
            provider="provider",
            model="model",
            timeout_seconds=2,
            executable=str(command),
        )
    assert not runtime_was_opened


def test_subscription_lane_accounts_for_directories_files_and_symlinks(
    tmp_path: Path,
) -> None:
    _write_node_fixture(tmp_path)
    (tmp_path / "empty").mkdir()
    bin_directory = tmp_path / "node_modules" / ".bin"
    bin_directory.mkdir()
    (bin_directory / "dependency").symlink_to("../dependency/index.js")
    command = _write_provider_command(tmp_path, "print('ok')\n")

    lane = SubscriptionLane(
        provider="provider",
        model="model",
        timeout_seconds=2,
        executable=str(command),
    )
    snapshot = lane._implementation_snapshot
    source_entries = list(tmp_path.rglob("*"))

    assert len(snapshot.entries) == len(source_entries)
    assert sum(
        entry.kind.value == "directory" for entry in snapshot.entries
    ) == sum(
        path.is_dir() and not path.is_symlink() for path in source_entries
    )
    assert sum(
        entry.kind.value == "symlink" for entry in snapshot.entries
    ) == sum(path.is_symlink() for path in source_entries)
    assert (snapshot.root / "package" / "empty").is_dir()
    for entry in snapshot.entries:
        if entry.kind.value != "symlink":
            assert not (
                (snapshot.root / entry.snapshot_path).stat().st_mode & 0o222
            )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("NODE_OPTIONS", "--require=/tmp/inject.js"),
        ("NODE_OPTIONS", "--import=/tmp/inject.mjs"),
        ("NODE_PATH", "/tmp/injected-modules"),
    ],
)
def test_subscription_lane_rejects_node_source_injection_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    command = _write_provider_command(tmp_path, "print('ok')\n")
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=f"{name} must be empty"):
        SubscriptionLane(
            provider="provider",
            model="model",
            timeout_seconds=2,
            executable=str(command),
        )


@pytest.mark.parametrize("case", ["missing", "null", "list"])
def test_subscription_lane_rejects_malformed_lock_packages(
    tmp_path: Path,
    case: str,
) -> None:
    _write_node_fixture(tmp_path)
    lock = _read_fixture_lock(tmp_path)
    if case == "missing":
        del lock["packages"]
    elif case == "null":
        lock["packages"] = None
    else:
        lock["packages"] = []
    _write_fixture_lock(tmp_path, lock)
    command = _write_provider_command(tmp_path, "print('ok')\n")

    with pytest.raises(ValueError, match="packages must be an object"):
        SubscriptionLane(
            provider="provider",
            model="model",
            timeout_seconds=2,
            executable=str(command),
        )


def test_subscription_lane_rejects_unlocked_installed_package(
    tmp_path: Path,
) -> None:
    _write_node_fixture(tmp_path)
    unlocked = tmp_path / "node_modules" / "unlocked"
    unlocked.mkdir()
    (unlocked / "package.json").write_text(
        '{"name":"unlocked","version":"1.0.0"}'
    )
    command = _write_provider_command(tmp_path, "print('ok')\n")

    with pytest.raises(ValueError, match="absent from the lock"):
        SubscriptionLane(
            provider="provider",
            model="model",
            timeout_seconds=2,
            executable=str(command),
        )


def test_subscription_lane_rejects_missing_required_locked_package(
    tmp_path: Path,
) -> None:
    _write_node_fixture(tmp_path)
    lock = _read_fixture_lock(tmp_path)
    lock["packages"]["node_modules/missing"] = {"version": "1.0.0"}
    _write_fixture_lock(tmp_path, lock)
    command = _write_provider_command(tmp_path, "print('ok')\n")

    with pytest.raises(ValueError, match="required locked package is absent"):
        SubscriptionLane(
            provider="provider",
            model="model",
            timeout_seconds=2,
            executable=str(command),
        )


def test_subscription_lane_allows_absent_incompatible_optional_platform_package(
    tmp_path: Path,
) -> None:
    _write_node_fixture(tmp_path)
    lock = _read_fixture_lock(tmp_path)
    incompatible_os = "darwin" if sys.platform == "win32" else "win32"
    lock["packages"]["node_modules/platform-only"] = {
        "optional": True,
        "os": [incompatible_os],
        "version": "1.0.0",
    }
    _write_fixture_lock(tmp_path, lock)
    command = _write_provider_command(tmp_path, "print('ok')\n")

    lane = SubscriptionLane(
        provider="provider",
        model="model",
        timeout_seconds=2,
        executable=str(command),
    )

    assert lane.complete("prompt").strip() == "ok"


@pytest.mark.parametrize("field_name", ["name", "version"])
def test_subscription_lane_rejects_installed_name_or_version_mismatch(
    tmp_path: Path,
    field_name: str,
) -> None:
    dependency = _write_node_fixture(tmp_path)
    package_manifest = dependency.parent / "package.json"
    package = json.loads(package_manifest.read_bytes())
    package[field_name] = "changed"
    package_manifest.write_text(json.dumps(package))
    command = _write_provider_command(tmp_path, "print('ok')\n")

    with pytest.raises(ValueError, match=field_name):
        SubscriptionLane(
            provider="provider",
            model="model",
            timeout_seconds=2,
            executable=str(command),
        )


def test_subscription_lane_rejects_traversing_lock_coordinate(
    tmp_path: Path,
) -> None:
    _write_node_fixture(tmp_path)
    lock = _read_fixture_lock(tmp_path)
    lock["packages"]["node_modules/../escape"] = {"version": "1.0.0"}
    _write_fixture_lock(tmp_path, lock)
    command = _write_provider_command(tmp_path, "print('ok')\n")

    with pytest.raises(ValueError, match="malformed lock package coordinate"):
        SubscriptionLane(
            provider="provider",
            model="model",
            timeout_seconds=2,
            executable=str(command),
        )


def test_subscription_lane_identifies_only_structural_installed_roots(
    tmp_path: Path,
) -> None:
    dependency = _write_node_fixture(tmp_path)
    dist = dependency.parent / "dist"
    dist.mkdir()
    (dist / "package.json").write_text(
        '{"name":"nested-metadata","version":"99.0.0"}'
    )
    scoped = tmp_path / "node_modules" / "@scope" / "tool"
    scoped.mkdir(parents=True)
    (scoped / "package.json").write_text(
        '{"name":"@scope/tool","version":"2.0.0"}'
    )
    lock = _read_fixture_lock(tmp_path)
    lock["packages"]["node_modules/@scope/tool"] = {"version": "2.0.0"}
    _write_fixture_lock(tmp_path, lock)
    command = _write_provider_command(tmp_path, "print('ok')\n")

    lane = SubscriptionLane(
        provider="provider",
        model="model",
        timeout_seconds=2,
        executable=str(command),
    )

    assert lane.complete("prompt").strip() == "ok"


def test_subscription_lane_reuses_one_snapshot_sequentially_and_concurrently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _write_provider_command(tmp_path, "print('ok')\n")
    lane = SubscriptionLane(
        provider="provider",
        model="model",
        timeout_seconds=2,
        executable=str(command),
    )
    snapshot = lane._implementation_snapshot
    observed_commands: list[tuple[str, ...]] = []
    shared_run_subprocess = lane_module.run_subprocess

    def record_run(**kwargs: Any) -> SubprocessCompletedProcess:
        observed_commands.append(tuple(kwargs["command"]))
        return shared_run_subprocess(**kwargs)

    monkeypatch.setattr(lane_module, "run_subprocess", record_run)

    assert lane.complete("first").strip() == "ok"
    assert lane.complete("second").strip() == "ok"
    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = tuple(executor.map(lane.complete, ("third", "fourth")))

    assert tuple(response.strip() for response in responses) == ("ok", "ok")
    assert len(observed_commands) == 4
    assert all(
        command_value[: len(snapshot.command_prefix)]
        == snapshot.command_prefix
        for command_value in observed_commands
    )
    assert snapshot.root.exists()


def test_subscription_lane_snapshot_lifetime_is_owned_by_lane(
    tmp_path: Path,
) -> None:
    command = _write_provider_command(tmp_path, "print('ok')\n")
    lane = SubscriptionLane(
        provider="provider",
        model="model",
        timeout_seconds=2,
        executable=str(command),
    )
    snapshot_root = lane._implementation_snapshot.root

    assert snapshot_root.exists()
    del lane
    gc.collect()
    assert not snapshot_root.exists()


def test_subscription_lane_identity_excludes_random_snapshot_root(
    tmp_path: Path,
) -> None:
    command = _write_provider_command(tmp_path, "print('ok')\n")
    first = SubscriptionLane(
        provider="provider",
        model="model",
        timeout_seconds=2,
        executable=str(command),
    )
    second = SubscriptionLane(
        provider="provider",
        model="model",
        timeout_seconds=2,
        executable=str(command),
    )

    assert (
        first._implementation_snapshot.root
        != second._implementation_snapshot.root
    )
    assert (
        first._implementation_snapshot.identity
        == second._implementation_snapshot.identity
    )


def test_captured_implementation_identity_layout_is_pinned() -> None:
    entries = (
        lane_module._CapturedImplementationEntry(
            source_path="dist",
            snapshot_path="package/dist",
            kind=lane_module._ImplementationEntryKind.DIRECTORY,
            mode=0o755,
            link_target=None,
            size=0,
            sha256=None,
        ),
        lane_module._CapturedImplementationEntry(
            source_path="dist/cli.js",
            snapshot_path="package/dist/cli.js",
            kind=lane_module._ImplementationEntryKind.REGULAR,
            mode=0o755,
            link_target=None,
            size=3,
            sha256="a" * 64,
        ),
        lane_module._CapturedImplementationEntry(
            source_path="node_modules/.bin/pi",
            snapshot_path="package/node_modules/.bin/pi",
            kind=lane_module._ImplementationEntryKind.SYMLINK,
            mode=0o777,
            link_target="../../dist/cli.js",
            size=17,
            sha256="b" * 64,
        ),
        lane_module._CapturedImplementationEntry(
            source_path="/opt/node/bin/node",
            snapshot_path="runtime/node",
            kind=lane_module._ImplementationEntryKind.REGULAR,
            mode=0o755,
            link_target=None,
            size=5,
            sha256="c" * 64,
        ),
    )

    assert (
        lane_module._implementation_identity_for(
            executable=Path("/opt/pi/dist/cli.js"),
            package_root=Path("/opt/pi"),
            entrypoint_path="package/dist/cli.js",
            runtime_source=Path("/opt/node/bin/node"),
            runtime_path="runtime/node",
            entries=entries,
        )
        == "f2d922b7dd1a3045104145fa866ccbe2ec77583d5c73cecb759012be476e1445"
    )


def test_real_pi_implementation_closure_constructs_when_installed() -> None:
    if shutil.which("pi") is None:
        pytest.skip("pi is not installed")

    lane = SubscriptionLane(
        provider="provider",
        model="model",
        timeout_seconds=10,
    )
    snapshot = lane._implementation_snapshot
    environment = lane_module._snapshot_environment(
        snapshot,
        lane._environment,
    )
    captured_version = run_subprocess(
        command=(*snapshot.command_prefix, "--version"),
        input_text="",
        timeout_seconds=10,
        environment=environment,
    )
    installed_version = run_subprocess(
        command=(lane.executable, "--version"),
        input_text="",
        timeout_seconds=10,
        environment=dict(lane._environment),
    )

    assert snapshot.entries
    assert captured_version.returncode == 0
    assert installed_version.returncode == 0
    assert captured_version.stdout.strip() == installed_version.stdout.strip()
    assert captured_version.stdout.strip()
    assert snapshot.runtime is not None
    assert str(snapshot.root) in str(snapshot.runtime)
    assert str(snapshot.root) in str(snapshot.entrypoint)
    assert lane.executable not in snapshot.command_prefix
    assert dict(lane.policy.transport)["implementation_policy"] == (
        "installed-node-runtime-closure-v2"
    )


def test_duplicate_key_retry_error_is_bounded() -> None:
    duplicate = "x" * 100_000
    lane = _ScriptedLane(
        [
            '{"label":"prose-no-code","rationale":"x",'
            f'"{duplicate}":1,"{duplicate}":2}}',
            "still invalid",
        ]
    )

    outcome = classify_one_repeat(lane, FailureFamily.PARSE, "prompt")

    assert outcome.failure is not None
    assert len(outcome.failure.detail) <= MAX_RESPONSE_ERROR_CHARS


def test_subscription_lane_bounds_nonzero_stderr_detail(
    tmp_path: Path,
) -> None:
    command = _write_provider_command(
        tmp_path,
        ("import sys\nsys.stderr.write('x' * 700)\nraise SystemExit(17)\n"),
    )
    lane = SubscriptionLane(
        provider="provider",
        model="model",
        timeout_seconds=2,
        executable=str(command),
    )

    with pytest.raises(LaneTransportError) as raised:
        lane.complete("prompt")

    assert raised.value.kind is TransportFailureKind.NONZERO_EXIT
    assert raised.value.detail == f"command exited 17: {'x' * 500}"


def _write_provider_command(
    directory: Path,
    source: str,
) -> Path:
    command = directory / "test-provider"
    command.write_text(f"#!{sys.executable}\n{source}")
    command.chmod(0o755)
    return command


def _write_node_fixture(directory: Path) -> Path:
    (directory / "package.json").write_text(
        '{"name":"fixture-provider","version":"1.0.0"}'
    )
    dependency_directory = directory / "node_modules" / "dependency"
    dependency_directory.mkdir(parents=True)
    (dependency_directory / "package.json").write_text(
        '{"name":"dependency","version":"1.0.0"}'
    )
    dependency = dependency_directory / "index.js"
    dependency.write_text("export default 'original';\n")
    _write_fixture_lock(
        directory,
        {
            "lockfileVersion": 3,
            "name": "fixture-provider",
            "packages": {
                "": {
                    "name": "fixture-provider",
                    "version": "1.0.0",
                },
                "node_modules/dependency": {"version": "1.0.0"},
            },
            "version": "1.0.0",
        },
    )
    return dependency


def _read_fixture_lock(directory: Path) -> dict[str, Any]:
    value = json.loads((directory / "npm-shrinkwrap.json").read_bytes())
    assert isinstance(value, dict)
    return value


def _write_fixture_lock(directory: Path, value: Mapping[str, Any]) -> None:
    (directory / "npm-shrinkwrap.json").write_text(
        json.dumps(value, separators=(",", ":"), sort_keys=True)
    )
