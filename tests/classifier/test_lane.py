from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from dr_code.classifier.classify import classify_one_repeat
from dr_code.classifier.lane import (
    MAX_RESPONSE_ERROR_CHARS,
    MAX_SUBSCRIPTION_OUTPUT_BYTES,
    LaneTransportError,
    SubscriptionLane,
    TransportFailureKind,
    parse_label_response,
)
from dr_code.classifier.taxonomy import FailureFamily


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


# The lane spawns through dr-exec's real engine below (sanctioned
# driver-body coverage). dr-exec owns spawn-path correctness — its own
# suite covers process-group teardown of descendants and errno-level start
# failures — so these tests assert only the lane's own contract: the command
# vector it builds, the granted environment it delivers, its
# attribution-keyed transport taxonomy, and its implementation-closure pin.


def test_subscription_lane_builds_the_expected_command(
    tmp_path: Path,
) -> None:
    command = _write_provider_command(
        tmp_path,
        (
            "import json, sys\n"
            "print(json.dumps({\n"
            "    'arguments': sys.argv[1:],\n"
            "    'stdin': sys.stdin.read(),\n"
            "}))\n"
        ),
    )
    lane = SubscriptionLane(
        provider="provider",
        model="model",
        timeout_seconds=30,
        executable=str(command),
    )

    response = json.loads(lane.complete("prompt"))

    assert response == {
        "arguments": _EXPECTED_ARGUMENTS,
        "stdin": "prompt",
    }


def test_subscription_lane_delivers_granted_environment_and_resolves_path(
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
        timeout_seconds=30,
        executable=command.name,
    )

    response = json.loads(lane.complete("prompt"))

    assert response == {
        "arguments": _EXPECTED_ARGUMENTS,
        "credential": "inherited",
        "stdin": "prompt",
    }


def test_subscription_lane_types_nonzero_exit_with_bounded_stderr(
    tmp_path: Path,
) -> None:
    command = _write_provider_command(
        tmp_path,
        ("import sys\nsys.stderr.write('x' * 700)\nraise SystemExit(17)\n"),
    )
    lane = SubscriptionLane(
        provider="provider",
        model="model",
        timeout_seconds=30,
        executable=str(command),
    )

    with pytest.raises(LaneTransportError) as raised:
        lane.complete("prompt")

    assert raised.value.kind is TransportFailureKind.NONZERO_EXIT
    assert raised.value.detail == f"command exited 17: {'x' * 500}"


def test_subscription_lane_types_timeout(tmp_path: Path) -> None:
    command = _write_provider_command(
        tmp_path,
        "import time\ntime.sleep(60)\n",
    )
    lane = SubscriptionLane(
        provider="provider",
        model="model",
        timeout_seconds=1,
        executable=str(command),
    )

    with pytest.raises(LaneTransportError) as raised:
        lane.complete("prompt")

    assert raised.value.kind is TransportFailureKind.TIMEOUT
    assert raised.value.detail == "command timed out after 1 seconds"


@pytest.mark.parametrize(
    "source",
    [
        (
            "import os\n"
            f"os.write(1, b'x' * {MAX_SUBSCRIPTION_OUTPUT_BYTES + 1})\n"
        ),
        (
            "import os\n"
            f"os.write(2, b'x' * {MAX_SUBSCRIPTION_OUTPUT_BYTES + 1})\n"
        ),
        (
            "import os\n"
            f"os.write(1, b'x' * "
            f"{MAX_SUBSCRIPTION_OUTPUT_BYTES // 2 + 1})\n"
            f"os.write(2, b'y' * "
            f"{MAX_SUBSCRIPTION_OUTPUT_BYTES // 2 + 1})\n"
        ),
    ],
    ids=["stdout", "stderr", "combined"],
)
def test_subscription_lane_types_output_limit(
    tmp_path: Path,
    source: str,
) -> None:
    command = _write_provider_command(tmp_path, source)
    lane = SubscriptionLane(
        provider="provider",
        model="model",
        timeout_seconds=30,
        executable=str(command),
    )

    with pytest.raises(LaneTransportError) as raised:
        lane.complete("prompt")

    assert raised.value.kind is TransportFailureKind.OUTPUT_LIMIT
    assert raised.value.detail == "command output exceeded limit"


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
        timeout_seconds=30,
        executable=first_command.name,
    )
    first_policy = dict(lane.policy.transport)
    monkeypatch.setenv("OPENAI_API_KEY", "second-secret")
    secret_rotated = SubscriptionLane(
        provider="provider",
        model="model",
        timeout_seconds=30,
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
        timeout_seconds=30,
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
    with pytest.raises(
        LaneTransportError,
        match="implementation closure changed after policy capture",
    ):
        lane.complete("prompt")


def test_subscription_lane_policy_detects_imported_sibling_mutation(
    tmp_path: Path,
) -> None:
    (tmp_path / "package.json").write_text(
        '{"name":"fixture-provider","version":"1.0.0"}'
    )
    helper = tmp_path / "provider_helper.py"
    helper.write_text("RESPONSE = 'first'\n")
    command = _write_provider_command(
        tmp_path,
        "from provider_helper import RESPONSE\nprint(RESPONSE)\n",
    )
    lane = SubscriptionLane(
        provider="provider",
        model="model",
        timeout_seconds=30,
        executable=str(command),
    )

    helper.write_text("RESPONSE = 'changed'\n")

    with pytest.raises(
        LaneTransportError,
        match="implementation closure changed after policy capture",
    ):
        _ = lane.policy


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


_EXPECTED_ARGUMENTS = [
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
]


def _write_provider_command(
    directory: Path,
    source: str,
) -> Path:
    command = directory / "test-provider"
    command.write_text(f"#!{sys.executable}\n{source}")
    command.chmod(0o755)
    return command
