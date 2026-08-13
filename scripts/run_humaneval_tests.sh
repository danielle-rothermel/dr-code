#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  scripts/run_humaneval_tests.sh [PYTEST_ARG...]

Runs dr-code's opt-in HumanEval+ adapter tests (pytest marker: humaneval).

Example:
  scripts/run_humaneval_tests.sh
  scripts/run_humaneval_tests.sh tests/humaneval/test_sampling.py
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

repo_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "${repo_root}"

declare -a pytest_args=("$@")
if [[ "${#pytest_args[@]}" -eq 0 ]]; then
    pytest_args=(tests/humaneval)
fi

exec uv run pytest -q \
    -o addopts='--import-mode=importlib' \
    -m humaneval \
    "${pytest_args[@]}"
