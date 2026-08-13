#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  scripts/run_addon_tests.sh [PYTEST_ARG...]

Runs all opt-in domain-extension tests in the drc_* packages.

Example:
  scripts/run_addon_tests.sh
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

repo_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "${repo_root}"

uv sync --locked --group dev --group addons >/dev/null

declare -a pytest_args=("$@")
if [[ "${#pytest_args[@]}" -eq 0 ]]; then
    pytest_args=(
        packages/drc-humaneval/tests
        packages/drc-synthetic/tests
        packages/drc-generation-corpus/tests
    )
fi

exec uv run pytest -q \
    -o addopts='--import-mode=importlib' \
    "${pytest_args[@]}"
