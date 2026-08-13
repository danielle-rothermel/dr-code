#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  scripts/run_addon_tests.sh [PYTEST_ARG...]

Runs all opt-in domain-extension tests: humaneval, synthetic, and
generation_corpus.

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

declare -a pytest_args=("$@")
if [[ "${#pytest_args[@]}" -eq 0 ]]; then
    pytest_args=(tests/humaneval tests/synthetic tests/generation_corpus)
fi

exec uv run pytest -q \
    -o addopts='--import-mode=importlib' \
    -m "humaneval or synthetic or generation_corpus" \
    "${pytest_args[@]}"
