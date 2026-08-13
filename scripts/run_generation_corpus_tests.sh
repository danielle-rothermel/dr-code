#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  scripts/run_synthetic_tests.sh [PYTEST_ARG...]

Runs drc-generation-corpus tests.

Example:
  scripts/run_generation_corpus_tests.sh
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
    pytest_args=(packages/drc-generation-corpus/tests)
fi

exec uv run --package drc-generation-corpus pytest -q \
    -o addopts='--import-mode=importlib' \
    "${pytest_args[@]}"
