#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  scripts/run_generation_corpus_tests.sh [PYTEST_ARG...]

Runs dr-code's opt-in generation corpus extraction tests (pytest marker:
generation_corpus).

Example:
  scripts/run_generation_corpus_tests.sh
  scripts/run_generation_corpus_tests.sh tests/generation_corpus/test_writer.py
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
    pytest_args=(tests/generation_corpus)
fi

exec uv run pytest -q \
    -o addopts='--import-mode=importlib' \
    -m generation_corpus \
    "${pytest_args[@]}"
