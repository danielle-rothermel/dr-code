#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  scripts/run_postgres_tests.sh [PYTEST_ARG...]

Runs dr-code's opt-in postgres-marked tests through dr-store's scratch-server
script. Set DR_STORE_ROOT to a dr-store checkout that provides
scripts/test-postgres.sh.

Example:
  export DR_STORE_ROOT=/path/to/dr-store
  scripts/run_postgres_tests.sh
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ -z "${DR_STORE_ROOT:-}" ]]; then
    cat <<'EOF' >&2
DR_STORE_ROOT is not set. Export the path to a dr-store checkout, then re-run:
  export DR_STORE_ROOT=/path/to/dr-store
  scripts/run_postgres_tests.sh
EOF
    exit 1
fi

scratch="${DR_STORE_ROOT}/scripts/test-postgres.sh"
if [[ ! -x "${scratch}" ]]; then
    cat <<EOF >&2
Expected an executable scratch-server script at:
  ${scratch}

Verify DR_STORE_ROOT points at a dr-store checkout root.
EOF
    exit 1
fi

repo_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "${repo_root}"

declare -a pytest_args=("$@")
if [[ "${#pytest_args[@]}" -eq 0 ]]; then
    pytest_args=(tests/evaluation/test_evidence_postgres.py)
fi

exec "${scratch}" -- uv run pytest -q -m postgres "${pytest_args[@]}"
