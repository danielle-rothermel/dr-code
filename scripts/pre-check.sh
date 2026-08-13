#!/usr/bin/env bash

set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "${repo_root}"
export UV_LOCKED=1

CACHE_DIR=".cache/pre-check"
mkdir -p "${CACHE_DIR}"

run_report() {
    local name="$1"
    local output_file="$2"
    shift 2

    printf '\n==> %s\n' "${name}"
    "$@" 2>&1 | tee "${output_file}"
    return "${PIPESTATUS[0]}"
}

run_viewer() {
    (
        cd -- "${repo_root}/viewer"
        CI=1 corepack pnpm "$@"
    )
}

run_report "locked environment" "${CACHE_DIR}/uv-sync.txt" \
    uv sync --locked

case "${1:-}" in
    "") ;;
    --fix)
        printf 'Running explicit autofixes...\n'
        uv run ruff check --fix .
        uv run ty check --fix
        uv run ruff format .
        ;;
    *)
        printf 'usage: %s [--fix]\n' "$0" >&2
        exit 2
        ;;
esac

printf 'Running final checks...\n'
status=0

run_report "ruff format" "${CACHE_DIR}/ruff-format.txt" \
    uv run ruff format --check . || status=1
run_report "ruff check" "${CACHE_DIR}/ruff-check.txt" \
    uv run ruff check . || status=1
run_report "ty check" "${CACHE_DIR}/ty-check.txt" \
    uv run ty check || status=1
run_report ".defs schema lint" "${CACHE_DIR}/defs-lint.txt" \
    uvx tombi@1.2.5 lint --offline \
        .defs/terms.toml .defs/contracts.toml || status=1
run_report "Python tests" "${CACHE_DIR}/pytest.txt" \
    uv run pytest || status=1

if command -v corepack >/dev/null 2>&1; then
    run_report "viewer install" "${CACHE_DIR}/viewer-install.txt" \
        run_viewer install --frozen-lockfile || status=1
    run_report "viewer typecheck" "${CACHE_DIR}/viewer-typecheck.txt" \
        run_viewer typecheck || status=1
    run_report "viewer build" "${CACHE_DIR}/viewer-build.txt" \
        run_viewer build || status=1
else
    printf '\n==> viewer checks failed (corepack not found)\n' >&2
    status=1
fi

printf '\nCheck output files: %s\n' "${CACHE_DIR}"

if [[ "${status}" -ne 0 ]]; then
    printf '\nFix all reported issues, then rerun:\n'
    printf '  scripts/pre-check.sh\n'
fi

exit "${status}"
