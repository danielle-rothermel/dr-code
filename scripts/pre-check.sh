#!/usr/bin/env bash

set -uo pipefail

CACHE_DIR=".cache/pre-check"

mkdir -p "${CACHE_DIR}"

run_silent() {
    local name="$1"
    local output_file="$2"
    shift 2

    printf '  %s\n' "${name}"
    "$@" >"${output_file}" 2>&1
}

run_report() {
    local name="$1"
    local output_file="$2"
    shift 2

    printf '\n==> %s\n' "${name}"
    "$@" 2>&1 | tee "${output_file}"
    return "${PIPESTATUS[0]}"
}

printf 'Running quiet autofixes...\n'
run_silent "ruff check --fix" "${CACHE_DIR}/ruff-check-fix.txt" \
    uv run ruff check --fix .
run_silent "ty check --fix" "${CACHE_DIR}/ty-check-fix.txt" \
    uv run ty check --fix
run_silent "ruff format" "${CACHE_DIR}/ruff-format.txt" \
    uv run ruff format .

printf '\nRunning final checks...\n'

status=0

run_report "ruff check" "${CACHE_DIR}/ruff-check.txt" \
    uv run ruff check . || status=1
run_report "ty check" "${CACHE_DIR}/ty-check.txt" \
    uv run ty check || status=1

if command -v pnpm >/dev/null 2>&1; then
    run_report "viewer install" "${CACHE_DIR}/viewer-install.txt" \
        pnpm -C viewer install --frozen-lockfile || status=1
    run_report "viewer typecheck" "${CACHE_DIR}/viewer-typecheck.txt" \
        pnpm -C viewer typecheck || status=1
    run_report "viewer build" "${CACHE_DIR}/viewer-build.txt" \
        pnpm -C viewer build || status=1
else
    printf '\n==> viewer checks skipped (pnpm not found; CI always runs them)\n'
fi

printf '\nCheck output files:\n'
printf '  %s\n' "${CACHE_DIR}/ruff-check.txt"
printf '  %s\n' "${CACHE_DIR}/ty-check.txt"
if command -v pnpm >/dev/null 2>&1; then
    printf '  %s\n' "${CACHE_DIR}/viewer-typecheck.txt"
    printf '  %s\n' "${CACHE_DIR}/viewer-build.txt"
fi

if [[ "${status}" -ne 0 ]]; then
    printf '\nFix all reported issues, then rerun:\n'
    printf '  scripts/pre-check.sh\n'
fi

exit "${status}"
