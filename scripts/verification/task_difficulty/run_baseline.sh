#!/usr/bin/env bash
# Run the full directional HumanEval baseline workflow and export git-tracked logs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BASELINE_NAME="${1:-pre-106}"

REVIEWED_CORPUS="${HOME}/drotherm/data/code-comp/generation-corpora/2026-08-08-reviewed/human_eval"
FIXTURE_CORPUS="${REPO_ROOT}/tests/fixtures/generation_corpus/human_eval"
REVIEWED_MANIFEST_SHA256="fc6a3e6bb33d446d3ccf98bd33a44b7c05225a79340c643859b0d68f9d3d0728"

CORPUS_BUNDLE="${DR_CODE_GENERATION_CORPUS_BUNDLE:-${REVIEWED_CORPUS}}"
MANIFEST_OVERRIDE="${DR_CODE_EXPECTED_MANIFEST_SHA256-__UNSET__}"
if [[ "${MANIFEST_OVERRIDE}" != "__UNSET__" ]]; then
  MANIFEST_SHA256="${MANIFEST_OVERRIDE}"
elif [[ "${CORPUS_BUNDLE}" == "${REVIEWED_CORPUS}" ]]; then
  MANIFEST_SHA256="${REVIEWED_MANIFEST_SHA256}"
else
  MANIFEST_SHA256=""
fi
RUN_DIR="${DR_CODE_TASK_DIFFICULTY_RUN_DIR:-${HOME}/drotherm/data/.codex/dr-code/task-difficulty-directional/runs/${BASELINE_NAME}}"
EXPORT_DIR="${REPO_ROOT}/scripts/verification/task_difficulty/baseline/${BASELINE_NAME}"
WORKERS="${DR_CODE_EVALUATION_WORKERS:-16}"
TIMEOUT="${DR_CODE_EVALUATION_TIMEOUT_SECONDS:-120}"
EVALUATION_VENV="${DR_CODE_EVALUATION_VENV:-${REPO_ROOT}/.evaluation-venv}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [baseline-name]

Run stages 01-04 of the directional HumanEval workflow, then export logs and
summaries into a git-tracked baseline directory.

Defaults:
  baseline name: pre-106
  corpus bundle: ${REVIEWED_CORPUS}
  manifest SHA-256: ${REVIEWED_MANIFEST_SHA256}
  run directory: ${HOME}/drotherm/data/.codex/dr-code/task-difficulty-directional/runs/<baseline-name>
  export directory: scripts/verification/task_difficulty/baseline/<baseline-name>

Fixture smoke run:
  DR_CODE_GENERATION_CORPUS_BUNDLE=${FIXTURE_CORPUS} \\
  DR_CODE_TASK_DIFFICULTY_RUN_DIR=/tmp/task-difficulty-fixture \\
  DR_CODE_DISPOSABLE_WORKER=1 \\
  $(basename "$0") fixture-smoke

Environment overrides:
  DR_CODE_GENERATION_CORPUS_BUNDLE
  DR_CODE_EXPECTED_MANIFEST_SHA256
  DR_CODE_TASK_DIFFICULTY_RUN_DIR
  DR_CODE_EVALUATION_WORKERS
  DR_CODE_EVALUATION_TIMEOUT_SECONDS
  DR_CODE_EVALUATION_PYTHON
  DR_CODE_EVALUATION_VENV
  DR_CODE_DISPOSABLE_WORKER=1   (required for stage 3)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

export DR_CODE_GENERATION_CORPUS_BUNDLE="${CORPUS_BUNDLE}"
export DR_CODE_TASK_DIFFICULTY_RUN_DIR="${RUN_DIR}"
if [[ -n "${MANIFEST_SHA256}" ]]; then
  export DR_CODE_EXPECTED_MANIFEST_SHA256="${MANIFEST_SHA256}"
else
  unset DR_CODE_EXPECTED_MANIFEST_SHA256
fi

mkdir -p "${RUN_DIR}" "${EXPORT_DIR}/logs"
RUN_LOG="${EXPORT_DIR}/logs/run.log"

cd "${REPO_ROOT}"

{
  echo "baseline=${BASELINE_NAME}"
  echo "corpus_bundle=${CORPUS_BUNDLE}"
  echo "run_dir=${RUN_DIR}"
  echo "export_dir=${EXPORT_DIR}"
  echo "manifest_sha256=${MANIFEST_SHA256:-<unpinned>}"
  echo "workers=${WORKERS}"
  echo "timeout_seconds=${TIMEOUT}"
  echo "git_rev=$(git rev-parse HEAD)"
  echo "git_branch=$(git branch --show-current)"
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee "${RUN_LOG}"

echo "Stage 1: build eligible corpus" | tee -a "${RUN_LOG}"
uv run python scripts/verification/task_difficulty/01_build_eligible_corpus.py \
  2>&1 | tee -a "${RUN_LOG}"

echo "Stage 2: select balanced sample" | tee -a "${RUN_LOG}"
uv run python scripts/verification/task_difficulty/02_select_balanced_sample.py \
  2>&1 | tee -a "${RUN_LOG}"

if [[ "${DR_CODE_DISPOSABLE_WORKER:-}" != "1" ]]; then
  cat <<EOF | tee -a "${RUN_LOG}"
Stage 3 skipped: set DR_CODE_DISPOSABLE_WORKER=1 before running evaluation.
Historical model output executes with the worker's permissions; use a disposable host.
EOF
  uv run python scripts/verification/task_difficulty/05_export_baseline.py \
    "${BASELINE_NAME}" \
    --workers "${WORKERS}" \
    --timeout-seconds "${TIMEOUT}" \
    2>&1 | tee -a "${RUN_LOG}"
  echo "Exported partial baseline (stages 1-2 only) to ${EXPORT_DIR}" | tee -a "${RUN_LOG}"
  exit 0
fi

if [[ -z "${DR_CODE_EVALUATION_PYTHON:-}" ]]; then
  if [[ ! -x "${EVALUATION_VENV}/bin/python3" ]]; then
    echo "Creating evaluation venv at ${EVALUATION_VENV}" | tee -a "${RUN_LOG}"
    python3 -m venv --copies "${EVALUATION_VENV}"
    uv pip install --python "${EVALUATION_VENV}/bin/python3" .
  fi
  export DR_CODE_EVALUATION_PYTHON="${EVALUATION_VENV}/bin/python3"
fi

echo "Stage 3: evaluate sample with ${DR_CODE_EVALUATION_PYTHON}" | tee -a "${RUN_LOG}"
uv run python scripts/verification/task_difficulty/03_evaluate_sample.py \
  --workers "${WORKERS}" \
  --timeout-seconds "${TIMEOUT}" \
  2>&1 | tee -a "${RUN_LOG}"

echo "Stage 4: summarize results" | tee -a "${RUN_LOG}"
set +e
uv run python scripts/verification/task_difficulty/04_summarize_results.py \
  --workers "${WORKERS}" \
  --timeout-seconds "${TIMEOUT}" \
  2>&1 | tee -a "${RUN_LOG}"
summarize_status=${PIPESTATUS[0]}
set -e
if [[ "${summarize_status}" -ne 0 ]]; then
  echo "Stage 4 exited ${summarize_status}; exporting available logs anyway" | tee -a "${RUN_LOG}"
fi

echo "Export baseline artifacts" | tee -a "${RUN_LOG}"
uv run python scripts/verification/task_difficulty/05_export_baseline.py \
  "${BASELINE_NAME}" \
  --workers "${WORKERS}" \
  --timeout-seconds "${TIMEOUT}" \
  2>&1 | tee -a "${RUN_LOG}"

echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${RUN_LOG}"
echo "Baseline exported to ${EXPORT_DIR}" | tee -a "${RUN_LOG}"
