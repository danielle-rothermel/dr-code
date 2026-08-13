#!/usr/bin/env bash
# Run the full directional HumanEval baseline workflow and export git-tracked results.
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
WORKERS="${DR_CODE_EVAL_WORKERS:-16}"
TIMEOUT="${DR_CODE_EVAL_TIMEOUT_SECONDS:-120}"
PREPROCESS_TIMEOUT="${DR_CODE_PREPROCESS_TIMEOUT_SECONDS:-600}"
TASKS_PER_GROUP="${DR_CODE_SAMPLE_TASKS_PER_GROUP:-40}"
EVAL_VENV="${DR_CODE_EVAL_VENV:-${REPO_ROOT}/.evaluation-venv}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [baseline-name]

Run stages 01-04 of the directional HumanEval workflow, then export run config and
results into a git-tracked baseline directory. Verbose stage logs are written
only under the run directory.

Defaults:
  baseline name: pre-106
  corpus bundle: ${REVIEWED_CORPUS}
  manifest SHA-256: ${REVIEWED_MANIFEST_SHA256}
  per-item preprocessing timeout: 600s (0 or none runs unbudgeted)
  tasks per sampling group: 40 (0 or all keeps every task)
  run directory: ${HOME}/drotherm/data/.codex/dr-code/task-difficulty-directional/runs/<baseline-name>
  export directory: scripts/verification/task_difficulty/baseline/<baseline-name>

Fixture smoke run:
  DR_CODE_GENERATION_CORPUS_BUNDLE=${FIXTURE_CORPUS} \\
  DR_CODE_TASK_DIFFICULTY_RUN_DIR=/tmp/task-difficulty-fixture \\
  $(basename "$0") fixture-smoke

Environment overrides:
  DR_CODE_GENERATION_CORPUS_BUNDLE
  DR_CODE_EXPECTED_MANIFEST_SHA256
  DR_CODE_TASK_DIFFICULTY_RUN_DIR
  DR_CODE_EVAL_WORKERS
  DR_CODE_EVAL_TIMEOUT_SECONDS
  DR_CODE_PREPROCESS_TIMEOUT_SECONDS (per-item stage-1 watchdog; 0/none disables)
  DR_CODE_SAMPLE_TASKS_PER_GROUP (stage-2 tasks per group; 0/all keeps every task)
  DR_CODE_EVAL_PYTHON
  DR_CODE_EVAL_VENV
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

mkdir -p "${RUN_DIR}/logs" "${EXPORT_DIR}"
RUN_LOG="${RUN_DIR}/logs/run.log"

cd "${REPO_ROOT}"

{
  echo "baseline=${BASELINE_NAME}"
  echo "corpus_bundle=${CORPUS_BUNDLE}"
  echo "run_dir=${RUN_DIR}"
  echo "export_dir=${EXPORT_DIR}"
  echo "manifest_sha256=${MANIFEST_SHA256:-<unpinned>}"
  echo "workers=${WORKERS}"
  echo "timeout_seconds=${TIMEOUT}"
  echo "preprocess_timeout_seconds=${PREPROCESS_TIMEOUT}"
  echo "tasks_per_group=${TASKS_PER_GROUP}"
  echo "git_rev=$(git rev-parse HEAD)"
  echo "git_branch=$(git branch --show-current)"
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee "${RUN_LOG}"

echo "Stage 1: build eligible corpus" | tee -a "${RUN_LOG}"
uv run python scripts/verification/task_difficulty/01_build_eligible_corpus.py \
  --workers "${WORKERS}" \
  --preprocess-timeout-seconds "${PREPROCESS_TIMEOUT}" \
  2>&1 | tee -a "${RUN_LOG}"

echo "Stage 2: select balanced sample" | tee -a "${RUN_LOG}"
uv run python scripts/verification/task_difficulty/02_select_balanced_sample.py \
  --tasks-per-group "${TASKS_PER_GROUP}" \
  2>&1 | tee -a "${RUN_LOG}"

if [[ -z "${DR_CODE_EVAL_PYTHON:-}" ]]; then
  if [[ ! -x "${EVAL_VENV}/bin/python3" ]]; then
    echo "Creating evaluation venv at ${EVAL_VENV}" | tee -a "${RUN_LOG}"
    python3 -m venv --copies "${EVAL_VENV}"
    uv pip install --python "${EVAL_VENV}/bin/python3" .
  fi
  export DR_CODE_EVAL_PYTHON="${EVAL_VENV}/bin/python3"
fi

echo "Stage 3: evaluate sample with ${DR_CODE_EVAL_PYTHON}" | tee -a "${RUN_LOG}"
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
