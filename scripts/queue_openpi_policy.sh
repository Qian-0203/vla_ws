#!/usr/bin/env bash
# Run one openpi policy's eval jobs through scripts/gpu_queue.py on this host:
# start one policy server per slot, queue (split, task) jobs over the slots, stop the servers.
#
#   OPENPI_POLICY_CONFIG=pi0_libero OPENPI_POLICY_DIR=gs://openpi-assets/checkpoints/pi0_libero \
#     NOTE=pi0_libero SLOTS=0,0,1,1,2,2,3,3 SPLITS=spatial/default,grounding/landmark_exchange \
#     bash scripts/queue_openpi_policy.sh
#
# Env: NOTE (run_id_note; tags the results files), SLOTS (GPU per slot, repeats = several servers
# per GPU), SPLITS (comma list; default = run_benchmark.sh's DEFAULT_SPLITS), PORT_BASE (8000),
# QUEUE_ARGS (extra flags for gpu_queue.py, e.g. "--retry-failed"). Re-running resumes: finished
# jobs are skipped and half-finished ones continue (--resume True).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${OPENPI_POLICY_CONFIG:?e.g. pi0_libero}"
: "${OPENPI_POLICY_DIR:?e.g. gs://openpi-assets/checkpoints/pi0_libero}"
: "${NOTE:?run_id_note, e.g. pi0_libero}"
SLOTS="${SLOTS:-0,0,1,1,2,2,3,3}"
PORT_BASE="${PORT_BASE:-8000}"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

LOG_DIR="${LOG_DIR:-${HOME}/logs/queue_${NOTE}}"
JOBS="${LOG_DIR}/jobs.tsv"
export PID_FILE="${LOG_DIR}/servers.pid"
mkdir -p "${LOG_DIR}"

export OPENPI_POLICY_CONFIG OPENPI_POLICY_DIR
stop_servers() { bash "${ROOT}/scripts/serve_openpi.sh" stop || true; }
trap stop_servers EXIT

splits_args=()
[[ -n "${SPLITS:-}" ]] && splits_args=(--splits "${SPLITS}")
# Language conditions outside the registry (make_eval_jobs.py --instruction-file), e.g.
# INSTRUCTION_FILES=screening/language_stress/candidates/target_swap.json,...; alone, only those jobs are queued
# (give them their own LOG_DIR so the split queue's markers and jobs.tsv are not touched).
INSTRUCTION_FILES="${INSTRUCTION_FILES:-}"
for f in ${INSTRUCTION_FILES//,/ }; do splits_args+=(--instruction-file "$f"); done
python3 "${ROOT}/scripts/make_eval_jobs.py" --note "${NOTE}" --family openpi \
  --checkpoint "${OPENPI_POLICY_DIR}" "${splits_args[@]}" >"${JOBS}"

GPUS="${SLOTS}" POLICY_PORT_BASE="${PORT_BASE}" bash "${ROOT}/scripts/serve_openpi.sh"

# Servers are resident, so no free-memory gate; stagger only spaces out container starts.
# shellcheck disable=SC2086
python3 "${ROOT}/scripts/gpu_queue.py" --jobs "${JOBS}" --slots "${SLOTS}" --port-base "${PORT_BASE}" \
  --log-dir "${LOG_DIR}" --stagger 20 ${QUEUE_ARGS:-}
