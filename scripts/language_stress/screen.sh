#!/usr/bin/env bash
# Screen language-stress candidates on ONE policy at a small budget, then print the summary table.
#
# Each candidate is a JSON in screening/language_stress/candidates/ (make_candidates.py), run through
# the normal eval entry point with --instruction_file, outside the benchmark registry. Same runners
# and variables as scripts/run_benchmark.sh:
#
#   CHECKPOINT=/path/to/openvla/ckpt RUN_NOTE=openvla-ours GPUS=0,1,2,3 bash scripts/language_stress/screen.sh
#   MODEL_FAMILY=openpi CHECKPOINT=gs://openpi-assets/checkpoints/pi05_libero RUN_NOTE=pi05_libero \
#     bash scripts/language_stress/screen.sh                  # start scripts/serve_openpi.sh first
#   CHECKPOINT=/path/qwenvla.pt STATS=/path/stats.json RUN_NOTE=qwenvla EVAL_RUNNER=scripts/run_eval_qwenvla.sh \
#     RESULTS_DIR=/var/tmp/$USER/vla_ws_runs/logs/results bash scripts/language_stress/screen.sh
#
# NUM_TRIALS_PER_TASK defaults to 5 here (50 rollouts per 10-task condition, ±7 pts SE): enough to
# see which candidates move a 98% policy by tens of points, not to report numbers. CANDIDATES selects
# a subset (space-separated condition names). Extra flags go to the runner (e.g. --resume True).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${CHECKPOINT:?Set CHECKPOINT (and MODEL_FAMILY / EVAL_RUNNER as for run_benchmark.sh)}"
: "${RUN_NOTE:?Set RUN_NOTE to a short policy label, e.g. openvla-ours, pi05_libero, qwenvla}"
EVAL_RUNNER="${EVAL_RUNNER:-${ROOT}/docker/openvla_libero/run_eval.sh}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT}/openvla/experiments/logs/results}"
export NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-5}"
CAND_DIR="${ROOT}/screening/language_stress/candidates"
SCREEN_NOTE="${RUN_NOTE}--screen"

if [[ -n "${CANDIDATES:-}" ]]; then
  read -r -a names <<<"${CANDIDATES}"
else
  names=(); for f in "${CAND_DIR}"/*.json; do names+=("$(basename "${f}" .json)"); done
fi

for name in "${names[@]}"; do
  host_path="${CAND_DIR}/${name}.json"
  [[ -f "${host_path}" ]] || { echo "No candidate ${host_path}; run make_candidates.py" >&2; exit 1; }
  # The Docker runner mounts this workspace at /workspace; the host QwenVLA runner reads host paths.
  if [[ "$(basename "${EVAL_RUNNER}")" == run_eval_qwenvla.sh ]]; then
    arg_path="${host_path}"
  else
    arg_path="/workspace/${host_path#"${ROOT}/"}"
  fi
  echo "=== ${name} (run note: ${SCREEN_NOTE}, ${NUM_TRIALS_PER_TASK} trials/task) ==="
  bash "${EVAL_RUNNER}" --task_suite_name libero_spatial --unnorm_key libero_spatial \
    --instruction_file "${arg_path}" --run_id_note "${SCREEN_NOTE}" "$@"
done

python3 "${ROOT}/scripts/language_stress/summarize.py" --results-dir "${RESULTS_DIR}" "${SCREEN_NOTE}"
