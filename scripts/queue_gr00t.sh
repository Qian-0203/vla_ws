#!/usr/bin/env bash
# Run GR00T N1.5's jobs through scripts/gpu_queue.py: one policy-server container per GPU slot
# (scripts/policy_servers/serve_gr00t.py, image openvla-libero:gr00t), the (split, task) queue on top, then teardown.
# The eval side is the generic client: `--model_family openpi --policy_obs raw --replan_steps 1` (NVIDIA's own LIBERO
# eval queries at every step and executes the first predicted action).
#
#   CKPT=$HOME/gr00t_ckpt/gr00t-n1.5-libero-spatial-posttrain NOTE=gr00t_n15_spatial SLOTS=0,1,2,3 \
#     SPLITS=spatial/default,grounding/landmark_exchange bash scripts/queue_gr00t.sh
#
# Env: CKPT (checkpoint dir), NOTE (run_id_note), SLOTS (GPU per slot, repeats allowed), SPLITS (default:
# run_benchmark.sh DEFAULT_SPLITS), TASK_IDS / TRIALS (smoke tests), PORT_BASE (8600), GR00T_DIR (checkout of
# NVIDIA/Isaac-GR00T at tag n1.5-release, default ~/Isaac-GR00T), SERVER_IMAGE (openvla-libero:gr00t), EVAL_IMAGE
# (openvla-libero:blackwell), DATA_CONFIG (examples.Libero.custom_data_config:LiberoDataConfig; the goal checkpoint
# uses ...:LiberoDataConfigMeanStd), DENOISING_STEPS (8), THREADS (CPU thread-pool cap per server, default 4),
# QUEUE_ARGS. Re-running resumes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CKPT:?checkpoint directory}"
: "${NOTE:?run_id_note}"
SLOTS="${SLOTS:-0,1,2,3}"
PORT_BASE="${PORT_BASE:-8600}"
GR00T_DIR="${GR00T_DIR:-$HOME/Isaac-GR00T}"
SERVER_IMAGE="${SERVER_IMAGE:-openvla-libero:gr00t}"
EVAL_IMAGE="${EVAL_IMAGE:-openvla-libero:blackwell}"
DATA_CONFIG="${DATA_CONFIG:-examples.Libero.custom_data_config:LiberoDataConfig}"
DENOISING_STEPS="${DENOISING_STEPS:-8}"
LOG_DIR="${HOME}/logs/queue_${NOTE}"
mkdir -p "${LOG_DIR}"
[[ -d "${GR00T_DIR}/gr00t" ]] || { echo "no Isaac-GR00T checkout at ${GR00T_DIR}" >&2; exit 1; }
[[ -f "${CKPT}/config.json" ]] || { echo "no config.json in ${CKPT}" >&2; exit 1; }

names=()
stop_servers() { for n in "${names[@]:-}"; do [[ -n "$n" ]] && docker rm -f "$n" >/dev/null 2>&1 || true; done; }
trap stop_servers EXIT

IFS=',' read -r -a GPU_ARR <<<"${SLOTS}"
for idx in "${!GPU_ARR[@]}"; do
  port=$((PORT_BASE + idx))
  name="gr00t_${NOTE}_${port}"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker run -d --rm --name "${name}" --network host --gpus "device=${GPU_ARR[$idx]}" --ipc=host \
    --user "$(id -u):$(id -g)" -w /gr00t \
    -e PYTHONPATH=/gr00t -e HOME=/tmp -e HF_HOME=/tmp/hf -e PYTHONNOUSERSITE=1 \
    -e OMP_NUM_THREADS="${THREADS:-4}" -e MKL_NUM_THREADS="${THREADS:-4}" \
    -e TF_NUM_INTRAOP_THREADS="${THREADS:-4}" -e TF_NUM_INTEROP_THREADS=2 \
    -v "${GR00T_DIR}:/gr00t:ro" -v "${ROOT}/scripts/policy_servers:/serve:ro" -v "${CKPT}:${CKPT}:ro" \
    "${SERVER_IMAGE}" python3 /serve/serve_gr00t.py --model_path "${CKPT}" --port "${port}" \
      --data_config "${DATA_CONFIG}" --denoising_steps "${DENOISING_STEPS}" >/dev/null
  names+=("${name}")
  echo "server ${idx}: GPU ${GPU_ARR[$idx]}, port ${port}, container ${name}"
done

echo "Waiting for the servers (model load) ..."
for idx in "${!GPU_ARR[@]}"; do
  port=$((PORT_BASE + idx))
  until (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null; do
    sleep 5
    docker ps -q --filter "name=${names[$idx]}" | grep -q . || { echo "server on port ${port} exited:" >&2; docker logs "${names[$idx]}" 2>&1 | tail -25 >&2 || true; exit 1; }
  done
  echo "port ${port} ready"
done

job_args=()
[[ -n "${SPLITS:-}" ]] && job_args+=(--splits "${SPLITS}")
[[ -n "${TASK_IDS:-}" ]] && job_args+=(--task-ids "${TASK_IDS}")
[[ -n "${TRIALS:-}" ]] && job_args+=(--trials "${TRIALS}")
python3 "${ROOT}/scripts/make_eval_jobs.py" --note "${NOTE}" --family openpi --checkpoint "${CKPT}" \
  --image "${EVAL_IMAGE}" --extra "--policy_obs raw --replan_steps 1" "${job_args[@]}" >"${LOG_DIR}/jobs.tsv"
# shellcheck disable=SC2086
python3 "${ROOT}/scripts/gpu_queue.py" --jobs "${LOG_DIR}/jobs.tsv" --slots "${SLOTS}" --port-base "${PORT_BASE}" \
  --log-dir "${LOG_DIR}" --stagger 20 ${QUEUE_ARGS:-}
