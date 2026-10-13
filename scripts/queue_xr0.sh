#!/usr/bin/env bash
# Run Xiaomi-Robotics-0's jobs through scripts/gpu_queue.py: one policy-server container per GPU slot
# (scripts/policy_servers/serve_xr0.py, image openvla-libero:xr0), the (split, task) queue on top, then teardown.
# The eval side is the generic client: `--model_family openpi --policy_obs raw --replan_steps 10` (the authors'
# LIBERO eval executes the whole 10-step chunk).
#
#   CKPT=$HOME/xr0_ckpt/Xiaomi-Robotics-0-LIBERO NOTE=xr0_libero SLOTS=0,1,2,3 \
#     SPLITS=spatial/default,grounding/landmark_exchange bash scripts/queue_xr0.sh
#
# Env: CKPT (checkpoint dir), NOTE (run_id_note), SLOTS (GPU per slot, repeats allowed), SPLITS (default:
# run_benchmark.sh DEFAULT_SPLITS), INSTRUCTION_FILES, TASK_IDS / TRIALS (smoke tests), PORT_BASE (8800),
# SERVER_IMAGE (openvla-libero:xr0), EVAL_IMAGE (openvla-libero:blackwell), ATTN (sdpa), THREADS (CPU thread-pool
# cap per server, default 4), LOG_DIR, QUEUE_ARGS. Re-running resumes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CKPT:?checkpoint directory}"
: "${NOTE:?run_id_note}"
SLOTS="${SLOTS:-0,1,2,3}"
PORT_BASE="${PORT_BASE:-8800}"
SERVER_IMAGE="${SERVER_IMAGE:-openvla-libero:xr0}"
EVAL_IMAGE="${EVAL_IMAGE:-openvla-libero:blackwell}"
ATTN="${ATTN:-sdpa}"
LOG_DIR="${LOG_DIR:-${HOME}/logs/queue_${NOTE}}"
mkdir -p "${LOG_DIR}"
[[ -f "${CKPT}/config.json" ]] || { echo "no config.json in ${CKPT}" >&2; exit 1; }

names=()
stop_servers() { for n in "${names[@]:-}"; do [[ -n "$n" ]] && docker rm -f "$n" >/dev/null 2>&1 || true; done; }
trap stop_servers EXIT

IFS=',' read -r -a GPU_ARR <<<"${SLOTS}"
for idx in "${!GPU_ARR[@]}"; do
  port=$((PORT_BASE + idx))
  name="xr0_${NOTE}_${port}"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker run -d --rm --name "${name}" --network host --gpus "device=${GPU_ARR[$idx]}" --ipc=host \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e HF_HOME=/tmp/hf -e PYTHONNOUSERSITE=1 -e TOKENIZERS_PARALLELISM=false \
    -e OMP_NUM_THREADS="${THREADS:-4}" -e MKL_NUM_THREADS="${THREADS:-4}" \
    -v "${ROOT}/scripts/policy_servers:/serve:ro" -v "${CKPT}:${CKPT}:ro" \
    "${SERVER_IMAGE}" python3 /serve/serve_xr0.py --model_path "${CKPT}" --port "${port}" --attn "${ATTN}" >/dev/null
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
INSTRUCTION_FILES="${INSTRUCTION_FILES:-}"
for f in ${INSTRUCTION_FILES//,/ }; do job_args+=(--instruction-file "$f"); done
[[ -n "${TASK_IDS:-}" ]] && job_args+=(--task-ids "${TASK_IDS}")
[[ -n "${TRIALS:-}" ]] && job_args+=(--trials "${TRIALS}")
python3 "${ROOT}/scripts/make_eval_jobs.py" --note "${NOTE}" --family openpi --checkpoint "${CKPT}" \
  --image "${EVAL_IMAGE}" --extra "--policy_obs raw --replan_steps 10" "${job_args[@]}" >"${LOG_DIR}/jobs.tsv"
# shellcheck disable=SC2086
python3 "${ROOT}/scripts/gpu_queue.py" --jobs "${LOG_DIR}/jobs.tsv" --slots "${SLOTS}" --port-base "${PORT_BASE}" \
  --log-dir "${LOG_DIR}" --stagger 20 ${QUEUE_ARGS:-}
