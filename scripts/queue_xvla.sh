#!/usr/bin/env bash
# Run X-VLA's jobs through scripts/gpu_queue.py: one X-VLA policy-server container per GPU slot
# (scripts/policy_servers/serve_xvla.py, image openvla-libero:xvla), the (split, task) queue on top, then teardown.
# The eval side is `run_libero_eval.py --model_family xvla` (absolute-pose control, 30-step chunks).
#
#   CKPT=$HOME/xvla_ckpt/X-VLA-Libero NOTE=xvla_libero SLOTS=0,1,2,3 MAX_STEPS=800 \
#     SPLITS=spatial/default,grounding/landmark_exchange bash scripts/queue_xvla.sh
#
# Env: CKPT (X-VLA checkpoint dir), NOTE (run_id_note), SLOTS (GPU per slot, repeats allowed), SPLITS (default:
# run_benchmark.sh DEFAULT_SPLITS), MAX_STEPS (episode horizon after the 10 settle steps; 0 = the suite's usual 220,
# X-VLA's own protocol uses 800), TASK_IDS / TRIALS (smoke tests), PORT_BASE (8500), XVLA_DIR (checkout of
# 2toINF/X-VLA, default ~/X-VLA), SERVER_IMAGE (openvla-libero:xvla), EVAL_IMAGE (openvla-libero:blackwell),
# LORA (adapter dir, optional), QUEUE_ARGS. Re-running resumes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CKPT:?checkpoint directory}"
: "${NOTE:?run_id_note}"
SLOTS="${SLOTS:-0,1,2,3}"
PORT_BASE="${PORT_BASE:-8500}"
XVLA_DIR="${XVLA_DIR:-$HOME/X-VLA}"
SERVER_IMAGE="${SERVER_IMAGE:-openvla-libero:xvla}"
EVAL_IMAGE="${EVAL_IMAGE:-openvla-libero:blackwell}"
LOG_DIR="${HOME}/logs/queue_${NOTE}"
mkdir -p "${LOG_DIR}"
[[ -d "${XVLA_DIR}/models" ]] || { echo "no X-VLA checkout at ${XVLA_DIR}" >&2; exit 1; }

names=()
stop_servers() { for n in "${names[@]:-}"; do [[ -n "$n" ]] && docker rm -f "$n" >/dev/null 2>&1 || true; done; }
trap stop_servers EXIT

lora_args=()
[[ -n "${LORA:-}" ]] && lora_args=(--lora_path "${LORA}")

IFS=',' read -r -a GPU_ARR <<<"${SLOTS}"
for idx in "${!GPU_ARR[@]}"; do
  port=$((PORT_BASE + idx))
  name="xvla_${NOTE}_${port}"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker run -d --rm --name "${name}" --network host --gpus "device=${GPU_ARR[$idx]}" --ipc=host \
    --user "$(id -u):$(id -g)" -w /xvla \
    -e PYTHONPATH=/xvla -e HOME=/tmp -e HF_HOME=/tmp/hf -e PYTHONNOUSERSITE=1 \
    -e OMP_NUM_THREADS="${THREADS:-4}" -e MKL_NUM_THREADS="${THREADS:-4}" \
    -e TF_NUM_INTRAOP_THREADS="${THREADS:-4}" -e TF_NUM_INTEROP_THREADS=2 \
    -v "${XVLA_DIR}:/xvla:ro" -v "${ROOT}/scripts/policy_servers:/serve:ro" -v "${CKPT}:${CKPT}:ro" \
    ${LORA:+-v "${LORA}:${LORA}:ro"} \
    "${SERVER_IMAGE}" python3 /serve/serve_xvla.py --model_path "${CKPT}" --port "${port}" "${lora_args[@]}" >/dev/null
  names+=("${name}")
  echo "server ${idx}: GPU ${GPU_ARR[$idx]}, port ${port}, container ${name}"
done

echo "Waiting for the servers (model load) ..."
for idx in "${!GPU_ARR[@]}"; do
  port=$((PORT_BASE + idx))
  until (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null; do
    sleep 5
    docker ps -q --filter "name=${names[$idx]}" | grep -q . || { echo "server on port ${port} exited:" >&2; docker logs "${names[$idx]}" 2>&1 | tail -20 >&2 || true; exit 1; }
  done
  echo "port ${port} ready"
done

job_args=()
[[ -n "${SPLITS:-}" ]] && job_args+=(--splits "${SPLITS}")
[[ -n "${TASK_IDS:-}" ]] && job_args+=(--task-ids "${TASK_IDS}")
[[ -n "${TRIALS:-}" ]] && job_args+=(--trials "${TRIALS}")
python3 "${ROOT}/scripts/make_eval_jobs.py" --note "${NOTE}" --family xvla --checkpoint "$(basename "${CKPT}")" \
  --image "${EVAL_IMAGE}" --extra "--max_steps ${MAX_STEPS:-0}" "${job_args[@]}" >"${LOG_DIR}/jobs.tsv"
# shellcheck disable=SC2086
python3 "${ROOT}/scripts/gpu_queue.py" --jobs "${LOG_DIR}/jobs.tsv" --slots "${SLOTS}" --port-base "${PORT_BASE}" \
  --log-dir "${LOG_DIR}" --stagger 20 ${QUEUE_ARGS:-}
