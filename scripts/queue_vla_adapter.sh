#!/usr/bin/env bash
# Run VLA-Adapter's jobs through scripts/gpu_queue.py: one policy-server container per slot (the authors' own
# eval code, scripts/policy_servers/serve_vla_adapter.py), the (split, task) queue on top, then tear down.
#
#   CKPT=$HOME/vla_adapter_ckpt/LIBERO-Spatial-Pro PRO=1 NOTE=vla_adapter_spatial_pro SLOTS=0,1,2,3 \
#     SPLITS=spatial/default,grounding/landmark_exchange bash scripts/queue_vla_adapter.sh
#
# Env: CKPT (checkpoint dir, mounted read-only), PRO (1 for the *-Pro checkpoints), NOTE (run_id_note),
# SLOTS (GPU per slot, repeats allowed), SPLITS (default: run_benchmark.sh DEFAULT_SPLITS), PORT_BASE (8000),
# VLA_ADAPTER_DIR (checkout of OpenHelix-Team/VLA-Adapter, default ~/vla_adapter), IMAGE_NAME (eval image),
# TASK_IDS / TRIALS (smoke tests: restrict tasks, trials per task), QUEUE_ARGS (extra gpu_queue.py flags). The loader rewrites its checkpoint directory, so every server gets a
# private working copy (small files copied, weights symlinked to the real checkpoint). Re-running resumes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CKPT:?checkpoint directory}"
: "${NOTE:?run_id_note}"
: "${PRO:?1 for a *-Pro checkpoint, 0 otherwise}"
SLOTS="${SLOTS:-0,1,2,3}"
PORT_BASE="${PORT_BASE:-8000}"
VLA_ADAPTER_DIR="${VLA_ADAPTER_DIR:-$HOME/vla_adapter}"
IMAGE_NAME="${IMAGE_NAME:-openvla-libero:blackwell}"
LOG_DIR="${HOME}/logs/queue_${NOTE}"
WORK_ROOT="${HOME}/vla_adapter_work"
mkdir -p "${LOG_DIR}" "${WORK_ROOT}"
[[ -f "${CKPT}/action_head--checkpoint.pt" ]] || { echo "no action_head--checkpoint.pt in ${CKPT}" >&2; exit 1; }
[[ -d "${VLA_ADAPTER_DIR}/prismatic" ]] || { echo "no VLA-Adapter checkout at ${VLA_ADAPTER_DIR}" >&2; exit 1; }

names=()
stop_servers() { for n in "${names[@]:-}"; do [[ -n "$n" ]] && docker rm -f "$n" >/dev/null 2>&1 || true; done; }
trap stop_servers EXIT

IFS=',' read -r -a GPU_ARR <<<"${SLOTS}"
for idx in "${!GPU_ARR[@]}"; do
  port=$((PORT_BASE + idx))
  name="vla_adapter_${NOTE}_${port}"
  work="${WORK_ROOT}/${NOTE}_${port}"
  rm -rf "${work}" && mkdir -p "${work}"
  for f in "${CKPT}"/*; do
    b="$(basename "$f")"
    case "$b" in
      model*.safetensors|*.pt) ln -s "$f" "${work}/$b" ;;
      *) cp -r "$f" "${work}/$b" ;;
    esac
  done
  docker rm -f "${name}" >/dev/null 2>&1 || true
  # --entrypoint: the image's own entrypoint rewrites /workspace/.libero, and the workspace is mounted read-only
  # here; the repo's .libero/config.yaml is already in place.
  docker run -d --rm --name "${name}" --network host --gpus "device=${GPU_ARR[$idx]}" --ipc=host \
    --user "$(id -u):$(id -g)" -w /vla_adapter --entrypoint python3 \
    -e PYTHONPATH=/vla_adapter:/workspace/LIBERO -e HOME=/tmp -e PYTHONNOUSERSITE=1 -e HF_HOME=/tmp/hf \
    -e KMP_DUPLICATE_LIB_OK=TRUE -e WANDB_MODE=disabled -e LIBERO_CONFIG_PATH=/workspace/.libero \
    -v "${ROOT}:/workspace:ro" -v "${VLA_ADAPTER_DIR}:/vla_adapter:ro" -v "${ROOT}/scripts/policy_servers:/serve:ro" \
    -v "${CKPT}:${CKPT}:ro" -v "${work}:${work}" \
    "${IMAGE_NAME}" /serve/serve_vla_adapter.py --checkpoint "${work}" --port "${port}" \
      --use_pro_version "${PRO}" --task_suite_name libero_spatial >/dev/null
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

splits_args=()
[[ -n "${SPLITS:-}" ]] && splits_args=(--splits "${SPLITS}")
[[ -n "${TASK_IDS:-}" ]] && splits_args+=(--task-ids "${TASK_IDS}")
[[ -n "${TRIALS:-}" ]] && splits_args+=(--trials "${TRIALS}")
python3 "${ROOT}/scripts/make_eval_jobs.py" --note "${NOTE}" --family openpi --checkpoint "${CKPT}" \
  --extra "--policy_obs raw --replan_steps 8" "${splits_args[@]}" >"${LOG_DIR}/jobs.tsv"
# shellcheck disable=SC2086
python3 "${ROOT}/scripts/gpu_queue.py" --jobs "${LOG_DIR}/jobs.tsv" --slots "${SLOTS}" --port-base "${PORT_BASE}" \
  --log-dir "${LOG_DIR}" --stagger 20 ${QUEUE_ARGS:-}
