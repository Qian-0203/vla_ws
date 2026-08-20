#!/usr/bin/env bash
# Single canonical launcher for every eval split -- replaces the old
# eval_<suite>.sh / eval_<suite>_multigpu.sh scripts (which only differed in
# hardcoded TASK_SUITE_NAME/UNNORM_KEY/RUN_ID_NOTE defaults). Split-specific
# knowledge (which task suite, which unnorm key, which prompt condition) lives
# in exactly one place -- openvla/experiments/robot/libero/eval_registry.py --
# selected here via --split; this script only owns Docker/GPU/checkpoint plumbing.
#
# Config precedence: CLI args to this script > variables already exported in
# your shell > MACHINE_CONFIG file (config/laptop.env, config/server.env) >
# the fallback defaults below.
#
# Usage:
#   # 1 GPU (default), using the laptop machine profile:
#   MACHINE_CONFIG=../../config/laptop.env bash run_eval.sh --split spatial/default
#
#   # Specific GPUs, sharding the suite's tasks round-robin across them:
#   MACHINE_CONFIG=../../config/server.env GPUS=0,1,2,3,4 bash run_eval.sh --split spatial_3bowl/irrelevant
#
#   # Resume an interrupted run, or list available splits:
#   bash run_eval.sh --split spatial/default --resume True
#   python ../../openvla/experiments/robot/libero/run_libero_eval.py --help
#
# Any flag not consumed by this script (--split, --condition, --task_ids,
# --resume, --overwrite, --run_id_note, ...) is forwarded verbatim to
# run_libero_eval.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -n "${MACHINE_CONFIG:-}" ]]; then
  # shellcheck disable=SC1090
  source "${MACHINE_CONFIG}"
fi

IMAGE_NAME="${IMAGE_NAME:-openvla-libero:cuda12.1}"
SERVER_ROOT="${SERVER_ROOT:-}"
CHECKPOINT="${CHECKPOINT:-${WORKSPACE_ROOT}/openvla/checkpoint/baseline_lora_libero_spatial_4gpu_b24_run004/openvla-7b+libero_spatial_no_noops+b24+lr-0.0005+lora-r32+dropout-0.0--image_aug}"
GPUS="${GPUS:-0}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-False}"
LOAD_IN_8BIT="${LOAD_IN_8BIT:-False}"
OPENVLA_ATTN_IMPLEMENTATION="${OPENVLA_ATTN_IMPLEMENTATION:-flash_attention_2}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
SEED="${SEED:-7}"

EXTRA_ARGS=("$@") # forwarded to run_libero_eval.py verbatim (e.g. --split spatial/default)

# --- checkpoint integrity check (unchanged from the original per-suite scripts) ---
if [[ -f "${CHECKPOINT}/model.safetensors.index.json" ]]; then
  python3 - "${CHECKPOINT}" <<'PY'
import glob, json, os, sys
checkpoint = sys.argv[1]
index_path = os.path.join(checkpoint, "model.safetensors.index.json")
with open(index_path) as f:
    index = json.load(f)
expected = int(index.get("metadata", {}).get("total_size", 0))
actual = sum(os.path.getsize(p) for p in glob.glob(os.path.join(checkpoint, "model-*.safetensors")))
if expected and actual < expected * 0.95:
    print(f"Checkpoint shards look incomplete: {actual/1e9:.2f} GB found, index declares {expected/1e9:.2f} GB.", file=sys.stderr)
    print(f"Checkpoint: {checkpoint}", file=sys.stderr)
    sys.exit(2)
PY
fi

# WORKSPACE_ROOT is bind-mounted to /workspace (not an identity mount, unlike
# SERVER_ROOT below) -- so a CHECKPOINT that lives inside the workspace must be
# translated to its /workspace-relative path before being handed to the
# container; a CHECKPOINT under SERVER_ROOT is already identity-mounted and
# needs no translation.
CONTAINER_CHECKPOINT="${CHECKPOINT}"
if [[ "${CHECKPOINT}" == "${WORKSPACE_ROOT}"/* ]]; then
  CONTAINER_CHECKPOINT="/workspace${CHECKPOINT#"${WORKSPACE_ROOT}"}"
fi

DOCKER_TTY_ARGS=()
[[ -t 0 && -t 1 ]] && DOCKER_TTY_ARGS=(-it)

# The image ships only the Mesa (software) GLVND EGL vendor ICD, so libEGL falls
# back to swrast and headless MuJoCo rendering fails. Supply the NVIDIA EGL ICD
# (the nvidia container runtime injects libEGL_nvidia.so.0 when graphics caps are on).
DOCKER_MOUNTS=(-v "${WORKSPACE_ROOT}:/workspace")
[[ -n "${SERVER_ROOT}" && -d "${SERVER_ROOT}" ]] && DOCKER_MOUNTS+=(-v "${SERVER_ROOT}:${SERVER_ROOT}:ro")
DOCKER_MOUNTS+=(-v "${SCRIPT_DIR}/10_nvidia_egl.json:/etc/glvnd/egl_vendor.d/10_nvidia.json:ro")

DOCKER_ENV_COMMON=(
  -e MUJOCO_GL=egl
  # `--gpus device=X` below makes exactly one GPU visible inside the container,
  # always as index 0 -- so the EGL device id is always 0 regardless of host id.
  -e MUJOCO_EGL_DEVICE_ID=0
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
  -e __EGL_VENDOR_LIBRARY_FILENAMES=/etc/glvnd/egl_vendor.d/10_nvidia.json
  -e HF_HOME=/workspace/.cache/huggingface
  # Multiple libs in this stack (torch, TF, mujoco) each link an OpenMP runtime;
  # their init order races and intermittently aborts with "OMP: Error #15". Allow
  # the duplicate runtime so the process continues (the documented workaround).
  -e KMP_DUPLICATE_LIB_OK=TRUE
  -e WANDB_MODE="${WANDB_MODE:-disabled}"
  -e OPENVLA_ATTN_IMPLEMENTATION="${OPENVLA_ATTN_IMPLEMENTATION}"
  # Run as the host user so eval outputs (logs, rollouts) aren't root-owned. HOME
  # and MPLCONFIGDIR point at writable, mounted paths since the container user has
  # no /etc/passwd entry or home directory.
  -e HOME=/workspace/.cache/home
  -e MPLCONFIGDIR=/workspace/.cache/matplotlib
  -e XDG_CACHE_HOME=/workspace/.cache
)

DOCKER_USER_ARGS=(--user "$(id -u):$(id -g)")

IFS=',' read -r -a GPU_ARR <<<"${GPUS}"
NUM_SHARDS="${#GPU_ARR[@]}"
LOG_DIR="${WORKSPACE_ROOT}/openvla/experiments/logs"
mkdir -p "${LOG_DIR}"

run_shard() {
  local gpu="$1" shard_idx="$2" num_shards="$3"
  local shard_args=()
  if ((num_shards > 1)); then
    shard_args=(--num_shards "${num_shards}" --shard_index "${shard_idx}")
  fi
  docker run --rm "${DOCKER_TTY_ARGS[@]}" "${DOCKER_USER_ARGS[@]}" --gpus "device=${gpu}" \
    --ipc=host \
    --shm-size=32g \
    "${DOCKER_ENV_COMMON[@]}" \
    "${DOCKER_MOUNTS[@]}" \
    -w /workspace/openvla \
    "${IMAGE_NAME}" \
    python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint "${CONTAINER_CHECKPOINT}" \
    --load_in_4bit "${LOAD_IN_4BIT}" \
    --load_in_8bit "${LOAD_IN_8BIT}" \
    --center_crop True \
    --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
    --seed "${SEED}" \
    --local_log_dir /workspace/openvla/experiments/logs \
    --use_wandb False \
    "${shard_args[@]}" \
    "${EXTRA_ARGS[@]}"
}

if ((NUM_SHARDS == 1)); then
  run_shard "${GPU_ARR[0]}" 0 1
else
  echo "Sharding across ${NUM_SHARDS} GPUs: ${GPUS}"
  STAMP="$(date +%Y_%m_%d-%H_%M_%S)"
  pids=()
  for idx in "${!GPU_ARR[@]}"; do
    gpu="${GPU_ARR[$idx]}"
    out="${LOG_DIR}/multigpu-${STAMP}-shard${idx}of${NUM_SHARDS}-gpu${gpu}.out"
    echo "  shard ${idx}/${NUM_SHARDS} -> GPU ${gpu} (log: ${out})"
    run_shard "${gpu}" "${idx}" "${NUM_SHARDS}" >"${out}" 2>&1 &
    pids+=("$!")
  done
  echo "Waiting for ${#pids[@]} shards to finish..."
  fail=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "Shard ${i} (GPU ${GPU_ARR[$i]}) FAILED -- see ${LOG_DIR}/multigpu-${STAMP}-shard${i}of${NUM_SHARDS}-gpu${GPU_ARR[$i]}.out" >&2
      fail=1
    fi
  done
  ((fail == 0)) || {
    echo "One or more shards failed." >&2
    exit 1
  }
  echo "All shards complete."
fi
