#!/usr/bin/env bash
# Evaluate the libero_spatial LoRA baseline on the libero_spatial_3bowl_hardneg variant:
# identical tasks/scenes but each scene has a THIRD akita_black_bowl (one extra
# distractor). Default (target-only) instructions, so this measures how an extra
# distractor bowl affects the ordinary-instruction baseline (cf. the 84% two-bowl
# default number in experiments/explicit_instructions_eval_results.md).
#
# The scene variant's suite name differs from the training dataset, so we pass
# --unnorm_key libero_spatial (resolves to libero_spatial_no_noops in the ckpt).
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-openvla-libero:cuda12.1}"
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SERVER_ROOT="${SERVER_ROOT:-/home/ec2-user/wenhan}"
CHECKPOINT="${CHECKPOINT:-${SERVER_ROOT}/openvla/checkpoints/baseline_lora_libero_spatial_4gpu_b24_run004/openvla-7b+libero_spatial_no_noops+b24+lr-0.0005+lora-r32+dropout-0.0--image_aug}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_spatial_3bowl_hardneg}"
UNNORM_KEY="${UNNORM_KEY:-libero_spatial}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
SEED="${SEED:-7}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
# Default instructions for the distractor experiment (target-only prompt).
USE_EXPLICIT_PROMPT="${USE_EXPLICIT_PROMPT:-False}"
if [[ "${USE_EXPLICIT_PROMPT}" == "True" ]]; then
  RUN_ID_NOTE="${RUN_ID_NOTE:-explicit_instructions_3bowl_hardneg}"
else
  RUN_ID_NOTE="${RUN_ID_NOTE:-default_instructions_3bowl_hardneg}"
fi
LOAD_IN_4BIT="${LOAD_IN_4BIT:-False}"
LOAD_IN_8BIT="${LOAD_IN_8BIT:-False}"
OPENVLA_ATTN_IMPLEMENTATION="${OPENVLA_ATTN_IMPLEMENTATION:-flash_attention_2}"

HOST_CHECKPOINT="${CHECKPOINT/#\/workspace/${WORKSPACE_ROOT}}"
HOST_CHECKPOINT="${HOST_CHECKPOINT/#${SERVER_ROOT}/${SERVER_ROOT}}"
if [[ -f "${HOST_CHECKPOINT}/model.safetensors.index.json" ]]; then
  python3 - "${HOST_CHECKPOINT}" <<'PY'
import glob
import json
import os
import sys

checkpoint = sys.argv[1]
index_path = os.path.join(checkpoint, "model.safetensors.index.json")
with open(index_path, "r") as f:
    index = json.load(f)

expected = int(index.get("metadata", {}).get("total_size", 0))
actual = sum(os.path.getsize(path) for path in glob.glob(os.path.join(checkpoint, "model-*.safetensors")))

if expected and actual < expected * 0.95:
    print(
        f"Checkpoint shards look incomplete: found {actual / 1e9:.2f} GB of safetensors "
        f"for an index declaring {expected / 1e9:.2f} GB.",
        file=sys.stderr,
    )
    print(f"Checkpoint: {checkpoint}", file=sys.stderr)
    sys.exit(2)
PY
fi

DOCKER_TTY_ARGS=()
if [[ -t 0 && -t 1 ]]; then
  DOCKER_TTY_ARGS=(-it)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DOCKER_MOUNTS=(-v "${WORKSPACE_ROOT}:/workspace")
if [[ -d "${SERVER_ROOT}" ]]; then
  DOCKER_MOUNTS+=(-v "${SERVER_ROOT}:${SERVER_ROOT}:ro")
fi
# The image ships only the Mesa (software) GLVND EGL vendor ICD, so libEGL falls
# back to swrast and headless MuJoCo rendering fails. Supply the NVIDIA EGL ICD
# (the nvidia container runtime injects libEGL_nvidia.so.0 when graphics caps are on).
DOCKER_MOUNTS+=(-v "${SCRIPT_DIR}/10_nvidia_egl.json:/etc/glvnd/egl_vendor.d/10_nvidia.json:ro")

DOCKER_ENV=(
  -e MUJOCO_GL=egl
  -e MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
  -e __EGL_VENDOR_LIBRARY_FILENAMES=/etc/glvnd/egl_vendor.d/10_nvidia.json
  -e HF_HOME=/workspace/.cache/huggingface
  -e KMP_DUPLICATE_LIB_OK=TRUE
  -e WANDB_MODE="${WANDB_MODE:-disabled}"
  -e OPENVLA_ATTN_IMPLEMENTATION="${OPENVLA_ATTN_IMPLEMENTATION}"
  -e HOME=/workspace/.cache/home
  -e MPLCONFIGDIR=/workspace/.cache/matplotlib
  -e XDG_CACHE_HOME=/workspace/.cache
)

DOCKER_USER_ARGS=(--user "$(id -u):$(id -g)")

DOCKER_GPU_ARGS=(--gpus all)
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  DOCKER_GPU_ARGS=(--gpus "device=${CUDA_VISIBLE_DEVICES}")
fi

docker run --rm "${DOCKER_TTY_ARGS[@]}" "${DOCKER_USER_ARGS[@]}" "${DOCKER_GPU_ARGS[@]}" \
  --ipc=host \
  --shm-size=32g \
  "${DOCKER_ENV[@]}" \
  "${DOCKER_MOUNTS[@]}" \
  -w /workspace/openvla \
  "${IMAGE_NAME}" \
  python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --load_in_4bit "${LOAD_IN_4BIT}" \
    --load_in_8bit "${LOAD_IN_8BIT}" \
    --task_suite_name "${TASK_SUITE_NAME}" \
    --unnorm_key "${UNNORM_KEY}" \
    --center_crop True \
    --use_explicit_prompt "${USE_EXPLICIT_PROMPT}" \
    --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
    --seed "${SEED}" \
    --num_shards "${NUM_SHARDS}" \
    --shard_index "${SHARD_INDEX}" \
    --run_id_note "${RUN_ID_NOTE}" \
    --local_log_dir /workspace/openvla/experiments/logs \
    --use_wandb False
