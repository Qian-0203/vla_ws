#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-openvla-libero:cuda12.1}"
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECKPOINT="${CHECKPOINT:-/workspace/openvla/checkpoint/baseline_lora_libero_spatial_4gpu_b24_run004/openvla-7b+libero_spatial_no_noops+b24+lr-0.0005+lora-r32+dropout-0.0--image_aug}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
SEED="${SEED:-7}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-False}"
LOAD_IN_8BIT="${LOAD_IN_8BIT:-False}"
OPENVLA_ATTN_IMPLEMENTATION="${OPENVLA_ATTN_IMPLEMENTATION:-flash_attention_2}"

HOST_CHECKPOINT="${CHECKPOINT/#\/workspace/${WORKSPACE_ROOT}}"
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

docker run --rm "${DOCKER_TTY_ARGS[@]}" --gpus all \
  --ipc=host \
  --shm-size=32g \
  -e MUJOCO_GL=egl \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -e HF_HOME=/workspace/.cache/huggingface \
  -e WANDB_MODE="${WANDB_MODE:-disabled}" \
  -e OPENVLA_ATTN_IMPLEMENTATION="${OPENVLA_ATTN_IMPLEMENTATION}" \
  -v "${WORKSPACE_ROOT}:/workspace" \
  -w /workspace/openvla \
  "${IMAGE_NAME}" \
  python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --load_in_4bit "${LOAD_IN_4BIT}" \
    --load_in_8bit "${LOAD_IN_8BIT}" \
    --task_suite_name libero_spatial \
    --center_crop True \
    --use_explicit_prompt True \
    --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
    --seed "${SEED}" \
    --run_id_note explicit_instructions \
    --local_log_dir /workspace/openvla/experiments/logs \
    --use_wandb False
