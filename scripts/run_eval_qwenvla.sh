#!/usr/bin/env bash
# Host-side counterpart of docker/openvla_libero/run_eval.sh for QwenVLA checkpoints.
#
# QwenVLA needs torch>=2.11 / transformers>=5.5, which the openvla-libero image (torch 2.2,
# transformers 4.40) cannot provide, so this runs the SAME entry point
# (openvla/experiments/robot/libero/run_libero_eval.py, --model_family qwenvla) in a host conda
# env instead of Docker. Everything split-specific still comes from eval_registry.py via --split.
#
# Usage:
#   CHECKPOINT=/path/to/qwenvla.pt STATS=/path/to/dataset_statistics.json \
#   GPUS=0,1,2,3 bash scripts/run_eval_qwenvla.sh --split spatial/default
#
# A GPU id may repeat (GPUS=4,4,5) to run two shards on one GPU (~18 GB each).
# Any flag not consumed here is forwarded verbatim to run_libero_eval.py.
set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${CHECKPOINT:?Set CHECKPOINT=/path/to/qwenvla/checkpoint.pt}"
: "${STATS:?Set STATS=/path/to/dataset_statistics.json}"
QWENVLA_REPO="${QWENVLA_REPO:-${HOME}/qwen-vla}"
CONDA_ENV="${CONDA_ENV:-qwen-vla}"
GPUS="${GPUS:-0}"
CENTER_CROP="${CENTER_CROP:-True}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
SEED="${SEED:-7}"
# robosuite 1.4 resets degrade after a few dozen on one env; 15 matches the QwenVLA reference eval.
ENV_RECREATE_EVERY="${ENV_RECREATE_EVERY:-15}"
# Outputs (logs, results JSONL, rollout videos). Default is local disk: large writes to an NFS
# $HOME can stall mid-run.
OUT_ROOT="${OUT_ROOT:-/var/tmp/${USER}/vla_ws_runs}"
export HF_HOME="${HF_HOME:-/var/tmp/${USER}/hf-cache}"

EXTRA_ARGS=("$@")

# Point LIBERO at this workspace's scene fork without touching ~/.libero (other LIBERO
# checkouts on this host read that one).
export LIBERO_CONFIG_PATH="${WORKSPACE_ROOT}/.libero-host"
mkdir -p "${LIBERO_CONFIG_PATH}" "${OUT_ROOT}/logs"
# Write-then-rename so concurrent invocations never read a half-written config.
CONFIG_TMP="$(mktemp "${LIBERO_CONFIG_PATH}/config.yaml.XXXXXX")"
cat >"${CONFIG_TMP}" <<YAML
assets: ${WORKSPACE_ROOT}/LIBERO/libero/libero/assets
bddl_files: ${WORKSPACE_ROOT}/LIBERO/libero/libero/bddl_files
benchmark_root: ${WORKSPACE_ROOT}/LIBERO/libero/libero
datasets: ${WORKSPACE_ROOT}/openvla
init_states: ${WORKSPACE_ROOT}/LIBERO/libero/libero/init_files
YAML
mv -f "${CONFIG_TMP}" "${LIBERO_CONFIG_PATH}/config.yaml"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export PYTHONPATH="${WORKSPACE_ROOT}/openvla:${WORKSPACE_ROOT}/LIBERO"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=3
export KMP_DUPLICATE_LIB_OK=TRUE WANDB_MODE="${WANDB_MODE:-disabled}"
# Cap CPU thread pools. torch/OpenMP/MKL default to one thread per core, and with many shards
# per host they oversubscribe the CPU: on a 256-core box with 22 shards this halved step speed
# (1.17 -> 0.54 s/step once capped). Rollouts are single-episode serial work; 4 is plenty.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

IFS=',' read -r -a GPU_ARR <<<"${GPUS}"
NUM_SHARDS="${#GPU_ARR[@]}"

run_shard() {
  local gpu="$1" shard_idx="$2" num_shards="$3"
  local shard_args=()
  if ((num_shards > 1)); then
    shard_args=(--num_shards "${num_shards}" --shard_index "${shard_idx}")
  fi
  # cwd = OUT_ROOT: rollout videos are written to ./rollouts/ relative to it.
  (cd "${OUT_ROOT}" && CUDA_VISIBLE_DEVICES="${gpu}" python \
    "${WORKSPACE_ROOT}/openvla/experiments/robot/libero/run_libero_eval.py" \
    --model_family qwenvla \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --dataset_statistics_path "${STATS}" \
    --qwenvla_repo "${QWENVLA_REPO}" \
    --center_crop "${CENTER_CROP}" \
    --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
    --env_recreate_every "${ENV_RECREATE_EVERY}" \
    --seed "${SEED}" \
    --local_log_dir "${OUT_ROOT}/logs" \
    --use_wandb False \
    "${shard_args[@]}" \
    "${EXTRA_ARGS[@]}")
}

if ((NUM_SHARDS == 1)); then
  run_shard "${GPU_ARR[0]}" 0 1
else
  echo "Sharding across ${NUM_SHARDS} processes on GPUs: ${GPUS}"
  STAMP="$(date +%Y_%m_%d-%H_%M_%S)"
  pids=()
  for idx in "${!GPU_ARR[@]}"; do
    gpu="${GPU_ARR[$idx]}"
    out="${OUT_ROOT}/logs/qwenvla-${STAMP}-shard${idx}of${NUM_SHARDS}-gpu${gpu}.out"
    echo "  shard ${idx}/${NUM_SHARDS} -> GPU ${gpu} (log: ${out})"
    run_shard "${gpu}" "${idx}" "${NUM_SHARDS}" >"${out}" 2>&1 &
    pids+=("$!")
    sleep 5 # stagger the 17 GB checkpoint reads
  done
  echo "Waiting for ${#pids[@]} shards to finish..."
  fail=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "Shard ${i} (GPU ${GPU_ARR[$i]}) FAILED -- see ${OUT_ROOT}/logs/qwenvla-${STAMP}-shard${i}of${NUM_SHARDS}-gpu${GPU_ARR[$i]}.out" >&2
      fail=1
    fi
  done
  ((fail == 0)) || {
    echo "One or more shards failed." >&2
    exit 1
  }
  echo "All shards complete."
fi
