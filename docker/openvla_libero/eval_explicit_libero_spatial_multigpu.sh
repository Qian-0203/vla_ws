#!/usr/bin/env bash
# Launch the explicit-instruction libero_spatial eval across multiple GPUs.
#
# Data parallelism: the 10 libero_spatial tasks are round-robin sharded across
# the listed GPUs (task_id % NUM_SHARDS == SHARD_INDEX), one container per GPU.
# Each shard writes its own log (…--shard{i}of{N}.txt) and console output to
# experiments/logs/. Per-task success rates are independent, so the suite-wide
# number is the mean of the 10 per-task rates across all shard logs.
#
# Usage:
#   bash eval_explicit_libero_spatial_multigpu.sh                # GPUs 0..7
#   GPUS="0,1,2,3" bash eval_explicit_libero_spatial_multigpu.sh # specific GPUs
#   NUM_TRIALS_PER_TASK=50 GPUS="0,1" bash eval_..._multigpu.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SINGLE_SCRIPT="${SCRIPT_DIR}/eval_explicit_libero_spatial.sh"

# Comma-separated GPU ids. There are only 10 tasks, so using more than 10 GPUs
# wastes resources (extra shards get no tasks).
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NUM_SHARDS="${#GPU_ARR[@]}"

LOG_DIR="${WORKSPACE_ROOT}/openvla/experiments/logs"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y_%m_%d-%H_%M_%S)"

echo "Launching ${NUM_SHARDS} shards across GPUs: ${GPUS}"
echo "Console logs: ${LOG_DIR}/multigpu-${STAMP}-shard*.out"

pids=()
for idx in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$idx]}"
  out="${LOG_DIR}/multigpu-${STAMP}-shard${idx}of${NUM_SHARDS}-gpu${gpu}.out"
  echo "  shard ${idx}/${NUM_SHARDS} -> GPU ${gpu}  (log: ${out})"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  NUM_SHARDS="${NUM_SHARDS}" \
  SHARD_INDEX="${idx}" \
    bash "${SINGLE_SCRIPT}" >"${out}" 2>&1 &
  pids+=("$!")
done

echo "Waiting for ${#pids[@]} shards to finish..."
fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "Shard ${i} (GPU ${GPU_ARR[$i]}) FAILED — see ${LOG_DIR}/multigpu-${STAMP}-shard${i}of${NUM_SHARDS}-gpu${GPU_ARR[$i]}.out" >&2
    fail=1
  fi
done

if [[ "${fail}" -ne 0 ]]; then
  echo "One or more shards failed." >&2
  exit 1
fi

echo "All shards complete. Per-task results are in ${LOG_DIR}/EVAL-*--shard*of${NUM_SHARDS}.txt"
echo "Suite-wide success rate = mean of the 10 'Current task success rate' lines across those logs."
