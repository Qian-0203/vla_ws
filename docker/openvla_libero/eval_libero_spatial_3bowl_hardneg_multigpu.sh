#!/usr/bin/env bash
# Launch the libero_spatial_3bowl_hardneg eval (extra-distractor variant) across multiple
# GPUs. The 10 tasks are round-robin sharded (task_id % NUM_SHARDS == SHARD_INDEX),
# one container per GPU. Each shard writes its own log to experiments/logs/.
#
# Usage:
#   bash eval_libero_spatial_3bowl_hardneg_multigpu.sh                    # GPUs 0..7
#   GPUS="0,1,2,3,4" bash eval_libero_spatial_3bowl_hardneg_multigpu.sh   # specific GPUs
#   USE_EXPLICIT_PROMPT=False GPUS="0,1,2,3,4" bash eval_libero_spatial_3bowl_hardneg_multigpu.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SINGLE_SCRIPT="${SCRIPT_DIR}/eval_libero_spatial_3bowl_hardneg.sh"

# Comma-separated GPU ids. There are only 10 tasks, so >10 GPUs wastes resources.
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NUM_SHARDS="${#GPU_ARR[@]}"

LOG_DIR="${WORKSPACE_ROOT}/openvla/experiments/logs"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y_%m_%d-%H_%M_%S)"

echo "Launching ${NUM_SHARDS} shards across GPUs: ${GPUS}"
echo "Console logs: ${LOG_DIR}/multigpu-3bowl-hardneg-${STAMP}-shard*.out"

pids=()
for idx in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$idx]}"
  out="${LOG_DIR}/multigpu-3bowl-hardneg-${STAMP}-shard${idx}of${NUM_SHARDS}-gpu${gpu}.out"
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
    echo "Shard ${i} (GPU ${GPU_ARR[$i]}) FAILED — see ${LOG_DIR}/multigpu-3bowl-hardneg-${STAMP}-shard${i}of${NUM_SHARDS}-gpu${GPU_ARR[$i]}.out" >&2
    fail=1
  fi
done

if [[ "${fail}" -ne 0 ]]; then
  echo "One or more shards failed." >&2
  exit 1
fi

echo "All shards complete. Per-task results are in ${LOG_DIR}/EVAL-*--shard*of${NUM_SHARDS}.txt"
echo "Suite-wide success rate = mean of the 10 'Current task success rate' lines across those logs."
