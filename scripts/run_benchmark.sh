#!/usr/bin/env bash
# Run every current (non-legacy) benchmark split on one checkpoint, then print
# the aggregated per-split success rates. Thin loop over run_eval.sh -- each
# split is exactly one `run_eval.sh --split ...` call, so anything that works
# there (MACHINE_CONFIG, GPUS, LOAD_IN_4BIT, CENTER_CROP, ...) works here too.
#
# Usage:
#   CHECKPOINT=/path/to/merged/openvla/ckpt bash scripts/run_benchmark.sh
#   CHECKPOINT=... GPUS=0,1,2,3 bash scripts/run_benchmark.sh --resume True   # continue after a crash
#   CHECKPOINT=... SPLITS="spatial/default" bash scripts/run_benchmark.sh        # just the sanity gate
#
# Extra args are forwarded to every run_eval.sh call. Results land in
# openvla/experiments/logs/results/{suite}--{condition}--{RUN_NOTE}*.jsonl.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CHECKPOINT:?Set CHECKPOINT=/path/to/your/merged/openvla/checkpoint}"
# Tags every results file with the checkpoint so runs of different models never collide.
RUN_NOTE="${RUN_NOTE:-$(basename "${CHECKPOINT}" | tr -c 'A-Za-z0-9._\n-' '_' | cut -c1-80)}"

# Ordered by how much each split moved the reference checkpoint's success rate
# (docs/benchmark_split_result.md), after the spatial/default sanity gate.
# Split 4b/4c instruction dicts only cover a task subset -- those ids are required.
declare -A TASK_IDS=(
  [grounding/target_cue_region]=0,1,3,5,6,7,8,9
  [grounding/target_cue_landmark]=3,5,7,9
  [grounding/target_cue_proximity_novel]=3,5,7,9
)
DEFAULT_SPLITS=(
  spatial/default                             # Split 1 baseline -- sanity gate
  spatial/positive_contrast                   # Split 1
  spatial/negative_contrast                   # Split 1
  grounding/target_cue_region                 # Split 4b
  grounding/target_cue_landmark               # Split 4b
  grounding/target_cue_proximity_novel        # Split 4c
  spatial_3bowl/landmark_with_hardneg_prompt  # Split 1x2
  spatial_3bowl/drawer_open                   # Split 3
  spatial_3bowl/landmark                      # Split 2
  spatial_3bowl/irrelevant                    # Split 2
  spatial_3bowl/semantic                      # Split 2
  grounding/surface_landmark                  # Split 4a gap-fill cell
  grounding/region_surface                    # Split 4a gap-fill cell
)
if [[ -n "${SPLITS:-}" ]]; then read -r -a RUN_SPLITS <<<"${SPLITS}"; else RUN_SPLITS=("${DEFAULT_SPLITS[@]}"); fi

export CHECKPOINT
for split in "${RUN_SPLITS[@]}"; do
  echo "=== ${split} (run note: ${RUN_NOTE}) ==="
  task_args=()
  [[ -n "${TASK_IDS[${split}]:-}" ]] && task_args=(--task_ids "${TASK_IDS[${split}]}")
  bash "${ROOT}/docker/openvla_libero/run_eval.sh" --split "${split}" --run_id_note "${RUN_NOTE}" \
    "${task_args[@]}" "$@"
done

python3 "${ROOT}/scripts/aggregate_results.py" --filter="--${RUN_NOTE}"
