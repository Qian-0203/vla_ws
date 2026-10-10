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
#   CHECKPOINT=... SPLITS=new bash scripts/run_benchmark.sh                      # only the 9 conditions added 2026-10-01
#   MODEL_FAMILY=openpi CHECKPOINT=gs://openpi-assets/checkpoints/pi05_libero RUN_NOTE=pi05_libero \
#     bash scripts/run_benchmark.sh                  # an openpi policy; start scripts/serve_openpi.sh first
#
# Extra args are forwarded to every run_eval.sh call. Results land in
# openvla/experiments/logs/results/{suite}--{condition}--{RUN_NOTE}*.jsonl.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CHECKPOINT:?Set CHECKPOINT=/path/to/your/merged/openvla/checkpoint}"
# Per-split launcher. QwenVLA checkpoints use the host runner instead of Docker:
#   EVAL_RUNNER=scripts/run_eval_qwenvla.sh RESULTS_DIR=/var/tmp/$USER/vla_ws_runs/logs/results
EVAL_RUNNER="${EVAL_RUNNER:-${ROOT}/docker/openvla_libero/run_eval.sh}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT}/openvla/experiments/logs/results}"
# Tags every results file with the checkpoint so runs of different models never collide.
RUN_NOTE="${RUN_NOTE:-$(basename "${CHECKPOINT}" | tr -c 'A-Za-z0-9._\n-' '_' | cut -c1-80)}"

# Grouped by probe, and ordered roughly by how much each group moved the reference checkpoint's
# success rate (docs/benchmark_split_result.md), after the spatial/default sanity gate.
# Split 4b/4c instruction dicts only cover a task subset -- those ids are required.
declare -A TASK_IDS=(
  [grounding/target_cue_region]=0,1,3,5,6,7,8,9
  [grounding/target_cue_landmark]=3,5,7,9
  [grounding/target_cue_proximity_novel]=3,5,7,9
  [grounding/target_cue_region_v2]=0,1,3,5,6,7,8,9
  [grounding/target_cue_region_v3]=0,1,3,5,6,7,8,9
  [grounding/target_cue_proximity_beside]=3,5,7,9
  [grounding/target_cue_proximity_near]=3,5,7,9
  [grounding/target_cue_proximity_adjacent]=3,5,7,9
)
# The 9 conditions added 2026-10-01, runnable on their own with SPLITS=new (docs/eval_log.md's
# launch commands for that batch use it). They are all in DEFAULT_SPLITS as well.
NEW_SPLITS=(
  spatial/length_control_infix spatial/length_control_suffix
  grounding/paraphrase_lexical grounding/paraphrase_syntactic
  grounding/target_cue_region_v2 grounding/target_cue_region_v3
  grounding/target_cue_proximity_beside grounding/target_cue_proximity_near grounding/target_cue_proximity_adjacent
)
DEFAULT_SPLITS=(
  spatial/default                             # Split 1 baseline -- sanity gate
  spatial/positive_contrast                   # Split 1
  spatial/negative_contrast                   # Split 1
  spatial/length_control_suffix               # Split 1 length control (for positive_contrast)
  spatial/length_control_infix                # Split 1 length control (for negative_contrast)
  spatial/negation_only                       # Split 1 negation-only reference (2026-10-10)
  grounding/target_cue_region                 # Split 4b
  grounding/target_cue_region_v2              # Split 4b, 2nd wording
  grounding/target_cue_region_v3              # Split 4b, 3rd wording
  grounding/target_cue_landmark               # Split 4b
  grounding/target_cue_proximity_novel        # Split 4c ("close to")
  grounding/target_cue_proximity_beside       # Split 4c synonym
  grounding/target_cue_proximity_near         # Split 4c synonym
  grounding/target_cue_proximity_adjacent     # Split 4c synonym
  spatial_3bowl/landmark_with_hardneg_prompt  # Split 1x2
  spatial_3bowl/drawer_open                   # Split 3
  spatial_3bowl/landmark                      # Split 2
  spatial_3bowl/irrelevant                    # Split 2
  spatial_3bowl/semantic                      # Split 2
  grounding/paraphrase_lexical                # Split 4 same-cue paraphrase
  grounding/paraphrase_syntactic              # Split 4 same-cue paraphrase
  grounding/surface_landmark                  # Split 4a gap-fill cell
  grounding/region_surface                    # Split 4a gap-fill cell
)
if [[ "${SPLITS:-}" == new ]]; then RUN_SPLITS=("${NEW_SPLITS[@]}")
elif [[ -n "${SPLITS:-}" ]]; then read -r -a RUN_SPLITS <<<"${SPLITS}"; else RUN_SPLITS=("${DEFAULT_SPLITS[@]}"); fi

export CHECKPOINT
for split in "${RUN_SPLITS[@]}"; do
  echo "=== ${split} (run note: ${RUN_NOTE}) ==="
  task_args=()
  [[ -n "${TASK_IDS[${split}]:-}" ]] && task_args=(--task_ids "${TASK_IDS[${split}]}")
  bash "${EVAL_RUNNER}" --split "${split}" --run_id_note "${RUN_NOTE}" \
    "${task_args[@]}" "$@"
done

python3 "${ROOT}/scripts/aggregate_results.py" --results-dir "${RESULTS_DIR}" --filter="--${RUN_NOTE}"
