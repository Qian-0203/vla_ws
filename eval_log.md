# OpenVLA · LIBERO-Spatial — Eval Log

**What this file is.** An append-only, chronological record of every real eval **launch**: what was
run, in what order, on what hardware/image/checkpoint, and where the outputs landed. It answers "when
was this run, in what batch, and how do I relaunch it" — not "what were the results" or "why does this
condition exist."

**What it is not.** Not the detailed per-condition record — instruction text, scene/distractor
placement, render comparisons, full per-task results, and analysis all live in
`benchmark_split_result.md`. Not the design doc — hypotheses, split rationale, and scene geometry live
in `benchmark_split_plan.md`. This file only needs a new entry when a real GPU eval is launched;
never edit or delete a prior entry, append instead (if a run is redone, add a new entry and note what
it supersedes).

**Update rule:** every time a real GPU eval finishes, append an entry here (batch date, hardware,
launch order, one-line result headline + rollouts per condition, results-file paths) **and** update
`benchmark_split_result.md` (the condition's detail + status + findings). `benchmark_split_plan.md`
only needs an update when a split's *definition* changes.

---

## 2026-08-1x — Baseline batch: Split 1 (`default`/`negative_contrast`) + Split 2/3 (`center_fixed_legacy`/`drawer_open`)

- **Hardware:** 5× H200 (GPUs 0–4), `openvla-libero:cuda12.1` (mujoco 2.3.2, robosuite 1.4.1, EGL headless)
- **Checkpoint:** `baseline_lora_libero_spatial_4gpu_b24_run004/openvla-7b+libero_spatial_no_noops+b24+lr-0.0005+lora-r32+dropout-0.0--image_aug`
- **Protocol:** seed 7, 50 trials/task, 10 tasks/condition (500 rollouts)

Launched, in order:

| # | Split/condition | Suite | Headline SR | Rollouts |
|--:|---|---|--:|--:|
| 1 | `spatial/default` | `libero_spatial` (2 bowls) | 84.0% | 420/500 |
| 2 | `spatial/negative_contrast` (originally "explicit") | `libero_spatial` (2 bowls) | 36.8% | 184/500 |
| 3 | `spatial_3bowl/center_fixed_legacy` | `libero_spatial_3bowl` (3 bowls) | 80.2% | 401/500 |
| 4 | `spatial_3bowl/drawer_open` | `libero_spatial_3bowl_open` (3 bowls + open drawer) | 60.0% raw / 73.1% adj (7-task) | 300/500 |

**Launch commands** (from `vla_ws/docker/openvla_libero/`; `USE_EXPLICIT_PROMPT` toggles condition 2):

```bash
# 1-2: baseline / negative_contrast (2 bowls)
USE_EXPLICIT_PROMPT=False GPUS="0,1,2,3,4" bash eval_explicit_libero_spatial_multigpu.sh
USE_EXPLICIT_PROMPT=True  GPUS="0,1,2,3,4" bash eval_explicit_libero_spatial_multigpu.sh

# 3: three bowls, default prompt
USE_EXPLICIT_PROMPT=False GPUS="0,1,2,3,4" bash eval_libero_spatial_3bowl_multigpu.sh

# 4: three bowls + open drawer, default prompt
USE_EXPLICIT_PROMPT=False GPUS="0,1,2,3,4" bash eval_libero_spatial_3bowl_open_multigpu.sh
```

**Status:** complete. `center_fixed_legacy`'s distractor placement was later found to confound task 6
(see `benchmark_split_plan.md` Split 2) and was retired as a definition — kept only as a labeled
historical result, not reused as the current `irrelevant` condition (see the 2026-08-19 batch below).

---

## 2026-08-19 — Split 1×2 batch: prompt variants + Split 2 redefinition + hard-negative combo

- **Hardware:** 4× RTX PRO 6000 Blackwell, `openvla-libero:blackwell` (sdpa — no flash-attn wheel for
  this arch)
- **Checkpoint/seed:** unchanged from the baseline batch above
- **Protocol:** seed 7, 50 trials/task, 10 tasks/condition (500 rollouts), sharded 4-way

Launched (best-reconstructed order — condition 1 completed earlier the same day, ~14:02, before this
session picked up Split 2, and was recovered/documented rather than re-run):

| # | Split/condition | Suite | Headline SR | Rollouts |
|--:|---|---|--:|--:|
| 1 | `spatial_3bowl/irrelevant` (redefined) | `libero_spatial_3bowl_neutral` | 88.8% | 444/500 |
| 2 | `spatial/positive_contrast` | `libero_spatial` (2 bowls) | 32.4% | 162/500 |
| 3 | `spatial_3bowl/semantic` | `libero_spatial_3bowl_semantic` | 84.8% | 424/500 |
| 4 | `spatial_3bowl/landmark` | `libero_spatial_3bowl_hardneg` | 80.6% | 403/500 |
| 5 | `spatial_3bowl/landmark_with_hardneg_prompt` (same scene as #4, `hardneg` prompt) | `libero_spatial_3bowl_hardneg` | 41.2% | 412/500 |

**Pre-run checks:** scene BDDL/init-state numeric verify + contact-sheet eyeball done before launch
for conditions 1, 3, 4 (new/regenerated scenes) — see `benchmark_split_result.md` §7 for the check
log and figures.

**Results files:**
```
results/libero_spatial_3bowl_neutral--default--shard{0..3}of4.jsonl
results/libero_spatial--positive_contrast--shard{0..3}of4.jsonl
results/libero_spatial_3bowl_semantic--default--shard{0..3}of4.jsonl
results/libero_spatial_3bowl_hardneg--default--shard{0..3}of4.jsonl
results/libero_spatial_3bowl_hardneg--hardneg--shard{0..3}of4.jsonl
```

**Status:** complete.

---

## 2026-08-20 — Split 4 batch: 4a's 2 gap-fill cells + 4b's target-cue-type probe

- **Hardware:** 4× RTX PRO 6000 Blackwell (same server as the 2026-08-19 batch), `openvla-libero:blackwell`
  (sdpa). Only GPUs 1 and 3 used — GPUs 0 and 2 were occupied by other concurrent work on this shared
  server at launch time, left untouched throughout.
- **Checkpoint/seed:** unchanged from the baseline batch (`baseline_lora_libero_spatial_4gpu_b24_run004`,
  seed 7).
- **Protocol:** 50 trials/task, sharded 2-way across GPUs 1/3 (round-robin task assignment produced
  uneven per-shard task counts for `target_cue_region` (3 vs. 5 tasks) and `target_cue_landmark` (0
  vs. 4 tasks) since sharding happens before the `--task_ids` filter narrows the set — correctness
  unaffected (verified below), only wall-clock balance).
- **Pre-launch fixes required** (both applied on `LIBERO` branch `worktree-split4-contact-sheets`,
  merged to `master`): `torch.load` in `verify_suite_init_states.py` and
  `Benchmark.get_task_init_states` both needed `weights_only=False` for current PyTorch (>=2.6
  flipped the default) — caught via a 1-trial smoke test before committing to the full run.

Launched, in order:

| # | Split/condition | Suite | Headline SR | Rollouts |
|--:|---|---|--:|--:|
| 1 | `grounding/surface_landmark` | `libero_spatial_grounding_surface_landmark` | 88.0% | 50/50 |
| 2 | `grounding/region_surface` | `libero_spatial_grounding_region_surface` | 92.0% | 50/50 |
| 3 | `grounding/target_cue_region` | `libero_spatial` (task_ids 0,1,3,5,6,7,8,9) | 17.0% | 400/400 |
| 4 | `grounding/target_cue_landmark` | `libero_spatial` (task_ids 3,5,7,9) | 30.5% | 200/200 |

**Verification before writing up results:** exact task-id coverage and per-task trial counts
cross-checked programmatically for #3/#4 (all requested task ids present, exactly 50/task, zero
duplicate `(task_id, episode_idx)` pairs across shards) — see `benchmark_split_result.md` §5.2 for
the full per-task table and analysis. Numbers cross-verified against
`scripts/aggregate_results.py --filter <suite>` (canonical aggregator), not just ad hoc counting.

**Results files:**
```
results/libero_spatial_grounding_surface_landmark--default--shard0of2.jsonl
results/libero_spatial_grounding_region_surface--default--shard0of2.jsonl
results/libero_spatial--target_cue_region--shard{0,1}of2.jsonl
results/libero_spatial--target_cue_landmark--shard{0,1}of2.jsonl
```

**Status:** complete. This finishes Split 4 (4a: 6/6 cells, 4b: 2/2 conditions) — see
`benchmark_split_result.md` §5 for the full write-up and §6 findings 10-11 for the headline result
(target cue-type rephrasing alone costs 50-67 pts, the largest effect measured in this project).

---

## Still queued (registry-ready, not yet launched)

_(none — Split 4's `path` distractor (Split 2) remains the only open item, see below)_

**Not registry-ready** (open design questions, `benchmark_split_plan.md` §9): Split 2's `path`
distractor.

---

## Config reference (unchanged across all batches above)

- Action un-norm stats: checkpoint key `libero_spatial_no_noops`. Variant suites pass
  `--unnorm_key libero_spatial` so their differing suite name still resolves to that key.
- The `modified_libero_rlds` RLDS path is training data and is **not** read at eval time.
- Scene-variant suites (canonical `libero_spatial` untouched):

  | Suite | Scene |
  |---|---|
  | `libero_spatial` | 2 bowls (stock) |
  | `libero_spatial_3bowl` | +1 bowl (`center_fixed_legacy`) |
  | `libero_spatial_3bowl_open` | +1 bowl, top drawer open |
  | `libero_spatial_3bowl_neutral` | +1 bowl, redefined `irrelevant` |
  | `libero_spatial_3bowl_semantic` | +1 bowl, `semantic` |
  | `libero_spatial_3bowl_hardneg` | +1 bowl, `landmark` / `landmark_with_hardneg_prompt` |
  | `libero_spatial_grounding_surface_landmark` | 2 bowls, distractor moved to a landmark region |
  | `libero_spatial_grounding_region_surface` | 2 bowls, distractor moved to a surface region |

- Helper scripts (`LIBERO/scripts/`): `gen_suite_init_states.py <suite>`,
  `verify_suite_init_states.py <suite>`, `render_suite_contact_sheet.py <suite>`,
  `compare_two_suites_init.py <suite_a> <suite_b> <outname>`.
- Artifacts: per-shard text logs and structured JSONL results under `openvla/experiments/logs/` (see
  `CLAUDE.md` "Results & logs" for exact naming); rollout videos under `openvla/rollouts/<date>/`;
  figures under `openvla/experiments/figures/`.
