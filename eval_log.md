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

## Still queued (registry-ready, not yet launched)

| Split/condition | Suite | Readiness |
|---|---|---|
| `grounding/surface_landmark` | new scene | init states verified; contact sheet not yet eyeballed |
| `grounding/region_surface` | new scene | init states verified; contact sheet not yet eyeballed |

**Not registry-ready** (open design questions, `benchmark_split_plan.md` §9): Split 2's `path`
distractor, Split 4b's cue-phrasing matrix.

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

- Helper scripts (`LIBERO/scripts/`): `gen_suite_init_states.py <suite>`,
  `verify_suite_init_states.py <suite>`, `render_suite_contact_sheet.py <suite>`,
  `compare_two_suites_init.py <suite_a> <suite_b> <outname>`.
- Artifacts: per-shard text logs and structured JSONL results under `openvla/experiments/logs/` (see
  `CLAUDE.md` "Results & logs" for exact naming); rollout videos under `openvla/rollouts/<date>/`;
  figures under `openvla/experiments/figures/`.
