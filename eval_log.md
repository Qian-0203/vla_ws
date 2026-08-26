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

## 2026-08-25 — VLM bowl-pointing probe (diagnostic, not a `run_eval.sh --split` launch)

- **Hardware:** 1x RTX PRO 6000 Blackwell (GPUs 0-1 of the 4-GPU server), `openvla-libero:blackwell`.
- **Checkpoint(s):** `baseline_lora_libero_spatial_4gpu_b24_run004` (this project's fine-tuned
  checkpoint) and, for comparison, the unmodified base `openvla/openvla-7b` downloaded fresh from HF
  Hub for this test (~15GB, not previously cached in this workspace).
- **What ran:** new standalone script `openvla/experiments/robot/libero/probe_bowl_pointing.py` —
  a VQA-style probe (numbered-bowl-marker image, ask the model in free text which number is the
  target) meant to separate vision-language grounding from action decoding on the distractor-mention
  conditions (`negative_contrast`, `positive_contrast`, `landmark_with_hardneg_prompt`). Not wired
  into `eval_registry.py`/`SPLITS` — it doesn't produce action rollouts.
- **Outcome:** dead end, confirmed 3 ways (full detail + exact numbers in
  `benchmark_split_result.md` §8): (1) the fine-tuned checkpoint's free-text generation always
  returns action-bin-range tokens regardless of the question, even for a content-free control
  ("What is 2+2?"); (2) the base `openvla-7b` does the same, ruling out this project's fine-tuning as
  the cause — it's structural to OpenVLA's action-only training recipe; (3) a restricted-logit
  comparison (bypassing generation) found the ranking among candidate answer tokens depends only on
  the image, never on the instruction text, across real/unrelated/mismatched prompts. Only 2 tasks
  were smoke-tested before this was established; the planned full 30-query battery (3 conditions x 10
  tasks) was deliberately not run since it would only reproduce the same negative result.
- **Artifacts:** 2 example annotated images (smoke test) at
  `openvla/experiments/figures/probe_bowl_pointing/`; structured smoke-test output at
  `openvla/experiments/logs/probe_bowl_pointing_smoketest/probe_bowl_pointing.jsonl` (gitignored,
  local only).

**Status:** closed — see `benchmark_split_result.md` §8 for the full write-up and cross-experiment
finding 12 for how this bears on findings 1-11's interpretation.

---

## 2026-08-25 — Qwen2-VL-7B-Instruct bowl-pointing probe (the §8 "open alternative", not a `run_eval.sh --split` launch)

- **Hardware:** 1x RTX PRO 6000 Blackwell (GPU 0 of the 4-GPU server, `g4-flex-20260824`),
  `openvla-libero:blackwell` image with `transformers==4.51.3` + `qwen-vl-utils` installed
  ephemerally via `pip install --user` inside the container (the image's pinned `transformers==4.40.1`
  predates Qwen2-VL support; the script's own docstring suggestion of an open-ended `>=4.49` pulls
  today's `5.15.1` instead and breaks an unrelated import — `openvla_utils.py`'s
  `AutoModelForVision2Seq`, removed in transformers 5.x — that `bowl_pointing_common.py` transitively
  imports through `libero_utils.py`/`robot_utils.py` despite never calling into it; pinning to
  `4.51.3` avoids both problems).
- **Model:** `Qwen/Qwen2-VL-7B-Instruct` (already cached locally, ~16GB, no OpenVLA weights
  involved) — a genuinely separate general-purpose VLM never trained on OpenVLA's action-only
  template, per the alternative noted at the end of §8.
- **What ran:** `openvla/experiments/robot/libero/probe_bowl_pointing_qwen.py` (new, uncommitted in
  the `openvla` fork), the full battery this time: all 3 conditions x 10 tasks = 30 queries. Smoke-
  tested on 1 query first.
- **Outcome:** full detail in `benchmark_split_result.md` §8.1. Headline: `negative_contrast` 60%,
  `positive_contrast` 40%, `hardneg` 60% — none convincingly above chance (50% for the 2-bowl
  conditions, 33% for 3-bowl `hardneg`), and the raw answers show a numeric-position bias (Qwen never
  answered "1" across all 10 `hardneg` queries; 7/10 `positive_contrast` answers were "2") rather than
  language-tracking behavior. Answers to the two 2-bowl conditions (identical images per task, since
  `bowl_pointing_common.py` caches renders by `(suite, task_id)` — only the prompt differs) differ on
  5/10 tasks between `negative_contrast` and `positive_contrast`, showing the model IS phrasing-
  sensitive, just not in a way that improves accuracy.
- **Artifacts:** 30 annotated images (all conditions/tasks, overwriting the 2 pre-existing OpenVLA
  smoke-test images at the same paths — same filename scheme, both scripts share `fig_dir`) at
  `openvla/experiments/figures/probe_bowl_pointing/`; structured full-battery output at
  `openvla/experiments/logs/probe_bowl_pointing_qwen/probe_bowl_pointing_qwen.jsonl` (gitignored,
  local only).

**Status:** closed — see `benchmark_split_result.md` §8.1 for the full write-up and cross-experiment
finding 13.

---

## 2026-08-26 — Bowl-pointing probe marker fix + Qwen2-VL re-run (not a `run_eval.sh --split` launch)

- **Trigger:** user inspection of the §8.1 image gallery flagged the markers as too large (occluding
  the bowl) and possibly misaligned. Settle-timing was tested as the hypothesis (`num_steps_wait`
  swept 0-120 on `libero_spatial` task 0) and ruled out — physics fully settles by step 5. A real,
  independent bug was found instead: `bowl_pointing_common.py`'s hand-projection of each bowl's 3D
  position (`robosuite.utils.camera_utils.project_points_from_world_to_camera`) placed
  `akita_black_bowl_1`'s marker ~46px off (on bare table, ~20% of the 224px frame) on `libero_spatial`
  task 0, confirmed against MuJoCo's own segmentation render as an independent ground truth. Root
  mechanism in the projection math wasn't identified; fix instead reads the segmentation mask's own
  pixel centroid per bowl, sidestepping the buggy function entirely. Markers also shrunk from filled
  `radius=15` circles to `radius=6` outline dots so they no longer cover the bowl.
- **Hardware:** 1x RTX PRO 6000 Blackwell (GPU 1 of the 4-GPU server, `g4-flex-20260824`),
  `openvla-libero:blackwell` image, same ephemeral `transformers==4.51.3` + `qwen-vl-utils` install as
  the 2026-08-25 Qwen run.
- **What ran:** `probe_bowl_pointing_qwen.py`, full battery (3 conditions x 10 tasks = 30 queries),
  identical model/prompts/scoring to the 2026-08-25 run — only the rendered images changed (fixed
  `bowl_pointing_common.py`, uncommitted in the `openvla` fork). Smoke-tested on task 0 first.
- **Outcome:** full detail in `benchmark_split_result.md` §8.2. Headline: all 3 conditions now score
  70% (was 60% / 40% / 60%) — clearly above chance (50%/50%/33%) where none convincingly cleared it
  before. `negative_contrast` and `positive_contrast` (identical images, wording differs) now agree on
  10/10 tasks (was 5/10) — the earlier "phrasing sensitivity" finding was very likely noise from bad
  markers, not a real wording effect.
- **Artifacts:** same paths as the 2026-08-25 Qwen run, overwritten in place:
  `openvla/experiments/figures/probe_bowl_pointing/` (30 images) and
  `openvla/experiments/logs/probe_bowl_pointing_qwen/probe_bowl_pointing_qwen.jsonl` (gitignored,
  local only). Code fix: `openvla/experiments/robot/libero/bowl_pointing_common.py` (uncommitted in
  the `openvla` fork as of this entry).

**Status:** closed — see `benchmark_split_result.md` §8.2 for the full write-up and cross-experiment
finding 14.

---

## 2026-08-26 — Bowl-pointing probe: `default` (no-distractor-mention) baseline added

- **Trigger:** user asked to add the `default` (no-distractor-mention) instruction as a comparison
  point alongside `negative_contrast`/`positive_contrast`/`hardneg` in the (now-fixed, see the
  2026-08-26 marker-fix entry above) bowl-pointing probe.
- **Code change:** `bowl_pointing_common.CONDITION_SUITES` gained `"default": ("libero_spatial",
  None)`; `render_and_annotate()` now also returns LIBERO's native `task.language` string so a
  `None` instruction dict (the existing `eval_registry.CONDITIONS` convention for "use the task's
  own language") resolves correctly in both probe scripts.
- **Hardware:** 1x RTX PRO 6000 Blackwell (GPU 1 of 4, `g4-flex-20260824`), same ephemeral
  `transformers==4.51.3` + `qwen-vl-utils` install as the prior two Qwen runs.
- **What ran:** `probe_bowl_pointing_qwen.py`, full battery, now 4 conditions x 10 tasks = 40 queries
  in one invocation (`default,negative_contrast,positive_contrast,hardneg`) so all four conditions'
  results land in the same run/file.
- **Outcome:** full detail in `benchmark_split_result.md` §8.3. Headline: `default` scores 5/10 (50%,
  exactly chance) — lower than both `negative_contrast` and `positive_contrast` (7/10, 70% each) on
  the identical images. The two distractor-mention phrasings each correctly resolve 2 tasks (task ids
  4 and 5, both target=bowl "1") that the target-only phrasing gets wrong, and are otherwise identical
  to each other and to `default` on every other task. So for this scene/model, distractor-mention
  phrasing is a disambiguating cue, not a difficulty source — see cross-experiment finding 15.
- **Artifacts:** same paths as the prior two Qwen runs, overwritten/extended in place:
  `openvla/experiments/figures/probe_bowl_pointing/` (+10 new `--default--` images) and
  `openvla/experiments/logs/probe_bowl_pointing_qwen/probe_bowl_pointing_qwen.jsonl` (gitignored,
  local only, now 40 lines). Code: `bowl_pointing_common.py`, `probe_bowl_pointing.py`,
  `probe_bowl_pointing_qwen.py` (all uncommitted in the `openvla` fork as of this entry).

**Status:** closed — see `benchmark_split_result.md` §8.3 for the full write-up and cross-experiment
finding 15.

---

## 2026-08-26 — Bowl-pointing probe: `hardneg_default` (3-bowl, no-distractor-mention) baseline added

- **Trigger:** user asked to extend the just-added `default` no-mention comparison (see the prior
  2026-08-26 entry) to the 3-bowl `hardneg` scene too.
- **Code change:** `bowl_pointing_common.CONDITION_SUITES` gained `"hardneg_default":
  ("libero_spatial_3bowl_hardneg", None)` — same 3-bowl scene as `hardneg`, LIBERO's own native
  (target-only) task language. Confirmed via smoke test that `task.language` is identical in form
  across the 2-bowl and 3-bowl suites (the extra distractor bowl doesn't change LIBERO's own
  description).
- **Hardware:** 1x RTX PRO 6000 Blackwell (GPU 1 of 4, `g4-flex-20260824`), same ephemeral
  `transformers==4.51.3` + `qwen-vl-utils` install as the prior Qwen runs.
- **What ran:** `probe_bowl_pointing_qwen.py`, full battery, now 5 conditions x 10 tasks = 50 queries
  in one invocation (`default,negative_contrast,positive_contrast,hardneg,hardneg_default`).
- **Outcome:** full detail in `benchmark_split_result.md` §8.3 (extended) and cross-experiment
  finding 16. Headline: `hardneg_default` scores 6/10 (60%, comfortably above the 33% chance floor) —
  lower than `hardneg`'s 7/10 (70%) but a much smaller gap than the 2-bowl `default` vs.
  `negative_contrast`/`positive_contrast` comparison (50% vs. 70%). The two 3-bowl conditions
  disagree on exactly 1 of 10 tasks (task id 5). Task id 0 is wrong in all 5 conditions across both
  scenes — the single hardest case in the whole battery; task id 2 is wrong only in the two 3-bowl
  conditions, isolating the extra bowl (not phrasing) as that task's specific difficulty.
- **Artifacts:** same paths as prior Qwen runs, extended in place:
  `openvla/experiments/figures/probe_bowl_pointing/` (+10 new `--hardneg_default--` images) and
  `openvla/experiments/logs/probe_bowl_pointing_qwen/probe_bowl_pointing_qwen.jsonl` (gitignored,
  local only, now 50 lines). Code: `bowl_pointing_common.py`, `probe_bowl_pointing.py`,
  `probe_bowl_pointing_qwen.py` (all uncommitted in the `openvla` fork as of this entry).

**Status:** closed — see `benchmark_split_result.md` §8.3 for the full write-up and cross-experiment
finding 16.

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
