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

## 2026-08-26 — Split 2 second redefinition: `irrelevant` and `semantic` scene authoring (not a `run_eval.sh --split` launch)

Scene/registry authoring only, at the user's request — no GPU eval launched. `irrelevant`'s bowl_3
was redefined a second time (new suite `libero_spatial_3bowl_front`: bowl_3 always at the table's
front edge, `table_center` fallback for 2 tasks) and `semantic`'s task 4 was redefined (new suite
`libero_spatial_3bowl_semantic2`: bowl_3 moved from `next_to_plate_region` to
`between_plate_ramekin_region`, closer to the drawer and inside the same distance band as the other
9 tasks). Full rationale and per-task coordinates: `benchmark_split_plan.md` §Split 2 ("Second
redefinition"). Prior definitions kept as `irrelevant_v1_legacy` / `semantic_v1_legacy` in the
registry so their 88.8%/84.8% numbers stay attributable — see `benchmark_split_result.md` §3, §7.

- **Hardware:** laptop/local (this machine), `openvla-libero:blackwell` image, CPU/EGL rendering only
  (no CUDA compute needed for scene generation/verification — see CLAUDE.md "Known pitfalls").
- Generated init states (`LIBERO/scripts/gen_suite_init_states.py`) and verified
  (`verify_suite_init_states.py`) for both new suites. First pass failed for both (real physical
  overlaps caught by the verifier, not just estimated from region-center distances — see
  `benchmark_split_result.md` §7 for the exact numbers); fixed and re-verified PASS, worst
  separation 0.122m for both.
- Both new suites are registry-ready (`spatial_3bowl/irrelevant`, `spatial_3bowl/semantic`) but
  **not yet run** — see "Still queued" below.

## 2026-08-27 — Contact-sheet render fix + `irrelevant` offset fine-tune (not a `run_eval.sh --split` launch)

Scene-authoring/tooling only, at the user's request — no GPU eval launched.

- **Render-settling fix, applied retroactively.** The 2026-08-26 entry above's contact sheets for
  `libero_spatial_3bowl_front`/`libero_spatial_3bowl_semantic2` had been captured via
  `gen_suite_init_states.py`'s own inline preview (no physics-settle step before capture), so some
  bowls showed mid-fall/floating instead of resting on the table — the same class of bug
  `render_suite_contact_sheet.py` was already fixed for one day earlier (see that script), but these
  two suites were authored the same day and missed the fix. Re-rendered both with
  `render_suite_contact_sheet.py` (read-only over the already-verified `.pruned_init` files, init
  states unchanged); both now show every bowl resting flat with a contact shadow. Detail:
  `benchmark_split_result.md` §7.
- **`irrelevant` offset fine-tune.** `table_front`/`table_center` pulled another 0.05m apart, within
  `libero_spatial_3bowl_front` only: `table_front` → (0.24,−0.01)–(0.26,0.01) for the 6 non-fallback
  tasks; `table_center` fallback → (−0.15,−0.01)–(−0.10,0.01) for the 4 fallback tasks (1, 3, 5, 6;
  task 2's own `table_center` bowl_1 target untouched). Stays ≈0.29m clear of `stove_region` either
  way. Regenerated, re-verified PASS (worst sep 0.122m, task 4, unrelated to bowl_3; every other task
  improved to 0.148–0.301m), re-rendered and eyeballed. Full rationale:
  `benchmark_split_plan.md` §Split 2 ("Offset fine-tune").
- **Correction to the 2026-08-26 entry above:** it says "`table_center` fallback for 2 tasks" — the
  actual count is 4 (tasks 1, 3, 5, 6; tasks 3 and 5 were caught in a second verify pass after that
  entry was written). Left as-is per this file's append-only convention; correct count is in
  `benchmark_split_plan.md`/`benchmark_split_result.md`.
- `spatial_3bowl/irrelevant` (suite `libero_spatial_3bowl_front`) is still **not yet run** — see
  "Still queued" below. `spatial_3bowl/semantic` (suite `libero_spatial_3bowl_semantic2`) unaffected
  by the offset fine-tune (`irrelevant`-only change), also still not yet run.

## 2026-08-27 — Split 2 redefinitions run: `irrelevant` (`libero_spatial_3bowl_front`) + `semantic` (`libero_spatial_3bowl_semantic2`)

- **Hardware:** 4× RTX PRO 6000 Blackwell (`g4-flex-20260824`), `openvla-libero:blackwell` (sdpa).
  Note: this shell had a stray `IMAGE_NAME=common-cu129-ubuntu-2204-nvidia-580-stage` exported
  (unrelated to this project, not set by anything in this repo), which silently overrode
  `config/server.env`'s `IMAGE_NAME` per `run_eval.sh`'s own documented precedence (exported vars
  beat the machine-config file) and made the first launch attempt fail outright (image not found,
  all 4 shards failed in seconds). Fixed by passing `IMAGE_NAME=openvla-libero:blackwell` explicitly
  on the launch command line.
- **Checkpoint/seed:** unchanged from the baseline batch, seed 7, 50 trials/task, 10 tasks/condition
  (500 rollouts), sharded 4-way.
- **Bug found and fixed mid-run:** `spatial_3bowl/irrelevant`'s first launch livelocked — all 4
  shards spun at ~99% CPU with zero GPU utilization and zero rollout progress for over an hour
  (stalled at 121/500), spamming a benign-looking `MjRenderContextOffscreen` cleanup exception
  thousands of times. Root cause: `LIBERO/libero/libero/envs/env_wrapper.py`'s `ControlEnv.reset()`
  had an *unbounded* `while not success: try: env.reset() except RandomizationError: pass` retry
  loop — a persistent `RandomizationError` (robosuite's placement-sampler exception) retries forever
  with no bound, each attempt constructing and immediately tearing down a render context (the source
  of the cleanup-exception spam). This is a pre-existing bug in the LIBERO fork, not specific to the
  new suites' scene geometry (confirmed: `gen_suite_init_states.py`/`verify_suite_init_states.py` had
  already exercised `env.reset()` 50/50 times per task on both new suites with zero failures). It
  plausibly explains why every past full-suite batch in this log undershot 500 (444/500, 424/500,
  403/500, 412/500, 162/500) without ever crashing loudly enough to investigate. Fixed by bounding
  the retry to 50 attempts before re-raising (commit in the `LIBERO` fork) so a persistent failure
  now surfaces as a clear crash instead of an invisible livelock. Killed the 4 hung containers
  (`docker stop`), applied the fix (bind-mounted, no image rebuild needed), and relaunched
  `spatial_3bowl/irrelevant` with `--resume True` — continued cleanly from the 121 already-recorded
  rollouts to 500/500 with no further hangs. `spatial_3bowl/semantic` launched fresh afterward,
  completed 500/500 with no hangs either.

| # | Split/condition | Suite | Headline SR | Rollouts |
|--:|---|---|--:|--:|
| 1 | `spatial_3bowl/irrelevant` (current, 2nd redefinition + offset fine-tune) | `libero_spatial_3bowl_front` | 85.2% | 500/500 |
| 2 | `spatial_3bowl/semantic` (current, 2nd redefinition) | `libero_spatial_3bowl_semantic2` | 85.2% | 500/500 |

**Results files:**
```
results/libero_spatial_3bowl_front--default--shard{0..3}of4.jsonl
results/libero_spatial_3bowl_semantic2--default--shard{0..3}of4.jsonl
```

**Status:** complete. Full per-task tables, Δ vs. baseline, and analysis in
`benchmark_split_result.md` §3.

---

## 2026-09-02 — Bowl-attraction probe: instrumented action rollouts (diagnostic, not a `run_eval.sh --split` launch)

- **Trigger:** user asked why a separate VLM (Qwen2-VL, §8.2-8.3) can resolve the distractor-mention
  referring expression well above chance while OpenVLA's actual task success collapses under the same
  phrasing — wanted the mechanism identified and directly tested, not just inferred from success-rate
  deltas.
- **Hardware:** Berkeley server (`config/berkeley.env`, 4x RTX PRO 6000 Blackwell), `openvla-libero:blackwell`,
  1 GPU (device 0).
- **Pre-launch investigation.** A live smoke test first hit a hard `MUJOCO_EGL_DEVICE_ID` /
  zero-EGL-devices error identical in shape to CLAUDE.md's documented "missing EGL libs" pitfall — but
  the fix package (`libnvidia-gl-580-server`, matching this box's driver) was already installed, so
  that wasn't it. Root-caused instead to a stray shell-exported `IMAGE_NAME=common-cu129-ubuntu-2204-nvidia-580-stage`
  (same failure mode as this file's 2026-08-27 entry) silently overriding `config/berkeley.env`'s
  default per `run_eval.sh`'s documented precedence (exported vars beat the machine-config file).
  Fixed by passing `IMAGE_NAME=openvla-libero:blackwell` explicitly; a 1-task/1-trial smoke test then
  ran clean end-to-end (`Success: True`) — the EGL rendering pipeline itself was never actually broken
  on this box, only the image selection.
- **What ran:** new standalone script `openvla/experiments/robot/libero/probe_bowl_attraction.py` —
  instruments real action rollouts (not a VQA probe, unlike §8's dead end) with per-step
  end-effector-to-bowl distance, to classify each episode by which bowl (if any) the arm actually
  reached for. 3 conditions x 10 episodes = 30 rollouts on task 5 ("on the ramekin", `libero_spatial`):
  `default`, `negative_contrast`, `target_cue_landmark`. Smoke-tested on 2 `default` episodes first.
- **Operational hiccup.** Two earlier launch attempts that timed out at the harness level (before
  switching to a proper background launch) left their `docker run --rm` containers running detached
  rather than actually terminating, so 3 identical copies of the full battery briefly ran concurrently
  on the same GPU. Caught via `docker ps`/`nvidia-smi`; the 2 orphans were stopped, keeping the
  properly-tracked run. No data corruption resulted (each process's structured JSONL output is named
  by its own start timestamp, so the 3 runs' records never intermixed) — only wasted GPU cycles during
  the overlap window.
- **Outcome:** full detail in `benchmark_split_result.md` §8.5 and cross-experiment finding 17.
  Headline: under `default`, the arm reaches for the target bowl first in 10/10 episodes; under both
  `negative_contrast` and `target_cue_landmark`, the dominant failure mode is the arm never coming
  within grasping range of *either* bowl (60% and 80% of episodes) rather than confidently grasping the
  distractor (30% and 0% respectively) — direct behavioral evidence for template-mismatch action
  collapse over distractor-driven misdirection as the primary mechanism, corroborating Split 4b from a
  new angle.
- **Artifacts:** `openvla/experiments/logs/probe_bowl_attraction/libero_spatial--t5--2026_09_02-07_49_03.jsonl`
  + `--summary.json` (gitignored, local only); 30 rollout videos under `openvla/rollouts/2026_09_02/`.
  Code: `openvla/experiments/robot/libero/probe_bowl_attraction.py` (new, uncommitted in the `openvla`
  fork as of this entry).

**Status:** closed — see `benchmark_split_result.md` §8.5.

---

## 2026-09-02 — Bowl-attraction probe extension: tasks 3, 7, 9 (diagnostic, not a `run_eval.sh --split` launch)

- **Trigger:** user asked whether the "template-mismatch action collapse" reading of the task-5-only
  bowl-attraction probe above was actually proven, or just inferred from one task's evidence. Flagged
  as the largest of three open gaps (sample size, no length-matched control, no mechanistic
  localization) in `benchmark_split_result.md` §8.6; this run closes the sample-size gap.
- **Hardware:** same Berkeley-profile server (4x RTX PRO 6000 Blackwell), `openvla-libero:blackwell`.
  All 3 tasks launched in parallel, one GPU each (devices 0/1/2), via `run_in_background` Bash calls
  (not raw backgrounded `docker run`, to avoid this same probe's earlier orphaned-container incident).
- **Pre-launch check.** The same stray shell-exported `IMAGE_NAME` override documented in this file's
  earlier bowl-attraction entry was still present in the shell — passed `IMAGE_NAME=openvla-libero:blackwell`
  explicitly again. Smoke-tested task 3 (2 `default` episodes, 2/2 succeeded, target-first) before
  committing to the full battery.
- **What ran:** `probe_bowl_attraction.py --task_id {3,7,9} --conditions default,negative_contrast,target_cue_landmark --num_trials 10`,
  matching task 5's original protocol exactly (same seed, same 3 conditions, same episode count). 90
  rollouts total, on top of task 5's existing 30 — 120 across the 4-task cohort.
- **Outcome:** full detail in `benchmark_split_result.md` §8.5 (extension) and the revised §8.6.
  Headline: pooled across all 4 tasks, "arm never approaches either bowl" remains the largest failure
  category (55.6% of `negative_contrast` failures, 61.3% of `target_cue_landmark` failures) — the
  original task-5 finding generalizes. But task 3 broke the pattern: its failures are almost entirely
  the arm correctly approaching the target bowl and still failing to complete the pick-and-place, a
  failure mode nearly absent from tasks 5/7/9. Per-task success rates track the real 50-trial numbers'
  direction/magnitude (task 9's `default` running low, 50% vs. 72%, is within n=10 noise).
- **Artifacts:** `openvla/experiments/logs/probe_bowl_attraction/libero_spatial--t{3,7,9}--2026_09_02-14_39_*.jsonl`
  + matching `--summary.json` per task (gitignored, local only); 90 rollout videos under
  `openvla/rollouts/2026_09_02/`; launch logs `openvla/experiments/logs/probe_bowl_attraction_launch/t{3,7,9}.out`.

**Status:** closed — see `benchmark_split_result.md` §8.5/§8.6. Sample-size gap closed; length-matched
control and mechanistic-localization gaps remain open (§8.6).

---

## 2026-09-03 — Bowl-attraction probe: full 10-task extension (diagnostic, not a `run_eval.sh --split` launch)

- **Trigger:** user asked for the approaching-first test to be run on all 10 `libero_spatial` tasks,
  not just the 4-task surface cohort — then narrowed the request to `negative_contrast` specifically.
  `target_cue_landmark` can't be extended past its existing 4 tasks (no prompt exists for tasks whose
  native phrasing already is "next to X," and tasks 2/4 have no landmark-family analog at all), so this
  run adds `default`+`negative_contrast` (both defined for all 10 tasks) on the 6 remaining tasks
  (0, 1, 2, 4, 6, 8).
- **Hardware:** same Berkeley-profile server, `openvla-libero:blackwell`. Launched in 2 waves as GPUs
  freed up (tasks 0/1/2/4 first across all 4 GPUs, then 6/8 on GPUs 0/1 as they became free), each via
  a tracked `run_in_background` Bash call, not a raw backgrounded `docker run` — no orphaned containers
  this time. Stray shell `IMAGE_NAME` override (recurring pitfall, see 2026-09-02 entries above) present
  again; passed `IMAGE_NAME=openvla-libero:blackwell` explicitly.
- **What ran:** `probe_bowl_attraction.py --task_id {0,1,2,4,6,8} --conditions default,negative_contrast --num_trials 10`
  each, 60 rollouts total, on top of the existing 3/5/7/9 data (280 rollouts across all 10 tasks
  combined for `default`+`negative_contrast`).
- **Outcome:** full detail in `benchmark_split_result.md` §8.5 (full-suite extension) and the revised
  §8.6. Headline: pooled across all 10 tasks, "arm never approaches either bowl" is now the **majority**
  failure mode (58.1% of `negative_contrast` failures, up from 55.6% at 4-task scale) — and pooled
  success rates (81% `default`, 38% `negative_contrast`) closely reproduce the real 500-trial eval
  (84.0%, 36.8%), validating the probe at full scale. Task 3's "approached correctly, still failed"
  mode persists at scale (27.4% of failures pooled) rather than washing out. **Task 8 anomaly:**
  `negative_contrast` (70%) scored above `default` (60%) here, opposite the real eval's −48pt drop;
  instructions verified correct against the raw JSONL, read as sampling noise on n=10, not investigated
  further.
- **Artifacts:** `openvla/experiments/logs/probe_bowl_attraction/libero_spatial--t{0,1,2,4,6,8}--2026_09_03-*.jsonl`
  + matching `--summary.json` per task (gitignored, local only); 60 rollout videos under
  `openvla/rollouts/2026_09_03/`; launch logs `openvla/experiments/logs/probe_bowl_attraction_launch/t{0,1,2,4,6,8}.out`.

**Status:** closed — see `benchmark_split_result.md` §8.5/§8.6. Sample-size gap now fully closed
(all 10 tasks); length-matched control and mechanistic-localization gaps remain open (§8.6).

---

## 2026-09-02 — Split 4c: Familiar vs. Novel Proximity-Cue Probe

- **Trigger:** resolves the open question 4b left behind (§4c of `benchmark_split_plan.md`) — whether
  `target_cue_landmark`'s ~50pt drop tracks the relation-type change (surface→proximity) or the exact
  lexical template ("next to X") matching a phrase used natively elsewhere in `libero_spatial`.
- **Hardware:** Berkeley server (`config/berkeley.env`), `openvla-libero:blackwell`, 4 GPUs
  (`GPUS=0,1,2,3`), round-robin-sharded.
- **Launched:** `MACHINE_CONFIG=config/berkeley.env GPUS=0,1,2,3 bash docker/openvla_libero/run_eval.sh
  --split grounding/target_cue_proximity_novel --task_ids 3 5 7 9`.
- **Operational note.** Task ids 3,5,7,9 mod 4 shards land on only 2 of the 4 residue classes, so
  shards 0 and 2 correctly received `[]` and exited immediately (0 episodes each, by design of
  round-robin sharding, not a bug) while shards 1 and 3 each ran 2 tasks x 50 trials = 100 episodes.
  A couple of earlier launch attempts hit a harness-level timeout before the run was properly
  backgrounded — no stray containers this time (unlike the same day's bowl-attraction probe); the
  kept run is the one reflected in the results below.
- **Outcome:** 200/200 rollouts, all 4 tasks. Pooled SR 53.0% (task 3: 56%, task 5: 66%, task 7: 64%,
  task 9: 26%) — see `benchmark_split_result.md` §5.3 and cross-experiment finding 18. Headline: the
  novel phrasing ("close to X") drops *less* than the familiar one ("next to X" / `target_cue_landmark`,
  30.5% pooled) — a −22.5pt Familiarity Gap in the opposite direction from the plan's predicted branch.
  Reusing a phrase seen at fine-tuning time did not protect the policy here; it hurt more than a phrase
  it had never seen at all.
- **Artifacts:**
  `openvla/experiments/logs/results/libero_spatial--target_cue_proximity_novel--shard{0,1,2,3}of4.jsonl`
  (shards 0/2 empty by design) + matching `.meta.json`; rollout videos under
  `openvla/rollouts/2026_09_02/`.

**Status:** closed — see `benchmark_split_result.md` §5.3.

---

## 2026-09-04 — Qwen bowl-pointing probe: sampled decoding re-run (diagnostic, not a `run_eval.sh --split` launch)

- **Trigger:** user request to sample multiple responses per query (not just one greedy decode) on
  the Qwen bowl-pointing VQA probe (§8.1-§8.4), to see the trend rather than a single point estimate.
- **Code change:** `probe_bowl_pointing_qwen.py` (`openvla` fork, uncommitted) — `query_qwen()` now
  draws `num_samples` generations per query via one `model.generate(..., do_sample=True,
  temperature=cfg.temperature, num_return_sequences=cfg.num_samples)` call instead of a single
  `do_sample=False` call; each record reports `sample_accuracy` (fraction of samples correct) and a
  majority-vote answer alongside the full per-sample list. `--num_samples 1 --temperature 0`
  reproduces the old greedy behavior exactly.
- **Hardware:** Berkeley server (`config/berkeley.env`), `openvla-libero:blackwell` image, ephemeral
  `transformers==4.51.3` + `qwen-vl-utils` install (same pattern as every prior Qwen run) — **GPU 2**
  specifically, not GPU 0/1 (both occupied at the time by an unrelated job under a different Linux
  user, `hense1219`; confirmed via `nvidia-smi`/`docker ps` before launching so as not to disturb it).
- **What ran:** smoke test first (1 task, 3 samples, temperature 0.7 — confirmed the code path end to
  end), then the full battery: same 5 conditions x 10 tasks as §8.3, `--num_samples 10 --temperature
  0.7` (500 generations total).
- **Outcome:** full detail in `benchmark_split_result.md` §8.7. Headline: **zero within-query
  disagreement across all 50 queries** — every query's 10 samples unanimously agree, so
  `sample_accuracy` is exactly 0.0 or 1.0 everywhere, never split. `default`/`hardneg`/
  `hardneg_default` reproduce their §8.3 greedy numbers exactly (50%/70%/60%); `negative_contrast`
  (70%→60%) and `positive_contrast` (70%→80%) each move by one task versus the old greedy table, and
  in both cases the new answer is itself unanimous across all 10 samples — not resolved noise, but an
  unexplained (disclosed, not investigated) difference between the old single-sequence `generate()`
  call and the new batched `num_return_sequences=10` call.
- **Artifacts:** `openvla/experiments/logs/probe_bowl_pointing_qwen/probe_bowl_pointing_qwen.jsonl`
  (overwritten in place, schema extended — see §8.7); annotated images unchanged (same render cache,
  `openvla/experiments/figures/probe_bowl_pointing/`). Code:
  `probe_bowl_pointing_qwen.py` (uncommitted in the `openvla` fork as of this entry).

**Status:** closed — see `benchmark_split_result.md` §8.7.

---

## 2026-09-04 — Qwen3-VL-8B-Instruct bowl-pointing probe (diagnostic, not a `run_eval.sh --split` launch)

- **Trigger:** user judged §8.7's Qwen2-VL-7B-Instruct accuracy (48-90%, chance 33-50%) still too
  low; asked whether a newer/stronger VLM (Qwen3-VL) raises the ceiling.
- **Code:** new `probe_bowl_pointing_qwen3.py` (`openvla` fork, uncommitted) — same structure as
  `probe_bowl_pointing_qwen.py`, swapped to `Qwen3VLForConditionalGeneration` /
  `Qwen/Qwen3-VL-8B-Instruct` (8B chosen over the also-available 32B-Instruct to keep the scale
  comparable to Qwen2-VL-7B-Instruct). Hit and fixed the same class of pitfall §8.1 first documented
  for Qwen2-VL: an open-ended `transformers>=4.57.0` pulled today's `5.16.1`, which removed
  `AutoModelForVision2Seq` and broke an unrelated transitive import through `libero_utils.py` ->
  `robot_utils.py` -> `openvla_utils.py`; pinned to `transformers==4.57.6` (newest release still on
  the 4.x line) to get both `qwen3_vl` support and the still-present `AutoModelForVision2Seq`. No
  `qwen_vl_utils` needed this time — `processor.apply_chat_template(..., tokenize=True,
  return_dict=True, return_tensors="pt")` handles image encoding directly.
- **Hardware:** Berkeley server (`config/berkeley.env`), `openvla-libero:blackwell` image — **GPU 1**
  (GPUs 0/2/3 were occupied by another Linux user's unrelated training job; GPU 1 also had a small
  concurrent job from a sibling session of this project, left running alongside since there was ~82GB
  of headroom; confirmed via `nvidia-smi`/`docker ps` before launching).
- **What ran:** smoke test first (1 task, 3 samples — correctly answered task 0, which Qwen2-VL got
  wrong in every prior run), then the full battery: same 5 conditions x 10 tasks x 10 samples,
  temperature 0.7 (500 generations).
- **Outcome:** full detail in `benchmark_split_result.md` §8.8. Headline: **not a uniform upgrade**.
  Large gains on the 2-bowl scene (`default` 50%→81%, `negative_contrast` 60%→90%, `positive_contrast`
  80%→84%) but a regression on the 3-bowl `hardneg` scene (`hardneg` 70%→48%, `hardneg_default`
  60%→42%) — every condition still clears its chance baseline on both models, so §8.6's core
  synthesis is untouched, but the newer model doesn't simply dominate the older one. Also the first
  run in this probe family with genuine within-query sampling variance: 5/50 queries show real
  sample-to-sample disagreement (vs. 0/50 for Qwen2-VL in §8.7). Task 3 ("on the cookie box") flips
  from correct to incorrect in all 5 conditions, unanimously — the single clearest regression.
- **Artifacts:** `openvla/experiments/logs/probe_bowl_pointing_qwen3/probe_bowl_pointing_qwen3.jsonl`
  (new file); annotated images shared/unchanged
  (`openvla/experiments/figures/probe_bowl_pointing/`). Code: `probe_bowl_pointing_qwen3.py` (new,
  uncommitted in the `openvla` fork as of this entry).

**Status:** closed — see `benchmark_split_result.md` §8.8.

---

## Still queued (registry-ready, not yet launched)

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
  | `libero_spatial_3bowl_neutral` | +1 bowl, `irrelevant_v1_legacy` (retired) |
  | `libero_spatial_3bowl_front` | +1 bowl, current `irrelevant` |
  | `libero_spatial_3bowl_semantic` | +1 bowl, `semantic_v1_legacy` (retired) |
  | `libero_spatial_3bowl_semantic2` | +1 bowl, current `semantic` |
  | `libero_spatial_3bowl_hardneg` | +1 bowl, `landmark` / `landmark_with_hardneg_prompt` |
  | `libero_spatial_grounding_surface_landmark` | 2 bowls, distractor moved to a landmark region |
  | `libero_spatial_grounding_region_surface` | 2 bowls, distractor moved to a surface region |

- Helper scripts (`LIBERO/scripts/`): `gen_suite_init_states.py <suite>`,
  `verify_suite_init_states.py <suite>`, `render_suite_contact_sheet.py <suite>`,
  `compare_two_suites_init.py <suite_a> <suite_b> <outname>`.
- Artifacts: per-shard text logs and structured JSONL results under `openvla/experiments/logs/` (see
  `CLAUDE.md` "Results & logs" for exact naming); rollout videos under `openvla/rollouts/<date>/`;
  figures under `openvla/experiments/figures/`.
