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
it supersedes). The reference sections at the end ("Servers & environments reference", "Config
reference") are maintained in place: they are the one home for machine/server setup, which the
benchmark docs deliberately don't carry.

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

## 2026-09-04 — Bowl-attraction probe: `target_cue_proximity_novel` added (diagnostic, not a `run_eval.sh --split` launch)

- **Trigger:** user asked to add Split 4c's novel-phrasing condition (`target_cue_proximity_novel`,
  "close to X") to the bowl-attraction probe and merge all of §8.5's results into one consolidated
  table instead of the scattered per-round subsections.
- **Hardware:** same Berkeley-profile server, `openvla-libero:blackwell`. GPUs 0/2/3 were heavily
  loaded by unrelated concurrent training jobs (99-100% util) at launch time; ran all 4 tasks (3, 5,
  7, 9) sequentially on GPU 1, the only lightly-loaded GPU, rather than parallelizing.
- **New pitfall hit and fixed — see `CLAUDE.md` "Known pitfalls."** Task 3 (launched first) succeeded,
  but tasks 5/7/9 (launched after) all failed identically with `ImportError: cannot import name
  'AutoModelForVision2Seq' from 'transformers'`. Root cause: `run_eval.sh`'s `HOME=/workspace/.cache/home`
  mount is a *shared*, host-persisted directory — some concurrent session had `pip install --user`ed
  `transformers==5.16.1` into it (confirmed via direct inspection of
  `.cache/home/.local/lib/python3.11/site-packages/transformers-5.16.1.dist-info`), silently
  overriding the image's correctly-pinned `4.40.1` for every container sharing that mount, including
  this one — not caused by anything this session did. Verified the image's own baked-in version was
  still correct (`docker run --rm <image> python3 -c "import transformers; print(...)"`, no `HOME`
  mount, printed `4.40.1`). Fixed by adding `-e PYTHONNOUSERSITE=1` to the container invocation
  (rather than touching or reinstalling into the shared directory, which another session may still
  need) and re-ran tasks 5/7/9 successfully.
- **What ran:** `probe_bowl_attraction.py --task_id {3,5,7,9} --conditions target_cue_proximity_novel --num_trials 10`,
  40 rollouts total, using the same protocol as every prior round.
- **Outcome:** full detail in the consolidated `benchmark_split_result.md` §8.5/§8.6. Headline: pooled
  success 50.0% (vs. real 53.0% for the same 4-task cohort, Split 4c) — the best-performing of the
  three off-template/distractor-mention conditions tested through this probe, matching Split 4c's
  finding that novel phrasing beats familiar-but-wrongly-bound phrasing. Its failure-mode shape
  (among failures: 60.0% neither-approached, 40.0% target-approached-but-failed, 0% distractor-first)
  is nearly identical to `target_cue_landmark`'s (61.3%/38.7%/0%) despite the large gap in overall
  success rate — a second, independent line of evidence that distractor-pull isn't the mechanism,
  since this condition structurally cannot exhibit it (no distractor mentioned, no misleading
  template association) yet still fails the same way.
- **Artifacts:** `openvla/experiments/logs/probe_bowl_attraction/libero_spatial--t{3,5,7,9}--2026_09_04-*.jsonl`
  + matching `--summary.json` per task (gitignored, local only); 40 rollout videos under
  `openvla/rollouts/2026_09_04/`; launch logs
  `openvla/experiments/logs/probe_bowl_attraction_launch/t{3,5,7,9}_novel.out` (5/7/9's first, failed
  attempt overwritten by the successful re-run at the same path).

**Status:** closed — see `benchmark_split_result.md` §8.5/§8.6 (now one consolidated table across all
4 conditions tested through this probe).

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

## 2026-09-04 — Mechanistic localization probe: task 5, `default`/`negative_contrast` (diagnostic, not a `run_eval.sh --split` launch) — retroactively logged 2026-09-06

- **Note on timing.** This run actually happened on 2026-09-04, in a prior session, but was never
  logged here or written up in `benchmark_split_result.md` at the time — discovered on 2026-09-06 as
  pre-existing output files while responding to a request to run the probe. Logged now, retroactively,
  rather than left undocumented.
- **Trigger:** `benchmark_split_result.md` §8.6 gap #3 ("no mechanistic localization" — §8.5's
  bowl-attraction probe shows *what* the arm does, not *where in the network* it goes wrong).
- **Hardware:** Berkeley-profile server (4x RTX PRO 6000 Blackwell), `openvla-libero:blackwell`
  (exact GPU/launch command not recoverable after the fact — no launch log was captured for this
  run).
- **What ran:** new standalone script `openvla/experiments/robot/libero/probe_mechanistic_localization.py`
  — instruments real action rollouts (not VQA) with a logit-lens (resolution layer) and
  attention-mass (vision-patch share) diagnostic at every action-token prediction. Two output files
  found: `compare1` (5 episodes x 2 conditions, 30 instrumented steps) superseded by `fullcompare1`
  (15 episodes x 2 conditions, **full 220-step episodes** — `--max_env_steps_to_instrument 220`, not
  the script's own 30-step default), on task 5, conditions `default`/`negative_contrast`.
- **Outcome:** full detail in `benchmark_split_result.md` §8.9 and finding 19. Headline: **null
  result** — the two conditions' resolution-layer and vision-attention diagnostics are
  indistinguishable (well under 0.5 stdev apart on every metric), but the run is confounded: the
  script's disclosed missing center-crop preprocessing step drove task success to 0/15 in *both*
  conditions, far below the real eval's 92-94%/2-4% contrast for this task, so the run may not contain
  the behavioral difference the diagnostics were meant to distinguish. Treated as "tried, came back
  null under a confound," not a clean close of gap #3.
- **Artifacts:** `openvla/experiments/logs/probe_mechanistic_localization/libero_spatial--t5--compare1--2026_09_04-13_57_46.jsonl`
  + `libero_spatial--t5--fullcompare1--2026_09_04-15_09_35.jsonl` (gitignored, local only; no rollout
  videos, this script doesn't record them). Code:
  `openvla/experiments/robot/libero/probe_mechanistic_localization.py` (new, uncommitted in the
  `openvla` fork as of this entry; its own docstring's "DRAFT / not yet run" header predates this run
  and is now stale).

**Status:** open — see `benchmark_split_result.md` §8.6 gap #3 / §8.9 for recommended next steps
(fix center-crop and re-run; condition diagnostics on §8.5's per-episode approach labels).

---

## 2026-09-06 — Mechanistic localization probe: re-confirmation run (diagnostic, not a `run_eval.sh --split` launch)

- **Trigger:** user asked to run `probe_mechanistic_localization.py`. Discovered the undocumented
  2026-09-04 run (previous entry) in the process; that data was written up instead of re-running the
  full battery, but this session's own smoke test and default-config run are logged here for
  completeness.
- **Hardware:** GCP server (`config/server.env`, 4x RTX PRO 6000 Blackwell), `openvla-libero:blackwell`,
  GPU 1 (idle at launch time; GPUs 0/2/3 were occupied by unrelated concurrent jobs, confirmed via
  `nvidia-smi`). Full precision (`--load_in_4bit False --load_in_8bit False`), `OPENVLA_ATTN_IMPLEMENTATION=sdpa`
  (HF's SDPA path auto-falls-back to eager attention when `output_attentions=True` is requested, with
  a one-time warning — no code change needed).
- **What ran:** smoke test (task 5, `default`, 1 episode, 3 instrumented steps) to confirm the draft
  script still executes correctly on the current image/checkpoint, then the script's own default
  config (task 5, `default`+`negative_contrast`, 3 trials each, 30 instrumented steps/episode — 6
  episodes, 42 token-level records total).
- **Outcome:** both runs completed cleanly, reproducing the same qualitative pattern as the
  2026-09-04 `fullcompare1` run (see previous entry / `benchmark_split_result.md` §8.9) — not
  separately analyzed, since §8.9's write-up is based on the larger, full-episode 2026-09-04 data.
- **Artifacts:** `openvla/experiments/logs/probe_mechanistic_localization/libero_spatial--t5--smoketest--2026_09_06-08_22_17.jsonl`
  + `libero_spatial--t5--2026_09_06-08_23_20.jsonl` (gitignored, local only).

**Status:** closed — see `benchmark_split_result.md` §8.9 (write-up based on the 2026-09-04 data).

---

## 2026-09-06 — Mechanistic localization probe: center-crop fix, then a second bug found and fixed, then a valid re-run (diagnostic, not a `run_eval.sh --split` launch)

- **Trigger:** user asked to add back the disclosed missing center-crop step (§8.9's recommended next
  step) and re-run.
- **Hardware:** GCP server (`config/server.env`), `openvla-libero:blackwell`, GPU 1 throughout (idle at
  every launch; GPUs 0/2/3 occupied by unrelated jobs, confirmed via `nvidia-smi` each time).
- **Isolation:** code changes made in an isolated worktree (`openvla/.claude/worktrees/mechloc-centercrop`,
  branch `worktree-mechloc-centercrop`) per this session's background-job convention, since the script
  needed editing (not just running). Final script and output `.jsonl` files copied back to the
  canonical `openvla/experiments/robot/libero/` and `openvla/experiments/logs/probe_mechanistic_localization/`
  paths once validated; the script stays uncommitted there, matching every sibling probe script's
  convention in this project.
- **What ran, in order:**
  1. Added center-crop (`crop_and_resize`, `crop_scale=0.9`, matching `openvla_utils.get_vla_action`
     exactly). Smoke-tested (2 episodes, full 220 steps), then ran the full battery (task 5,
     `default`+`negative_contrast`, 15 episodes each, full 220-step episodes, `--run_id_note ccfix_full`).
     Result: `default` success rose from 0/15 to only 1/15 (6.7%) — real improvement, nowhere near the
     real eval's 92-94%, so a second problem clearly remained.
  2. Root-caused the remainder with a cheap controlled test (not a full rollout): same single frame,
     decoded once via `predict_action()`'s fast path (`sdpa`) and once via the probe's diagnostic path
     (`generate(..., output_attentions=True)`, which HF silently falls back to `eager` attention for,
     since `sdpa` doesn't support returning attention weights). 4 of 7 action dims differed on the very
     first prediction of a fresh episode — this bf16 checkpoint's coarse action-bin argmax is sensitive
     enough to the `sdpa`/`eager` kernels' differing float summation order that the diagnostic path had
     been silently studying a different (and evidently much worse) policy than the one actually being
     evaluated, on top of the center-crop bug.
  3. Redesigned `get_action_with_diagnostics` as two passes: pass 1 (`generate()`, no diagnostic flags)
     decides the real executed action; pass 2 is a separate, non-incremental forward pass teacher-forced
     on those exact tokens (`vla(input_ids=prompt+generated_tokens, ..., output_attentions=True,
     output_hidden_states=True)`), used only to read hidden-states/attentions — its own argmax is
     recorded (`diag_argmax_matches_executed`, a new field) but never used to act. Smoke-tested (3
     episodes, full 220 steps — 3/3 success, immediately consistent with the real eval), then ran the
     full battery again (`--run_id_note tf_full`).
- **Outcome:** full detail in `benchmark_split_result.md` §8.10 and finding 19. Headline: success rates
  now match the real eval (`default` 14/15 = 93.3% vs. 92-94%; `negative_contrast` 0/15 = 0% vs. 2-4%)
  — the probe is finally measuring the behavior it was built to measure. On that valid contrast, the
  resolution-layer and vision-attention diagnostics still show no condition-level difference (~0.5
  stdev apart) — the same qualitative null as the 2026-09-04 run, now unconfounded, so a materially
  stronger result. New side finding: the `sdpa`/`eager` argmax-mismatch rate is large (55-88%) for the
  6 continuous action dims and small (10-12%) for the gripper, tightly tracking each dimension's
  logit-lens resolution layer (late-resolving = small decision margin = kernel-sensitive) — consistent
  in both prompt conditions, so a property of the checkpoint's general calibration, not of the failing
  condition specifically.
- **Artifacts:** `openvla/experiments/logs/probe_mechanistic_localization/libero_spatial--t5--ccfix_full--2026_09_06-08_53_13.jsonl`
  (center-crop-only fix, superseded), `libero_spatial--t5--tf_full--2026_09_06-10_14_08.jsonl` (both
  fixes, the validated run), plus `*ccfix_smoketest*`/`*tf_smoketest*` (gitignored, local only, no
  rollout videos). Code: `openvla/experiments/robot/libero/probe_mechanistic_localization.py` (updated,
  still uncommitted in the `openvla` fork as of this entry).

**Status:** closed — see `benchmark_split_result.md` §8.10. Gap #3 (§8.6) now has a real, unconfounded
null result rather than a confounded one; the per-episode-approach-conditioning idea from §8.9 remains
the natural next step if this line of investigation continues.

---

## 2026-09-06 — Mechanistic localization probe: regrouped by §8.5's approach behavior instead of prompt condition (diagnostic, not a `run_eval.sh --split` launch)

- **Trigger:** user asked to redo the §8.10 diagnostics grouped by `probe_bowl_attraction.py`'s
  per-episode approach label (target/distractor/neither) rather than by prompt condition, per §8.9/
  §8.10's own recommendation that a per-condition average might be washing out a per-episode effect.
- **Hardware:** GCP server (`config/server.env`), `openvla-libero:blackwell`, GPU 1 throughout.
- **First attempt (failed) — separate `probe_bowl_attraction.py` re-run doesn't give a joinable
  label.** Ran it fresh on the exact same task/conditions/episodes/seed as §8.10's `tf_full` run,
  expecting identical trajectories. It wasn't: `default` success was 14/15 in both runs but a
  *different* episode failed (ep 13 here vs. ep 10 in `tf_full`), and `negative_contrast` success was
  1/15 here vs. 0/15 there. Two separate process launches of nominally the same computation diverged
  over the 220-step closed loop from ordinary GPU non-determinism (unseeded cuDNN/cuBLAS kernel
  selection) — not an attention-implementation issue this time, just run-to-run variance. Per-episode
  joins across two separately-launched probe scripts are therefore not valid on this checkpoint at
  this rollout length, even with matching seed/checkpoint/greedy-decoding.
- **Fix:** instrumented the same per-step eef-to-bowl-center distance bookkeeping
  `probe_bowl_attraction.py` uses directly into `probe_mechanistic_localization.py`'s own rollout loop
  (identical `near_thresh_m=0.08`, identical "first bowl within threshold" derivation), so the approach
  label and the mechanistic diagnostics now come from the same trajectory by construction. Smoke-tested
  (2 episodes), then ran the full battery a fourth time (`--run_id_note dist_full`) — episode outcomes
  reproduced §8.10's `tf_full` run exactly, confirming this script's own rollouts are internally
  reproducible (the divergence above was specific to comparing across the two different scripts).
- **Outcome:** full detail in `benchmark_split_result.md` §8.11 and finding 19 (updated). Headline: a
  first pooled-across-conditions regroup looked like a real effect (`target_first` episodes' mean
  vision-attention 0.114 vs. `distractor_first`/`neither`'s ~0.096) — until checking group composition
  showed it was a confound (`target_first` is 15/17 `default` episodes, so the group difference mostly
  restated the already-known condition-level gap). The properly isolated test — within
  `negative_contrast` only, comparing behavior with prompt condition held fixed — found
  `target_first`/`distractor_first`/`neither` at 0.095/0.096/0.098, indistinguishable (`target_first`
  n=2, underpowered). Resolution layer showed a small, inconclusive gap (`distractor_first` ~0.02-0.03
  lower) not treated as a finding given n=2 for `target_first`. Side observation: 2 of 15
  `negative_contrast` episodes reached the target bowl first and still failed — the same
  "approached-but-failed" pattern §8.5/§8.6 documented for task 3, here at a lower (13%) rate.
- **Artifacts:** `openvla/experiments/logs/probe_bowl_attraction/libero_spatial--t5--joinmechloc--2026_09_06-11_22_47.jsonl`
  (the non-joinable separate re-run, kept for the record) and
  `openvla/experiments/logs/probe_mechanistic_localization/libero_spatial--t5--dist_full--2026_09_06-11_40_47.jsonl`
  (the combined, self-consistent run analyzed above); gitignored, local only. Code:
  `openvla/experiments/robot/libero/probe_mechanistic_localization.py` (updated again, still
  uncommitted in the `openvla` fork as of this entry).

**Status:** closed — see `benchmark_split_result.md` §8.11. Gap #3 (§8.6) has now been tested at both
the condition level (§8.10) and the per-episode-behavior level (§8.11, after correcting a confound) —
both null. Remaining open avenues: a different diagnostic, or extending past task 5.

---

## 2026-09-08 — Mechanistic localization probe: episode-level reanalysis + length-matched rerun (diagnostic, not a `run_eval.sh --split` launch)

- **Trigger:** user asked to continue digging into the mechanistic localization probe; identified that
  §8.10/§8.11's "no condition-level difference" conclusion rested on comparing means against pooled
  per-token stdev rather than the correct episode-level unit (n=15/condition), and that §8.9's
  phase-based breakdown was only ever run on confounded (0%-success) data.
- **Part 1 — Tier-0 reanalysis (no new rollouts).** Re-analyzed §8.11's existing `dist_full` JSONL at
  the episode level: vision-attention shows a large, highly significant condition-level difference
  (`default` vs. `negative_contrast`, p<0.0001, Cohen's d≈1.9) that the token-pooled comparison had
  masked; resolution-layer's null holds. Flagged a likely prompt-length confound (`negative_contrast`'s
  instruction is ~12 words longer than `default`'s) as the probable cause, since attention is
  softmax-normalized over the whole sequence and more text tokens mechanically dilute vision's share.
- **Part 2 — length-matched rerun to test the confound.** Hardware: same GCP server, GPU 0 (GPU 1 was
  occupied by an unrelated `openpi` process — left untouched), `openvla-libero:blackwell`. Ran
  `probe_mechanistic_localization.py --task_id 5 --conditions
  target_cue_landmark,target_cue_proximity_novel --num_trials 15 --max_env_steps_to_instrument 220`
  (both conditions word-for-word the same length as `default`, per §8.6 gap 2) — smoke-tested (1
  episode/condition, 5 steps) then the full battery. Success: `target_cue_landmark` 1/15 (6.7%, vs. real
  eval's 0/10), `target_cue_proximity_novel` 10/15 (66.7%, vs. real eval's 5/10) — both consistent with
  the real eval within small-n noise.
- **Outcome:** length dilution is NOT sufficient to explain §8.12's finding — `target_cue_landmark`
  (matched length) still shows a significant vision-attention drop (p=0.0023, d=1.15) and, unlike
  `negative_contrast`, a significant resolution-layer shift too (p=0.0095, d=-0.62); `target_cue_proximity_novel`
  (matched length, high success) shows neither. But pooling all 4 conditions' 60 episodes together, both
  diagnostics track episode length/success generally (`corr(length, vision_attn)=-0.675`; pooled
  success-vs-fail p<0.0001 for vision-attn, p=0.0002 for resolution-layer) — largely because a failed
  episode runs to the 220-step cap while a successful one ends early, mechanically coupling length to
  outcome regardless of condition. The one test immune to that coupling (a fixed steps-0-9 window,
  identical extent in every episode) still shows a small, borderline `target_cue_landmark`-specific
  signal (p=0.029) absent in `target_cue_proximity_novel` (p=0.19) — suggestively aligned with finding
  18's "template binding, not template matching," but one borderline p-value among many comparisons run
  across §8.9-§8.13, not yet a finding.
- **Artifacts:** `openvla/experiments/logs/probe_mechanistic_localization/libero_spatial--t5--lenmatch_full--2026_09_08-04_11_24.jsonl`
  (analyzed run); smoketest jsonl deleted after passing; both gitignored, local only. Full detail in
  `benchmark_split_result.md` §8.12-§8.13.

**Status:** open, narrowed. Gap #3 (§8.6) still has no diagnostic that cleanly localizes the "never
commits" mechanism; the clearest remaining lead is §8.13's small early-window, template-binding-
consistent hint, which needs either more episodes or a diagnostic designed to avoid the length/outcome
coupling (e.g. counting a fixed window backward from episode end instead of forward from episode start).

---

## 2026-09-08 — Mechanistic localization probe: early-window replication at n=50 (diagnostic, not a `run_eval.sh --split` launch)

- **Trigger:** user asked to replicate §8.13's borderline (p=0.029, n=15) early-window vision-attention
  finding with more episodes.
- **Method:** the early window only needs env_step 10-19 (10 real steps after warmup), so
  `--max_env_steps_to_instrument 10` truncates every episode there directly — >20x cheaper per episode
  than the up-to-220-step runs, and unlike §8.13's post hoc restriction this makes the length/outcome
  coupling (§8.13 Result B) structurally impossible rather than merely avoided, since every episode is
  now exactly 10 steps by construction. Confirmed 50 pre-sampled init states exist per task (matching
  `NUM_TRIALS_PER_TASK=50`'s default), so ran `--num_trials 50` — hardware: same GCP server, GPU 0,
  `openvla-libero:blackwell`. Conditions: `default`, `target_cue_landmark`, `target_cue_proximity_novel`
  (task 5). Smoke-tested (2 episodes) then the full 150-episode battery, which completed in under a
  minute given the tiny per-episode cost.
- **Outcome:** `target_cue_landmark`'s early-window vision-attention deficit replicates and strengthens
  (p=0.029→0.0009, n=15→50, d=0.62) — no longer a single borderline test. But `target_cue_proximity_novel`,
  expected to be an unaffected control, also differs from `default` at n=50 (p=0.039) — in the opposite
  direction (higher, not lower, vision-attention) — so the clean "only the template-bound phrase shows
  an effect" story doesn't fully hold. Resolution-layer's direction flips between the whole-episode
  measurement (§8.13: `target_cue_landmark` resolves *later* than `default`) and this early-window-only
  one (resolves *earlier*, p=0.019) — informative given this design structurally rules out the
  length/outcome confound, meaning the two measurements are picking up genuinely different phenomena.
  Full detail and a tentative "confident misexecution" interpretation in `benchmark_split_result.md` §8.14.
- **Artifacts:** `openvla/experiments/logs/probe_mechanistic_localization/libero_spatial--t5--earlywin_n50--2026_09_08-04_46_21.jsonl`
  (1,500 records: 50 episodes × 3 conditions × 10 steps each); smoketest jsonl deleted after passing;
  both gitignored, local only.

**Status:** open, further narrowed. The early-window signal is now a replicated, non-borderline result,
but more complicated than first thought — next step is a diagnostic that can distinguish "confidently
executing the wrong motion" from "confidently executing the right one," which resolution-layer/attention-
mass alone cannot do.

---

## 2026-09-10 — New GCP server stood up; `spatial/default` re-run as environment validation

- **Trigger:** old `berkeley` GCP instance (`g4-flex-20260824`) was replaced by a fresh instance at a
  new IP (`ssh` alias `berkeley` repointed); needed a full environment rebuild plus one real eval to
  confirm the new box reproduces known results before trusting it for further work (incl. the §8.15
  diagnostic this was blocked on).
- **Environment setup:** fresh Ubuntu 22.04 instance, 4x RTX PRO 6000 Blackwell Server Edition (98GB
  VRAM each, driver 580.178.04) — `nvidia-container-toolkit` was present but Docker Engine was not;
  installed Docker CE + configured the nvidia runtime. Hit the "fresh server missing MuJoCo's EGL
  libs" pitfall (`CLAUDE.md` "Known pitfalls") on first render smoke test — `libnvidia-gl-580-server`
  wasn't installed despite the compute driver being present; installed it and regenerated the CDI spec
  (`nvidia-ctk cdi generate`), confirmed with a manual EGL render smoke test before touching the real
  eval pipeline. Built `openvla-libero:blackwell` from the current `Dockerfile.blackwell` (clean
  build, ~2 min given the server's own fast network). Cloned all three repos (`vla_ws` public HTTPS;
  `openvla`/`LIBERO` private, via SSH agent forwarding to this session's local GitHub key). Checkpoint
  (~15GB) transferred laptop-to-server over `rsync`, bottlenecked by the laptop's home upload
  bandwidth (~300-900kB/s, degrading over the transfer) — took ~4.5 hours; confirmed via a parallel-
  stream test that this was genuine raw uplink capacity, not a fixable per-connection limit, so no
  faster alternative was available this session. `preflight.py` clean on completion. Server's local
  `openvla` `main` fast-forwarded (not pushed upstream) to include the not-yet-merged confidence-
  diagnostic commit from PR #1 (§8.15), so that diagnostic can now actually be run here.
- **Smoke test:** task 5, 1 trial — success, confirming the full pipeline (Docker, GPU passthrough,
  EGL rendering, checkpoint load, action decoding, video/JSONL logging) end-to-end.
- **Outcome — full `spatial/default` re-run (500 rollouts, all 4 GPUs sharded) reproduces the
  documented baseline:** 84.4% (422/500) vs. the original 84.0% (420/500) — within the documented
  ±3.3pt pooled noise band (`benchmark_split_result.md` §0). Per-task numbers move by up to ±6-10pts in
  either direction (task 9: 72%→66%, task 6: 90%→96%), consistent with ordinary run-to-run GPU
  non-determinism already documented (§8.11) rather than anything server-specific.
- **Outcome:** full detail in `benchmark_split_result.md` §2 (`default` setting, replication note
  added). Results: `openvla/experiments/logs/results/libero_spatial--default--newserver_validation--shard{0..3}of4.jsonl`
  (server-local, gitignored); rollout videos `openvla/rollouts/2026_09_10/` (server-local).

**Status:** closed. New server validated as a drop-in replacement for the retired `berkeley` instance —
`config/server.env`'s existing settings (GPUs, image name, precision) needed no changes, since the new
box matches the same hardware profile. Unblocks §8.15's confidence diagnostic (bf16 GPU access now
available).

---

## 2026-09-10 — §8.15 confidence diagnostic launched on the new server (diagnostic, not a `run_eval.sh --split` launch)

- **Trigger:** new server validated (previous entry); ran the confidence diagnostic (`final_layer_margin`/
  `final_layer_entropy`) that had been implemented and blocked on GPU access since PR #1.
- **Method:** manual `docker run` (not `run_eval.sh`, which only drives `run_libero_eval.py`) invoking
  `probe_mechanistic_localization.py` directly with the same mounts/env as `run_eval.sh`. Same cheap,
  confound-free design as the 2026-09-08 early-window run: `--max_env_steps_to_instrument 10
  --num_trials 50`, task 5, `default`/`target_cue_landmark`/`target_cue_proximity_novel`, GPU 0 (server
  idle, confirmed via `docker ps`). Smoke-tested (2 episodes, confirmed `final_layer_margin`/
  `final_layer_entropy` present and sane) then the full 150-episode battery, which completed in under a
  minute.
- **Outcome:** clean null on both tests this diagnostic was built for — episode-level mean confidence
  (margin/entropy) doesn't differ by condition (p=0.15-0.65, small/null effect sizes), and the
  per-episode correlation between confidence and progress-toward-target doesn't either (p=0.29-0.54) —
  where there's any separation, it runs opposite §8.14's tentative "confident misexecution" hypothesis
  (`target_cue_landmark` has the *weakest* confidence-tracks-correctness relationship of the three, not
  a reversed one). Full detail in `benchmark_split_result.md` §8.15.
- **Artifacts:** `openvla/experiments/logs/probe_mechanistic_localization/libero_spatial--t5--confdiag_earlywin_n50--2026_09_10-09_52_30.jsonl`
  (1,500 records; server-local, gitignored); smoketest jsonl deleted after passing.

**Status:** closed. §8.15's diagnostic is the first in this line of investigation built to directly test
"confident-correct vs. confident-wrong motion" rather than infer it, and it comes back null — narrows
gap #3 (§8.6) further without resolving it. Remaining leads: the real, replicated vision-attention/
resolution-layer early-window effects (§8.14) still lack a mechanistic explanation of *why* they occur;
a longer instrumented window or more episodes could sharpen this diagnostic's noisy per-episode
correlation estimate (sd≈0.41-0.48 on n=9 within-episode points) further.

---

## 2026-09-30 → 2026-10-01 — `pi05_libero` on the whole benchmark (22 conditions, first non-OpenVLA policy)

- **Trigger:** first cross-model run. `run_libero_eval.py` gained `--model_family openpi`
  (openvla `07e34ec`, keepalive fix `a5f9a97`), so an openpi policy can be driven through the same
  registry, scenes, init states, and seed as the OpenVLA reference. Run metadata records `07e34ec`
  for `spatial/default`, `positive_contrast`, and `negative_contrast`, and `a5f9a97` for the rest.
- **Hardware:** `berkeley` (GCP host `g4-flexstart-mig-uswest1-9dxc`, 4× RTX PRO 6000 Blackwell,
  driver 580.178.04; see "Machines" below), `openvla-libero:blackwell`, `config/server.env`. The GPUs
  were shared with another user's training containers for the whole run, and with the OpenVLA batch
  in the next entry.
- **Policy:** `gs://openpi-assets/checkpoints/pi05_libero`, served by 4 openpi policy servers on the
  host (`scripts/serve_openpi.sh`, GPU *i* → port 8000+*i*); shard *i* of each split talks to server
  *i*. Differences from the OpenVLA path: agentview + wrist images, resize-with-pad, proprio state,
  replan every 5 steps. `env.seed(0)`, seed 7, 50 trials/task, and init states are unchanged.
  **`pi05_libero` was trained on all four LIBERO suites (40 tasks), not only `libero_spatial`.**
- **Launched** (tmux, `~/pipeline_pi05.sh`, run note `pi05_libero`, every call `--resume True`):
  1. Smoke test, `spatial/default --task_ids 5 --num_trials_per_task 2` (run note `pi05_smoke`): 2/2.
  2. `MODEL_FAMILY=openpi CHECKPOINT=gs://openpi-assets/checkpoints/pi05_libero RUN_NOTE=pi05_libero
     bash scripts/run_benchmark.sh --resume True` — the 13 `DEFAULT_SPLITS`, in script order
     (2026-09-30 18:29 → 2026-10-01 02:46 UTC).
  3. The same with `SPLITS=new` — the 9 conditions authored 2026-10-01 (02:46 → 07:42 UTC).
- **An earlier attempt was stopped and is superseded by this one:** the first `pi05` pipeline
  (`~/pipe_pi05.log`) was stopped because the websocket keepalive dropped the first query while the
  server JIT-compiled. The client now runs with `ping_interval=None`.
- **Outcome:** 22 conditions, 8,300/8,300 rollouts, no errors in the pipeline log.

  | Split | SR | Rollouts | | Split | SR | Rollouts |
  |---|--:|--:|---|---|--:|--:|
  | `spatial/default` | 98.4% | 492/500 | | `spatial/length_control_infix` | 99.2% | 496/500 |
  | `spatial/positive_contrast` | 92.2% | 461/500 | | `spatial/length_control_suffix` | 99.0% | 495/500 |
  | `spatial/negative_contrast` | 87.0% | 435/500 | | `grounding/paraphrase_lexical` | 97.2% | 486/500 |
  | `grounding/target_cue_region` | 75.5% | 302/400 | | `grounding/paraphrase_syntactic` | 97.8% | 489/500 |
  | `grounding/target_cue_landmark` | 87.0% | 174/200 | | `grounding/target_cue_region_v2` | 77.5% | 310/400 |
  | `grounding/target_cue_proximity_novel` | 96.0% | 192/200 | | `grounding/target_cue_region_v3` | 74.8% | 299/400 |
  | `spatial_3bowl/landmark_with_hardneg_prompt` | 66.2% | 331/500 | | `grounding/target_cue_proximity_beside` | 97.0% | 194/200 |
  | `spatial_3bowl/drawer_open` | 43.4% | 217/500 | | `grounding/target_cue_proximity_near` | 96.0% | 192/200 |
  | `spatial_3bowl/landmark` | 72.0% | 360/500 | | `grounding/target_cue_proximity_adjacent` | 97.0% | 194/200 |
  | `spatial_3bowl/irrelevant` | 80.6% | 403/500 | | | | |
  | `spatial_3bowl/semantic` | 91.0% | 455/500 | | | | |
  | `grounding/surface_landmark` | 100.0% | 50/50 | | | | |
  | `grounding/region_surface` | 100.0% | 50/50 | | | | |

  Full per-task tables and metrics: `benchmark_split_result.md` §9.
- **Results:** `openvla/experiments/logs/results/*--pi05_libero--shard{0..3}of4.jsonl` + `.meta.json`
  (server-local, gitignored), mirrored every 30 min to the private HF dataset
  `Qian0203/vla_ws-eval-results` by `~/sync_results_hf.sh`. Pipeline log: `~/pipe_pi05_v2.log`.

**Status:** closed. The 4 openpi policy servers were still up when this entry was written (2026-10-01
12:00 UTC); stop them with `bash scripts/serve_openpi.sh stop`.

---

## 2026-09-30 → 2026-10-01 — OpenVLA on the 9 conditions authored 2026-10-01 (in progress when logged: 3/9 finished)

- **Hardware / checkpoint:** same host and image as the previous entry, bf16, sdpa. Checkpoint
  `openvla/checkpoint/openvla-7b-libero-spatial-lora-r32` (fetched from the private HF repo
  `Qian0203/openvla-7b-libero-spatial-lora-r32` with `scripts/hf_checkpoint.sh download`). Run
  metadata records openvla commit `7f3884d` for `length_control_infix` and `a5f9a97` for the splits
  after it.
- **Launched** (tmux `eval`, `~/pipeline_2026_10_01.sh`, 2026-09-30 18:26 UTC):
  `CHECKPOINT=... SPLITS=new bash scripts/run_benchmark.sh --resume True`, run note
  `openvla-7b-libero-spatial-lora-r32`, 4 GPUs sharded. It ran concurrently with the pi05 batch above
  until 07:42 UTC and with another user's jobs throughout, so each 500-rollout split took about
  5 hours.
- **Outcome so far** (as of 2026-10-01 12:00 UTC):

  | Split | SR | Rollouts | Finished (UTC) |
  |---|--:|--:|---|
  | `spatial/length_control_infix` | 62.8% | 314/500 | 2026-10-01 00:01 |
  | `spatial/length_control_suffix` | 53.6% | 268/500 | 2026-10-01 05:33 |
  | `grounding/paraphrase_lexical` | 72.4% | 362/500 | 2026-10-01 10:20 |
  | `grounding/paraphrase_syntactic` | running | — | started 10:20 |
  | `grounding/target_cue_region_v2`, `_v3` | queued in the same pipeline | — | — |
  | `grounding/target_cue_proximity_beside`, `_near`, `_adjacent` | queued in the same pipeline | — | — |

  Detail for the three finished splits: `benchmark_split_result.md` §2 (length controls) and §5.5
  (`paraphrase_lexical`).
- **Results:** `openvla/experiments/logs/results/libero_spatial--{condition}--openvla-7b-libero-spatial-lora-r32--shard{0..3}of4.jsonl`
  (server-local, gitignored; mirrored to `Qian0203/vla_ws-eval-results`). Pipeline log:
  `~/pipe_openvla.log`.

**Status:** open. The remaining 6 splits get their own entry when the pipeline finishes.

---

## 2026-10-01 — OpenVLA batch on the 9 new conditions finished (completes the entry above)

- **Same launch as the previous entry**, no relaunch and no resume needed: the pipeline ran through
  all 9 splits and exited 0 at 2026-10-01 23:42 UTC. No shard failure and no CUDA OOM in any shard log.
- **Outcome** (all on openvla `a5f9a97`):

  | Split | SR | Rollouts | Ran (UTC, 2026-10-01) |
  |---|--:|--:|---|
  | `grounding/paraphrase_syntactic` | 82.2% | 411/500 | 10:20 → 14:24 |
  | `grounding/target_cue_region_v2` | 20.0% | 80/400 | 14:26 → 16:42 |
  | `grounding/target_cue_region_v3` | 18.8% | 75/400 | 16:42 → 19:05 |
  | `grounding/target_cue_proximity_beside` | 38.0% | 76/200 | 19:06 → 20:30 |
  | `grounding/target_cue_proximity_near` | 49.0% | 98/200 | 20:30 → 21:59 |
  | `grounding/target_cue_proximity_adjacent` | 37.5% | 75/200 | 21:59 → 23:42 |

  Together with the three splits in the previous entry: 9 conditions, 3,400/3,400 rollouts. Detail:
  `benchmark_split_result.md` §2 and §5.5.
- **Results:** same paths as the previous entry; the HF mirror `Qian0203/vla_ws-eval-results` holds
  all 11,702 rollouts in the results directory as of 2026-10-02 01:46 UTC.
- **Correction to the two entries above** (they are left as written): the "another user's training
  containers" on the GPUs were not another user's. The 14 `hybridil:train` containers mount
  `/home/qian/hybridil` and were started from the `qian` account (about 2 GB of GPU memory each). The
  other user on the host, `vaclis`, ran non-container training jobs of about 65 GB per GPU on the
  afternoon and evening of 2026-10-01 (first seen at 15:47 UTC).
- **openpi servers:** stopped 2026-10-01 19:36 UTC. `scripts/serve_openpi.sh stop` killed only the
  wrapper shells and left the `uv` / `serve_policy.py` children holding about 9 GB per GPU; those were
  terminated by hand. The script has since been fixed to stop the whole process group.

**Status:** closed. Every condition in the registry's current set now has an OpenVLA and a
`pi05_libero` number.

---

## Still queued (registry-ready, not yet launched)

**Nothing is waiting to be launched.** All 9 conditions authored 2026-10-01 are finished for both
OpenVLA and `pi05_libero` (entries above).

**Not registry-ready** (open design questions, `benchmark_split_plan.md` §9): Split 2's `path`
distractor.

---

## Servers & environments reference

This section records where each batch above ran and how to stand up a new machine. It is maintained
in place: update it when a machine is added or retired, and leave the dated entries above unchanged.

### Machines

| Machine | GPUs | Image / attention | Precision | Used for |
|---|---|---|---|---|
| Original H200 server (`SERVER_ROOT=/home/ec2-user/wenhan`, checkpoint outside the workspace) — retired | 5× H200 (GPUs 0–4) | `openvla-libero:cuda12.1` / FlashAttention-2 | bf16 | 2026-08-1x baseline batch |
| 4× Blackwell server (instance name not recorded) | 4× RTX PRO 6000 Blackwell | `openvla-libero:blackwell` / sdpa | bf16 | 2026-08-19 and 2026-08-20 batches |
| GCP `g4-flex-20260824`, ssh alias `berkeley` — replaced 2026-09-10 | 4× RTX PRO 6000 Blackwell Server Edition, 98 GB each, compute cap 12.0 | `openvla-libero:blackwell` / sdpa | bf16 | 2026-08-25 → 2026-09-08 entries |
| GCP replacement instance, same `berkeley` alias — replaced by 2026-09-30 | same hardware, driver 580.178.04 | `openvla-libero:blackwell` / sdpa | bf16 | 2026-09-10 entries |
| GCP `g4-flexstart-mig-uswest1-9dxc`, same `berkeley` alias — current. Shared host: other users' jobs run on the same GPUs. Workspace `/home/qian/vla_ws`, checkpoint `openvla/checkpoint/openvla-7b-libero-spatial-lora-r32`, openpi checkout `/home/qian/openpi` | same hardware, driver 580.178.04 | `openvla-libero:blackwell` / sdpa | bf16 | 2026-09-30 onward |
| Laptop | 1× RTX 5060 Laptop, 8 GB, compute cap 12.0 | `openvla-libero:blackwell` / sdpa | 4-bit only | scene authoring, init states, contact sheets, smoke tests (no reported number comes from here) |

Entries above that cite `config/berkeley.env` or `config/server.env` both refer to the 4× Blackwell
GCP profile.

### Machine profiles

Since 2026-09-29, machine-specific `config/*.env` files are no longer tracked in git. The settings
they held:

- **4× Blackwell GCP (`berkeley.env` / `server.env`):** `GPUS=0,1,2,3`,
  `IMAGE_NAME=openvla-libero:blackwell`, `OPENVLA_ATTN_IMPLEMENTATION=sdpa`, `LOAD_IN_4BIT=False`,
  `NUM_TRIALS_PER_TASK=50`, `SEED=7`. The workspace is deployed at `/home/qian/vla_ws` with the
  checkpoint inside it (`openvla/checkpoint/...`), so no `SERVER_ROOT` is needed.
- **H200 server:** see `config/server.env.example` (tracked).
- **Laptop:** see `config/laptop.env` (tracked).

The 84.0% baseline reproduces across hardware and attention kernels: 84.0% on H200 with
FlashAttention-2 (2026-08-1x) vs. 84.4% on Blackwell with sdpa (2026-09-10).

### Standing up a new server (procedure used on 2026-09-10)

1. Install Docker CE and `nvidia-container-toolkit`, and configure the NVIDIA runtime.
2. Check that MuJoCo's EGL rendering libraries are present. Compute-only cloud drivers ship without
   them, and the failure looks like
   `MUJOCO_EGL_DEVICE_ID ... must be an integer between 0 and -1`. To fix it, install
   `libnvidia-gl-<driver version>-server` and run
   `sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml`. No Docker restart is needed.
   Confirm with a manual EGL render before touching the eval pipeline.
3. Run `git clone --recursive https://github.com/Qian-0203/vla_ws.git`. Since 2026-09-29, `openvla`
   and `LIBERO` are submodules; before that date, each was cloned separately over SSH.
4. Build the image that matches the GPU's compute capability: `Dockerfile.blackwell` for 12.0, and
   `Dockerfile` below that. The build takes about 2 minutes on a server network. Build fresh rather
   than trusting an existing tag, because a tagged image once drifted to `mujoco==3.10.0` and broke
   all env stepping.
5. Transfer the checkpoint (about 15 GB). Using `rsync` from the laptop took about 4.5 hours, limited
   by the home uplink. Since 2026-10-01 the checkpoint is meant to go through a private Hugging Face
   model repo instead. The laptop uploads it once with
   `bash scripts/hf_checkpoint.sh upload <merged ckpt dir> <hf_user>/<repo>`. Each new server then
   runs `bash scripts/hf_checkpoint.sh download <hf_user>/<repo>`, which lands it under
   `openvla/checkpoint/<repo>`. Both ends need `uv` and a one-time
   `uvx --from huggingface_hub hf auth login`. The reference checkpoint is uploaded as the private
   repo `Qian0203/openvla-7b-libero-spatial-lora-r32`, which took about 5 minutes from the laptop on
   2026-10-01. So the 4.5-hour `rsync` was limited by the single connection, not by the home uplink.
6. Run `python3 scripts/preflight.py`.
7. Smoke test: `run_eval.sh --split spatial/default --task_ids 5 --num_trials_per_task 1`.
8. Validation: run the full `spatial/default` split. It must land within about ±3.3 pts of 84.0%
   before the machine is trusted.

Containers on one machine share `.cache/home` as `HOME`. A `pip install --user` from any session
therefore leaks into all of them. `run_eval.sh` has set `PYTHONNOUSERSITE=1` since 2026-09-29 to
prevent this; ad hoc `docker run` probes need the same flag.

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
