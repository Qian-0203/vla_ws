# OpenVLA · LIBERO-Spatial Benchmark Plan

Canonical, human-readable spec for the evaluation suite: what each split tests, why, and how it's
built. The executable source of truth for which (task suite, unnorm key, prompt condition) a split
actually runs is `openvla/experiments/robot/libero/eval_registry.py::SPLITS` — this document and
that registry must agree; if you change one, change the other. Operational details (environment
setup, exact commands, log/result locations) live in `CLAUDE.md` and are linked, not repeated, here.

**This is the plan and process doc.** It answers "what is each split, what's the hypothesis, what's
built." It does **not** track whether a condition has actually been run or what the numbers were —
that's `benchmark_split_result.md` (status/progress dashboard + cross-experiment analysis) and
`eval_results.md` (append-only, full per-experiment write-ups — the ultimate source for any number).
When a split's definition changes, update this file. When an eval finishes, update the other two.

## 1. Research motivation

`libero_spatial`-fine-tuned OpenVLA (7B, LoRA r32) reaches 84.0% success on its own training
distribution (2 identical bowls, target-only prompts). The open question: **how much of that is
robust spatial/language grounding vs. overfitting to a narrow prompt-and-scene distribution?**
Each split below changes exactly one axis — prompt phrasing, distractor placement, scene clutter,
or the type of spatial cue used — while holding the model, checkpoint, and seed fixed, so drops
in success rate can be attributed to a specific capability gap rather than confounded together.

## 2. Splits overview

| Split | Probes | Registry status |
|---|---|---|
| 1. Prompt Sensitivity | Does naming/negating a distractor in the prompt help or hurt? | 3/3 conditions implemented |
| 2. Distractor Placement | Does *where* an extra distractor sits matter more than its presence? | 3/4 conditions implemented (`path` not authored); `irrelevant` redefined to fix a path-proximity confound (see Split 2 below) |
| 3. Scene Complexity | Does added clutter (open drawer) degrade the policy, or just block the arm? | Implemented |
| 4. Surface vs. Landmark Grounding | Does the policy rely on landmark proximity vs. surface/region cues? | 4a: cells implemented (4/6 reuse existing data, 2/6 new scenes); 4b: not built |

"Implemented" = task suite + prompts exist in the registry and can be run with one `run_eval.sh`
command. Whether a condition has actually been *run* is tracked in `benchmark_split_result.md`, not
here — see that doc's status overview and §9 (open design questions) below for what's outstanding.

## 3. Splits in detail

### Split 1 — Prompt Sensitivity Probe

| Field | Description |
|---|---|
| Goal | Whether describing/negating a distractor in language helps or hurts disambiguation |
| Hypothesis | Naming a distractor without practice at contrastive language *confuses* rather than helps |
| Changed variable | Prompt text only |
| Controlled variables | Scene, bowl positions, init states, checkpoint, seed |
| Conditions | `default` (target only) · `positive_contrast` (mentions distractor, no negation) · `negative_contrast` (names + negates distractor, previously called "explicit") |
| Tasks | All 10 `libero_spatial` tasks |
| Trials | 50/task/condition, seed 7 |
| Metrics | See formulas below |
| Output | `EVAL-libero_spatial-openvla-*--{condition}.txt` + `results/libero_spatial--{condition}.jsonl` |

**Metrics** (SR = success rate, higher is better; all "Drop" metrics: positive = condition hurt):

```
Negative Contrast Drop  = SR(default) - SR(negative_contrast)
Distractor Mention Drop = SR(default) - SR(positive_contrast)
Negation-specific Drop  = SR(positive_contrast) - SR(negative_contrast)
```

Registry: `spatial/default`, `spatial/positive_contrast`, `spatial/negative_contrast`.

### Split 2 — Distractor Placement Probe

| Field | Description |
|---|---|
| Goal | Whether an extra distractor's *position* — not just its presence — drives failures |
| Hypothesis | Distractors near the target's own landmark ("landmark") hurt more than neutral ("irrelevant") or other-landmark ("semantic") placement |
| Changed variable | Position of a 3rd `akita_black_bowl` distractor |
| Controlled variables | Target bowl position, 2nd (original) distractor, prompt = `default` (except the combo condition below), checkpoint, seed |
| Conditions | `irrelevant` (per-task region, off the reach path, distance-matched to `semantic`/`landmark`) · `semantic` (named landmark different from target's) · `landmark` (near target's OWN landmark, farther away — "hard negative") · `path` (between target and plate) — **not yet authored, see §9** |
| Tasks | All 10 tasks, scene = corresponding `libero_spatial_3bowl*` suite |
| Trials | 50/task/condition, seed 7 |
| Metrics | `Distractor-type Drop = SR(spatial/default) - SR(condition)`, plus per-task deltas |
| Output | `EVAL-libero_spatial_3bowl*-openvla-*.txt` + matching `results/*.jsonl` |

Registry: `spatial_3bowl/irrelevant` (suite `libero_spatial_3bowl_neutral`, redefined — see below),
`spatial_3bowl/center_fixed_legacy` (suite `libero_spatial_3bowl`, the retired single-fixed-
coordinate definition, kept only so `eval_results.md`'s original Exp 2 number stays attributable),
`spatial_3bowl/semantic` (suite `libero_spatial_3bowl_semantic`), `spatial_3bowl/landmark` (suite
`libero_spatial_3bowl_hardneg`). All reuse `unnorm_key=libero_spatial` since the checkpoint was
trained on the 2-bowl scene. `spatial_3bowl/landmark_with_hardneg_prompt` additionally swaps in the
`hardneg` prompt condition on the same scene, combining Split 1 x Split 2.

**Distractor types & purpose.** The four conditions aren't just four placements — each targets a
different failure mode:

| Condition | Placement | Tests |
|---|---|---|
| `irrelevant` | Neutral, off the target-to-plate reach path, not tied to any relational language | Object-count robustness — does an extra bowl hurt regardless of where it sits |
| `semantic` | At a named landmark region (`next_to_X`/`on_X`) that is **not** the target's own landmark | Relation disambiguation — does sitting at *some* nameable relational spot pull the policy even when that landmark doesn't match the prompt |
| `landmark` | Near the target's **own** landmark, farther away than the real target (hard negative) | Fine-grained spatial grounding — can the policy still pick the closer, correct bowl when a plausible look-alike sits nearby |
| `path` | Between the target and the plate | Trajectory interference — physically fouls the transport path, independent of language grounding. **Not yet authored** (§9) |

For several tasks `irrelevant` and `semantic` land on the *same coordinate* (e.g. task 0's
`next_to_box_region`) — the two conditions are distinguished by *why* that region was chosen (least-
crowded neutral spot vs. deliberately a named non-target landmark), not by geometry alone.

**Bowl coordinates** (all ranges are `[x1,y1]`–`[x2,y2]` meters relative to `main_table`'s
origin; shared region catalog, identical across suites unless noted):

| Region | Range |
|---|---|
| `plate_region` | (0.05,0.19)–(0.07,0.21) |
| `next_to_plate_region` | (0.00,0.30)–(0.02,0.32) |
| `box_region` | (0.06,0.02)–(0.08,0.04) |
| `next_to_box_region` | (0.12,−0.08)–(0.14,−0.06) |
| `between_plate_ramekin_region` | (−0.06,0.19)–(−0.04,0.21) |
| `ramekin_region` | (−0.21,0.19)–(−0.19,0.21) |
| `next_to_ramekin_region` | (−0.19,0.31)–(−0.17,0.33) |
| `table_center` | (−0.10,−0.01)–(−0.05,0.01) |
| `table_front` | (0.19,−0.01)–(0.21,0.01) |
| `cabinet_region` | (0.02,−0.28)–(0.04,−0.26) |
| `stove_region` | (−0.42,−0.15)–(−0.40,−0.13) |

Per-task 3-bowl placement (bowl_1 = target, bowl_2 = original distractor, bowl_3 = 3rd bowl).
Object-relative placements (`cookies_1`, `wooden_cabinet_1_top_side`, `flat_stove_1_cook_region`,
`glazed_rim_porcelain_ramekin_1`, task 4's `In wooden_cabinet_1_top_region`) have no `main_table`
coordinate in the BDDL, so they're listed by object name, not fabricated numbers. bowl_1/bowl_2
are identical across all four bowl_3 columns below (only bowl_3 differs):

| id | target | bowl_1 | bowl_2 | bowl_3 `irrelevant` (redefined) | bowl_3 `center_fixed_legacy` | bowl_3 `semantic` | bowl_3 `landmark` (`hardneg_region`, per-task) |
|--:|---|---|---|---|---|---|---|
| 0 | between plate & ramekin | `between_plate_ramekin_region` | `next_to_ramekin_region` | `next_to_box_region` | `table_center` | `next_to_box_region` | (−0.070,0.040)–(−0.050,0.060) |
| 1 | next to ramekin | `next_to_ramekin_region` | `next_to_box_region` | `table_center` | `table_center` | `next_to_plate_region` | (−0.210,0.040)–(−0.190,0.060) |
| 2 | table center | `table_center` | `next_to_plate_region` | `next_to_ramekin_region` | `table_front`* | `next_to_ramekin_region` | (−0.290,0.010)–(−0.270,0.030) |
| 3 | on cookie box | `cookies_1` (obj) | `wooden_cabinet_1_top_side` (obj) | `next_to_ramekin_region` | `table_center` | `next_to_ramekin_region` | (0.180,−0.070)–(0.200,−0.050) |
| 4 | in top drawer | `In wooden_cabinet_1_top_region` (obj) | `wooden_cabinet_1_top_side` (obj) | `next_to_ramekin_region` | `table_center` | `next_to_plate_region` | (−0.060,−0.070)–(−0.040,−0.050) |
| 5 | on ramekin | `glazed_rim_porcelain_ramekin_1` (obj) | `cookies_1` (obj) | `table_center` | `table_center` | `next_to_plate_region` | (−0.210,0.040)–(−0.190,0.060) |
| 6 | next to cookie box | `next_to_box_region` | `flat_stove_1_cook_region` (obj) | `next_to_ramekin_region` | `table_center` | `next_to_ramekin_region` | (−0.030,−0.120)–(−0.010,−0.100) |
| 7 | on stove | `flat_stove_1_cook_region` (obj) | `wooden_cabinet_1_top_side` (obj) | `next_to_box_region` | `table_center` | `next_to_box_region` | (−0.310,−0.010)–(−0.290,0.010) |
| 8 | next to plate | `next_to_plate_region` | `next_to_ramekin_region` | `table_front` | `table_center` | `next_to_box_region` | (0.190,0.080)–(0.210,0.100) |
| 9 | on wooden cabinet | `wooden_cabinet_1_top_side` (obj) | `flat_stove_1_cook_region` (obj) | `next_to_ramekin_region` | `table_center` | `next_to_ramekin_region` | (−0.010,−0.060)–(0.010,−0.040) |

\* Task 2's `center_fixed_legacy` bowl_3 uses `table_front` instead of `table_center` only because
bowl_1 already occupies `table_center` there.

**Design confound found, then fixed.** `center_fixed_legacy`'s bowl_3 sits at the *same fixed
absolute coordinate* in 9 of 10 tasks, regardless of the target's location — it is not "neutral"
relative to every task's reach path. This produced task 6's outlier result in `eval_results.md`
Exp 2 (90%→44%): `table_center` (−0.075,0.0) sits roughly on the straight line between
`next_to_box_region` (0.13,−0.07) and `plate_region` (0.06,0.20) — a rough perpendicular-distance
estimate puts it ~0.18m off that line, comparable to the ~0.115m bowl diameter plus gripper
clearance.

`irrelevant` was redefined (new suite `libero_spatial_3bowl_neutral`) to fix this: for each task,
bowl_3 is placed in whichever of the 6 named regions independently confirmed safe for free bowl
placement (never a fixture base — `cabinet_region`/`stove_region`/`ramekin_region`/`box_region`
are excluded, they're where the cabinet/stove/ramekin/cookie-box objects themselves sit) gives the
largest clearance from *both* the target-to-plate reach path (≥0.20m in every task) and bowl_2's
own position, picking the least-reused region on ties to keep some diversity. Verified via
`LIBERO/scripts/verify_suite_init_states.py` — passes, worst separation 0.122m (task 4's
persistent bowl_1/bowl_2 constraint present in every suite, unrelated to bowl_3), initially caught
one real overlap (task 1: `table_front` was only 0.099m from bowl_2's `next_to_box_region`, fixed
by reassigning to `table_center`). One residual imperfection, disclosed rather than hidden:
`next_to_ramekin_region` is reused for 5 of 10 tasks since only 6 safe regions exist and several
tasks exclude 2-3 of them (own target + bowl_2's region) — "neutral" leans toward the ramekin
vicinity more than an ideal design would. Distances to target range 0.236–0.626m, comparable to or
larger than `semantic`'s 0.33–0.50m and clearly farther than `landmark`'s 0.15–0.29m, so the
ordering (`landmark` closest → `semantic` mid → `irrelevant` farthest) is intact. `landmark`'s
bowl_3 gets a bespoke `hardneg_region` per task and never shared this flaw. `center_fixed_legacy`
is kept in the registry only as a retired label so `eval_results.md`'s original Exp 2 number stays
attributable — it is **not** reused as the current `irrelevant` definition.

### Split 3 — Scene Complexity Probe

| Field | Description |
|---|---|
| Goal | Whether clutter/occlusion degrades the policy, independent of scene feasibility |
| Hypothesis | Clutter costs real success rate, but naive deltas overstate it by conflating policy failure with physically blocked trajectories |
| Changed variable | Wooden cabinet's top drawer: closed vs. open |
| Controlled variables | 3-bowl scene (`center_fixed_legacy` placement — see Split 2's redefinition note), prompt = `default`, checkpoint, seed |
| Conditions | `drawer_closed` (= Split 2's `center_fixed_legacy`, not the current `irrelevant`) · `drawer_open` |
| Tasks | All 10 tasks; **tasks 3, 6, 7 excluded from the adjusted metric** — rollout review showed the open drawer physically blocks their trajectories (`eval_results.md` Exp 3; see `benchmark_split_result.md` for the methodology caveat on this exclusion) |
| Trials | 50/task/condition, seed 7 |
| Metrics | `Raw Clutter Drop = SR(drawer_closed) - SR(drawer_open)` over all 10 tasks; `Adjusted Clutter Drop` = same formula restricted to the 7 feasible tasks {0,1,2,4,5,8,9} |
| Output | `EVAL-libero_spatial_3bowl_open-openvla-*.txt` + matching `results/*.jsonl` |

Registry: `spatial_3bowl/drawer_open` (suite `libero_spatial_3bowl_open`).

### Split 4 — Surface vs. Landmark Grounding Probe

Two sub-parts with different requirements — kept separate rather than forced into one shape.

**4a. Grounding-by-scene probe.** Categorize each (target, distractor) pair already in the
scene catalog by relation family, and measure success/attraction per category — no prompt
change, this is purely about what's physically present.

| Field | Description |
|---|---|
| Goal | Whether the policy's target/distractor confusion depends on relation type (landmark/surface/region), not just distance |
| Hypothesis | Landmark-target scenes are more attraction-prone than surface/region-target scenes, since landmark grounding requires relative (not just absolute) reasoning |
| Changed variable | (target relation family) x (distractor relation family) |
| Controlled variables | Prompt = `default`, checkpoint, seed |
| Conditions | 6 cells: {landmark, surface, region} target x {landmark, surface} distractor (containment target excluded — only 1 task, `top_drawer`, and no distractor-placement variants exist for it) |
| Tasks | 4 cells (10 tasks total) already exist inside baseline `libero_spatial` — see `eval_registry.GROUNDING_PROBE_CELLS`; 2 cells (`surface`/`landmark` and `region`/`surface`) needed a new scene (existing distractor bowl moved to a different pre-existing region; canonical `libero_spatial` untouched) |
| Trials | 50/task/cell, seed 7 |
| Metrics | Final success, first-pick accuracy, wrong-bowl pick rate, distractor attraction rate (see §9 — the latter three need per-step contact logging not yet implemented) |
| Output | `libero_spatial` results filtered by `--task_ids` for the 4 existing cells; `EVAL-libero_spatial_grounding_*-openvla-*.txt` for the 2 new ones |

Registry: existing-scene cells run via `--split spatial/default --task_ids <ids from
GROUNDING_PROBE_CELLS>`; new cells are `grounding/surface_landmark`, `grounding/region_surface`.

**4b. Target-cue x distractor-cue prompt matrix.** The original spec's 3x3 matrix ("Pick the
bowl next to the ramekin, not the one next to the cookie box...") requires describing the SAME
physical target/distractor pair truthfully via 3 different cue types each. **Not built** — see
§9, this needs either new scene geometry or a relaxed truthfulness constraint, not a decision to
make silently.

## 4. Baselines

The single reference number everything compares against: `spatial/default` — 2 bowls, default
prompt, `libero_spatial` scene, 84.0% (420/500), `eval_results.md` Exp 1. Split 3's adjusted
comparison uses a recomputed 7-task baseline (84.9%, 297/350) instead — see Split 3's row above.

## 5. Rollout & seed protocol

- 50 trials/task, seed 7, fixed across every split (matches the original baseline run).
- Init states are pre-sampled per task (`.pruned_init` files, torch tensors of shape
  `(50, state_dim)`); every condition on the same scene replays the identical 50 init states, so
  differences are attributable to the changed variable, not sampling noise.
- Success = LIBERO's own goal predicate for that task (`env.step` returns `done=True`).
- At n=50/task, 1 standard error is roughly ±5-7 points — treat single-task deltas below ~10
  points as noise (stated explicitly in `eval_results.md`, carried forward here).

## 6. Aggregation rule

Suite-wide success rate = **mean of the 10 (or n) per-task success rates**, not a flat count over
all rollouts (only differs from a flat average when a run is incomplete or task-filtered).
`scripts/aggregate_results.py` implements this once, canonically, from the JSONL results files.

## 7. Expected outputs & naming

- Per-run text log: `openvla/experiments/logs/EVAL-{task_suite_name}-openvla-{timestamp}[--{run_id_note}][--shard{i}of{N}].txt`
- Structured per-rollout results (used for resume + aggregation): `openvla/experiments/logs/results/{task_suite_name}--{condition}[--{run_id_note}][--shard{i}of{N}].jsonl`
- Run metadata (config, checkpoint, git commit, env, timestamp): sibling `*.meta.json`
- Rollout videos: `openvla/rollouts/{date}/`
- See `CLAUDE.md` for the exact commands that produce these.

## 8. Design-level limitations & confounders

- **Manual feasibility calls aren't automated.** Split 3's task-3/6/7 exclusion was decided by
  watching rollout videos, not an automated "is this trajectory kinematically feasible" check. A
  different reviewer might draw the line differently — documented as a limitation of the
  methodology, not silently treated as ground truth.
- **Fixed `unnorm_key` across all scene variants.** Every Split 2/3/4 suite keeps the checkpoint's
  original `unnorm_key` (`libero_spatial_no_noops`) even though the scene changed — correct for
  action un-normalization (the action space didn't change), but means none of these variants test
  whether the *policy* generalizes to a re-trained distribution, only whether a fixed policy holds
  up to distribution shift.
- **`center_fixed_legacy` is a single fixed coordinate** (`table_center`/`table_front`) reused
  across all 10 tasks, not a per-task-neutral position — see Split 2's confound note above. It's
  kept only as a retired label for `eval_results.md`'s original Exp 2 number, never reused as the
  current `irrelevant` definition.

## 9. Open design questions (do not implement without a decision)

1. **Split 2 `path` distractor**: needs a new per-task numeric region (not a reused named
   region), since "between target and plate" is geometrically different for every task. Proposed
   approach: midpoint between the target's region center and `plate_region`, verified via
   `LIBERO/scripts/gen_suite_init_states.py` + `verify_suite_init_states.py`'s overlap check
   before trusting it. Deferred rather than rushed without enough render/verify iterations.
2. **Split 4b cue-phrasing matrix**: the existing scene catalog gives each bowl exactly one
   truthful spatial description; a 3x3 truthful-phrasing matrix isn't obtainable by relabeling.
   Either (a) build new scenes where a bowl is genuinely near multiple landmark/surface/region
   cues simultaneously, or (b) relax "truthful" to "salient but approximate" and accept some
   prompts describe a real but non-primary relation. Needs a call from whoever owns the research
   design before either is built.
3. **First-pick accuracy / wrong-bowl pick rate / distractor attraction rate** (referenced by
   Split 4a): these need per-step end-effector-to-object contact/proximity logging that
   `run_libero_eval.py` does not currently record (it only logs final success + video). Adding
   this is a moderate scope change to the rollout loop — worth doing once Split 4a's cell
   mapping is validated as useful, not preemptively.

## 10. Execution checklist

1. Pick a split id from `eval_registry.SPLITS` (or add one — see `CLAUDE.md`'s "how to add a
   benchmark split").
2. Run `python scripts/preflight.py` once per machine to confirm GPU/checkpoint/image are ready.
3. If the scene is new or its render hasn't been eyeballed yet, render its contact sheet and check
   it before spending GPU time (`LIBERO/scripts/render_suite_contact_sheet.py <suite>` — see
   `CLAUDE.md`'s "how to add a benchmark split").
4. `MACHINE_CONFIG=config/<machine>.env bash docker/openvla_libero/run_eval.sh --split <id>`
5. `python scripts/aggregate_results.py --filter <suite name>` to get the per-task/overall table.
6. Compare against the baseline/formulas in the relevant split's row above.
7. Record the result in `eval_results.md` (append — never overwrite existing experiments) **and**
   update `benchmark_split_result.md`'s status tables + findings. Every real run touches both.
