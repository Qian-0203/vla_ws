# OpenVLA · LIBERO-Spatial Benchmark Plan

Canonical, human-readable spec for the evaluation suite: what each split tests, why, and how it's
built. The executable source of truth for which (task suite, unnorm key, prompt condition) a split
actually runs is `openvla/experiments/robot/libero/eval_registry.py::SPLITS` — this document and
that registry must agree; if you change one, change the other. Operational details (environment
setup, exact commands, log/result locations) live in `CLAUDE.md` and are linked, not repeated, here.

**This is the plan and process doc.** It answers "what is each split, what's the hypothesis, what's
built." It does **not** track whether a condition has actually been run, what its detailed setting/
results/analysis were, or when it was launched — that's `benchmark_split_result.md` (detailed
per-condition setting, render compare, results, analysis — the authoritative source for any number)
and `eval_log.md` (append-only launch log: what ran, in what order, on what hardware). When a split's
definition changes, update this file. When an eval finishes, update the other two.

## 1. Research motivation

`libero_spatial`-fine-tuned OpenVLA (7B, LoRA r32) reaches 84.0% success on its own training
distribution (2 identical bowls, target-only prompts). The open question: **how much of that is
robust spatial/language grounding vs. overfitting to a narrow prompt-and-scene distribution?**
Each split below changes exactly one axis — prompt phrasing, distractor placement, scene clutter,
or the type of spatial cue used — while holding the model, checkpoint, and seed fixed, so drops
in success rate can be attributed to a specific capability gap rather than confounded together.

Since 2026-09-30 the same splits have also been run on other LIBERO-Spatial policies (ten checkpoints by
2026-10-13, §4), each read against its own `spatial/default`.

## 2. Splits overview

| Split | Probes | Registry status |
|---|---|---|
| 1. Prompt Sensitivity | Does naming/negating a distractor in the prompt help or hurt? | 3/3 conditions implemented, plus 2 length controls (authored 2026-10-01, run 2026-10-01) |
| 2. Distractor Placement | Does *where* an extra distractor sits matter more than its presence? | 3/4 conditions implemented (`path` not authored); `irrelevant` and `semantic` each redefined a second time (see Split 2 below) -- both current suites run 2026-08-27, 85.2% each |
| 3. Scene Complexity | Does added clutter (open drawer) degrade the policy, or just block the arm? | Implemented |
| 4. Surface vs. Landmark Grounding | Does the policy rely on landmark proximity vs. surface/region cues? | 4a: cells implemented (4/6 reuse existing data, 2/6 new scenes); 4b: implemented as a target-cue-type probe (`grounding/target_cue_region`, `grounding/target_cue_landmark`); 4c: `grounding/target_cue_proximity_novel`. All run. 4b/4c paraphrase controls (7 conditions) authored and run 2026-10-01 (`benchmark_split_result.md` §5.5). **Erratum E1 (2026-10-11): the three region-cue conditions (`target_cue_region`, `_v2`, `_v3`) used swapped-axis direction words and are withdrawn; corrected `*_img` conditions are authored and were run on the six public checkpoints only (`benchmark_split_result.md` §13)** |
| 5. Landmark Exchange | Does the policy deliver the bowl the sentence describes in the *current* scene, or the bowl at the coordinate the sentence was trained on? | Implemented and run 2026-10-12 as an **exploratory probe, not part of the official data** (`grounding/landmark_exchange`; `benchmark_split_result.md` §14) |
| 6. Instruction-file language conditions (`target_swap`, `no_location`; not `SPLITS` entries) | Does the policy deliver the bowl a sentence names when it names the *other* bowl, and can it do the task when the location is removed? | Defined in `screening/language_stress/candidates/` (Split 6 below); promoted from the 2026-10-10 screen and run at 50 trials/task on all ten checkpoints (`benchmark_split_result.md` §12, §11 addendum) |
| VLM Bowl-Pointing / Bowl-Attraction Probes (§11, not a `SPLITS` entry) | Is the distractor-mention collapse (Splits 1/2) a grounding failure or an action-decoding failure? | Standalone diagnostic scripts, run directly — not registered in `eval_registry.py`. Both probes run to completion on their respective batteries; open gaps listed in §11.3 |

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

**Length control (authored 2026-10-01; run 2026-10-01, results in `benchmark_split_result.md` §2).** Both contrast conditions make the prompt
longer *and* push it off the fine-tuning template, so their drop can't yet be pinned on either one
(§11.3 gap 2). The two length controls keep every task's native wording verbatim. Each adds a
content-free politeness clause (no object, location, or second referent) at the position the
contrast clause occupies, sized to the contrast clause's mean length in the checkpoint's own
Llama-2 tokens:

| Condition id | Stands in for | Added text (identical for all 10 tasks) | Extra tokens vs. native |
|---|---|---|---|
| `length_control_infix` | `negative_contrast` (+9..13, mean +11.1) | "..., if it is not too much trouble for you, and place it on the plate" | +11 |
| `length_control_suffix` | `positive_contrast` (+11..15, mean +13.1) | "... and place it on the plate; thank you so very much in advance for your help with this" | +13 |

```
Length-only Drop        = SR(default) - SR(length_control_*)
Content Drop (residual) = SR(length_control_infix)  - SR(negative_contrast)
                          SR(length_control_suffix) - SR(positive_contrast)
```

A length-only drop inside the ±3.3pt pooled noise band means the contrast conditions' ~50pt
damage comes from *what* the added clause says. A large length-only drop means prompt length or
generic off-template text is itself a sufficient cause. Registry: `spatial/length_control_infix`,
`spatial/length_control_suffix` (all 10 tasks, 500 rollouts each).

**Negation-only reference (authored 2026-10-10).** `negative_contrast` names the target *and* negates
the distractor, so a policy can ignore the negated clause and still read the target's own location.
`negation_only` removes that route: the target is identified only by negating the distractor's
location, and the target's own location is never stated.

| Condition | Wording (task 7) |
|---|---|
| `default` | "pick up the black bowl on the stove and place it on the plate" |
| `negative_contrast` | "pick up the black bowl on the stove, not the one on top of the wooden cabinet, and place it on the plate" |
| `negation_only` | "pick up the black bowl that is not on top of the wooden cabinet and place it on the plate" |

It is truthful only in the stock 2-bowl scene, where exactly one other black bowl exists. The
distractor phrases are the ones `negative_contrast` uses. Hypothesis: a policy that grounds negation
scores near `default`; one that keys on the location words it is given moves toward the distractor or
fails to commit. Metric: `Negation-only Drop = SR(default) − SR(negation_only)`. It was selected by the
language-stress screen (`benchmark_split_result.md` §12) as the one candidate that collapsed both the
official OpenVLA checkpoint and pi05_libero. Registry: `spatial/negation_only` (all 10 tasks, 500
rollouts).

### Split 2 — Distractor Placement Probe

| Field | Description |
|---|---|
| Goal | Whether an extra distractor's *position* — not just its presence — drives failures |
| Hypothesis | Distractors near the target's own landmark ("landmark") hurt more than neutral ("irrelevant") or other-landmark ("semantic") placement |
| Changed variable | Position of a 3rd `akita_black_bowl` distractor |
| Controlled variables | Target bowl position, 2nd (original) distractor, prompt = `default` (except the combo condition below), checkpoint, seed |
| Conditions | `irrelevant` (fixed at the table's front edge, off the reach path, farthest of the three) · `semantic` (named landmark different from target's) · `landmark` (near target's OWN landmark, farther away — "hard negative") · `path` (between target and plate) — **not yet authored, see §9** |
| Tasks | All 10 tasks, scene = corresponding `libero_spatial_3bowl*` suite |
| Trials | 50/task/condition, seed 7 |
| Metrics | `Distractor-type Drop = SR(spatial/default) - SR(condition)`, plus per-task deltas |
| Output | `EVAL-libero_spatial_3bowl*-openvla-*.txt` + matching `results/*.jsonl` |

Registry: `spatial_3bowl/irrelevant` (suite `libero_spatial_3bowl_front`, current definition — see
below), `spatial_3bowl/irrelevant_v1_legacy` (suite `libero_spatial_3bowl_neutral`, the first
redefinition, kept only so `benchmark_split_result.md`'s 88.8% number stays attributable),
`spatial_3bowl/center_fixed_legacy` (suite `libero_spatial_3bowl`, the retired single-fixed-
coordinate definition, kept only so `benchmark_split_result.md`'s original Exp 2 number stays attributable),
`spatial_3bowl/semantic` (suite `libero_spatial_3bowl_semantic2`, current definition — see below),
`spatial_3bowl/semantic_v1_legacy` (suite `libero_spatial_3bowl_semantic`, kept only so
`benchmark_split_result.md`'s 84.8% number stays attributable), `spatial_3bowl/landmark` (suite
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

**Axis orientation (confirmed against the renders 2026-10-11; see `benchmark_split_result.md`
Erratum E1).** In the image the policy sees, **+x points toward the camera (front of the table) and
+y points to the image right**; −x is toward the robot base (back) and −y is image-left. The
robot-facing frame therefore has its left on +y. Earlier text in this plan assumed "+x = right,
+y = back"; that was wrong and is corrected below. All left/right/front/back words in prompts should
be written in the image frame (left/right as seen; back = toward the robot base).

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
| `table_center` | (−0.10,−0.01)–(−0.05,0.01)\* |
| `table_front` | (0.19,−0.01)–(0.21,0.01)\* |
| `cabinet_region` | (0.02,−0.28)–(0.04,−0.26) |
| `stove_region` | (−0.42,−0.15)–(−0.40,−0.13) |

\* These two are the catalog *default*, still exact in `irrelevant_v1_legacy`,
`center_fixed_legacy`, and task 2's own `table_center` bowl_1 target everywhere. Inside
`libero_spatial_3bowl_front` specifically (current `irrelevant`), the 2026-08-27 offset fine-tune
moved both 0.05m further apart for bowl_3's placement only: `table_front` → (0.24,−0.01)–(0.26,0.01)
in the 6 non-fallback task files, `table_center` → (−0.15,−0.01)–(−0.10,0.01) in the 4 fallback task
files (1, 3, 5, 6) — see the redefinition notes below.

Per-task 3-bowl placement (bowl_1 = target, bowl_2 = original distractor, bowl_3 = 3rd bowl).
Object-relative placements (`cookies_1`, `wooden_cabinet_1_top_side`, `flat_stove_1_cook_region`,
`glazed_rim_porcelain_ramekin_1`, task 4's `In wooden_cabinet_1_top_region`) have no `main_table`
coordinate in the BDDL, so they're listed by object name, not fabricated numbers. bowl_1/bowl_2
are identical across all bowl_3 columns below (only bowl_3 differs). Current (`v2`) columns are
what `spatial_3bowl/irrelevant` and `spatial_3bowl/semantic` resolve to today; `v1`/legacy columns
are retired definitions kept only so their old result numbers stay attributable — see the
redefinition notes below the table:

| id | target | bowl_1 | bowl_2 | bowl_3 `irrelevant` (v2, current) | bowl_3 `irrelevant_v1_legacy` | bowl_3 `center_fixed_legacy` | bowl_3 `semantic` (v2, current) | bowl_3 `semantic_v1_legacy` | bowl_3 `landmark` (`hardneg_region`, per-task) |
|--:|---|---|---|---|---|---|---|---|---|
| 0 | between plate & ramekin | `between_plate_ramekin_region` | `next_to_ramekin_region` | `table_front` | `next_to_box_region` | `table_center` | `next_to_box_region` | `next_to_box_region` | (−0.070,0.040)–(−0.050,0.060) |
| 1 | next to ramekin | `next_to_ramekin_region` | `next_to_box_region` | `table_center`* | `table_center` | `table_center` | `next_to_plate_region` | `next_to_plate_region` | (−0.210,0.040)–(−0.190,0.060) |
| 2 | table center | `table_center` | `next_to_plate_region` | `table_front` | `next_to_ramekin_region` | `table_front`** | `next_to_ramekin_region` | `next_to_ramekin_region` | (−0.290,0.010)–(−0.270,0.030) |
| 3 | on cookie box | `cookies_1` (obj) | `wooden_cabinet_1_top_side` (obj) | `table_center`*** | `next_to_ramekin_region` | `table_center` | `next_to_ramekin_region` | `next_to_ramekin_region` | (0.180,−0.070)–(0.200,−0.050) |
| 4 | in top drawer | `In wooden_cabinet_1_top_region` (obj) | `wooden_cabinet_1_top_side` (obj) | `table_front` | `next_to_ramekin_region` | `table_center` | `between_plate_ramekin_region`**** | `next_to_plate_region` | (−0.060,−0.070)–(−0.040,−0.050) |
| 5 | on ramekin | `glazed_rim_porcelain_ramekin_1` (obj) | `cookies_1` (obj) | `table_center`*** | `table_center` | `table_center` | `next_to_plate_region` | `next_to_plate_region` | (−0.210,0.040)–(−0.190,0.060) |
| 6 | next to cookie box | `next_to_box_region` | `flat_stove_1_cook_region` (obj) | `table_center`* | `next_to_ramekin_region` | `table_center` | `next_to_ramekin_region` | `next_to_ramekin_region` | (−0.030,−0.120)–(−0.010,−0.100) |
| 7 | on stove | `flat_stove_1_cook_region` (obj) | `wooden_cabinet_1_top_side` (obj) | `table_front` | `next_to_box_region` | `table_center` | `next_to_box_region` | `next_to_box_region` | (−0.310,−0.010)–(−0.290,0.010) |
| 8 | next to plate | `next_to_plate_region` | `next_to_ramekin_region` | `table_front` | `table_front` | `table_center` | `next_to_box_region` | `next_to_box_region` | (0.190,0.080)–(0.210,0.100) |
| 9 | on wooden cabinet | `wooden_cabinet_1_top_side` (obj) | `flat_stove_1_cook_region` (obj) | `table_front` | `next_to_ramekin_region` | `table_center` | `next_to_ramekin_region` | `next_to_ramekin_region` | (−0.010,−0.060)–(0.010,−0.040) |

\* Tasks 1 and 6's `irrelevant` (v2) fall back to `table_center` because their own bowl_1/bowl_2
already sit within ~0.10m of `table_front` there (verified overlap, see below).
\*\* Task 2's `center_fixed_legacy` bowl_3 uses `table_front` instead of `table_center` only because
bowl_1 already occupies `table_center` there.
\*\*\* Tasks 3 and 5's `irrelevant` (v2) also fall back to `table_center` for the same reason (their
`cookies_1`/box-region objects sit close enough to `table_front` to overlap; caught by
`verify_suite_init_states.py`, worst sep 0.107m < the 0.12m threshold before the fallback).
\*\*\*\* Task 4's `semantic` (v2) could not use `next_to_box_region` like the other tasks that share
its landmark identity — the open drawer's 3D footprint (`z` up to 1.232) collides with it (min sep
0.063m). `between_plate_ramekin_region` (task 0's real target landmark) is the closest verified-safe
alternative.

**Design confound found, then fixed.** `center_fixed_legacy`'s bowl_3 sits at the *same fixed
absolute coordinate* in 9 of 10 tasks, regardless of the target's location — it is not "neutral"
relative to every task's reach path. This produced task 6's outlier result in `benchmark_split_result.md`
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
is kept in the registry only as a retired label so `benchmark_split_result.md`'s original Exp 2 number stays
attributable — it is **not** reused as the current `irrelevant` definition.

**Second redefinition (current).** Reusing `next_to_ramekin_region` for half the tasks left
`irrelevant` (v1) sitting at *another task's real target landmark* — not ideal for a condition
meant to carry no relational meaning at all. `irrelevant` was redefined again (new suite
`libero_spatial_3bowl_front`, registry condition unchanged, v1 kept as `irrelevant_v1_legacy`):
bowl_3 now sits at the single, literal front edge of the table (`table_front`) in every task,
falling back to `table_center` only where verification found a real overlap (tasks 1, 3, 5, 6 —
each has a bowl already within ~0.10-0.13m of `table_front`). Verified via
`verify_suite_init_states.py` — passes, worst separation 0.122m, matching v1's own worst case.

The same review flagged `semantic`'s task 4 (in the top drawer) as an outlier: its bowl_3 sat at
`next_to_plate_region`, ~0.58m from the drawer — outside the 0.33–0.50m band the other 9 tasks
land in, so it functioned more like a neutral placement than a genuine semantic distractor.
`semantic` was redefined (new suite `libero_spatial_3bowl_semantic2`, v1 kept as
`semantic_v1_legacy`): only task 4 changed, to `between_plate_ramekin_region` (~0.48m from the
drawer — task 0's real target landmark, and the closest verified-safe named landmark once
`next_to_box_region` turned out to physically collide with the open drawer, see the table
footnote above). All 9 other tasks are untouched.

**Offset fine-tune (2026-08-27).** `irrelevant`'s two anchor coordinates were pulled another 0.05m
apart — `table_front` moved to the actual front edge (x 0.19–0.21 → 0.24–0.26, the 6 non-fallback
tasks), `table_center`'s fallback moved further back (x −0.10––0.05 → −0.15––0.10, the 4 fallback
tasks 1/3/5/6 only — task 2's own `table_center` bowl_1 target is untouched). `stove_region` sits at
x −0.42––0.40, y −0.15––0.13; the fallback's new position stays ≈0.29m from it either way, well
clear. Re-verified via `verify_suite_init_states.py` — still PASS, worst separation 0.122m (task 4,
unrelated to bowl_3), every other task's separation improved (0.148–0.301m, up from 0.122–0.276m).
Contact sheet re-rendered and eyeballed: bowl_3 visibly farther front (front tasks) / farther back
(fallback tasks), still resting flat, no overlaps.

Both suites now have a real eval run (2026-08-27, 500/500 rollouts each, 85.2% overall for both) —
see `eval_log.md` and `benchmark_split_result.md` §3 for results and analysis. That launch also
surfaced and fixed a pre-existing unbounded-retry livelock bug in the LIBERO fork's
`env_wrapper.py::ControlEnv.reset()` (see `eval_log.md`'s 2026-08-27 entry) — plausibly the cause of
every earlier full-suite batch in this project undershooting its nominal 500 rollouts.

### Split 3 — Scene Complexity Probe

| Field | Description |
|---|---|
| Goal | Whether clutter/occlusion degrades the policy, independent of scene feasibility |
| Hypothesis | Clutter costs real success rate, but naive deltas overstate it by conflating policy failure with physically blocked trajectories |
| Changed variable | Wooden cabinet's top drawer: closed vs. open |
| Controlled variables | 3-bowl scene (`center_fixed_legacy` placement — see Split 2's redefinition note), prompt = `default`, checkpoint, seed |
| Conditions | `drawer_closed` (= Split 2's `center_fixed_legacy`, not the current `irrelevant`) · `drawer_open` |
| Tasks | All 10 tasks; **tasks 3, 6, 7 excluded from the adjusted metric** — rollout review showed the open drawer physically blocks their trajectories (see `benchmark_split_result.md` §4 for per-task detail and the methodology caveat on this exclusion) |
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
| Metrics | Final success, first-pick accuracy, wrong-bowl pick rate, distractor attraction rate (per-step contact logging now exists as a standalone probe, not integrated into this split's own eval loop — see §11.2) |
| Output | `libero_spatial` results filtered by `--task_ids` for the 4 existing cells; `EVAL-libero_spatial_grounding_*-openvla-*.txt` for the 2 new ones |

Registry: existing-scene cells run via `--split spatial/default --task_ids <ids from
GROUNDING_PROBE_CELLS>`; new cells are `grounding/surface_landmark`, `grounding/region_surface`.

**4b. Target Cue-Type Probe (redesigned).** The original spec was a 3x3 target-cue x
distractor-cue prompt matrix ("Pick the bowl next to the ramekin, not the one next to the cookie
box..."), requiring the SAME physical target/distractor pair to be described truthfully via 3 cue
types each — flagged as an open question (§9) because a genuine 3x3 needs either new scene
geometry (most cells can't be truthfully phrased 3 ways without moving fixtures) or a relaxed
truthfulness constraint, and because mentioning the distractor at all pushes into Split 1's
territory. **Resolved** by decoupling the two axes rather than forcing a joint matrix:

- **Distractor-mention axis is dropped from 4b entirely.** Split 1 already answered this
  question for this checkpoint (findings 1, 5 in `benchmark_split_result.md` §6): mentioning a
  second bowl's location at all, negated or not, costs 47-52 pts overall. Any 4b design that
  mentions the distractor would mostly re-measure that floor effect, not cue *type* — the two are
  confounded and 4b should isolate the one Split 1 hasn't tested: cue type, holding "distractor
  unmentioned" fixed (matching every other split's `default`-prompt convention).
- **Target-cue axis becomes the whole probe.** For each task, rephrase *only* the target
  description using an alternate cue type, scene and init states completely unchanged from
  `spatial/default` — this isolates "does describing the same bowl at the same place via a
  different cue type change success," independent of scene content (that's 4a's axis) and
  independent of distractor mention (that's Split 1's axis).

**Truthfulness tiers.** Not every (native family -> alternate cue type) rephrasing is available
without moving the bowl — each task's target sits in exactly one real place, and only some
alternate phrasings remain honest descriptions of that place:

| Native family | -> landmark cue | -> surface cue | -> region cue |
|---|---|---|---|
| landmark | exact (already `default`) | infeasible — bowl isn't resting on anything | approximate — table-zone phrasing is always constructible |
| surface | approximate — bowl is co-located with (resting on) the named object, so "next to X" is a defensible if loose reading | exact (already `default`) | approximate — same as above |
| region | infeasible — no nameable object sits near `table_center` | infeasible — nothing under the bowl | exact (already `default`) |

"Approximate" cells are used and their relaxation is disclosed in the results write-up, not
silently treated as equivalent to "exact" — same disclosure norm as Split 2's `irrelevant`
redefinition. "Infeasible" cells are skipped, not faked.

This yields exactly 2 new conditions, both reusing the stock `libero_spatial` scene/init
states/BDDL untouched (zero new geometry, zero new render-check burden — same scene `spatial/default`
was already checked against):

| Condition | Applies to (task ids) | New phrasing per task |
|---|---|---|
| `target_cue_region` | 0, 1, 3, 5, 6, 7, 8, 9 (landmark- and surface-family tasks; excludes task 2 — already native region — and task 4, containment) | table-zone description instead of the native landmark/surface phrasing |
| `target_cue_landmark` | 3, 5, 7, 9 (surface-family tasks only; landmark-family tasks already use this cue natively, region-family task 2 has no nearby nameable object) | "next to `<the object the bowl rests on>`" instead of "on `<object>`" |

> ⚠ **Erratum E1 (2026-10-11): the table below is the WRONG-AXIS wording that was run in
> 2026-09/10 and is withdrawn.** It was derived from the assumption "+x = robot-right, +y =
> table-back", which the renders show is swapped (+x = front, +y = image-right). 7 of its 8 tasks
> describe the wrong place (only task 3 is true). The corrected wording follows this table. Results
> from the old wording are not region-cue results; see `benchmark_split_result.md` Erratum E1.

Proposed instruction text (region-zone wording derived from each region's `(x,y)` sign in the
shared catalog above — positive x = robot-right, positive y = table-back, inferred from the
relative positions of `stove_region`/`cabinet_region`/`box_region` vs. `ramekin_region`/
`plate_region`; **not yet confirmed against a fresh render, sanity-check before running** — see
checklist item below):

| id | native target phrase | `target_cue_region` | `target_cue_landmark` |
|--:|---|---|---|
| 0 | between the plate and the ramekin | at the back of the table, just left of center | — (already landmark) |
| 1 | next to the ramekin | at the far back-left of the table | — (already landmark) |
| 3 | on the cookie box | near the center of the table | next to the cookie box |
| 5 | on the ramekin | at the back-left of the table | next to the ramekin |
| 6 | next to the cookie box | at the front-right of the table | — (already landmark) |
| 7 | on the stove | at the front-left of the table | next to the stove |
| 8 | next to the plate | at the far back of the table | — (already landmark) |
| 9 | on the wooden cabinet | at the front of the table, just right of center | next to the wooden cabinet |

**Corrected `target_cue_region_img` wording (image frame; authored 2026-10-11, not yet run).** Same
zones, same task subset (0, 1, 3, 5–9), direction words checked against the rendered init states
(`docs/figures/region_wording_check.png`, `scripts/check_region_wording.py`: 8/8 tasks). Only the
direction words change; task 3 is unchanged. `_v2_img` / `_v3_img` keep the v2 ("... area/part of
the table") and v3 (side-first) styles; all three live in `instructions.py`.

| id | native target phrase | where the bowl is (image) | `target_cue_region_img` (v1) |
|--:|---|---|---|
| 0 | between the plate and the ramekin | back half, right of center | at the back of the table, just right of center |
| 1 | next to the ramekin | far right, toward the back | at the far back-right of the table |
| 3 | on the cookie box | center | near the center of the table |
| 5 | on the ramekin | right side, toward the back | at the back-right of the table |
| 6 | next to the cookie box | front, slightly left | at the front of the table, just left of center |
| 7 | on the stove | back-left (next to robot base) | at the back-left of the table |
| 8 | next to the plate | far right, mid-depth | at the far right of the table, midway between front and back |
| 9 | on the wooden cabinet | far left edge (cabinet top) | at the far left of the table |

Task 9's bowl sits on top of the cabinet at the left edge of the table, so "of the table" is a loose
description and its depth is left unstated on purpose; task 0's bowl is mid-back rather than at the
very back.

**Metrics:**

```
Region-cue Drop (per task)     = SR(spatial/default, task) - SR(target_cue_region, task)
Landmark-cue Drop (per task)   = SR(spatial/default, task) - SR(target_cue_landmark, task)
Region-cue Drop, landmark-family pool = mean over tasks {0,1,6,8}
Region-cue Drop, surface-family pool  = mean over tasks {3,5,7,9}
```

The family-pooled comparison directly tests 4a's hypothesis from the language side: does forcing
a *landmark-located* bowl into region-style language cost more than forcing a *surface-located*
bowl into region-style language? If landmark-family's region-cue drop is meaningfully larger than
surface-family's, that's evidence the policy leans on landmark-style relational language
specifically, not just target-family-appropriate language in general.

Trials: 50/task/condition, seed 7, same protocol as every other split — reuses `spatial/default`'s
existing per-task numbers as the "native cue" baseline for both metrics (no re-run of that
condition needed, same pattern as 4a's 4 reused cells).

**Registry: implemented and run** (results: `benchmark_split_result.md` §5.2). The two instruction dicts
(`LIBERO_SPATIAL_TARGET_CUE_REGION_INSTRUCTIONS`, `LIBERO_SPATIAL_TARGET_CUE_LANDMARK_INSTRUCTIONS`)
and the two `eval_registry.SPLITS`/`CONDITIONS` entries (`grounding/target_cue_region`,
`grounding/target_cue_landmark`) exist on `openvla` `main` (originally developed on branch
`worktree-split4-target-cue-probe`) — both on suite `libero_spatial`, no new BDDL, no new
init states, no new contact-sheet check needed. Verified: `python3 -m py_compile` on both files,
plus a runtime check that `CONDITIONS['target_cue_region']`/`CONDITIONS['target_cue_landmark']`
resolve to dicts of exactly 8 and 4 task names respectively, matching the table above. Both
conditions **must** be run with `--task_ids` restricted to their covered subset (`0 1 3 5 6 7 8 9`
and `3 5 7 9`) — `run_libero_eval.py` asserts `task.name in instruction_map` and does not skip
missing tasks, so running the full 10-task suite would hard-fail on tasks 2/4 (region_cue) or
0/1/2/4/6/8 (landmark_cue). Still needed before this is "run" per this doc's own bar: (i) sanity-
check the region-zone directional wording against a render (see §9; done 2026-10-11, it failed: Erratum E1), (ii) merge/deploy the branch,
(iii) launch via `run_eval.sh --split grounding/target_cue_region --task_ids 0 1 3 5 6 7 8 9` and
the analogous command for `grounding/target_cue_landmark`.

**4c. Familiar vs. Novel Proximity-Cue Probe.** 4b's `target_cue_landmark` condition found a ~50pt
drop from rephrasing a surface-family target ("on the ramekin") as its landmark-cue analogue ("next
to the ramekin") — but that analogue happens to reuse "next to X," the *exact* phrasing template
tasks 0/1/6/8 already use natively (just with a different `X`). That leaves the drop's cause
ambiguous between two readings: (a) the policy is damaged by the relation-type change itself
(surface -> proximity semantics), or (b) it's damaged by the sentence not matching any template it
was fine-tuned on, and "next to X" only looks like a special case because it happens to coincide
with a template used elsewhere. 4c disentangles these by holding relation type fixed at "proximity"
and varying only whether the wording is a familiar (in-distribution) or novel (never-seen) template.

| Field | Description |
|---|---|
| Goal | Whether `target_cue_landmark`'s drop tracks the relation-type change (surface->proximity) or the specific lexical template ("next to X") matching training data |
| Hypothesis | If a novel proximity synonym drops *about as much* as the familiar one, the damage is relation-type-driven; if it drops *substantially more*, exact template familiarity — not relation type — is doing most of the work (the more likely outcome given 4b's broader "templated pattern matching" finding) |
| Changed variable | wording familiarity only (familiar "next to X" vs. novel "close to X"), relation type (proximity) and truthfulness (disclosed-approximate) held fixed |
| Controlled variables | scene, init states, target, distractor (never mentioned), checkpoint, seed — identical to `spatial/default` and `target_cue_landmark` |
| Conditions | 3-way comparison per task, reusing 2 of 3 legs from already-collected data |
| Tasks | same 4 surface-family tasks as 4b's landmark cue: 3, 5, 7, 9 |
| Trials | 50/task/condition, seed 7 (only `proximity_novel` needs a new run — 200 rollouts) |
| Metrics | see below |

| Label | Condition id | Example (task 5) | Cue type | Truthfulness | Familiarity | Status |
|---|---|---|---|---|---|---|
| surface_native | `default` (reused, `spatial/default`) | "on the ramekin" | surface | exact | native — this *is* the fine-tuning-time phrasing | ✅ already run (94% on task 5; 80.5% pooled over 3/5/7/9, see `benchmark_split_result.md` §5.2) |
| proximity_familiar | `target_cue_landmark` (reused) | "next to the ramekin" | landmark/proximity | approximate (disclosed) | familiar — "next to X" is tasks 0/1/6/8's native template, just with a different `X` | ✅ already run (10% on task 5; 30.5% pooled, see §5.2) |
| proximity_novel | `target_cue_proximity_novel` (**new**) | "close to the ramekin" | landmark/proximity | approximate (disclosed) | novel — "close to X" appears in none of the 10 native `libero_spatial` prompts | implemented, not yet run |

Full per-task phrasing (all 4 tasks, same truthfulness disclosure as 4b's landmark cue — the bowl is
co-located with, not literally beside, the named object):

| id | native (surface_native) | proximity_familiar | proximity_novel |
|--:|---|---|---|
| 3 | on the cookie box | next to the cookie box | close to the cookie box |
| 5 | on the ramekin | next to the ramekin | close to the ramekin |
| 7 | on the stove | next to the stove | close to the stove |
| 9 | on the wooden cabinet | next to the wooden cabinet | close to the wooden cabinet |

**Metrics:**

```
Novel-cue Drop     = SR(default, 4 tasks) - SR(target_cue_proximity_novel)
Familiar-cue Drop  = SR(default, 4 tasks) - SR(target_cue_landmark)          [already computed, 50.0 pts]
Familiarity Gap    = SR(target_cue_landmark) - SR(target_cue_proximity_novel)
```

A `Familiarity Gap` near 0 supports the relation-type reading; a `Familiarity Gap` clearly outside
the ~±7pt single-condition noise band (§5) supports the template-familiarity reading.

**Registry: implemented and run** (results: `benchmark_split_result.md` §5.3). `LIBERO_SPATIAL_TARGET_CUE_PROXIMITY_NOVEL_INSTRUCTIONS` in
`openvla/experiments/robot/libero/instructions.py` and the `target_cue_proximity_novel` entries in
`eval_registry.CONDITIONS`/`SPLITS` (split id `grounding/target_cue_proximity_novel`) are on
`openvla`'s `main` branch.
Verified: `python3 -m py_compile` on both files, plus a runtime check that
`CONDITIONS['target_cue_proximity_novel']` resolves to a dict of exactly 4 task names matching the
table above and `resolve_split('grounding/target_cue_proximity_novel')` returns
`('libero_spatial', 'libero_spatial', 'target_cue_proximity_novel')`. Must be run with
`--task_ids 3 5 7 9` (same constraint as `target_cue_landmark` — `run_libero_eval.py` asserts
`task.name in instruction_map` and does not skip missing tasks). Launch command:
```
... run_eval.sh --split grounding/target_cue_proximity_novel --task_ids 3 5 7 9
```

**4b/4c paraphrase controls (authored 2026-10-01; run 2026-10-01, results in `benchmark_split_result.md` §5.5).** Every 4b/4c number so far rests
on one hand-written phrasing per task, and 4b only measured *cross*-cue rewordings. These
conditions add (i) the matrix's missing diagonal, meaning the target described in its **own**
native cue type, exactly truthfully, in different words, and (ii) extra wordings for the region
cue and for 4c's novel proximity cue. Distractor never mentioned; scene and init states identical
to `spatial/default`.

| Condition id | What it tests | Example (task 5 unless noted) | Tasks | Rollouts |
|---|---|---|---|--:|
| `paraphrase_lexical` | same cue, one synonym swap / argument reorder | "on top of the ramekin"; task 1: "beside the ramekin"; task 0: "between the ramekin and the plate" | 0–9 | 500 |
| `paraphrase_syntactic` | same cue, native words, relative clause | "that is on the ramekin" | 0–9 | 500 |
| `target_cue_region_v2` (withdrawn, E1; corrected: `_v2_img`) | region cue, 2nd wording (same zones as `target_cue_region`) | "in the back-left area of the table" | 0,1,3,5–9 | 400 |
| `target_cue_region_v3` (withdrawn, E1; corrected: `_v3_img`) | region cue, 3rd wording (side-first) | "on the left side of the table, toward the back" | 0,1,3,5–9 | 400 |
| `target_cue_proximity_beside` | novel proximity synonym | "beside the ramekin" | 3,5,7,9 | 200 |
| `target_cue_proximity_near` | novel proximity synonym | "near the ramekin" | 3,5,7,9 | 200 |
| `target_cue_proximity_adjacent` | novel proximity synonym | "adjacent to the ramekin" | 3,5,7,9 | 200 |

`paraphrase_lexical`'s per-task edit: task 0 argument reorder; task 2 "table center" -> "the
middle of the table"; task 4 "in" -> "inside"; tasks 1/6/8 "next to" -> "beside"; tasks 3/5/7/9
"on" -> "on top of". The last group is also how `negative_contrast` rewords tasks 3/5/9's target,
so this condition also bounds how much of that drop comes from the target rewording rather than
the added clause. `paraphrase_syntactic`'s only extra edit is task 2's "from table center" -> "that
is at table center". The region variants reuse `target_cue_region`'s zone per task, so they inherit
its truthfulness check (**E1: and its error: those zones were mislabeled, so the old v2/v3 are not truthful either**). All four proximity synonyms ("close to", "beside", "near", "adjacent to")
appear in none of the 10 native prompts.

```
Same-cue Paraphrase Drop = SR(default) - SR(paraphrase_{lexical,syntactic})
Cue-type Drop (excess)   = SR(paraphrase_lexical) - SR(target_cue_region*)       [same tasks; E1: only meaningful with the `_img` wording]
Familiarity Gap (robust) = mean over 4 novel synonyms of SR(novel) - SR(target_cue_landmark)
```

If the same-cue paraphrases stay near `default`, 4b's collapse is about the cue type. If they drop
nearly as far, the checkpoint fails on any surface-form change, which is the stronger
template-matching claim. The spread across the 3 region and 4 novel-proximity wordings is the
per-phrasing variance that 4b/4c's single-wording numbers could not show. Registry:
`grounding/paraphrase_{lexical,syntactic}`, `grounding/target_cue_region_v{2,3}`,
`grounding/target_cue_proximity_{beside,near,adjacent}`. The subset conditions must be run with
the `--task_ids` listed above.

**Deprioritized stretch option — the original 3x3 target x distractor mention matrix.** Kept as an
explicitly optional follow-on, not part of 4b's core design: a full truthful 3x3 would need a
purpose-built pilot scene (e.g. relocating `cookies_1` adjacent to `ramekin_region` so a single
bowl sitting on the cookie box is simultaneously "on the cookie box" (surface), "next to the
ramekin" (landmark), and "at the back-right of the table" (region; the 2026-10 text said back-left, which was the E1 error) — all exactly true at once,
for both target and distractor). That's new-geometry work at the same scope as Split 2's
unauthored `path` distractor (§9) — a new per-task numeric region, not reusable from the shared
catalog, needing `gen_suite_init_states.py` + `verify_suite_init_states.py` + a contact-sheet
eyeball before trusting it. Given the expected floor effect from distractor mention (see above),
this is unlikely to isolate cue-type effects cleanly even if built, so it's not recommended as the
next use of engineering time unless the target-cue-only results above turn out to need it as a
follow-up.

### Split 5 — Landmark Exchange Probe (authored and pre-registered 2026-10-12)

> **Status: exploratory (author's decision 2026-10-12).** Run once; excluded from the official data and
> from the paper's evidence base until promoted (conditions at the end of `benchmark_split_result.md` §14).

| Field | Description |
|---|---|
| Goal | Separate "the sentence selects the bowl it describes in the current scene" from "the sentence selects a trained coordinate" and from "the scene alone selects the action". Every other split leaves the described bowl at the coordinate its sentence was trained on, so none of them can tell these apart (`benchmark_split_result.md` §12, batch E: all ten `target_swap` sentences name a bowl that sits at another task's trained target coordinate) |
| Changed variable | Scene only: the cookie box and the ramekin exchange places in every task (cookie box → `ramekin_region`, ramekin → `box_region`). Sentences are each task's original `libero_spatial` sentence, unchanged |
| Controlled variables | Prompt = native sentence, checkpoint, seed 7, fixed 50 init states per task, plate/cabinet/stove untouched |
| Tasks | 4 tasks in suite `libero_spatial_landmark_exchange`: id 0 "next to the ramekin" (libero_spatial id 1), id 1 "on the cookie box" (3), id 2 "on the ramekin" (5), id 3 "next to the cookie box" (6). Two layouts: `S_next` (tasks 0 and 3: both bowls on the table, one beside each exchanged landmark) and `S_on` (tasks 1 and 2: each bowl rests on a moved landmark). Tasks 0/3 and 1/2 share one layout and differ only in the sentence |
| Construction | `bowl_1` is always the bowl the sentence now describes (the goal stays `On(bowl_1, plate)`); `bowl_2` is at, or on the landmark at, the coordinate the sentence was trained on. The two exchanged landmarks have the same size class and no yaw region, so no new geometry is introduced; every coordinate was already used by another task |
| Trials | 50/task, seed 7 (200 rollouts per model); OpenVLA official, OpenVLA ours (r32) and pi05_libero |
| Metrics | Per rollout, from the outcome fields (`aggregate_results.py --outcomes`): described bowl delivered / trained-coordinate bowl delivered / trained-coordinate bowl lifted only / described bowl lifted only / no grasp (split by first bowl approached) |

Registry: `grounding/landmark_exchange` (suite `libero_spatial_landmark_exchange`, `unnorm_key=libero_spatial`,
condition `default`; no new instruction dict, the sentence is the BDDL file name).

**Pre-registered readings (written before any rollout of this split was read).** "Wrong bowl" below is
`bowl_2`, the bowl at the sentence's trained coordinate. Thresholds are descriptive cut-offs, not tests.

| Reading | What a model that does this shows |
|---|---|
| Grounds the sentence in the scene | Described bowl delivered in at least 70% of rollouts on every task; wrong bowl delivered rarely |
| Sentence → trained coordinate | Described bowl delivered in about 0% on every task; wrong bowl approached first / lifted / delivered in most rollouts, on both layouts |
| Scene decides, sentence ignored | Within each shared-layout pair (0/3, 1/2) the same physical bowl is delivered under both sentences, so the two success rates sum to about 100%, and the wrong bowl is delivered in the complementary task |
| Does not act | Most failures approach neither bowl (the §8.5 / batch E failure for OpenVLA) |

No directional prediction is made for any model; the probe exists to classify them. Known limits: two
layouts only, four tasks, one seed, one exchange (ramekin ↔ cookie box). The stove ↔ cabinet exchange
(tasks 7/9) was not built: the cabinet region carries a yaw rotation the stove region does not, so an
exchange needs a new region definition and a fresh render check.

### Split 6 — Instruction-file language conditions (`target_swap`, `no_location`)

Promoted from the language-stress screen (`benchmark_split_result.md` §12) and run at full size. They stay
outside `SPLITS`: `target_swap` needs `swap_target` (the scored bowl changes), which only an instruction file
can set, and `run_libero_eval.py` refuses a registry condition with a screening condition's name. Both run as
`run_libero_eval.py --task_suite_name libero_spatial --unnorm_key libero_spatial --instruction_file
screening/language_stress/candidates/<name>.json` (or `--instruction-file` in the queue scripts).

| Field | `target_swap` | `no_location` |
|---|---|---|
| Change | Sentence names the *other* bowl in the stock 2-bowl scene, by its location (task 7: "pick up the black bowl on top of the wooden cabinet and place it on the plate") | Location phrase removed: "pick up the black bowl and place it on the plate", same for all 10 tasks |
| Scored bowl | The bowl the sentence names (`swap_target: true`; the goal predicate is retargeted to the other bowl) | The original target (with the location removed, both bowls fit the sentence, so the condition has no unique correct answer) |
| Hypothesis (from the JSON) | A policy that executes the layout's trained motion instead of reading the sentence scores near 0 | A high score means the layout alone identifies the task, i.e. the policy need not read the location |
| Controlled | Scene, init states, checkpoint, seed 7, 50 trials/task, all 10 tasks | same |
| File sha256 | `3cccd982…aa26` | `5df13c4f…9296` |
| Metrics | SR; per-rollout outcome classes (below): named bowl delivered / trained bowl delivered / lifted only / no bowl lifted (split by first bowl approached) | SR; same classes, with "target" = original target |

**Known limit.** Every `target_swap` sentence names a bowl that sits at another task's trained target coordinate
(`benchmark_split_result.md` §12, batch E), so success cannot separate "reads the sentence and finds the described
bowl" from "maps the sentence to a trained coordinate". Split 5 was built to separate them and is exploratory.

## 4. Baselines

The single reference number everything compares against: `spatial/default` — 2 bowls, default
prompt, `libero_spatial` scene, 84.0% (420/500), see `benchmark_split_result.md` §2.

**Other checkpoints (added 2026-10-13).** The benchmark has been run on ten checkpoints: our LoRA r32, the
official `openvla-7b-finetuned-libero-spatial`, QwenVLA, `pi05_libero`, `pi0_libero`, OpenVLA-OFT, VLA-Adapter
Spatial and Spatial-Pro, X-VLA and GR00T N1.5 (`benchmark_split_result.md` summary, §9–§13). Each checkpoint's
baseline is its own `spatial/default`; a drop is that checkpoint's `default` minus the condition, computed on the
same tasks (tasks 3, 5, 7, 9 for the cue-rephrasing conditions, 8 tasks for the region cues). Columns of different
checkpoints are not compared directly, because their action protocols differ (§8). Split 3's
adjusted comparison uses a recomputed 7-task baseline (84.9%, 297/350) instead — see Split 3's row
above.

## 5. Rollout & seed protocol

- 50 trials/task, seed 7, fixed across every split (matches the original baseline run).
- Init states are pre-sampled per task (`.pruned_init` files, torch tensors of shape
  `(50, state_dim)`); every condition on the same scene replays the identical 50 init states, so
  differences are attributable to the changed variable, not sampling noise.
- Success = LIBERO's own goal predicate for that task (`env.step` returns `done=True`).
- At n=50/task, 1 standard error is roughly ±5-7 points — treat single-task deltas below ~10
  points as noise (stated explicitly in `benchmark_split_result.md` §0, carried forward here).

## 6. Aggregation rule

Suite-wide success rate = **mean of the 10 (or n) per-task success rates**, not a flat count over
all rollouts (only differs from a flat average when a run is incomplete or task-filtered).
`scripts/aggregate_results.py` implements this once, canonically, from the JSONL results files.

**Outcome classes (which-bowl logging, every model family, added 2026-10-11; older runs are success-only).** Each rollout records which
bowls are on the plate at the end, the first bowl the end effector comes within 8 cm of, and the first bowl
lifted (more than 3 cm above its settled height while touching the gripper). `aggregate_results.py --outcomes`
turns these into: target delivered / other bowl delivered / other bowl lifted, not delivered / target lifted only /
no bowl lifted. "Target" is the bowl the run scores (the named bowl under `target_swap`). `success` is unchanged by
the logging (per-episode identical on the runs checked, `benchmark_split_result.md` §2).

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
- **Different policies, different protocols.** The ten checkpoints differ in action chunking (1 to 30 actions per
  query), cameras and proprioception (OFT uses two cameras + proprio), action space (X-VLA predicts absolute
  end-effector poses), attention implementation (GR00T N1.5 with SDPA instead of flash-attention-2) and training
  data (`pi0_libero` and X-VLA-Libero are trained on all four LIBERO suites; the others on LIBERO-Spatial).
  QwenVLA ran on another machine with `env_recreate_every=15`. Each family keeps the protocol of its own full run;
  `benchmark_split_result.md` §13 lists them.
- **One seed, one run per condition.** All numbers are seed 7, 50 trials/task. Where a condition was repeated (our
  r32's 11 re-run conditions, pi05's which-bowl re-run), the two runs differ by 0–3.2 pts
  (`docs/data_status_2026-10-13.md` §3).
- **`drawer_open` mixes clutter with physical blocking.** The open drawer blocks the arm on some tasks for some
  policies (`benchmark_split_result.md` §4, §11), and the third bowl uses the retired `center_fixed_legacy` placement.
- **`center_fixed_legacy` is a single fixed coordinate** (`table_center`/`table_front`) reused
  across all 10 tasks, not a per-task-neutral position — see Split 2's confound note above. It's
  kept only as a retired label for `benchmark_split_result.md`'s original Exp 2 number, never
  reused as the current `irrelevant` definition.

## 9. Open design questions (do not implement without a decision)

1. **Split 2 `path` distractor**: needs a new per-task numeric region (not a reused named
   region), since "between target and plate" is geometrically different for every task. Proposed
   approach: midpoint between the target's region center and `plate_region`, verified via
   `LIBERO/scripts/gen_suite_init_states.py` + `verify_suite_init_states.py`'s overlap check
   before trusting it. Deferred rather than rushed without enough render/verify iterations.
2. **Split 4b cue-phrasing matrix — resolved, implemented, and run** (`benchmark_split_result.md` §5.2). Decision: drop the
   distractor-mention axis (confounded with Split 1's already-established mention penalty — see
   Split 4's 4b section above), keep only a target-cue-type axis (landmark/surface/region
   rephrasing of the same unmentioned target), and use relaxed/disclosed truthfulness (option (b))
   rather than new scene geometry for that axis. Full design — conditions, per-task prompts,
   metrics — is written up in Split 4's 4b section above; the two instruction dicts and two
   `SPLITS`/`CONDITIONS` entries are implemented on `openvla` `main`
   (merged from `worktree-split4-target-cue-probe`). The pre-run checklist, kept for the record: (i)
   sanity-check the inferred region-zone directional wording (back-left, front-right, etc.) — **done 2026-10-11 after the runs: the wording failed (Erratum E1)**;
   against `libero_spatial`'s existing contact sheet (`LIBERO/scratch_render/libero_spatial/`) —
   those phrasings were derived from BDDL coordinate signs, not confirmed visually; (ii) merge the
   branch; (iii) launch both conditions with `--task_ids` restricted per their `SPLITS`
   description (required — `run_libero_eval.py` hard-fails on a task id missing from the
   condition's instruction dict). The original full 3x3 (with distractor mention) is kept only as
   an explicitly deprioritized stretch option, not a commitment.
3. **First-pick accuracy / wrong-bowl pick rate / distractor attraction rate** (referenced by
   Split 4a) — **partially resolved, see §11.2.** `run_libero_eval.py` itself still doesn't record
   per-step end-effector-to-object proximity (only final success + video), so this remains
   unimplemented for Split 4a's full 6-cell design. Instead of extending the main eval loop, a
   standalone diagnostic script (`probe_bowl_attraction.py`, §11.2) added exactly this
   instrumentation for a different, narrower purpose (the distractor-mention collapse probe), and
   reuses it on 4 of `libero_spatial`'s 10 tasks. Whether to generalize that script (or port its
   instrumentation into the main loop) for Split 4a's own 6 cells is still open — not done
   preemptively, same reasoning as before.

## 10. Execution checklist

1. Pick a split id from `eval_registry.SPLITS` (or add one — see `CLAUDE.md`'s "how to add a
   benchmark split").
2. Run `python scripts/preflight.py` once per machine to confirm GPU/checkpoint/image are ready.
3. If the scene is new or its render hasn't been eyeballed yet, render its contact sheet and check
   it before spending GPU time (`LIBERO/scripts/render_suite_contact_sheet.py <suite>` — see
   `CLAUDE.md`'s "how to add a benchmark split").
4. `MACHINE_CONFIG=config/<machine>.env bash docker/openvla_libero/run_eval.sh --split <id>` (machine
   profiles and server setup: `eval_log.md` "Servers & environments reference")
5. `python scripts/aggregate_results.py --filter <suite name>` to get the per-task/overall table.
6. Compare against the baseline/formulas in the relevant split's row above.
7. Append an entry to `eval_log.md` (launch batch, hardware, order, results-file paths) **and**
   update `benchmark_split_result.md` (the condition's detail section, status tables, findings).
   Every real run touches both — never overwrite existing sections in either.

## 11. VLM Bowl-Pointing / Bowl-Attraction Probes — grounding-vs-action-decoding diagnostics

Neither probe below is a `SPLITS` registry entry — both are standalone scripts under
`openvla/experiments/robot/libero/`, run directly (not via `run_eval.sh --split`), because the
question they answer (does the distractor-mention success-rate collapse in Splits 1/2 reflect
broken grounding, or broken action decoding downstream of intact grounding?) can't be answered from
task-success numbers alone, however they're sliced. Full numbers, the mid-investigation
marker-placement bug, and per-task detail all live in `benchmark_split_result.md` §8 — this section
only records the design: goal, hypothesis, method, and what's still open.

### 11.1 VQA bowl-pointing probe

| Field | Description |
|---|---|
| Goal | Determine whether OpenVLA's task-success collapse under distractor-mention prompts (Split 1/2 findings) is a broken vision-language grounding signal, or an intact grounding signal that only the action-decoding head fails to express |
| Hypothesis | If the checkpoint can be made to answer "which bowl matches this instruction" in free text instead of actions, grounding and action-decoding become separately measurable |
| Method | Render each task's exact episode-0 init state, overlay a shuffled ID number on each bowl (Set-of-Mark), project 3D bowl position to pixels, call `.generate()` directly (bypassing action-token decoding) with the real instruction reformatted as a pointing question. Ground truth = the task's BDDL goal target, invariant across scene variants used here |
| Result on OpenVLA | **Dead end**, confirmed 3 independent ways (fine-tuned checkpoint free-text, base checkpoint free-text, restricted-logit comparison): this model family has no text-output channel causally responsive to language at the queried point — it unconditionally emits action-bin tokens regardless of the question. Structural to OpenVLA's action-only continued-pretraining recipe, not a symptom of this project's fine-tune or distractor-mention conditions specifically |
| Pivot | Same render/annotate/score pipeline repointed at `Qwen/Qwen2-VL-7B-Instruct` — a general-purpose VLM never trained on OpenVLA's action-only template — to test whether the referring expression is resolvable *in principle*, independent of which model is asked |
| Conditions (Qwen only) | `default` / `hardneg_default` (native LIBERO language, distractor never mentioned) · `negative_contrast` · `positive_contrast` · `hardneg` (distractor named, negated or not) |
| Tasks | All 10 `libero_spatial` tasks (2-bowl conditions); all 10 `libero_spatial_3bowl_hardneg` tasks (3-bowl conditions) |
| Trials | 1 query/task/condition — a VQA probe, not a rollout, so no seed/trial replication |
| Metrics | Parsed-answer accuracy vs. chance (50% for 2-bowl, 33% for 3-bowl) |
| Scripts | `probe_bowl_pointing.py` (OpenVLA, dead end), `probe_bowl_pointing_qwen.py` (Qwen, pursued), `bowl_pointing_common.py` (shared render/annotate/score, holds the marker-placement fix) — none registered in `eval_registry.py` |
| Status | Full 5-condition x 10-task battery (50 queries) run to completion on Qwen2-VL-7B-Instruct; numbers, the marker-placement bug found and fixed mid-run, and the per-task render table are in `benchmark_split_result.md` §8.1-§8.4. Extended 2026-09-04 to sampled (not greedy-only) decoding — 10 samples/query at temperature 0.7, reporting per-query sample accuracy and majority vote instead of one `do_sample=False` draw — see §8.7. Same day, re-run on Qwen3-VL-8B-Instruct to test whether a newer/stronger VLM raises the accuracy ceiling: not a uniform upgrade — large 2-bowl-scene gains, a regression on the 3-bowl `hardneg` scene — see §8.8 |

### 11.2 Bowl-attraction probe

| Field | Description |
|---|---|
| Goal | §11.1 established OpenVLA's only language-responsive output is action tokens themselves — so read that channel directly instead of inferring mechanism from success-rate deltas: does the arm move toward the wrong/distractor bowl (misdirection), or fail to commit to any bowl at all (template-mismatch collapse)? |
| Hypothesis | Prompt deviation from the fine-tuning template collapses the action decoder's ability to commit to *any* target, with distractor-directed misdirection present but secondary — the same template-mismatch mechanism Split 4b's success-rate pattern already pointed to, tested here from the action-behavior side instead |
| Method | Run the real `get_action()`/environment-step loop from `run_libero_eval.py` (same model, image preprocessing, seed/init-state protocol) with one addition: at every post-settle step, read each `akita_black_bowl_*` body's 3D position and its distance to the end-effector. Per episode, record the running-min distance to each bowl and `first_bowl_approached` (first step, if any, the gripper comes within 8cm of a given bowl — target, distractor, or `None`) |
| Controlled variables | Scene = `libero_spatial` (2-bowl); checkpoint; seed-7 init states identical to the corresponding real 50-trial eval (episode `i` here = init state `i` there) |
| Conditions | `default` · `negative_contrast` (Split 1) · `target_cue_landmark` (Split 4b) — reuses existing condition/instruction definitions, no new prompt text |
| Tasks | `default`/`negative_contrast`: all 10 `libero_spatial` tasks (extended from task 5 alone, to Split 4b's 4-task surface cohort {3,5,7,9}, to the full suite, closing the sample-size gap in three steps). `target_cue_landmark`: still only the 4-task surface cohort {3,5,7,9} — it can't be authored for tasks whose native phrasing is already "next to X", or for tasks 2/4 which have no landmark-family analog |
| Trials | 10 episodes/task/condition — 100 rollouts/condition pooled for `default`/`negative_contrast` (10 tasks), 40 for `target_cue_landmark` (4 tasks) |
| Metrics | Episode success; among failed episodes only: target-approached-first vs. distractor-approached-first vs. neither-approached, both per-task and pooled |
| Scripts | `probe_bowl_attraction.py` — not registered in `eval_registry.py` |
| Status | `default`/`negative_contrast` run to completion on all 10 tasks (2026-09-03); `target_cue_landmark` still 4-task only. Pooled failure-mode breakdown, the task-3 exception (arm reaches the target correctly and still fails to complete the pick-and-place), the full-suite pooled numbers (closely reproduce the real 500-trial eval: 81%/38% here vs. 84.0%/36.8% real), and the synthesis against §11.1's Qwen result are in `benchmark_split_result.md` §8.5-§8.6 |

### 11.3 Open gaps

Carried forward from `benchmark_split_result.md` §8.6's revised synthesis:

1. ~~**Sample size.**~~ **Closed (2026-09-03).** §11.2's `default`/`negative_contrast` cells were
   extended in three steps (1 task -> 4 -> all 10 `libero_spatial` tasks, 200 rollouts pooled across
   the two conditions) specifically to close this gap — see §11.2's Status row. The picture sharpened
   rather than changed direction: "arm never approaches either bowl" is the outright majority failure
   mode at full scale (58.1% of `negative_contrast` failures), and task 3's distinct "approached
   correctly, still failed" mode persists (27.4% of failures) rather than washing out as a one-task
   artifact.
2. **No length/complexity control.** `negative_contrast` adds a whole extra clause ("...not the one
   on top of...") versus `target_cue_landmark`'s single-word swap ("on" -> "next to"). Nothing run
   so far separates "deviates from the fine-tuning template" from "prompt is simply longer" — both
   predict the same symptoms observed in §11.2. Would need a same-length, template-adjacent
   paraphrase condition to isolate. Unaddressed by the sample-size extension.
   **Controls authored 2026-10-01 (all run 2026-10-01; see `benchmark_split_result.md` §2 and §5.5):** Split 1's `length_control_infix`/`_suffix`
   (token-matched, content-free clause on native wording) and Split 4's same-cue paraphrases
   (`paraphrase_lexical`/`_syntactic`, +0..3 tokens). Together they separate length from
   template deviation at the success-rate level.
3. **No mechanistic localization.** §11.2 shows *what the arm does* (approach behavior), not
   *where in the network* it goes wrong — vision encoder, language projector, or action-token head
   could each independently produce "never commits to a target," and task 3's approach-then-fail
   pattern is a separate, untested localization question (likely downstream, in grasp/lift/place
   control, but not confirmed). **Pursued as its own probe (2026-09-04 → 2026-09-10): narrowed, not
   closed.** Design, runs, and current conclusion live in `mechanistic_localization.md`.
4. **Narrow condition sweep (task sweep now closed for 2 of 3 conditions).** `default`/
   `negative_contrast` now cover all 10 tasks, but `target_cue_landmark` is still 4-task only (see
   §11.2's Tasks row for why it can't be trivially extended), and `positive_contrast`,
   `target_cue_region`, `hardneg`, and every Split 2/3 scene variant remain entirely untested by
   this probe.
