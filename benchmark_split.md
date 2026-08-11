# OpenVLA · LIBERO-Spatial Benchmark Specification

Canonical, human-readable spec for the evaluation suite. The executable source of truth for
which (task suite, unnorm key, prompt condition) a split actually runs is
`openvla/experiments/robot/libero/eval_registry.py::SPLITS` — this document and that registry
must agree; if you change one, change the other. Operational details (environment setup, exact
commands, log/result locations) live in `CLAUDE.md` and are linked, not repeated, here.

**This file is the plan; `eval_results.md` is the previous record.** Every condition below is something we
intend to (or already did) run. Each split's **Progress** subsection states, per condition,
whether it's only planned/authored or whether it has actually been executed — and if executed,
the exact success-rate numbers, sourced from `eval_results.md`. If a number appears here, it was
copied from there, not recomputed; `eval_results.md` remains the append-only, authoritative
per-experiment writeup. Nothing in this file should be read as "already run" unless its Progress
table says so explicitly.

## 1. Research motivation

`libero_spatial`-fine-tuned OpenVLA (7B, LoRA r32) reaches 84.0% success on its own training
distribution (2 identical bowls, target-only prompts). The open question: **how much of that is
robust spatial/language grounding vs. overfitting to a narrow prompt-and-scene distribution?**
Each split below changes exactly one axis — prompt phrasing, distractor placement, scene clutter,
or the type of spatial cue used — while holding the model, checkpoint, and seed fixed, so drops
in success rate can be attributed to a specific capability gap rather than confounded together.

## 2. Splits overview

| Split | Probes | Registry status | Data status |
|---|---|---|---|
| 1. Prompt Sensitivity | Does naming/negating a distractor in the prompt help or hurt? | 3/3 conditions implemented | 2/3 run (`default`, `negative_contrast`) |
| 2. Distractor Placement | Does *where* an extra distractor sits matter more than its presence? | 3/4 conditions implemented (`path` not authored) | 0/4 confirmed run (`irrelevant`'s data exists but its condition label is disputed — see Split 2's Progress table) |
| 3. Scene Complexity | Does added clutter (open drawer) degrade the policy, or just block the arm? | Implemented | Run (both conditions) |
| 4. Surface vs. Landmark Grounding | Does the policy rely on landmark proximity vs. surface/region cues? | 4a: cells implemented (4/6 reuse existing data, 2/6 new scenes); 4b: not built | 4/6 cells derivable from existing Split-1 data; 2/6 not run |

"Implemented" = task suite + prompts exist in the registry and can be run with one `run_eval.sh`
command. "Run" = rollouts actually executed and success-rate numbers exist in `eval_results.md`.
Implemented ≠ run — see each split's **Progress** subsection below and §10 for what's outstanding.

## 3. Splits in detail

### Split 1 — Prompt Sensitivity Probe

| Field | Description |
|---|---|
| Goal | Whether describing/negating a distractor in language helps or hurts disambiguation |
| Hypothesis | Naming a distractor without practice at contrastive language *confuses* rather than helps (confirmed, see §8) |
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

**Progress**

| Condition | Status | Overall SR | Rollouts | Source |
|---|---|--:|--:|---|
| `default` | ✅ run | 84.0% | 420/500 | `eval_results.md` Exp 1 |
| `negative_contrast` | ✅ run | 36.8% | 184/500 | `eval_results.md` Exp 1 |
| `positive_contrast` | ⬜ planned, not run | — | — | prompts exist in `instructions.py::LIBERO_SPATIAL_POSITIVE_CONTRAST_INSTRUCTIONS`, never executed |

Computed from the above: `Negative Contrast Drop = 84.0 - 36.8 = 47.2 pts`. `Distractor Mention
Drop` and `Negation-specific Drop` are blocked on `positive_contrast` — needs a GPU run (see
CLAUDE.md's one-GPU workflow; deferred to a server session, this laptop is scoped to content
authoring only).

### Split 2 — Distractor Placement Probe

| Field | Description |
|---|---|
| Goal | Whether an extra distractor's *position* — not just its presence — drives failures |
| Hypothesis | Distractors near the target's own landmark ("landmark") hurt more than neutral ("irrelevant") placement |
| Changed variable | Position of a 3rd `akita_black_bowl` distractor |
| Controlled variables | Target bowl position, 2nd (original) distractor, prompt = `default`, checkpoint, seed |
| Conditions | `irrelevant` (neutral table center/front) · `semantic` (named landmark different from target's) · `landmark` (near target's OWN landmark, farther away — "hard negative") · `path` (between target and plate) — **not yet authored, see §10** |
| Tasks | All 10 tasks, scene = corresponding `libero_spatial_3bowl*` suite |
| Trials | 50/task/condition, seed 7 |
| Metrics | `Distractor-type Drop = SR(spatial/default) - SR(condition)`, plus per-task deltas |
| Output | `EVAL-libero_spatial_3bowl*-openvla-*.txt` + matching `results/*.jsonl` |

Registry: `spatial_3bowl/irrelevant` (suite `libero_spatial_3bowl`), `spatial_3bowl/semantic`
(suite `libero_spatial_3bowl_semantic`, new), `spatial_3bowl/landmark` (suite
`libero_spatial_3bowl_hardneg`). All three reuse `unnorm_key=libero_spatial` since the checkpoint
was trained on the 2-bowl scene. `spatial_3bowl/landmark_with_hardneg_prompt` additionally swaps
in the `hardneg` prompt condition on the same scene, combining Split 1 x Split 2.

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
are identical across `irrelevant`/`semantic`/`landmark` (only bowl_3 differs):

| id | target | bowl_1 | bowl_2 | bowl_3 `irrelevant` | bowl_3 `semantic` | bowl_3 `landmark` (`hardneg_region`, per-task) |
|--:|---|---|---|---|---|---|
| 0 | between plate & ramekin | `between_plate_ramekin_region` | `next_to_ramekin_region` | `table_center` | `next_to_box_region` | (−0.070,0.040)–(−0.050,0.060) |
| 1 | next to ramekin | `next_to_ramekin_region` | `next_to_box_region` | `table_center` | `next_to_plate_region` | (−0.210,0.040)–(−0.190,0.060) |
| 2 | table center | `table_center` | `next_to_plate_region` | `table_front`* | `next_to_ramekin_region` | (−0.290,0.010)–(−0.270,0.030) |
| 3 | on cookie box | `cookies_1` (obj) | `wooden_cabinet_1_top_side` (obj) | `table_center` | `next_to_ramekin_region` | (0.180,−0.070)–(0.200,−0.050) |
| 4 | in top drawer | `In wooden_cabinet_1_top_region` (obj) | `wooden_cabinet_1_top_side` (obj) | `table_center` | `next_to_plate_region` | (−0.060,−0.070)–(−0.040,−0.050) |
| 5 | on ramekin | `glazed_rim_porcelain_ramekin_1` (obj) | `cookies_1` (obj) | `table_center` | `next_to_plate_region` | (−0.210,0.040)–(−0.190,0.060) |
| 6 | next to cookie box | `next_to_box_region` | `flat_stove_1_cook_region` (obj) | `table_center` | `next_to_ramekin_region` | (−0.030,−0.120)–(−0.010,−0.100) |
| 7 | on stove | `flat_stove_1_cook_region` (obj) | `wooden_cabinet_1_top_side` (obj) | `table_center` | `next_to_box_region` | (−0.310,−0.010)–(−0.290,0.010) |
| 8 | next to plate | `next_to_plate_region` | `next_to_ramekin_region` | `table_center` | `next_to_box_region` | (0.190,0.080)–(0.210,0.100) |
| 9 | on wooden cabinet | `wooden_cabinet_1_top_side` (obj) | `flat_stove_1_cook_region` (obj) | `table_center` | `next_to_ramekin_region` | (−0.010,−0.060)–(0.010,−0.040) |

\* Task 2's `irrelevant` bowl_3 uses `table_front` instead of `table_center` only because bowl_1
already occupies `table_center` there.

**Design confound found while building this table**: `irrelevant`'s bowl_3 sits at the *same
fixed absolute coordinate* in 9 of 10 tasks, regardless of the target's location — it is not
"neutral" relative to every task's reach path. This is exactly what produced task 6's outlier
result in `eval_results.md` Exp 2 (90%→44%): `table_center` (−0.075,0.0) sits roughly on the
straight line between `next_to_box_region` (0.13,−0.07) and `plate_region` (0.06,0.20) — a rough
perpendicular-distance estimate puts it ~0.18m off that line, comparable to the ~0.115m bowl
diameter plus gripper clearance. `landmark`'s bowl_3, by contrast, gets a bespoke `hardneg_region`
per task, so it doesn't share this flaw. See §9/§10 for what this means for trusting `irrelevant`'s
numbers, and the advice in §10 item 4.

**Progress**

| Condition | Status | Overall SR | Rollouts | Source |
|---|---|--:|--:|---|
| `irrelevant` | ⬜ **not confirmed as run under this definition** | (80.2% exists, see caveat) | 401/500 | `eval_results.md` Exp 2 ("3 bowls") measured a 3rd bowl fixed at `table_center`/`table_front`; whether that run was performed *as* the "irrelevant" condition (vs. incidentally landing on this geometry) is disputed — treat the 80.2% number as real 3-bowl data, not as validated "irrelevant" condition data, until re-run or reconciled |
| `semantic` | ⬜ scene authored, not run | — | — | BDDL + init states generated/verified on this machine this pass; policy never invoked |
| `landmark` | ⬜ scene verified+rendered this pass, not run | — | — | BDDL/init states pre-existed but had never been checked with the current pipeline; regenerated (byte-identical to the committed data, confirming determinism), verified (min sep 0.121m vs. 0.12m threshold — **passes but narrowly**, task 3 is the tightest), contact sheet eyeballed — no overlaps visible. Old per-suite launch scripts existed but were never invoked for a real run; no rollouts recorded anywhere. Earlier drafts of this doc incorrectly claimed this was run — corrected |
| `landmark_with_hardneg_prompt` | ⬜ not run | — | — | Split 1 x Split 2 combination, not built/run |
| `path` | ⬜ not authored | — | — | open design question, §10 |

No condition in this split has confirmed, unambiguous rollout data yet. `Distractor-type Drop` is
not computable for any condition until `irrelevant` is reconciled and the others are run — deferred
to a server session.

### Split 3 — Scene Complexity Probe

| Field | Description |
|---|---|
| Goal | Whether clutter/occlusion degrades the policy, independent of scene feasibility |
| Hypothesis | Clutter costs real success rate, but naive deltas overstate it by conflating policy failure with physically blocked trajectories |
| Changed variable | Wooden cabinet's top drawer: closed vs. open |
| Controlled variables | 3-bowl scene (`irrelevant` placement), prompt = `default`, checkpoint, seed |
| Conditions | `drawer_closed` (= Split 2's `irrelevant`) · `drawer_open` |
| Tasks | All 10 tasks; **tasks 3, 6, 7 excluded from the adjusted metric** — rollout review showed the open drawer physically blocks their trajectories (`eval_results.md` Exp 3) |
| Trials | 50/task/condition, seed 7 |
| Metrics | `Raw Clutter Drop = SR(drawer_closed) - SR(drawer_open)` over all 10 tasks; `Adjusted Clutter Drop` = same formula restricted to the 7 feasible tasks {0,1,2,4,5,8,9} |
| Output | `EVAL-libero_spatial_3bowl_open-openvla-*.txt` + matching `results/*.jsonl` |

Registry: `spatial_3bowl/drawer_open` (suite `libero_spatial_3bowl_open`).

**Progress**

| Condition | Status | Raw SR (10 tasks) | Adjusted SR (7 tasks) | Rollouts | Source |
|---|---|--:|--:|--:|---|
| `drawer_closed` (= Split 2 `irrelevant`) | ✅ run | 80.2% | 84.3% | 401/500, 295/350 | `eval_results.md` Exp 2/3 |
| `drawer_open` | ✅ run | 60.0% | 73.1% | 300/500, 256/350 | `eval_results.md` Exp 3 |

`Raw Clutter Drop = 80.2 - 60.0 = 20.2 pts`. `Adjusted Clutter Drop = 84.3 - 73.1 = 11.1 pts` (vs.
the original 2-bowl baseline recomputed on the same 7 tasks, 84.9%, the drop is 11.7 pts — see §4).
Both numbers are final; this split is fully run. The task-3/6/7 exclusion was a manual
rollout-video review, not an automated feasibility check — see §9 (confounders) and §10 (open
question) for why that's a limitation worth flagging, not silently treating as solved.

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
| Metrics | Final success, first-pick accuracy, wrong-bowl pick rate, distractor attraction rate (see §10 — the latter three need per-step contact logging not yet implemented) |
| Output | `libero_spatial` results filtered by `--task_ids` for the 4 existing cells; `EVAL-libero_spatial_grounding_*-openvla-*.txt` for the 2 new ones |

Registry: existing-scene cells run via `--split spatial/default --task_ids <ids from
GROUNDING_PROBE_CELLS>`; new cells are `grounding/surface_landmark`, `grounding/region_surface`.

**Progress**

4 of 6 cells reuse rollouts already collected under `spatial/default` (Split 1, Exp 1) — no new
GPU run needed, just re-aggregating existing per-task numbers by relation-family cell. The other
2 cells need their own new-scene suites run. Per-task `default` success rates below are copied
from `eval_results.md` Exp 1's table (task 4, containment, is excluded from this matrix per §3).

| (target, distractor) family | Tasks (id: SR) | Cell SR | Status |
|---|---|--:|---|
| (landmark, landmark) | 0: 92%, 1: 84%, 8: 84% | **86.7%** | ✅ derived from existing data |
| (region, landmark) | 2: 92% | **92.0%** | ✅ derived from existing data |
| (surface, surface) | 3: 84%, 5: 94%, 7: 72%, 9: 72% | **80.5%** | ✅ derived from existing data |
| (landmark, surface) | 6: 90% | **90.0%** | ✅ derived from existing data |
| (surface, landmark) | new suite, task 0 | — | ⬜ scene authored + init states generated this pass, not run |
| (region, surface) | new suite, task 0 | — | ⬜ scene authored + init states generated this pass, not run |

**Preliminary observation** (from the 4 derived cells only, n is small — 1-4 tasks/cell, treat as
suggestive not conclusive): landmark-target cells (86.7%, 92.0%, 90.0%) are **not** lower than the
surface-target cell (80.5%) — if anything the opposite of the stated hypothesis ("landmark-target
scenes are more attraction-prone"). This is a real signal worth noting, but it's confounded with
per-task variation the hypothesis wasn't designed to control for (e.g. task 7's 72% may reflect
something task-specific, not its relation family) — the 2 missing cells (needing an actual GPU
run) are what would let this be tested properly rather than eyeballed from 4 small buckets.

**4b. Target-cue x distractor-cue prompt matrix.** The original spec's 3x3 matrix ("Pick the
bowl next to the ramekin, not the one next to the cookie box...") requires describing the SAME
physical target/distractor pair truthfully via 3 different cue types each. **Not built** — see
§10, this needs either new scene geometry or a relaxed truthfulness constraint, not a decision to
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

## 8. Preliminary findings (from `eval_results.md`, full detail there)

1. **Negative-contrast prompts badly hurt** the policy (-47.2 pts overall; up to -90 on some
   tasks) — the model does not know how to use a "not the one X" clause, it seems to actively
   confuse it. This is why Split 1 also needs the untested `positive_contrast` condition: is it
   the *negation* specifically, or any distractor mention, that hurts?
2. **One extra distractor barely dents overall success** (-3.8 pts) but concentrates almost
   entirely on one task (next to cookie box: 90% -> 44%) where the extra bowl lands on the
   direct path. This is the empirical motivation for Split 2's `path` condition — and, on closer
   inspection of the actual coordinates (see Split 2's Bowl coordinates table), also evidence that
   the fixed `table_center` spot used for every task isn't a uniformly "neutral" placement, since
   it happens to sit near task 6's reach path specifically.
3. **Open-drawer clutter's raw drop (-20.2 pts) overstates the policy effect** — about half is 3
   tasks where the drawer physically blocks the arm's path, not a perception failure. Adjusted
   (7 feasible tasks): -11.1 to -11.7 pts, a real but smaller robustness gap.
4. **Re-aggregating existing Split 1 data by relation family (Split 4a) does not support the
   landmark-attraction hypothesis** — the 3 landmark-target cells (86.7%, 92.0%, 90.0%) score at
   or above the one derived surface-target cell (80.5%), not below it. Weak evidence (1-4 tasks
   per cell, confounded with per-task variation) — see Split 4's Progress table — but notable
   enough that the 2 still-unrun new-scene cells matter for actually testing this.

## 9. Limitations & confounders

- The task-3/6/7 exclusion in Split 3 was decided by manually watching rollout videos, not an
  automated "is this trajectory kinematically feasible" check. A different reviewer might draw
  the line differently. Documented as a limitation, not silently treated as ground truth.
- `positive_contrast`, `semantic`, `grounding/surface_landmark`, and `grounding/region_surface`
  were authored this pass (prompts and/or BDDL scenes + verified init states) but never run
  through the policy — the phrasing/placement choices are reasoned but unvalidated. `landmark`
  (`libero_spatial_3bowl_hardneg`) has existed in the registry from before this pass; this pass
  verified its init states for the first time (min separation 0.121m against a 0.12m threshold —
  passes, but with the narrowest margin of any suite so far, task 3 specifically) and confirmed
  the checked-in data is deterministically reproducible, but it still has no rollouts recorded
  anywhere in `eval_results.md`.
- All current variants keep the checkpoint's original `unnorm_key` (`libero_spatial_no_noops`)
  even though the scene changed — correct for action un-normalization (the action space didn't
  change), but means none of these variants test whether the *policy* generalizes to a
  re-trained distribution, only whether a fixed policy holds up to distribution shift.
- `spatial_3bowl/irrelevant`'s bowl_3 is a single fixed coordinate (`table_center`/`table_front`)
  reused across all 10 tasks, not a per-task-neutral position — see Split 2's confound note. This
  is a real design gap, not just a labeling dispute: whatever this condition is called, the same
  data can't cleanly separate "does an extra distractor hurt" from "does this specific spot
  happen to sit on this specific task's path."

## 10. Open design questions (do not implement without a decision)

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
4. **Split 2 `irrelevant`'s "neutral" position isn't actually distance-controlled**: a single
   fixed `table_center` coordinate is not equally far from every task's target-to-plate path (see
   Split 2's confound note). Two fixes are on the table, not yet decided between: (a) treat this
   the same way Split 3 handled drawer-blocked tasks — run it, then compute per-task whether
   bowl_3 intersects a rough target-to-plate line, and report both raw and path-adjusted overalls;
   or (b) redefine `irrelevant` as a per-task position chosen to be a fixed *distance* from the
   target's own landmark in a direction away from the plate, so "neutral" means distance-matched
   to `landmark`/`semantic` rather than a single shared coordinate. (b) is more work (new BDDL
   geometry per task, same effort as authoring `path`) but removes the confound at the source
   instead of correcting for it after the fact. No GPU time should be spent validating `irrelevant`
   until one of these is chosen.

## 11. Execution checklist

1. Pick a split id from `eval_registry.SPLITS` (or add one — see `CLAUDE.md`'s "how to add a
   benchmark split").
2. Run `python scripts/preflight.py` once per machine to confirm GPU/checkpoint/image are ready.
3. `MACHINE_CONFIG=config/<machine>.env bash docker/openvla_libero/run_eval.sh --split <id>`
4. `python scripts/aggregate_results.py --filter <suite name>` to get the per-task/overall table.
5. Compare against the baseline/adjusted formulas in the relevant split's row above.
6. Record the result in `eval_results.md` (append — never overwrite existing experiments).

**Queued for the next GPU (server) session** — registry-ready, content authored/verified on this
laptop, no rollouts yet:

| Split id | Suite ready? |
|---|---|
| `spatial/positive_contrast` | yes |
| `spatial_3bowl/irrelevant` | **decide the redefinition question in §10 item 4 first** — re-running as-is repeats the same confound |
| `spatial_3bowl/semantic` | yes (init states verified this pass) |
| `spatial_3bowl/landmark` | yes (pre-existing suite; init states verified+rendered this pass, margin is tight — 0.121m vs. 0.12m threshold on task 3 — worth a second look if it turns out to matter) |
| `spatial_3bowl/landmark_with_hardneg_prompt` | yes |
| `grounding/surface_landmark` | yes (init states verified this pass) |
| `grounding/region_surface` | yes (init states verified this pass) |

Not registry-ready yet (open design questions, §10): Split 2's `path` distractor, Split 4b's
cue-phrasing matrix.
