# OpenVLA · LIBERO-Spatial Benchmark — Detailed Results

**What this file is.** The detailed record for every split/condition: the exact eval setting
(instruction/prompt text, scene render, distractor placement), render comparisons against baseline,
full per-task results, and the analysis/takeaway for each. It also carries the status dashboard
(run vs. not, computed drop metrics) and the cross-experiment findings as they accumulate. This file
is the authoritative source for *what a condition's setting was and what its numbers were*.

**What it is not.** Not the launch record — *when* something was run, in what batch/order, on what
hardware, and the exact relaunch command live in `eval_log.md`. Not the split design/rationale either
— hypotheses, why a split exists, and open design questions live in `benchmark_split_plan.md`.

**Update rule: every time a real GPU eval finishes, update this file** (add/extend the condition's
detail section, the status table, and the cross-experiment findings if the picture changed) **and
append an entry to `eval_log.md`.** `benchmark_split_plan.md` only needs an update when a split's
*definition* changes (new condition, redefined placement, newly authored scene).

## 0. Common eval setting

Applies to every condition below unless a section says otherwise.

- **Model:** `openvla-7b` LoRA (r32) fine-tuned on `libero_spatial_no_noops` (bf16 + FlashAttention-2
  on H200; sdpa on Blackwell — see `eval_log.md` per batch).
- **Base suite:** `libero_spatial` — 10 tasks; each = *"pick up the black bowl \<location\> and place
  it on the plate."*
- **Protocol:** 50 trials/task = 500 rollouts per condition, seed 7. Success = LIBERO's own goal
  predicate for that task.
- **Task ids** are identical across every suite, so columns/rows line up 1:1 everywhere below:

  | id | target location | id | target location |
  |--:|---|--:|---|
  | 0 | between the plate and the ramekin | 5 | on the ramekin |
  | 1 | next to the ramekin | 6 | next to the cookie box |
  | 2 | table center | 7 | on the stove |
  | 3 | on the cookie box | 8 | next to the plate |
  | 4 | in the top drawer of the wooden cabinet | 9 | on the wooden cabinet |

- **Noise:** at 50 trials/task, 1 standard error ≈ ±5–7 pts on a single task, ≈ ±3.3 pts pooled over
  500 — treat single-task swings ≲10 pts, or overall swings ≲3–4 pts, as noise.
- **Baseline reference:** `spatial/default` — 2 bowls, default prompt, `libero_spatial` scene, 84.0%
  (420/500) — everything below compares against this unless noted.

## 1. Status overview

| Split | Registry status | Data status |
|---|---|---|
| 1. Prompt Sensitivity | 3/3 conditions implemented | 3/3 run (`default`, `negative_contrast`, `positive_contrast`) |
| 2. Distractor Placement | 3/4 conditions implemented (`path` not authored) | 4/4 implemented conditions run (`irrelevant`, `semantic`, `landmark`, `landmark_with_hardneg_prompt`; the old `irrelevant` data survives relabeled as `center_fixed_legacy`); only unauthored `path` remains |
| 3. Scene Complexity | Implemented | Run (both conditions) |
| 4. Surface vs. Landmark Grounding | 4a: cells implemented, 2 new-scene cells' contact sheets now checked (§7); 4b: implemented as a target cue-type probe | 4/6 cells derivable from existing Split-1 data; 2/6 not run; 4b's 2 conditions not run |

---

## 2. Split 1 — Prompt Sensitivity Probe

**Question.** The 2-bowl scene has a target + 1 identical distractor bowl. Does *telling the model
which bowl to avoid* help, and does *how* it's told (mention vs. negate) matter? Only the **prompt
string** changes across conditions; scene and init states are identical.

| Condition | Status | Overall SR | Rollouts |
|---|---|--:|--:|
| `default` — names only the target | ✅ run | **84.0%** | 420/500 |
| `positive_contrast` — states distractor's location as fact, no negation | ✅ run | **32.4%** | 162/500 |
| `negative_contrast` — names + negates distractor ("…not the one…") | ✅ run | **36.8%** | 184/500 |

Computed: `Negative Contrast Drop = 84.0 − 36.8 = 47.2 pts`. `Distractor Mention Drop = 84.0 − 32.4 =
51.6 pts`. `Negation-specific Drop = 32.4 − 36.8 = −4.4 pts` — negative, meaning the negation clause
is not the source of the damage; bare mention of a second location does effectively all of it alone.

**Split fully run.**

### Setting: `default` vs. `negative_contrast`

- **Default:** *"pick up the black bowl on the stove and place it on the plate."*
- **Negative contrast:** also names the distractor — *"…, not the one on top of the wooden cabinet, …"*
- Prompts: `openvla/experiments/robot/libero/instructions.py::LIBERO_SPATIAL_NEGATIVE_CONTRAST_INSTRUCTIONS`
  (historically named "explicit").
- Scene: stock `libero_spatial` (2 bowls), unmodified — no render check needed.

| id | target | distractor named in prompt | Default | Neg. contrast | Δ |
|--:|---|---|--:|--:|--:|
| 0 | between the plate and the ramekin | next to the ramekin | 92% | 94% | +2 |
| 1 | next to the ramekin | next to the cookie box | 84% | 32% | −52 |
| 2 | table center | next to the plate | 92% | 2% | **−90** |
| 3 | on the cookie box | on top of the wooden cabinet | 84% | 52% | −32 |
| 4 | in the top drawer | on top of the cabinet | 76% | 64% | −12 |
| 5 | on the ramekin | on top of the cookie box | 94% | 4% | **−90** |
| 6 | next to the cookie box | on the stove | 90% | 72% | −18 |
| 7 | on the stove | on top of the wooden cabinet | 72% | 4% | −68 |
| 8 | next to the plate | next to the ramekin | 84% | 36% | −48 |
| 9 | on the wooden cabinet | on the stove | 72% | 8% | −64 |

<details><summary>Exact negative-contrast prompts used (distractor clause in <b>bold</b>)</summary>

| Default target | Negative-contrast instruction |
|---|---|
| between the plate and the ramekin | pick up the black bowl between the plate and the ramekin, **not the one next to the ramekin**, and place it on the plate |
| table center | pick up the black bowl at the center of the table, **not the one next to the plate**, and place it on the plate |
| in the top drawer of the cabinet | pick up the black bowl inside the top drawer of the wooden cabinet, **not the one on top of the cabinet**, and place it on the plate |
| next to the cookie box | pick up the black bowl next to the cookie box, **not the one on the stove**, and place it on the plate |
| next to the plate | pick up the black bowl next to the plate, **not the one next to the ramekin**, and place it on the plate |
| next to the ramekin | pick up the black bowl next to the ramekin, **not the one next to the cookie box**, and place it on the plate |
| on the cookie box | pick up the black bowl on top of the cookie box, **not the one on top of the wooden cabinet**, and place it on the plate |
| on the ramekin | pick up the black bowl on top of the ramekin, **not the one on top of the cookie box**, and place it on the plate |
| on the stove | pick up the black bowl on the stove, **not the one on top of the wooden cabinet**, and place it on the plate |
| on the wooden cabinet | pick up the black bowl on top of the wooden cabinet, **not the one on the stove**, and place it on the plate |

</details>

**Analysis.** The "…not the one on X…" clause *confuses* the policy instead of disambiguating it
(−47.2 pts overall). Damage is worst where the clause names a salient surface (ramekin, table center,
stove, cabinet). The only unhurt task (0) was already phrased relationally.

### Setting: `positive_contrast`

- **Positive contrast:** states the distractor's location as a plain fact, no negation — *"…the other
  black bowl is on X."*
- Prompts: `openvla/experiments/robot/libero/instructions.py::LIBERO_SPATIAL_POSITIVE_CONTRAST_INSTRUCTIONS`.
- Scene: stock `libero_spatial` (2 bowls), unmodified — no render check needed.

| id | target | Default | Positive contrast | Negative contrast | Δ (pos − default) | Δ (neg − pos) |
|--:|---|--:|--:|--:|--:|--:|
| 0 | between the plate and the ramekin | 92% | 92% | 94% | 0 | +2 |
| 1 | next to the ramekin | 84% | 8% | 32% | **−76** | +24 |
| 2 | table center | 92% | 14% | 2% | **−78** | −12 |
| 3 | on the cookie box | 84% | 38% | 52% | −46 | +14 |
| 4 | in the top drawer | 76% | 50% | 64% | −26 | +14 |
| 5 | on the ramekin | 94% | 2% | 4% | **−92** | +2 |
| 6 | next to the cookie box | 90% | 40% | 72% | −50 | +32 |
| 7 | on the stove | 72% | 12% | 4% | −60 | −8 |
| 8 | next to the plate | 84% | 68% | 36% | −16 | −32 |
| 9 | on the wooden cabinet | 72% | 0% | 8% | **−72** | +8 |

**Analysis.** The negation-specific drop is *negative* — adding "not the one…" on top of a bare
mention slightly **helps** rather than hurts (+4.4 pts), and that's within noise at this trial count
anyway. Nearly all of `negative_contrast`'s −47 pt damage survives with the negation removed entirely
(−51.6 pts here). So the earlier hypothesis ("negation confuses the policy") was misattributed: the
confusion comes from **naming a second spatial location in the prompt at all** — the checkpoint was
fine-tuned on target-only prompts and has no practice grounding a second referent, negated or not.
Task-level pattern matches `negative_contrast` closely (tasks 1, 2, 5, 9 hit hardest in both).

Results: `results/libero_spatial--positive_contrast--shard{0..3}of4.jsonl`.

---

## 3. Split 2 — Distractor Placement Probe

**Question.** Does an extra distractor's *position* — not just its presence — drive failures? Each
condition keeps the ordinary (target-only) prompt and adds a **third** `akita_black_bowl` at a
different kind of location relative to the target. Coordinate catalog and per-task placement table
for every condition live in `benchmark_split_plan.md` §Split 2 — this section covers rendered
outcome, results, and analysis per condition.

| Condition | Status | Overall SR | Rollouts |
|---|---|--:|--:|
| `irrelevant` (redefined) | ✅ run | 88.8% | 444/500 |
| `center_fixed_legacy` (retired definition) | ✅ run | 80.2% | 401/500 |
| `semantic` | ✅ run | 84.8% | 424/500 |
| `landmark` | ✅ run | 80.6% | 403/500 |
| `landmark_with_hardneg_prompt` (Split 1×2 combo) | ✅ run | 41.2% | 412/500 |
| `path` | ⬜ not authored | — | — |

Computed `Distractor-type Drop = SR(spatial/default) − SR(condition)`:

| Condition | Drop | Interpretation |
|---|--:|---|
| `irrelevant` | −4.8 pts | Negative — costs nothing; scores *above* baseline, no task collapse |
| `semantic` | −0.8 pts | Negative — no measurable cost from a distractor at an unrelated named landmark |
| `landmark` | +3.4 pts | The one real-cost result — concentrated almost entirely in two tasks |
| `landmark_with_hardneg_prompt` | +42.8 pts (vs. baseline); +39.4 pts vs. `landmark` on the identical scene | Adding a disambiguating prompt to the `landmark` scene does not rescue the affected tasks and wrecks the rest of the suite |

**Split's implemented conditions are fully run; only the unauthored `path` condition remains.**

### Setting: `center_fixed_legacy` (retired)

**Question.** Keep the default prompt, add a second distractor bowl at a single fixed absolute
coordinate (`table_center`/`table_front`) reused across all 10 tasks. Retired after this run revealed
a path-proximity confound (see analysis below); kept only as a labeled historical result.

- Distractor placement: single fixed coordinate, not per-task — see `benchmark_split_plan.md` §Split 2
  "Design confound found, then fixed" for the exact geometry.
- Scene: `libero_spatial_3bowl`. State vector grows 92→105 dims (extra free body). Verified across all
  500 states: worst bowl-to-bowl distance 0.122 m (bowl ⌀ ≈ 0.115 m) — no overlaps, valid heights.

| id | target | 2-bowl (`default`) | 3-bowl (`center_fixed_legacy`) | Δ |
|--:|---|--:|--:|--:|
| 0 | between the plate and the ramekin | 92% | 76% | −16 |
| 1 | next to the ramekin | 84% | 90% | +6 |
| 2 | table center | 92% | 94% | +2 |
| 3 | on the cookie box | 84% | 86% | +2 |
| 4 | in the top drawer | 76% | 84% | +8 |
| 5 | on the ramekin | 94% | 84% | −10 |
| 6 | next to the cookie box | 90% | 44% | **−46** |
| 7 | on the stove | 72% | 82% | +10 |
| 8 | next to the plate | 84% | 88% | +4 |
| 9 | on the wooden cabinet | 72% | 74% | +2 |

**Render compare** (left = 2 bowls `libero_spatial`, right = 3 bowls `libero_spatial_3bowl`; rows =
task ids 0–9, each panel the real episode-0 state restored via `set_init_state`):

![2-bowl vs 3-bowl init states](openvla/experiments/figures/compare_2v3bowl_grid.png)

**Analysis.** Overall success is **robust** to one extra distractor (−3.8 pts) — but the loss is
**concentrated**: task 6 (*next to the cookie box*) collapses 90%→44% because the fixed-coordinate
bowl lands between the cookie box and the plate, only slightly farther from the cookie box than the
target. Rough perpendicular-distance estimate: `table_center` (−0.075, 0.0) sits ~0.18 m off the
straight line from `next_to_box_region` to `plate_region` — comparable to the ~0.115 m bowl diameter
plus gripper clearance. This became the empirical motivation for the `irrelevant` redefinition below.

### Setting: `irrelevant` (redefined)

**Question.** Split 2's control: a 3rd bowl placed somewhere not tied to any relational language,
per-task chosen for maximum clearance from both the reach path and the 2nd bowl — replacing
`center_fixed_legacy`'s single fixed coordinate.

- Suite: `libero_spatial_3bowl_neutral`. Per-task region assignment and the redefinition rationale are
  in `benchmark_split_plan.md` §Split 2.
- Render check: `verify_suite_init_states.py` → `PASS`, worst separation 0.122 m; contact sheet
  eyeballed — 3 distinct bowls per task, no overlaps.

| id | target | Default (2-bowl) | Irrelevant (3-bowl) | Δ |
|--:|---|--:|--:|--:|
| 0 | between the plate and the ramekin | 92% | 86% | −6 |
| 1 | next to the ramekin | 84% | 90% | +6 |
| 2 | table center | 92% | 96% | +4 |
| 3 | on the cookie box | 84% | 90% | +6 |
| 4 | in the top drawer | 76% | 96% | **+20** |
| 5 | on the ramekin | 94% | 96% | +2 |
| 6 | next to the cookie box | 90% | 94% | +4 |
| 7 | on the stove | 72% | 88% | **+16** |
| 8 | next to the plate | 84% | 84% | 0 |
| 9 | on the wooden cabinet | 72% | 68% | −4 |

**Render compare** (left = `libero_spatial`, right = `libero_spatial_3bowl_neutral`; rows = task ids
0–9):

![2-bowl vs irrelevant](openvla/experiments/figures/compare_2v3bowl_neutral_grid.png)

**Analysis.** Unlike `center_fixed_legacy` (−3.8 pts, one task collapsing), the redefined neutral
placement doesn't cost anything at all — it scores *above* baseline (+4.8 pts, ~3 pooled SE, so
plausibly a real if modest effect rather than pure noise) with no single-task collapse. Tasks 4 and 7
gain the most (+20, +16); not investigated further at the rollout-video level.

Results: `results/libero_spatial_3bowl_neutral--default--shard{0..3}of4.jsonl`.

### Setting: `semantic`

**Question.** Places the 3rd bowl at a **named landmark that is not the target's own** — testing
whether sitting at *any* nameable relational spot pulls the policy, even when that landmark doesn't
match the prompt.

- Suite: `libero_spatial_3bowl_semantic`. Per-task landmark assignment in
  `benchmark_split_plan.md` §Split 2.
- Render check: BDDL + init states authored and verified this pass; contact sheet eyeballed — 3
  distinct bowls per task, no overlaps/clipping, drawer open only for task 4 (expected — task 4's
  target lives inside the drawer).

| id | target | 3rd bowl's landmark | Success | n |
|--:|---|---|--:|--:|
| 0 | between the plate and the ramekin | next to the cookie box | 86% | 50 |
| 1 | next to the ramekin | next to the plate | 72% | 50 |
| 2 | table center | next to the ramekin | 96% | 50 |
| 3 | on the cookie box | next to the ramekin | 90% | 50 |
| 4 | in the top drawer | next to the plate | 88% | 50 |
| 5 | on the ramekin | next to the plate | 86% | 50 |
| 6 | next to the cookie box | next to the ramekin | 94% | 50 |
| 7 | on the stove | next to the box | 88% | 50 |
| 8 | next to the plate | next to the box | 80% | 50 |
| 9 | on the wooden cabinet | next to the ramekin | 68% | 50 |

**Render compare** (left = `libero_spatial`, right = `libero_spatial_3bowl_semantic`; rows = task ids
0–9):

![2-bowl vs semantic](openvla/experiments/figures/compare_2v3bowl_semantic_grid.png)

**Analysis.** A semantic-but-irrelevant distractor barely moves overall success (+0.8 pts — well
under the ~±3.3 pt pooled SE). The two weakest tasks, 1 (72%) and 9 (68%), are still comfortably above
the collapse range seen for prompt-side manipulations, so even the softest spots here look like
ordinary scene-to-scene noise. Contrasted with the prompt-side conditions in Split 1 (−51.6 pts for a
bare-mention prompt) and `center_fixed_legacy` (extra bowl at a neutral spot: −3.8 pts), this is the
strongest evidence yet that failures are driven by **what the prompt says**, not by **what's sitting
on the table** — a distractor's presence and rough position barely register unless language draws
attention to it.

Results: `results/libero_spatial_3bowl_semantic--default--shard{0..3}of4.jsonl`.

### Setting: `landmark`

**Question.** Split 2's hardest confusability test: the 3rd bowl sits near the target's **own**
landmark — the same relational word the prompt uses — farther from the exact target point than the
real bowl. A genuine look-alike for "the bowl near X," unlike `semantic` (different landmark) or
`irrelevant` (no landmark).

- Suite: `libero_spatial_3bowl_hardneg`. Per-task `hardneg_region` coordinates in
  `benchmark_split_plan.md` §Split 2.
- Render check: BDDL/init states pre-existed but were unchecked with the current pipeline;
  regenerated (byte-identical, confirming determinism), verified (min separation 0.121 m vs. the
  0.12 m threshold — passes narrowly, task 3 tightest), contact sheet eyeballed — task 3's close pair
  confirmed as two separate bowls, not merged.

| id | target | Default (2-bowl) | Landmark (3-bowl) | Δ |
|--:|---|--:|--:|--:|
| 0 | between the plate and the ramekin | 92% | 48% | **−44** |
| 1 | next to the ramekin | 84% | 92% | +8 |
| 2 | table center | 92% | 98% | +6 |
| 3 | on the cookie box | 84% | 88% | +4 |
| 4 | in the top drawer | 76% | 84% | +8 |
| 5 | on the ramekin | 94% | 92% | −2 |
| 6 | next to the cookie box | 90% | 80% | −10 |
| 7 | on the stove | 72% | 86% | +14 |
| 8 | next to the plate | 84% | 84% | 0 |
| 9 | on the wooden cabinet | 72% | 54% | **−18** |

**Render compare** (left = `libero_spatial`, right = `libero_spatial_3bowl_hardneg`; rows = task ids
0–9 — same scene also backs `landmark_with_hardneg_prompt` below, only the prompt differs):

![2-bowl vs landmark](openvla/experiments/figures/compare_2v3bowl_hardneg_grid.png)

**Analysis.** The overall drop (−3.4 pts) looks mild — comparable to `center_fixed_legacy`'s plain
extra distractor (−3.8 pts) — but that headline number hides a **concentrated, task-specific effect**:
task 0 collapses 92%→48% and task 9 drops 72%→54%, both far beyond the ~±7 pt single-task noise band
at n=50; every other task is flat or actually improves. This is the first Split 2 condition where the
distractor's *placement relative to the target's own landmark* — not just presence or placement at
some other landmark — produces a real, attributable failure. Together with `semantic`, this narrows
*where* scene-driven failures come from: proximity to the target's own landmark matters; unrelated
landmarks and neutral placement don't.

Results: `results/libero_spatial_3bowl_hardneg--default--shard{0..3}of4.jsonl`.

### Setting: `landmark_with_hardneg_prompt` (Split 1×2 combo)

**Question.** Can a disambiguating prompt — *"pick up the black bowl closest to X, not the one farther
from it, …"* — rescue `landmark`'s task 0/9 collapse? Same scene as `landmark`
(`libero_spatial_3bowl_hardneg`), condition = `hardneg` prompt
(`instructions.py::LIBERO_SPATIAL_HARDNEG_INSTRUCTIONS`).

- Scene: identical to `landmark`, no changes — no new render check needed.
- Prompt: adds an explicit closer/farther disambiguation clause on top of the `landmark` scene.

| id | target | Landmark scene, default prompt | + disambiguating prompt | Δ |
|--:|---|--:|--:|--:|
| 0 | between the plate and the ramekin | 48% | 38% | −10 |
| 1 | next to the ramekin | 92% | 34% | **−58** |
| 2 | table center | 98% | 50% | **−48** |
| 3 | on the cookie box | 88% | 62% | −26 |
| 4 | in the top drawer | 84% | 80% | −4 |
| 5 | on the ramekin | 92% | 38% | **−54** |
| 6 | next to the cookie box | 80% | 52% | −28 |
| 7 | on the stove | 86% | 4% | **−82** |
| 8 | next to the plate | 84% | 48% | −36 |
| 9 | on the wooden cabinet | 54% | 6% | **−48** |

**Analysis.** The disambiguating prompt does not rescue the two tasks that actually needed help —
task 0 barely changes (48%→38%) and task 9 gets *worse*, not better (54%→6%) — while every task that
was fine without it collapses instead, several catastrophically (task 7: 86%→4%, task 1: 92%→34%,
task 5: 92%→38%). Net effect is strongly negative (−39.4 pts on the same scene) and lands the overall
number (41.2%) close to Split 1's `negative_contrast` on the plain 2-bowl scene (36.8%) — consistent
with Split 1's finding that *any* prompt referencing a second bowl/location hurts, regardless of
whether the scene actually contains a confusable distractor. Combining a hard scene with a hard prompt
doesn't compound narrowly on the hard cases; the prompt damage dominates and spreads to tasks the
scene alone never touched. Strongest evidence in the project that language, not scene design, is the
primary lever on this checkpoint's failures.

Results: `results/libero_spatial_3bowl_hardneg--hardneg--shard{0..3}of4.jsonl`.

---

## 4. Split 3 — Scene Complexity Probe

**Question.** Keep the `center_fixed_legacy` 3-bowl scene and add clutter/occlusion by **opening the
wooden cabinet's top ("first") drawer** in every task. Does a protruding open drawer degrade the
policy further, or does it just block the arm?

### Blocked tasks — tasks 3, 6, 7 excluded from the adjusted overall

Rollout review showed that for **task 3** (*on the cookie box*), **task 6** (*next to the cookie
box*), and **task 7** (*on the stove*), the open top drawer **physically blocks the trajectory**: the
protruding drawer sits directly in the pick-and-place corridor these tasks must traverse, so the arm
cannot complete the motion regardless of what the policy predicts. Failures on these three tasks
therefore measure the *scene's physical feasibility*, not the *policy's robustness*, and are excluded
from the adjusted overall below. (Task 6's 0% is the clearest case — a hard geometric block, not a
perception error.) **Data-quality caveat:** this exclusion was a manual rollout-video review, not an
automated feasibility check — treat the adjusted number as reviewed-and-reasoned, not
machine-verified ground truth.

**Raw** (all 10 tasks) and **adjusted** (7 feasible tasks, 350 rollouts) overalls:

| Condition (default prompt) | Raw overall (10 tasks) | Adjusted overall (7 tasks†) |
|---|--:|--:|
| 3 bowls, drawer closed (`center_fixed_legacy`) | 80.2% (401/500) | 84.3% (295/350) |
| 3 bowls, **drawer open** | 60.0% (300/500) | **73.1%** (256/350) |
| **Δ (open − closed)** | −20.2 pts | **−11.1 pts** |

† Same-subset comparison: tasks {0, 1, 2, 4, 5, 8, 9} in both conditions. Against the original 2-bowl
baseline on the same 7 tasks (84.9%, 297/350), the adjusted drop is **−11.7 pts** — vs. the raw
headline of −24.0, which conflates policy degradation with physically infeasible tasks.

| id | target | 3-bowl (drawer closed) | +drawer open | Δ | |
|--:|---|--:|--:|--:|---|
| 0 | between the plate and the ramekin | 76% | 66% | −10 | |
| 1 | next to the ramekin | 90% | 86% | −4 | |
| 2 | table center | 94% | 80% | −14 | |
| 3 | on the cookie box | 86% | 42% | −44 | ‡ blocked |
| 4 | in the top drawer | 84% | 84% | 0 | |
| 5 | on the ramekin | 84% | 84% | 0 | |
| 6 | next to the cookie box | 44% | 0% | −44 | ‡ blocked |
| 7 | on the stove | 82% | 46% | −36 | ‡ blocked |
| 8 | next to the plate | 88% | 68% | −20 | |
| 9 | on the wooden cabinet | 74% | 44% | −30 | |

‡ Trajectory physically obstructed by the open drawer — excluded from the adjusted overall.

**Setting.** Suite `libero_spatial_3bowl_open`, same `center_fixed_legacy` bowl placement as Split 2
+ cabinet top-drawer joint forced open. Verified across all 500 open-drawer init states: bowls never
overlap (worst separation 0.122 m) and the drawer joint is open (qpos −0.141 m) in every trial. Adding
the drawer state does not change the state-vector size (105 dims) — the drawer joint already exists in
the cabinet model; only its position changes.

**Render compare** (left = drawer closed `libero_spatial_3bowl`, right = drawer open
`libero_spatial_3bowl_open`; rows = task ids 0–9):

![3-bowl: drawer closed vs open](openvla/experiments/figures/compare_3bowl_closed_vs_open_grid.png)

**Analysis.** The raw −20.2 pt drop overstates the policy effect: nearly half of it comes from the
three blocked tasks (3, 6, 7), where no policy could succeed because the open drawer occupies the
motion corridor. On the seven tasks that remain physically feasible, the open drawer still costs the
policy **−11.1 pts** vs. the closed-drawer scene — a genuine (occlusion/robustness) effect,
concentrated around the cabinet: task 9 (*on the wooden cabinet*, −30) and task 8 (*next to the
plate*, −20) drop well beyond noise, with smaller dips on 2 and 0. The two unaffected tasks (4 *in the
top drawer*, 5 *on the ramekin*) are telling: task 4's target is inside the drawer, so an open drawer
is expected there. The protruding drawer occludes the cabinet side of the table, which is exactly
where the still-degraded targets sit or are approached — but the effect is a robustness gap (~−11
pts), not the collapse the raw number suggests.

**Split fully run.**

---

## 5. Split 4 — Surface vs. Landmark Grounding Probe

**4a.** 4 of 6 cells reuse rollouts already collected under `spatial/default` (Split 1) — no new GPU
run needed, just re-aggregating existing per-task numbers by relation-family cell. Per-task `default`
success rates below are copied from Split 1's `default` table (task 4, containment, is excluded from
this matrix — see `benchmark_split_plan.md` §3).

| (target, distractor) family | Tasks (id: SR) | Cell SR | Status |
|---|---|--:|---|
| (landmark, landmark) | 0: 92%, 1: 84%, 8: 84% | **86.7%** | ✅ derived from existing data |
| (region, landmark) | 2: 92% | **92.0%** | ✅ derived from existing data |
| (surface, surface) | 3: 84%, 5: 94%, 7: 72%, 9: 72% | **80.5%** | ✅ derived from existing data |
| (landmark, surface) | 6: 90% | **90.0%** | ✅ derived from existing data |
| (surface, landmark) | new suite, task 0 | — | ⬜ scene authored + init states verified + contact sheet checked (§7), not run |
| (region, surface) | new suite, task 0 | — | ⬜ scene authored + init states verified + contact sheet checked (§7), not run |

**Preliminary observation** (from the 4 derived cells only, n is small — 1-4 tasks/cell, treat as
suggestive not conclusive): landmark-target cells (86.7%, 92.0%, 90.0%) are **not** lower than the
surface-target cell (80.5%) — if anything the opposite of the stated hypothesis ("landmark-target
scenes are more attraction-prone"). This is a real signal worth noting, but it's confounded with
per-task variation the hypothesis wasn't designed to control for (e.g. task 7's 72% may reflect
something task-specific, not its relation family) — the 2 missing cells (needing an actual GPU run)
are what would let this be tested properly rather than eyeballed from 4 small buckets.

**4b.** Redesigned as a target cue-type probe and implemented (`grounding/target_cue_region`,
`grounding/target_cue_landmark` on `openvla` branch `worktree-split4-target-cue-probe`), not yet
run — see `benchmark_split_plan.md` Split 4's 4b section.

**Still queued:** `grounding/surface_landmark`, `grounding/region_surface` — registry-ready, init
states verified, contact sheets checked (§7), no rollouts yet. `grounding/target_cue_region`,
`grounding/target_cue_landmark` — implemented on `openvla` branch `worktree-split4-target-cue-probe`
(pushed, not merged), no rollouts yet.

---

## 6. Cross-experiment findings

1. **Negative-contrast prompts badly hurt** the policy (-47.2 pts overall; up to -90 on some
   tasks) — the model does not know how to use a "not the one X" clause, it seems to actively
   confuse it.
2. **One extra distractor barely dents overall success** (-3.8 pts, `center_fixed_legacy`) but
   concentrates almost entirely on one task (next to cookie box: 90% -> 44%) where the extra bowl
   lands on the direct path. This was the empirical motivation for Split 2's `irrelevant`
   redefinition — the fixed `table_center` spot used for every task wasn't a uniformly "neutral"
   placement, since it happens to sit near task 6's reach path specifically (see
   `benchmark_split_plan.md` Split 2's confound note).
3. **Open-drawer clutter's raw drop (-20.2 pts) overstates the policy effect** — about half is 3
   tasks where the drawer physically blocks the arm's path, not a perception failure. Adjusted
   (7 feasible tasks): -11.1 to -11.7 pts, a real but smaller robustness gap.
4. **Re-aggregating existing Split 1 data by relation family (Split 4a) does not support the
   landmark-attraction hypothesis** — the 3 landmark-target cells (86.7%, 92.0%, 90.0%) score at
   or above the one derived surface-target cell (80.5%), not below it. Weak evidence (1-4 tasks
   per cell, confounded with per-task variation) — see §5's table — but notable enough that the 2
   still-unrun new-scene cells matter for actually testing this.
5. **The negation clause wasn't the source of the damage.** Bare mention of the distractor's
   location, with no "not the one…" clause, is just as bad (-51.6 pts) as mentioning + negating it
   (-47.2 pts) — `positive_contrast` (32.4%) even scores slightly *below* `negative_contrast`
   (36.8%). The policy has no practice grounding a second referent at all; negation specifically
   isn't the issue.
6. **Split 2's first real result cut against its own hypothesis.** A distractor placed at an
   unrelated named landmark (`semantic`) costs ~0 pts (84.8% vs. 84.0% baseline) — well inside
   noise. Combined with finding 5, the pattern was that failures track *what the prompt says*, not
   *what's on the table*.
7. **The harder version of the test — `landmark` — does show a real effect, but a concentrated
   one.** A distractor near the target's *own* landmark costs -3.4 pts overall (80.6% vs. 84.0%),
   similar in size to finding 2's plain extra bowl. But like finding 2, that headline number hides
   a task-specific collapse: task 0 (92%→48%) and task 9 (72%→54%) account for nearly all of it,
   every other task flat or improved. Revised picture: scene changes barely matter *in general*,
   but a distractor placed to be a genuine near-miss for the target's own landmark can badly hurt
   specific tasks — proximity-to-own-landmark is the one scene manipulation so far with a real,
   attributable (if concentrated) cost.
8. **Split 2's three-way distractor-position comparison is complete, and the pattern holds.**
   `irrelevant` (+4.8 pts) and `semantic` (+0.8 pts) both cost nothing — if anything they trend
   positive with no task collapsing. Only `landmark` (-3.4 pts, concentrated in 2 tasks) shows a
   real effect. Combined with findings 5-6, the picture across this entire project so far: failures
   come from **language that references a second location** (catastrophic, -47 to -52 pts) or from
   **a distractor that's a genuine look-alike for the target's own described location**
   (task-concentrated, tens of pts on the affected tasks) — not from scene clutter, extra-object
   presence, or a distractor merely sitting at *some* named place in general.
9. **A disambiguating prompt does not rescue the `landmark` scene's task-0/9 collapse — it makes
   things much worse overall.** `landmark_with_hardneg_prompt` (same hard-negative scene as finding
   7, plus "closest to X, not the one farther") scores 41.2%, a further -39.4 pts on top of
   `landmark`'s already-reduced 80.6%. Tasks 0 and 9 barely improve or get worse (48%→38%, 54%→6%),
   while every task that was fine without the prompt collapses instead (task 7: 86%→4%). Language
   that references a second location dominates regardless of whether the scene actually contains a
   confusable distractor — it doesn't compound narrowly on the hard cases, it spreads damage to the
   whole suite. Strongest evidence in the project that this checkpoint's failures are fundamentally
   about prompt grounding, not scene design.

## 7. Render / contact-sheet check log

Every new or previously-unchecked scene gets its init-state contact sheet rendered
(`LIBERO/scripts/render_suite_contact_sheet.py <suite>`) and eyeballed before spending GPU time on
it. Where a run adds a distractor to the 2-bowl baseline, a side-by-side comparison is also
rendered (`LIBERO/scripts/compare_two_suites_init.py libero_spatial <suite> <outname>` — left/blue
= 2-bowl baseline, right/orange = the 3-bowl variant, one row per task id 0–9) and saved under
`openvla/experiments/figures/` for permanent reference (the raw per-suite contact sheets under
`LIBERO/scratch_render/` are scratch and get overwritten each pass). The figures themselves are
embedded inline in each condition's section above (§3-4).

| Suite | Numeric verify (`verify_suite_init_states.py`) | Visual eyeball | Result |
|---|---|---|---|
| `libero_spatial_3bowl_neutral` (`irrelevant`) | PASS, worst sep 0.122m | ✅ | 3 distinct bowls per task, no overlaps |
| `libero_spatial_3bowl_semantic` (`semantic`) | verified | ✅ | 3 distinct bowls per task, no overlaps/clipping, drawer open only on task 4 (expected — task 4's target lives in the drawer) |
| `libero_spatial_3bowl_hardneg` (`landmark` / `landmark_with_hardneg_prompt`, same scene) | PASS but narrow — min sep 0.121m vs. 0.12m threshold, task 3 tightest | ✅ | 3 distinct bowls per task; task 3's close pair confirmed as two separate bowls, not merged |
| `libero_spatial_grounding_surface_landmark` (`grounding/surface_landmark`) | PASS, sep 0.393m | ✅ | Single task (`on_the_ramekin`). Cross-checked exact xyz against the BDDL, not just pixels: target `akita_black_bowl_1` = (-0.210, 0.192, z=1.080) — inside `ramekin_region` (-0.21,0.19)-(-0.19,0.21), elevated (on top of the ramekin, as intended). Distractor `akita_black_bowl_2` = (0.116, -0.067, z=0.970) — inside `next_to_box_region` (0.12,-0.08)-(0.14,-0.06) (0.004m outside on x, negligible), flat on the table (not elevated) — confirms it moved off `cookies_1` (was elevated/surface in the original task 5) to a landmark placement, as designed. No overlap/clipping. |
| `libero_spatial_grounding_region_surface` (`grounding/region_surface`) | PASS, sep 0.210m | ✅ | Single task (`from_table_center`). Target `akita_black_bowl_1` = (-0.075, 0.003, z=0.970) — inside `table_center` (-0.10,-0.01)-(-0.05,0.01), flat on table (region cue, as intended). Distractor `akita_black_bowl_2` = (-0.263, -0.137, z=1.010) — y matches `stove_region`'s -0.14 almost exactly, x offset from the stove's base anchor (-0.41) is consistent with `flat_stove_1_cook_region` being the stove's own top surface (same region used by tasks 6/9's distractors elsewhere in the suite), elevated (on top of the stove) — confirms surface placement, moved off `next_to_plate_region` (landmark in the original task 2). No overlap/clipping. |

`libero_spatial` (baseline) and `libero_spatial_3bowl`/`libero_spatial_3bowl_open`
(`center_fixed_legacy`/`drawer_open`) predate this render-before-run practice being tracked here;
no issues have surfaced in their data, but no dedicated check is logged. The two grounding suites
above are now checked and ready to run — see Split 4a's row in the status table.

Note on method: for these two single-task suites, plain pixel-diffing the new render against the
original `libero_spatial` task's render (t5 for surface_landmark, t2 for region_surface) was tried
first and was inconclusive — moving one bowl shifts shadows/specular highlights across a large
fraction of a 256x256 frame, so a large diff bounding box doesn't distinguish "one bowl moved" from
"something is wrong." Reading the actual simulator joint `qpos` for each bowl (as tabulated above)
and comparing against the BDDL region catalog is unambiguous and is the more reliable check for a
single-object scene change — recommended over pixel-diffing for any future single-task gap-fill
suite in this project.
