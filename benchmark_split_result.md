# OpenVLA · LIBERO-Spatial Benchmark — Results Dashboard

**What this file is.** A living status dashboard: per-split progress (run vs. not, success rates,
computed drop metrics), the scene-render/contact-sheet checks done before each real GPU run, and
the cross-experiment findings/analysis as they accumulate. It answers "what's the current state of
each split and what have we learned so far."

**What it is not.** Not the detailed per-experiment record — every number here is copied from
`eval_results.md`, never recomputed independently, and `eval_results.md` (append-only, one section
per experiment, full per-task tables + narrative takeaway) is the authoritative source if the two
ever seem to disagree. Not the split design/rationale either — hypotheses, conditions, scene
geometry, and open design questions live in `benchmark_split_plan.md`.

**Update rule: every time a real GPU eval finishes, update this file (status table + findings) and
`eval_results.md` (append the experiment).** `benchmark_split_plan.md` only needs an update when a
split's *definition* changes (new condition, redefined placement, newly authored scene).

## 1. Status overview

| Split | Registry status | Data status |
|---|---|---|
| 1. Prompt Sensitivity | 3/3 conditions implemented | 3/3 run (`default`, `negative_contrast`, `positive_contrast`) |
| 2. Distractor Placement | 3/4 conditions implemented (`path` not authored) | 4/4 implemented conditions run (`irrelevant`, `semantic`, `landmark`, `landmark_with_hardneg_prompt`; the old `irrelevant` data survives relabeled as `center_fixed_legacy`); only unauthored `path` remains |
| 3. Scene Complexity | Implemented | Run (both conditions) |
| 4. Surface vs. Landmark Grounding | 4a: cells implemented (4/6 reuse existing data, 2/6 new scenes); 4b: not built | 4/6 cells derivable from existing Split-1 data; 2/6 not run |

## 2. Split 1 — Prompt Sensitivity Probe

| Condition | Status | Overall SR | Rollouts | Source |
|---|---|--:|--:|---|
| `default` | ✅ run | 84.0% | 420/500 | `eval_results.md` Exp 1 |
| `negative_contrast` | ✅ run | 36.8% | 184/500 | `eval_results.md` Exp 1 |
| `positive_contrast` | ✅ run | 32.4% | 162/500 | `eval_results.md` Exp 4 |

Computed: `Negative Contrast Drop = 84.0 - 36.8 = 47.2 pts`. `Distractor Mention Drop = 84.0 - 32.4 =
51.6 pts`. `Negation-specific Drop = 32.4 - 36.8 = -4.4 pts` — negative, meaning the negation clause
is not the source of Exp 1's damage; bare mention of a second location does effectively all of it on
its own (see `eval_results.md` Exp 4 for the full per-task breakdown).

**Split fully run.**

## 3. Split 2 — Distractor Placement Probe

| Condition | Status | Overall SR | Rollouts | Source |
|---|---|--:|--:|---|
| `irrelevant` | ✅ run | 88.8% | 444/500 | `eval_results.md` Exp 7 |
| `center_fixed_legacy` | ✅ run (retired definition) | 80.2% | 401/500 | `eval_results.md` Exp 2 ("3 bowls"); old single-fixed-coordinate placement, kept under this label so the number stays attributable — **not** the current `irrelevant` |
| `semantic` | ✅ run | 84.8% | 424/500 | `eval_results.md` Exp 5 |
| `landmark` | ✅ run | 80.6% | 403/500 | `eval_results.md` Exp 6 |
| `landmark_with_hardneg_prompt` | ✅ run | 41.2% | 412/500 | `eval_results.md` Exp 8 |
| `path` | ⬜ not authored | — | — | open design question, `benchmark_split_plan.md` §9 |

Computed `Distractor-type Drop = SR(spatial/default) - SR(condition)`:

| Condition | Drop | Interpretation |
|---|--:|---|
| `irrelevant` | -4.8 pts | Negative — costs nothing; scores *above* baseline, no task collapse |
| `semantic` | -0.8 pts | Negative — no measurable cost from a distractor at an unrelated named landmark |
| `landmark` | +3.4 pts | The one real-cost result — concentrated almost entirely in two tasks (task 0: -44pts, task 9: -18pts), both far outside single-task noise |
| `landmark_with_hardneg_prompt` | +42.8 pts (vs. `spatial/default`); +39.4 pts vs. `landmark` on the identical scene | Adding a disambiguating prompt to the `landmark` scene does not rescue tasks 0/9 and wrecks the rest of the suite instead |

See `eval_results.md` Exp 5/6/7/8 for per-task breakdowns and the three-way comparison table.

**Split's implemented conditions are fully run; only the unauthored `path` condition remains.**

## 4. Split 3 — Scene Complexity Probe

| Condition | Status | Raw SR (10 tasks) | Adjusted SR (7 tasks) | Rollouts | Source |
|---|---|--:|--:|--:|---|
| `drawer_closed` (= Split 2 `center_fixed_legacy`) | ✅ run | 80.2% | 84.3% | 401/500, 295/350 | `eval_results.md` Exp 2/3 |
| `drawer_open` | ✅ run | 60.0% | 73.1% | 300/500, 256/350 | `eval_results.md` Exp 3 |

`Raw Clutter Drop = 80.2 - 60.0 = 20.2 pts`. `Adjusted Clutter Drop = 84.3 - 73.1 = 11.1 pts` (vs.
the original 2-bowl baseline recomputed on the same 7 tasks, 84.9%, the drop is 11.7 pts).

**Data-quality caveat:** the task-3/6/7 exclusion from the adjusted metric was a manual
rollout-video review (drawer physically blocks the arm's path on those three), not an automated
feasibility check. A different reviewer might draw the line differently — treat the adjusted number
as reviewed-and-reasoned, not machine-verified ground truth.

**Split fully run.**

## 5. Split 4 — Surface vs. Landmark Grounding Probe

**4a.** 4 of 6 cells reuse rollouts already collected under `spatial/default` (Split 1, Exp 1) — no
new GPU run needed, just re-aggregating existing per-task numbers by relation-family cell. Per-task
`default` success rates below are copied from `eval_results.md` Exp 1's table (task 4, containment,
is excluded from this matrix — see `benchmark_split_plan.md` §3).

| (target, distractor) family | Tasks (id: SR) | Cell SR | Status |
|---|---|--:|---|
| (landmark, landmark) | 0: 92%, 1: 84%, 8: 84% | **86.7%** | ✅ derived from existing data |
| (region, landmark) | 2: 92% | **92.0%** | ✅ derived from existing data |
| (surface, surface) | 3: 84%, 5: 94%, 7: 72%, 9: 72% | **80.5%** | ✅ derived from existing data |
| (landmark, surface) | 6: 90% | **90.0%** | ✅ derived from existing data |
| (surface, landmark) | new suite, task 0 | — | ⬜ scene authored + init states generated, not run |
| (region, surface) | new suite, task 0 | — | ⬜ scene authored + init states generated, not run |

**Preliminary observation** (from the 4 derived cells only, n is small — 1-4 tasks/cell, treat as
suggestive not conclusive): landmark-target cells (86.7%, 92.0%, 90.0%) are **not** lower than the
surface-target cell (80.5%) — if anything the opposite of the stated hypothesis ("landmark-target
scenes are more attraction-prone"). This is a real signal worth noting, but it's confounded with
per-task variation the hypothesis wasn't designed to control for (e.g. task 7's 72% may reflect
something task-specific, not its relation family) — the 2 missing cells (needing an actual GPU run)
are what would let this be tested properly rather than eyeballed from 4 small buckets.

**4b.** Not built — see `benchmark_split_plan.md` §9.

**Still queued:** `grounding/surface_landmark`, `grounding/region_surface` — both registry-ready,
init states verified, no rollouts yet.

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
   per cell, confounded with per-task variation) — see §5's Progress table — but notable enough
   that the 2 still-unrun new-scene cells matter for actually testing this.
5. **The negation clause wasn't the source of Exp 1's damage.** Bare mention of the distractor's
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
`LIBERO/scratch_render/` are scratch and get overwritten each pass).

| Suite | Numeric verify (`verify_suite_init_states.py`) | Visual eyeball | Result |
|---|---|---|---|
| `libero_spatial_3bowl_neutral` (`irrelevant`) | PASS, worst sep 0.122m | ✅ | 3 distinct bowls per task, no overlaps |
| `libero_spatial_3bowl_semantic` (`semantic`) | verified | ✅ | 3 distinct bowls per task, no overlaps/clipping, drawer open only on task 4 (expected — task 4's target lives in the drawer) |
| `libero_spatial_3bowl_hardneg` (`landmark` / `landmark_with_hardneg_prompt`, same scene) | PASS but narrow — min sep 0.121m vs. 0.12m threshold, task 3 tightest | ✅ | 3 distinct bowls per task; task 3's close pair confirmed as two separate bowls, not merged |

`libero_spatial` (baseline) and `libero_spatial_3bowl`/`libero_spatial_3bowl_open`
(`center_fixed_legacy`/`drawer_open`) predate this render-before-run practice being tracked here;
no issues have surfaced in their data, but no dedicated check is logged. `grounding/surface_landmark`
and `grounding/region_surface` have generated + verified init states but have **not** had a contact
sheet eyeballed yet — do that before running them.

**2-bowl baseline vs. 3-bowl `irrelevant`** (left = `libero_spatial`, right =
`libero_spatial_3bowl_neutral`; rows = task ids 0–9):

![2-bowl vs irrelevant](openvla/experiments/figures/compare_2v3bowl_neutral_grid.png)

**2-bowl baseline vs. 3-bowl `semantic`** (left = `libero_spatial`, right =
`libero_spatial_3bowl_semantic`; rows = task ids 0–9):

![2-bowl vs semantic](openvla/experiments/figures/compare_2v3bowl_semantic_grid.png)

**2-bowl baseline vs. 3-bowl `landmark`** (left = `libero_spatial`, right =
`libero_spatial_3bowl_hardneg`; rows = task ids 0–9 — same scene backs `landmark_with_hardneg_prompt`,
only the prompt differs):

![2-bowl vs landmark](openvla/experiments/figures/compare_2v3bowl_hardneg_grid.png)

## 8. What's been run / still queued

**Run 2026-08-19 (server session, 4× RTX PRO 6000 Blackwell):** `spatial/positive_contrast`,
`spatial_3bowl/irrelevant`, `spatial_3bowl/semantic`, `spatial_3bowl/landmark`,
`spatial_3bowl/landmark_with_hardneg_prompt` — see `eval_results.md` Exp 4-8 and §2-3 above.

**Still queued** — registry-ready, content authored/verified, no rollouts yet:

| Split id | Suite ready? |
|---|---|
| `grounding/surface_landmark` | init states verified; contact sheet not yet eyeballed |
| `grounding/region_surface` | init states verified; contact sheet not yet eyeballed |

**Not registry-ready** (open design questions, `benchmark_split_plan.md` §9): Split 2's `path`
distractor, Split 4b's cue-phrasing matrix.
