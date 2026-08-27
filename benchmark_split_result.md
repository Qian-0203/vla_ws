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
| 2. Distractor Placement | 3/4 conditions implemented (`path` not authored) | All implemented conditions run. `irrelevant` and `semantic` each redefined a second time (see §3) — current suites (`libero_spatial_3bowl_front` 85.2%, `libero_spatial_3bowl_semantic2` 85.2%, both 500/500) now run; prior data survives relabeled `irrelevant_v1_legacy` (88.8%) / `semantic_v1_legacy` (84.8%); original fixed-coordinate data survives as `center_fixed_legacy`. Only unauthored `path` remains beyond that |
| 3. Scene Complexity | Implemented | Run (both conditions) |
| 4. Surface vs. Landmark Grounding | 4a: all 6 cells implemented; 4b: implemented as a target cue-type probe | ✅ fully run — 6/6 cells (4a), 2/2 conditions (4b) |
| VLM Bowl-Pointing Probe (§8, not a `SPLITS` entry) | Script implemented (`probe_bowl_pointing.py`) | OpenVLA itself: dead end confirmed on 3 angles — no language-responsive text channel. Qwen2-VL-7B alternative (§8.1): a marker-placement bug (§8.2) was found and fixed; re-run scores 70% on both 2-bowl distractor-mention conditions and `hardneg` (was 40-60%). `default`/`hardneg_default` no-mention baselines added (§8.3): 50% (2-bowl, exactly chance) and 60% (3-bowl, above chance) — distractor-mention phrasing is a mild *disambiguating* cue for Qwen in both scenes, not a difficulty source |

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
outcome, results, and analysis per condition. For the same numbers reorganized **by task** instead —
placement + render + instruction + SR side by side for `default`/`irrelevant`/`semantic`/`landmark`,
one table per task — see `split2_distractor_comparison.md`.

| Condition | Status | Overall SR | Rollouts |
|---|---|--:|--:|
| `irrelevant` (current, `libero_spatial_3bowl_front`) | ✅ run | 85.2% | 500/500 |
| `semantic` (current, `libero_spatial_3bowl_semantic2`) | ✅ run | 85.2% | 500/500 |
| `landmark` | ✅ run | 80.6% | 403/500 |
| `landmark_with_hardneg_prompt` (Split 1×2 combo) | ✅ run | 41.2% | 412/500 |
| `path` | ⬜ not authored | — | — |

Retired condition definitions (`irrelevant_v1_legacy`, `semantic_v1_legacy`, `center_fixed_legacy`)
were run and superseded by the current `irrelevant`/`semantic` above; their numbers are preserved in
git history and `eval_log.md`, not repeated here.

Computed `Distractor-type Drop = SR(spatial/default) − SR(condition)`:

| Condition | Drop | Interpretation |
|---|--:|---|
| `irrelevant` (current) | −1.2 pts | Negative — costs nothing; scores *above* baseline. Task 6 the one soft spot (−14 pts on that task alone), no other task collapses |
| `semantic` (current) | −1.2 pts | Negative — no measurable cost; no task collapses (weakest: task 1 at 72%) |
| `landmark` | +3.4 pts | The one real-cost result — concentrated almost entirely in two tasks |
| `landmark_with_hardneg_prompt` | +42.8 pts (vs. baseline); +39.4 pts vs. `landmark` on the identical scene | Adding a disambiguating prompt to the `landmark` scene does not rescue the affected tasks and wrecks the rest of the suite |

### Per-task render table: `default` vs. `irrelevant` vs. `semantic` vs. `hardneg`

One row per task id, one column per Split 2 condition (episode-0 init state, post physics-settle —
see §7 — so no floating bowls). `default` = `libero_spatial` (2 bowls); `irrelevant`/`semantic` are
the **current** suites (`libero_spatial_3bowl_front`/`libero_spatial_3bowl_semantic2`, 140x140
resized — not center-cropped — from the raw 256x256 render, since a center crop cuts off
`irrelevant`'s 3rd bowl at the table's front edge); `hardneg` (`landmark`'s scene) is
`libero_spatial_3bowl_hardneg`, unchanged since first authored. All copied to
`openvla/experiments/figures/per_task_render/` since the source `LIBERO/scratch_render/<suite>/`
dirs are scratch and can be overwritten by a future pass.

| id | target | `default` | `irrelevant` | `semantic` | `hardneg` |
|--:|---|---|---|---|---|
| 0 | between the plate and the ramekin | ![](openvla/experiments/figures/per_task_render/default_t0.png) | ![](openvla/experiments/figures/per_task_render/irrelevant_v2_thumb_t0.png) | ![](openvla/experiments/figures/per_task_render/semantic_v2_thumb_t0.png) | ![](openvla/experiments/figures/per_task_render/hardneg_t0.png) |
| 1 | next to the ramekin | ![](openvla/experiments/figures/per_task_render/default_t1.png) | ![](openvla/experiments/figures/per_task_render/irrelevant_v2_thumb_t1.png) | ![](openvla/experiments/figures/per_task_render/semantic_v2_thumb_t1.png) | ![](openvla/experiments/figures/per_task_render/hardneg_t1.png) |
| 2 | table center | ![](openvla/experiments/figures/per_task_render/default_t2.png) | ![](openvla/experiments/figures/per_task_render/irrelevant_v2_thumb_t2.png) | ![](openvla/experiments/figures/per_task_render/semantic_v2_thumb_t2.png) | ![](openvla/experiments/figures/per_task_render/hardneg_t2.png) |
| 3 | on the cookie box | ![](openvla/experiments/figures/per_task_render/default_t3.png) | ![](openvla/experiments/figures/per_task_render/irrelevant_v2_thumb_t3.png) | ![](openvla/experiments/figures/per_task_render/semantic_v2_thumb_t3.png) | ![](openvla/experiments/figures/per_task_render/hardneg_t3.png) |
| 4 | in the top drawer of the wooden cabinet | ![](openvla/experiments/figures/per_task_render/default_t4.png) | ![](openvla/experiments/figures/per_task_render/irrelevant_v2_thumb_t4.png) | ![](openvla/experiments/figures/per_task_render/semantic_v2_thumb_t4.png) | ![](openvla/experiments/figures/per_task_render/hardneg_t4.png) |
| 5 | on the ramekin | ![](openvla/experiments/figures/per_task_render/default_t5.png) | ![](openvla/experiments/figures/per_task_render/irrelevant_v2_thumb_t5.png) | ![](openvla/experiments/figures/per_task_render/semantic_v2_thumb_t5.png) | ![](openvla/experiments/figures/per_task_render/hardneg_t5.png) |
| 6 | next to the cookie box | ![](openvla/experiments/figures/per_task_render/default_t6.png) | ![](openvla/experiments/figures/per_task_render/irrelevant_v2_thumb_t6.png) | ![](openvla/experiments/figures/per_task_render/semantic_v2_thumb_t6.png) | ![](openvla/experiments/figures/per_task_render/hardneg_t6.png) |
| 7 | on the stove | ![](openvla/experiments/figures/per_task_render/default_t7.png) | ![](openvla/experiments/figures/per_task_render/irrelevant_v2_thumb_t7.png) | ![](openvla/experiments/figures/per_task_render/semantic_v2_thumb_t7.png) | ![](openvla/experiments/figures/per_task_render/hardneg_t7.png) |
| 8 | next to the plate | ![](openvla/experiments/figures/per_task_render/default_t8.png) | ![](openvla/experiments/figures/per_task_render/irrelevant_v2_thumb_t8.png) | ![](openvla/experiments/figures/per_task_render/semantic_v2_thumb_t8.png) | ![](openvla/experiments/figures/per_task_render/hardneg_t8.png) |
| 9 | on the wooden cabinet | ![](openvla/experiments/figures/per_task_render/default_t9.png) | ![](openvla/experiments/figures/per_task_render/irrelevant_v2_thumb_t9.png) | ![](openvla/experiments/figures/per_task_render/semantic_v2_thumb_t9.png) | ![](openvla/experiments/figures/per_task_render/hardneg_t9.png) |

**Split's implemented conditions are fully run; only the unauthored `path` condition remains.**

### Setting: `irrelevant` (current)

**Question.** Split 2's control: a 3rd bowl placed somewhere not tied to any relational language,
to isolate whether an extra bowl's mere *presence* costs anything, independent of where it sits.
Redefined twice (see `benchmark_split_plan.md` §Split 2 for the full history) — this current
definition places bowl_3 at the literal front edge of the table (`table_front`) in every task,
instead of a per-task "least-crowded named region" pick that the first redefinition used, which
reused `next_to_ramekin_region` (another task's real target landmark) for 5/10 tasks and undercut
the goal of a placement carrying no relational meaning at all. Four tasks (1, 3, 5, 6) fall back to
`table_center` where `table_front` overlaps an existing bowl.

**2026-08-27 offset fine-tune.** Both anchor coordinates were nudged another 5cm apart, within the
`libero_spatial_3bowl_front` suite only (not the shared catalog default): `table_front` +0.05m further
toward the front edge (x: 0.19–0.21 → 0.24–0.26), used in the 6 non-fallback task files; `table_center`
0.05m further back (x: −0.10––0.05 → −0.15––0.10), used only in the 4 fallback task files (1, 3, 5,
6) — task 2's own `table_center` bowl_1 target (its "from table center" semantic anchor) was left
untouched since it's not part of this bowl_3 placement. The `table_center` fallback stays ≈0.29m from
`stove_region` (−0.42,−0.15)–(−0.40,−0.13) either way — moving it back doesn't approach the stove.

- Suite: `libero_spatial_3bowl_front`. Full per-task region table and the redefinition rationale are
  in `benchmark_split_plan.md` §Split 2 ("Second redefinition").
- Render check: `verify_suite_init_states.py` → `PASS`, worst separation 0.122 m (task 4, unrelated
  to bowl_3 — same persistent bowl_1/bowl_2 constraint present in every 3-bowl suite); every other
  task's separation improved with the wider offsets (0.148–0.301 m, vs. 0.122–0.276 m before the
  fine-tune). Contact sheet re-rendered and eyeballed — 3 distinct bowls per task, no
  overlaps/clipping, all resting flat; bowl_3 visibly farther toward the front edge (front tasks) or
  farther back (fallback tasks) than the pre-fine-tune render.

| id | target | Default (2-bowl) | Irrelevant (3-bowl) | Δ |
|--:|---|--:|--:|--:|
| 0 | between the plate and the ramekin | 92% | 88% | −4 |
| 1 | next to the ramekin | 84% | 84% | 0 |
| 2 | table center | 92% | 94% | +2 |
| 3 | on the cookie box | 84% | 96% | **+12** |
| 4 | in the top drawer | 76% | 88% | **+12** |
| 5 | on the ramekin | 94% | 86% | −8 |
| 6 | next to the cookie box | 90% | 76% | **−14** |
| 7 | on the stove | 72% | 92% | **+20** |
| 8 | next to the plate | 84% | 80% | −4 |
| 9 | on the wooden cabinet | 72% | 68% | −4 |

**Analysis.** 500/500 rollouts, overall 85.2% vs. the 84.0% baseline — Drop = −1.2 pts, essentially
free, continuing the pattern of every Split 2 condition except `landmark`. Task 6 (next to the cookie
box) is the one real soft spot (−14 pts) — notably the same task that collapsed hardest under
`center_fixed_legacy`'s confounded single-coordinate placement (90%→44%, see the design-confound note
in `benchmark_split_plan.md`); it no longer collapses, but it's still this condition's weakest point,
worth watching if a third redefinition ever touches this task. Task 7 (+20) and tasks 3/4 (+12 each)
gain the most, matching `irrelevant_v1_legacy`'s pattern of tasks 4 and 7 being the biggest gainers —
consistent gains on the same tasks across two different neutral-placement definitions suggests this
is a real property of those tasks' scenes, not placement-specific noise.

Results: `results/libero_spatial_3bowl_front--default--shard{0..3}of4.jsonl`.

### Setting: `semantic` (current)

**Question.** Places the 3rd bowl at a **named landmark that is not the target's own** — testing
whether sitting at *any* nameable relational spot pulls the policy, even when that landmark doesn't
match the prompt. Redefined once (see `benchmark_split_plan.md` §Split 2 for the full history): task
4's (in the top drawer) bowl_3 moved from `next_to_plate_region` (~0.58m from the target — outside
the 0.33-0.50m band the other 9 tasks land in, so it behaved more like a neutral placement than a
genuine semantic distractor) to `between_plate_ramekin_region` (~0.48m, task 0's real target
landmark). All 9 other tasks unchanged.

- Suite: `libero_spatial_3bowl_semantic2`. Redefinition rationale in `benchmark_split_plan.md`
  §Split 2 ("Second redefinition").
- Render check: `verify_suite_init_states.py` → `PASS`, worst separation 0.122 m (task 4;
  `next_to_box_region` was tried first and failed at 0.063 m — the open drawer's footprint collides
  with it); contact sheet eyeballed — 3 distinct bowls per task, no overlaps/clipping.

| id | target | 3rd bowl's landmark | Success | n | Δ vs `semantic_v1_legacy` |
|--:|---|---|--:|--:|--:|
| 0 | between the plate and the ramekin | next to the cookie box | 86% | 50 | 0 |
| 1 | next to the ramekin | next to the plate | 72% | 50 | 0 |
| 2 | table center | next to the ramekin | 96% | 50 | 0 |
| 3 | on the cookie box | next to the ramekin | 90% | 50 | 0 |
| 4 | in the top drawer | between the plate and the ramekin | 92% | 50 | **+4** |
| 5 | on the ramekin | next to the plate | 86% | 50 | 0 |
| 6 | next to the cookie box | next to the ramekin | 94% | 50 | 0 |
| 7 | on the stove | next to the box | 88% | 50 | 0 |
| 8 | next to the plate | next to the box | 80% | 50 | 0 |
| 9 | on the wooden cabinet | next to the ramekin | 68% | 50 | 0 |

**Analysis.** 500/500 rollouts, overall 85.2% vs. the 84.0% baseline — Drop = −1.2 pts, in line with
every Split 2 condition except `landmark`. The 9 unchanged tasks land bit-for-bit identical to
`semantic_v1_legacy`'s numbers (same seed, same scene) — a clean internal-consistency check that the
redefinition genuinely isolated task 4. Task 4 itself improved +4 pts moving its bowl_3 from the
out-of-band `next_to_plate_region` to the in-band `between_plate_ramekin_region`, a small move in the
direction of "closer semantic distractor costs slightly more," though well within noise for a single
task at n=50. Task 1 (next to the ramekin, 72%) remains this condition's weakest task, unchanged from
`semantic_v1_legacy` since task 1's scene didn't change.

Results: `results/libero_spatial_3bowl_semantic2--default--shard{0..3}of4.jsonl`.

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

**Split fully run** — both 4a's two missing cells and 4b (redesigned as a target cue-type probe)
completed 2026-08-20. See `eval_log.md` for the launch batch.

### 5.0 — What this split is testing, in plain language

Every task in this suite is "pick up the black bowl \<somewhere\> and place it on the plate." The
"\<somewhere\>" is always describable in one of three ways, depending on what's physically next to
the bowl in that task's scene:

| Relation family | What it means | Example (task id) |
|---|---|---|
| **landmark** | bowl sits *beside* another named object | "next to the ramekin" (task 1), "next to the cookie box" (task 6), "next to the plate" (task 8), "between the plate and the ramekin" (task 0) |
| **surface** | bowl sits *on top of* another named object | "on the cookie box" (task 3), "on the ramekin" (task 5), "on the stove" (task 7), "on the wooden cabinet" (task 9) |
| **region** | bowl sits in an open area with nothing nearby to name | "table center" (task 2) |

Split 4 asks the same underlying question two different ways:

- **4a asks it from the *scene* side:** does it matter, physically, whether the target bowl and the
  (unmentioned) distractor bowl each happen to be a landmark-type, surface-type, or region-type
  placement? The prompt text stays the standard "pick up the black bowl \<location\>" the whole
  time — only where the bowls actually sit in the scene changes.
- **4b asks it from the *prompt* side:** for a target bowl that is physically sitting in one place
  (say, resting on the cookie box), does it matter whether the instruction *describes* that same
  spot with surface-style words ("on the cookie box"), landmark-style words ("next to the cookie
  box"), or region-style words ("near the center of the table")? Nothing about the scene moves —
  only the sentence changes, and it's checked to still be a true description of where the bowl is.

**Bottom line (read the full analysis in §5.1/§5.2 for the numbers):** the scene-side manipulation
(4a) barely matters — success rates stay within about 8 points of each other no matter which
relation family the target or distractor happens to be. The prompt-side manipulation (4b) is the
single biggest effect measured anywhere in this project, 50–67 points of success lost by rewording
a completely true, completely unambiguous sentence about a bowl whose position never changed and
whose distractor is never mentioned. Put together, this says the checkpoint isn't reasoning about
*where things are* — it's pattern-matching the specific sentence template it saw during fine-tuning,
and breaks badly the moment that template is swapped for an equally true one it wasn't trained on.

### 5.1 — 4a: Grounding-by-scene probe (complete, all 6 cells)

4 of 6 cells reuse rollouts already collected under `spatial/default` (Split 1) — no new GPU run
needed for those, just re-aggregating existing per-task numbers by relation-family cell. The other 2
cells (`grounding/surface_landmark`, `grounding/region_surface`) are new 50-trial/task runs on the
gap-fill scenes described in §7's render-check log. Per-task `default` success rates are copied from
Split 1's `default` table (task 4, containment, is excluded from this matrix — see
`benchmark_split_plan.md` §3).

| (target, distractor) family | Tasks (id: SR) | Cell SR | Status |
|---|---|--:|---|
| (landmark, landmark) | 0: 92%, 1: 84%, 8: 84% | **86.7%** | ✅ derived from existing data |
| (region, landmark) | 2: 92% | **92.0%** | ✅ derived from existing data |
| (surface, surface) | 3: 84%, 5: 94%, 7: 72%, 9: 72% | **80.5%** | ✅ derived from existing data |
| (landmark, surface) | 6: 90% | **90.0%** | ✅ derived from existing data |
| (surface, landmark) | new suite, task 0: 88% (44/50) | **88.0%** | ✅ run |
| (region, surface) | new suite, task 0: 92% (46/50) | **92.0%** | ✅ run |

**Conclusion (now conclusive, all 6 cells run — supersedes the earlier "preliminary" read).**
Pooling cells by target family (simple mean of that family's cell SRs, same convention as the
original 4-cell read):

| Target family | Cells (SR) | Family mean |
|---|---|--:|
| landmark | (landmark,landmark)=86.7%, (landmark,surface)=90.0% | **88.35%** |
| surface | (surface,surface)=80.5%, (surface,landmark)=88.0% | **84.25%** |
| region | (region,landmark)=92.0%, (region,surface)=92.0% | **92.0%** |

The hypothesis ("landmark-target scenes are more attraction-prone than surface/region-target
scenes") is **not supported** — landmark-target (88.35%) scores *above* surface-target (84.25%),
and region-target scores highest of all three (92.0%). The 2 new cells confirm rather than
overturn the earlier 4-cell reading: whatever drives this checkpoint's failures, it isn't the
target's own relation-family under a fixed, correctly-phrased prompt. Given 4b's results below,
the far larger effect by orders of magnitude is *prompt phrasing*, not scene relation type.

### 5.2 — 4b: Target Cue-Type Probe (complete)

**Question.** Holding the scene and distractor completely fixed (distractor never mentioned), does
rephrasing *only* how the target is described — landmark ("next to X") vs. surface ("on X") vs.
region (table-zone) — change success, independent of what relation family the scene actually is?
Full design (truthfulness tiers, which tasks get which rephrasing and why) is in
`benchmark_split_plan.md` Split 4's 4b section.

**Result: this is the single largest effect measured anywhere in this project so far, and it comes
from a manipulation that never mentions a second bowl at all.**

**Concretely, what changed.** Task 5's bowl physically rests on top of the ramekin, and nothing
about the scene or robot's view differs between these three runs — only the sentence:

| Condition | Instruction given to the policy |
|---|---|
| `spatial/default` (native, "surface" phrasing) | "pick up the black bowl on the ramekin and place it on the plate" |
| `target_cue_landmark` | "pick up the black bowl next to the ramekin and place it on the plate" |
| `target_cue_region` | "pick up the black bowl at the back-left of the table and place it on the plate" |

All three sentences are true descriptions of the exact same bowl in the exact same spot. Success
on this one task went 94% → 10% → 28% across those three sentences respectively — see the
per-task table below for every task's exact wording pair (full instruction dicts:
`LIBERO_SPATIAL_TARGET_CUE_REGION_INSTRUCTIONS` / `LIBERO_SPATIAL_TARGET_CUE_LANDMARK_INSTRUCTIONS`
in `openvla/experiments/robot/libero/instructions.py`).

| Condition | Tasks | Overall SR | Rollouts |
|---|---|--:|--:|
| `target_cue_region` | 0,1,3,5,6,7,8,9 (8 tasks) | **17.0%** | 400/400 |
| `target_cue_landmark` | 3,5,7,9 (4 tasks) | **30.5%** | 200/200 |

Per-task, against the `spatial/default` baseline for the same task:

| id | target (native phrasing) | Default SR | `target_cue_region` | Δ | `target_cue_landmark` | Δ |
|--:|---|--:|--:|--:|--:|--:|
| 0 | between the plate and the ramekin (landmark) | 92% | 58% | −34 | — | — |
| 1 | next to the ramekin (landmark) | 84% | 10% | −74 | — | — |
| 3 | on the cookie box (surface) | 84% | 28% | −56 | 52% | −32 |
| 5 | on the ramekin (surface) | 94% | 28% | −66 | 10% | **−84** |
| 6 | next to the cookie box (landmark) | 90% | 2% | **−88** | — | — |
| 7 | on the stove (surface) | 72% | 0% | −72 | 44% | −28 |
| 8 | next to the plate (landmark) | 84% | 10% | −74 | — | — |
| 9 | on the wooden cabinet (surface) | 72% | 0% | −72 | 16% | −56 |

**Metrics** (SR(default) restricted to the same task subset each condition covers, matching
`benchmark_split_plan.md`'s formulas):

```
Region-cue Drop   = SR(default, 8 tasks: 84.0%) − SR(target_cue_region: 17.0%)   = 67.0 pts
Landmark-cue Drop = SR(default, 4 tasks: 80.5%) − SR(target_cue_landmark: 30.5%) = 50.0 pts

Region-cue Drop, landmark-family pool {0,1,6,8}: 87.5% → 20.0%  = 67.5 pts
Region-cue Drop, surface-family pool  {3,5,7,9}: 80.5% → 14.0%  = 66.5 pts
```

**Analysis.** Three findings, in order of how surprising they are:

1. **The effect size is enormous** — 50 to 67 points overall, with several tasks collapsing to
   0-10%. This is *larger* than Split 1's `negative_contrast`/`positive_contrast` drops (47.2 /
   51.6 pts), the previous largest effect in the project — and 4b's prompts never reference a
   second bowl or location at all. Purely restating the *same true fact about the same bowl* in a
   different relation-family's words is enough to devastate the policy.
2. **No landmark-vs-surface asymmetry in the region-cue condition** — landmark-family (67.5 pt
   drop) and surface-family (66.5 pt drop) are damaged almost identically when forced into
   region-style phrasing. This rules out the original hypothesis's language-side analogue
   (landmark language being harder to compute than surface language) — the damage tracks
   *phrasing-template mismatch*, not relation-type difficulty.
3. **Landmark-cue (the "approximate/disclosed" relaxation) hurts less than region-cue for the same
   4 surface-family tasks** (50.0 vs. 66.5 pts) — describing a bowl resting on X as "next to X"
   survives better than describing it by table position. Plausibly because "next to X" is at least
   *structurally* the phrasing template the checkpoint was fine-tuned on for other tasks (landmark
   phrasing appears natively in 4/10 training tasks), whereas region-style phrasing
   ("at the back-left of the table") never appears in any of the 10 original `libero_spatial`
   prompts in any form.

Together with 4a's conclusion above, this reframes the whole split's headline finding: **the
policy's apparent "grounding" is templated surface-pattern matching on the exact phrasing structure
seen at fine-tuning time, not a semantic understanding of landmark/surface/region relations** — the
scene's actual relation family barely matters (4a, ≤8 pt spread across all 6 cells) while the
prompt's relation *wording* matters enormously (4b, 50-67 pt drops) even with the referenced fact
held perfectly true and the distractor never mentioned.

Results: `results/libero_spatial--target_cue_region--shard{0,1}of2.jsonl`,
`results/libero_spatial--target_cue_landmark--shard{0,1}of2.jsonl`,
`results/libero_spatial_grounding_surface_landmark--default--shard0of2.jsonl`,
`results/libero_spatial_grounding_region_surface--default--shard0of2.jsonl`.

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
4. **Split 4a is complete (all 6 cells) and confirms the landmark-attraction hypothesis is wrong,
   not just under-tested.** Target-family-pooled SR: landmark 88.35%, surface 84.25%, region 92.0%
   — landmark-target is *not* lower than surface-target, and region-target scores highest of all
   three. The 2 new-scene cells ((surface,landmark)=88.0%, (region,surface)=92.0%) landed close to
   their family's existing cells rather than overturning the pattern — see §5.1.
5. **The negation clause wasn't the source of the damage.** Bare mention of the distractor's
   location, with no "not the one…" clause, is just as bad (-51.6 pts) as mentioning + negating it
   (-47.2 pts) — `positive_contrast` (32.4%) even scores slightly *below* `negative_contrast`
   (36.8%). The policy has no practice grounding a second referent at all; negation specifically
   isn't the issue.
6. **Split 2's first real result cut against its own hypothesis, and the redefined `semantic` confirms
   it.** A distractor placed at an unrelated named landmark costs ~0 pts either way: `semantic_v1_legacy`
   84.8% (+0.8 vs. baseline), current `semantic` 85.2% (+1.2 vs. baseline) — both well inside noise.
   `semantic_v1_legacy`'s task 4 had sat ~0.58m from the target, outside the 0.33-0.50m band the other
   9 tasks land in (likely behaving like a neutral placement rather than a genuine semantic distractor
   for that one task); the redefined `semantic` (`libero_spatial_3bowl_semantic2`, run 2026-08-27,
   500/500) moved it to ~0.48m, in-band, and the 9 unchanged tasks landed bit-for-bit identical while
   task 4 ticked up +4 pts — a properly in-band semantic distractor costs nothing either. Combined with
   finding 5, the pattern holds: failures track *what the prompt says*, not *what's on the table*.
7. **The harder version of the test — `landmark` — does show a real effect, but a concentrated
   one.** A distractor near the target's *own* landmark costs -3.4 pts overall (80.6% vs. 84.0%),
   similar in size to finding 2's plain extra bowl. But like finding 2, that headline number hides
   a task-specific collapse: task 0 (92%→48%) and task 9 (72%→54%) account for nearly all of it,
   every other task flat or improved. Revised picture: scene changes barely matter *in general*,
   but a distractor placed to be a genuine near-miss for the target's own landmark can badly hurt
   specific tasks — proximity-to-own-landmark is the one scene manipulation so far with a real,
   attributable (if concentrated) cost.
8. **Split 2's three-way distractor-position comparison is complete, and the pattern holds under both
   the original and the redefined `irrelevant`/`semantic`.** Current `irrelevant` (+1.2 pts) and
   `semantic` (+1.2 pts) — as well as their retired `irrelevant_v1_legacy` (+4.8 pts) and
   `semantic_v1_legacy` (+0.8 pts) predecessors — all cost nothing; if anything they trend positive
   with no overall-task collapse (`irrelevant`'s task 6 is the one notable single-task soft spot,
   -14 pts, but the suite as a whole doesn't collapse). Only `landmark` (-3.4 pts, concentrated in 2
   tasks) shows a real effect. Combined with findings 5-6, the picture across this entire project so
   far: failures
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
10. **The single largest effect in the whole project: rephrasing the target's cue type — with the
    distractor never mentioned — costs 50 to 67 pts.** Split 4b held the scene, distractor, and the
    referenced fact's truth completely fixed, and varied only whether the target was described via
    its native relation-family wording or an alternate one (region-zone or a disclosed-approximate
    landmark phrasing). `target_cue_region` (8 tasks): 84.0%→17.0% (−67.0 pts). `target_cue_landmark`
    (4 surface-family tasks): 80.5%→30.5% (−50.0 pts). This *exceeds* finding 1's negative-contrast
    damage (−47.2 pts) despite never introducing a second referent — see §5.2.
11. **Findings 1-9 already showed language dominates scene; finding 10 shows it's not really about
    "a second referent" at all — it's template matching on exact phrasing.** Region-cue damage is
    statistically indistinguishable between landmark-family targets (−67.5 pts) and surface-family
    targets (−66.5 pts) — ruling out relation-type difficulty as the mechanism. And landmark-cue
    phrasing (present in 4/10 training tasks' native language) hurts less than region-cue phrasing
    (present in none) on the identical 4 tasks (−50.0 vs. −66.5 pts). Combined with finding 9 (a
    disambiguating prompt spreads damage to unaffected tasks rather than fixing the hard ones), the
    project's overall picture is now: this checkpoint's "grounding" is narrow surface-pattern
    matching against the exact phrasing templates seen at fine-tuning time — not scene-relation
    reasoning (4a, ≤8 pt spread) and not really referent-counting either (a single bowl, truthfully
    described in unfamiliar words, is nearly as damaging as describing two).
12. **Findings 1-11 diagnose failure patterns from success-rate deltas alone, and that turns out to
    be the only tool available.** A direct attempt to separate grounding from action decoding — ask
    the checkpoint, via VQA, which bowl it thinks the target is instead of asking it to act — hit a
    structural dead end confirmed 3 independent ways (see §8): both the fine-tuned checkpoint and the
    unmodified base `openvla-7b` unconditionally emit action-bin tokens after the eval prompt
    template regardless of what's asked (even a content-free control question), and a
    generation-free restricted-logit comparison showed no ranking change across real vs. unrelated
    vs. mismatched instructions for the same image. This is a consequence of OpenVLA's action-only
    continued-pretraining recipe, not something specific to distractor-mention phrasing — but it
    means every interpretation above (11 in particular: "template matching, not real grounding")
    rests on outcome-level evidence and cannot currently be corroborated by directly interrogating
    the model's internal grounding.
13. **The §8 dead end's deferred alternative was pursued (§8.1) and didn't rescue the interpretation
    either way.** A genuinely separate general-purpose VLM (`Qwen2-VL-7B-Instruct`, never trained on
    OpenVLA's action-only template) shown the identical numbered-bowl images/instructions scored
    40-60% against 33-50% chance baselines across all 3 conditions — not convincingly above chance —
    and its raw answers show a numeric/positional bias (zero "1" answers across all 10 `hardneg`
    queries; a 7/10 skew toward "2" on `positive_contrast`) rather than instruction-tracking, even
    though the same model *does* change its answer on 5/10 tasks when only the prompt (not the image)
    differs between `negative_contrast` and `positive_contrast`. So this project's probe rendering
    itself may be too weak a stimulus to cleanly separate "is this language ambiguous" from "is this
    model's grounding broken" — finding 12's caveat (findings 1-11 rest on outcome-level evidence
    alone) still stands, now for a different reason: the direct-interrogation approach hit a second,
    independent dead end even switching model families.
14. **Finding 13 was itself a rendering artifact — the probe's marker placement had a real bug, and
    fixing it flipped the §8.1 conclusion.** User-driven inspection of the probe images found the
    markers were both too large (occluding the target bowl) and, for at least one bowl per scene,
    projected up to ~46px off the bowl's true rendered position (confirmed against MuJoCo's own
    segmentation render as ground truth — see §8.2). After fixing marker placement (segmentation-based
    centroid instead of a hand-projected 3D→2D transform) and shrinking the markers so they no longer
    cover the bowl, re-running the identical Qwen2-VL battery scored 70% on all 3 conditions — clearly
    above chance, versus 40-60% (not convincingly above chance) before the fix. The `negative_contrast`
    /`positive_contrast` phrasing-sensitivity result in finding 13 also reversed: the two conditions
    now agree on 10/10 tasks (previously 5/10 disagreed), meaning that "phrasing sensitivity" was very
    likely the model reacting to noisy markers, not real wording sensitivity. Net effect: §8.1's
    "language may be ambiguous even to a capable VLM" alternative is weaker than it looked — a capable
    VLM resolves the referring expression fine once the stimulus itself isn't broken — but this still
    doesn't corroborate findings 1-11 about OpenVLA specifically (finding 12's dead end for that model
    family is untouched by this fix).
15. **Adding a `default` (no-distractor-mention) baseline to the corrected bowl-pointing probe (§8.3)
    shows distractor-mention phrasing is not inherently harder to ground — for Qwen it's easier.**
    `default` (LIBERO's own target-only task language, distractor never mentioned) scored 5/10 (50%,
    exactly chance) on `libero_spatial`'s 2-bowl scenes, versus 7/10 (70%) for both
    `negative_contrast` and `positive_contrast` on the identical images — the two distractor-mention
    phrasings each correctly resolve 2 tasks (both target=bowl "1") that the target-only phrasing gets
    wrong. So the distractor mention itself is, if anything, a disambiguating cue for a capable VLM,
    not a source of difficulty. This narrows the "maybe the phrasing is just inherently harder to
    ground" reading of findings 1/5/9/10 further than finding 14 already did — it's specifically
    OpenVLA's behavior under that phrasing that's unexplained by the language itself being hard,
    since a different, capable model finds the *same* phrasing easier than no phrasing at all.
16. **The same no-mention-vs-mention comparison holds at 3 bowls, but the gap shrinks and the scene
    itself clears chance easily either way.** `hardneg_default` (3-bowl scene, distractor family never
    mentioned) scored 6/10 (60%, comfortably above the 33% chance floor) versus `hardneg`'s 7/10
    (70%) — the two conditions disagree on exactly 1 of 10 tasks (task id 5, where the mention flips a
    wrong guess to right), a much smaller mention-effect than the 2-bowl comparison's 2/10 (finding
    15). Task id 0 is the single hardest case across the entire 5-condition, 50-query battery — wrong
    regardless of scene or phrasing — while task id 2 is wrong only in the two 3-bowl conditions,
    isolating the extra distractor bowl (not wording) as that task's specific difficulty. Combined with
    finding 15: a capable VLM's accuracy on this referring expression is barely dented by adding a
    third bowl (60% vs. 50%, no-mention baselines) and is never *hurt* by naming the distractor family,
    in either scene — a further data point against reading OpenVLA's findings 1-11 collapse as evidence
    that the language or clutter itself is intrinsically hard to ground.

## 7. Render / contact-sheet check log

Every new or previously-unchecked scene gets its init-state contact sheet rendered
(`LIBERO/scripts/render_suite_contact_sheet.py <suite>`) and eyeballed before spending GPU time on
it. Where a run adds a distractor to the 2-bowl baseline, a side-by-side comparison is also
rendered (`LIBERO/scripts/compare_two_suites_init.py libero_spatial <suite> <outname>` — left/blue
= 2-bowl baseline, right/orange = the 3-bowl variant, one row per task id 0–9) and saved under
`openvla/experiments/figures/` for permanent reference (the raw per-suite contact sheets under
`LIBERO/scratch_render/` are scratch and get overwritten each pass). The figures themselves are
embedded inline in each condition's section above (§3-4).

**2026-08-26 render-settling fix.** `render_suite_contact_sheet.py` rendered immediately after
`set_init_state()` with no settle steps, so some per-task renders (including cells in §3's per-task
render table) showed bowls still mid-fall/floating from their sampled init height instead of resting
on the surface — `compare_two_suites_init.py` never had this problem since it already stepped a
dummy action first. Both scripts now settle for 10 steps with the same no-op action
`run_libero_eval.py` uses via `cfg.num_steps_wait`, so every render reflects what the real eval
actually sees. Regenerated: the 4 suites feeding §3's per-task render table (`libero_spatial`,
`libero_spatial_3bowl_neutral`, `libero_spatial_3bowl_semantic`, `libero_spatial_3bowl_hardneg`) and
their thumbnails; not regenerated (unaffected — already had a settle step, unchanged appearance at
10 vs. the prior 12 steps): the five `compare_*_grid.png` figures embedded in §3-4.

**2026-08-27 follow-up.** `libero_spatial_3bowl_front`/`libero_spatial_3bowl_semantic2` missed the
2026-08-26 pass above (authored the same day, but their contact sheets had been captured via
`gen_suite_init_states.py`'s own inline preview, which has no settle step at all — same root cause,
different script). Caught when the `libero_spatial_3bowl_front` grid was eyeballed again and a bowl
was visibly still falling in one panel. Re-rendered both with `render_suite_contact_sheet.py`
(read-only over the already-generated/verified `.pruned_init` files, so `verify_suite_init_states.py`'s
PASS results below are unaffected — this only redid the preview image, not the init states); both
now show every bowl resting flat with contact shadows.

| Suite | Numeric verify (`verify_suite_init_states.py`) | Visual eyeball | Result |
|---|---|---|---|
| `libero_spatial_3bowl_neutral` (`irrelevant_v1_legacy`) | PASS, worst sep 0.122m | ✅ | 3 distinct bowls per task, no overlaps |
| `libero_spatial_3bowl_semantic` (`semantic_v1_legacy`) | verified | ✅ | 3 distinct bowls per task, no overlaps/clipping, drawer open only on task 4 (expected — task 4's target lives in the drawer) |
| `libero_spatial_3bowl_front` (`irrelevant`, current, 2026-08-26) | First pass FAIL (`table_front` alone) — worst sep 0.107m, tasks 3 & 5 overlap. Fixed by falling back tasks 1, 3, 5, 6 to `table_center`; re-verify PASS, worst sep 0.122m | ✅ (re-eyeballed 2026-08-27 after the settle-timing fix above) | 3 distinct bowls per task, no overlaps/clipping, every bowl resting flat with a contact shadow (first-pass grid had one still mid-fall); 3rd bowl visibly isolated near the table's front edge in every panel |
| `libero_spatial_3bowl_front` offset fine-tune (2026-08-27) | `table_front` +5cm front / `table_center` fallback +5cm back (still ≈0.29m from `stove_region`); re-verify PASS, worst sep 0.122m (task 4, unchanged, unrelated to bowl_3), all other tasks 0.148–0.301m (up from 0.122–0.276m) | ✅ | Bowl_3 visibly farther toward the front edge in the 6 front tasks, farther back (away from the cluster) in the 4 fallback tasks; still resting flat, no overlaps/clipping |
| `libero_spatial_3bowl_semantic2` (`semantic`, current, 2026-08-26) | First pass FAIL (task 4 at `next_to_box_region`) — worst sep 0.063m, open drawer's 3D footprint collides. Fixed by moving task 4 to `between_plate_ramekin_region`; re-verify PASS, worst sep 0.122m | ✅ (re-eyeballed 2026-08-27 after the settle-timing fix above) | 3 distinct bowls per task, no overlaps/clipping, every bowl resting flat with a contact shadow; task 4's 3rd bowl visibly near the plate/ramekin cluster instead of the far corner |
| `libero_spatial_3bowl_hardneg` (`landmark` / `landmark_with_hardneg_prompt`, same scene) | PASS but narrow — min sep 0.121m vs. 0.12m threshold, task 3 tightest | ✅ | 3 distinct bowls per task; task 3's close pair confirmed as two separate bowls, not merged |
| `libero_spatial_grounding_surface_landmark` (`grounding/surface_landmark`) | PASS, sep 0.393m | ✅ | Single task (`on_the_ramekin`). Cross-checked exact xyz against the BDDL, not just pixels: target `akita_black_bowl_1` = (-0.210, 0.192, z=1.080) — inside `ramekin_region` (-0.21,0.19)-(-0.19,0.21), elevated (on top of the ramekin, as intended). Distractor `akita_black_bowl_2` = (0.116, -0.067, z=0.970) — inside `next_to_box_region` (0.12,-0.08)-(0.14,-0.06) (0.004m outside on x, negligible), flat on the table (not elevated) — confirms it moved off `cookies_1` (was elevated/surface in the original task 5) to a landmark placement, as designed. No overlap/clipping. |
| `libero_spatial_grounding_region_surface` (`grounding/region_surface`) | PASS, sep 0.210m | ✅ | Single task (`from_table_center`). Target `akita_black_bowl_1` = (-0.075, 0.003, z=0.970) — inside `table_center` (-0.10,-0.01)-(-0.05,0.01), flat on table (region cue, as intended). Distractor `akita_black_bowl_2` = (-0.263, -0.137, z=1.010) — y matches `stove_region`'s -0.14 almost exactly, x offset from the stove's base anchor (-0.41) is consistent with `flat_stove_1_cook_region` being the stove's own top surface (same region used by tasks 6/9's distractors elsewhere in the suite), elevated (on top of the stove) — confirms surface placement, moved off `next_to_plate_region` (landmark in the original task 2). No overlap/clipping. |

`libero_spatial` (baseline) and `libero_spatial_3bowl`/`libero_spatial_3bowl_open`
(`center_fixed_legacy`/`drawer_open`) predate this render-before-run practice being tracked here;
no issues have surfaced in their data, but no dedicated check is logged. The two grounding suites
above were checked here before running — both have since been run, see §5.1.

Note on method: for these two single-task suites, plain pixel-diffing the new render against the
original `libero_spatial` task's render (t5 for surface_landmark, t2 for region_surface) was tried
first and was inconclusive — moving one bowl shifts shadows/specular highlights across a large
fraction of a 256x256 frame, so a large diff bounding box doesn't distinguish "one bowl moved" from
"something is wrong." Reading the actual simulator joint `qpos` for each bowl (as tabulated above)
and comparing against the BDDL region catalog is unambiguous and is the more reliable check for a
single-object scene change — recommended over pixel-diffing for any future single-task gap-fill
suite in this project.

## 8. VLM Bowl-Pointing Probe — grounding-vs-action-decoding diagnostic (2026-08-25)

**Question.** Every distractor-mention condition (`negative_contrast`, `positive_contrast`,
`landmark_with_hardneg_prompt`) collapses this checkpoint's task success rate, but end-to-end
success can't say *why*: is vision-language grounding itself broken once a second referent is
mentioned, or is grounding fine and only the action-decoding head falls apart on out-of-distribution
phrasing? This probe tried to isolate the two by showing the model the same scene with each black
bowl overlaid with a random number (Set-of-Mark style) and asking it, in free text, which numbered
bowl matches the (unmodified) failing instruction — swapping "output an action" for "output a
number" while holding scene and language fixed. Ground truth: every task's BDDL goal predicate names
`akita_black_bowl_1` as the target, invariant across all scene variants used here.

**Method.** New standalone script `openvla/experiments/robot/libero/probe_bowl_pointing.py` (not a
benchmark split — it's a VQA probe, not an action rollout, so it isn't wired into
`eval_registry.py`). For a given task: render the exact episode-0 init state the real eval would see
(same settle-step count/dummy action as `run_libero_eval.py`), project each bowl's true 3D position
into the rendered frame via robosuite `camera_utils`, overlay a shuffled marker number per bowl, then
call the checkpoint's `.generate()` directly (bypassing `predict_action()`'s action-token-only
decoding) with the condition's real instruction text reformatted as "which numbered bowl...".

**Result: the probe is not viable on this model family — confirmed three ways, smoke-tested on 2
tasks before committing to a full run.**

1. **Free-text generation on the fine-tuned checkpoint always returns action-bin tokens, regardless
   of the question.** Both a real bowl-pointing query and a content-free control ("What is 2+2?")
   decoded to garbage text (e.g. `'論˚塔▓貴军忠'`). Inspecting the raw token ids showed they fall in
   `[31744, 31999]` — exactly the tail-of-vocabulary range (`vocab_size=32000`) this checkpoint
   reserves for its 256 action bins. The model isn't answering badly; it isn't answering at all —
   it unconditionally emits action tokens after the `"...Out:"` + empty-token position no matter
   what precedes it.
2. **The base, pre-LIBERO-finetune `openvla/openvla-7b` does the same thing.** Downloaded fresh and
   ran the identical free-text check — same action-bin-range token ids for both the real and control
   prompts. This rules out "this project's LoRA fine-tune broke it": OpenVLA is continue-pretrained
   from Prismatic-VLM exclusively on the template `"In: What action should the robot take to
   {instruction}?\nOut:"` → action tokens across all of Open X-Embodiment, which appears to collapse
   the model's conditional distribution at that template's completion point onto the action-token
   subspace for *any* input. This is structural to the OpenVLA checkpoint family under this prompt
   template, not a symptom of the distractor-mention failures being investigated.
3. **A restricted-logit comparison (bypassing generation entirely) found no language-conditioned
   signal either.** Single forward pass per query, comparing raw next-token logits for just the
   candidate digit tokens ("1"/"2"/"3", ids 29896/29906/29941) instead of letting the model generate
   freely. Tested per image against three prompt variants: the real matching instruction, an
   unrelated question, and the *other* task's mismatched instruction. The ranking among the three
   candidates was **identical across all three prompt variants for a given image** (e.g. target-1
   image → `['3','2','1']` every time; target-2 image → `['3','1','2']` every time) — the ranking
   tracks only *which image*, never *what was asked*. Whatever tiny logit gap exists here isn't
   responsive to language at all, so it can't be read as a grounding signal.

**Conclusion.** OpenVLA (base or LIBERO-finetuned) has no text-output channel that's causally
responsive to language input at the point this diagnostic needs to query it — a consequence of its
action-only continued-pretraining recipe, not something specific to this project's distractor-mention
conditions. Grounding cannot be separated from action decoding via a VQA-style probe on this model
family; end-to-end task success remains the only measurable signal for this checkpoint. Given the
dead end was confirmed on 3 independent angles (2 checkpoints x free-text, plus restricted-logit), the
full 30-query battery (all 3 conditions x 10 tasks) was not run — it would only reproduce the same
negative result at 15x the cost. The script itself (`probe_bowl_pointing.py`) is left in place and
functions correctly end-to-end (rendering, 3D→pixel projection, marker overlay, structured
JSONL/figure output all verified working) — it would be directly reusable if a future checkpoint
without this action-token collapse becomes available to test, or repurposed for the "genuinely
separate general-purpose VLM" alternative noted below.

Two smoke-test example images (annotated, as fed to the model, with different target numbers to
confirm the marker-shuffle logic): `openvla/experiments/figures/probe_bowl_pointing/`.

**Open alternative (not pursued, deferred pending decision).** A genuinely separate general-purpose
VLM — one never trained on OpenVLA's action-only template — could still be shown the same
numbered-bowl images and instructions to test whether the referring expression is resolvable *in
principle*. This answers a different question than originally posed (not "does this checkpoint's
grounding survive independently of decoding" but "is the language itself unambiguous to a capable
VLM") and requires standing up a new model dependency — deferred rather than pursued opportunistically.

### 8.1 Qwen2-VL-7B-Instruct run (2026-08-25) — the alternative above, pursued

**Superseded by §8.2 (2026-08-26).** The rendering used for every image in this section had a
marker-placement bug — see §8.2 for the fix and a corrected re-run. Left in place as the historical
record of what was actually run and concluded at the time; do not use the accuracy numbers below as
current.

**Setup.** `openvla/experiments/robot/libero/probe_bowl_pointing_qwen.py` (new, uncommitted in the
`openvla` fork) reuses `bowl_pointing_common.py`'s identical render/annotate/score pipeline —
same numbered-bowl images, same instructions, same ground truth — but queries
`Qwen/Qwen2-VL-7B-Instruct` via ordinary free-text `.generate()` instead of the OpenVLA checkpoint.
Needed `transformers==4.51.3` + `qwen-vl-utils` installed ephemerally over the eval image's pinned
`4.40.1` (an open-ended `>=4.49` pulls today's `5.15.1` instead, which removed
`AutoModelForVision2Seq` and broke an unrelated import `bowl_pointing_common.py` pulls in
transitively through `libero_utils.py` → `robot_utils.py` → `openvla_utils.py`, despite never
calling into OpenVLA code). Full battery: 3 conditions x 10 tasks = 30 queries, no parse failures.

**Result: not resolvable in principle either — Qwen's answers track a numeric/positional bias, not the instruction.**

| Condition | Scene | Accuracy | Chance level |
|---|---|---|---|
| `negative_contrast` | `libero_spatial` (2 bowls) | 6/10 (60%) | 50% |
| `positive_contrast` | `libero_spatial` (2 bowls) | 4/10 (40%) | 50% |
| `hardneg` | `libero_spatial_3bowl_hardneg` (3 bowls) | 6/10 (60%) | 33% |

None of these clear chance convincingly, and the raw answer distributions explain why:

- **`hardneg`: Qwen answered "1" zero times across all 10 queries**, regardless of where the target
  actually was (target was bowl "1" on task 0, which it got wrong) — parsed answers were six "2"s and
  four "3"s only. A pure numeric/positional preference, not target-tracking.
- **`positive_contrast`: 7/10 answers were "2"**, vs. an even 5/5 split for `negative_contrast` on the
  *same 10 images* (render is cached per `(suite, task_id)`, so `negative_contrast` and
  `positive_contrast` show Qwen the identical picture per task — only the instruction wording
  differs). So the model **is** sensitive to the prompt (5/10 tasks flip answer between the two
  conditions on an unchanged image) — it's just not sensitive in a way that tracks correctness.

**Conclusion.** The bowl-pointing referring expression is not cleanly resolvable even by a capable,
independently-trained general-purpose VLM under this rendering (small overhead markers on a
256x256 clip, single frame, no interaction) — so the OpenVLA failure pattern documented in
findings 1, 5, 9, 10 (finding 12) cannot be presumed to reflect *unambiguous* language that OpenVLA
alone fails to ground. This doesn't rescue OpenVLA's action-space failures (Qwen isn't asked to act,
and OpenVLA's action-only decoding collapse in §8 is a separate, model-family-specific problem) — it
narrows what §8's dead end was blocking: even with a working text channel, this probe's rendering
may be too weak a stimulus (marker size/contrast, single static frame, no zoom) to isolate language
grounding cleanly, independent of which model answers it. A follow-up would need to strengthen the
stimulus (larger/higher-contrast markers, multiple viewpoints) before concluding anything about
grounding difficulty from accuracy numbers alone.

Annotated images (all 30, overwriting the two pre-existing OpenVLA smoke-test images at the same
filenames — harmless, since the annotation only depends on scene geometry, not which model is being
probed): `openvla/experiments/figures/probe_bowl_pointing/`. Structured output:
`openvla/experiments/logs/probe_bowl_pointing_qwen/probe_bowl_pointing_qwen.jsonl` (gitignored,
local only).

### 8.2 Marker-placement bug found and fixed — Qwen re-run (2026-08-26)

**Trigger.** User inspection of the §8.1 gallery flagged two problems by eye: the filled marker
circles (radius 15px on a 224px image) were large enough to fully occlude the target bowl, and some
markers looked like they weren't centered on their bowl at all. The initial hypothesis was that
`num_steps_wait=10` doesn't give bowls (which spawn slightly above the table and fall) enough time
to settle before the frame used for marker projection is captured.

**That specific hypothesis was ruled out.** Re-rendering `libero_spatial` task 0 with
`num_steps_wait` swept from 0 to 120 and logging each bowl's z-height every step showed physics
fully stable by step 5 (z stops changing entirely); the marker position computed at step 10 is
pixel-identical to step 120. Settle timing was not the mechanism.

**But a real, independent bug was confirmed anyway.** `render_and_annotate()` (in
`bowl_pointing_common.py`) computed marker pixel positions by hand-projecting each bowl's 3D
`sim.data.body_xpos` through `robosuite.utils.camera_utils.project_points_from_world_to_camera()`.
Cross-checking that projection against MuJoCo's own segmentation render (`sim.render(...,
segmentation=True)` — which pixels the renderer itself assigned to each body's geoms, so it can't be
wrong about where an object actually drew) at task 0, step 120:

| Object | Hand-projected pixel (224-space) | Segmentation ground truth | Error |
|---|---|---|---|
| `akita_black_bowl_2` | (33, 106) | (31, 114) | ~9 px |
| `plate_1` | — | — | ~3 px |
| **`akita_black_bowl_1`** | **(57, 84)** | **(56, 130)** | **~46 px (~20% of the frame)** |

`akita_black_bowl_1`'s hand-projected marker landed squarely on bare table — querying the
segmentation mask at that exact pixel returned body `"table"`, not the bowl — while the true bowl
sat 46px away. Single-point vs. batched projection gave identical (wrong) output, ruling out a
batching bug; `body_xpos`, `geom_xpos`, and `qpos` for the two bowls were internally consistent with
each other (same asset, same quaternion convention, both matched their own `qpos`); the flip
convention was verified by applying the identical `[::-1, ::-1]` transform to both the image and the
segmentation mask before comparing. The discrepancy is isolated to
`project_points_from_world_to_camera()`'s output for this specific 3D point vs. this camera
transform — root mechanism not identified, but not needed, since ground truth from segmentation is
trustworthy by construction (same draw call as the RGB frame).

**Fix (`bowl_pointing_common.py`, uncommitted in the `openvla` fork as of this write-up).** Replaced
the hand-projection with `_bowl_pixel_centroid()`: reads back each bowl's segmentation mask
directly and uses its pixel centroid as the marker position, sidestepping
`project_points_from_world_to_camera()` entirely. Also fixed the occlusion problem: markers changed
from filled `radius=15` circles to `radius=6` outline dots with a crosshair, with the number label
offset above-right of the dot (white-haloed for legibility) instead of drawn inside a filled shape
covering the bowl.

**Qwen2-VL-7B-Instruct re-run, full battery (3 conditions x 10 tasks), identical model/prompts/scoring, only the images changed:**

| Condition | Before fix (§8.1) | After fix | Chance level |
|---|---|---|---|
| `negative_contrast` | 6/10 (60%) | **7/10 (70%)** | 50% |
| `positive_contrast` | 4/10 (40%) | **7/10 (70%)** | 50% |
| `hardneg` | 6/10 (60%) | **7/10 (70%)** | 33% |

All three conditions now clear their chance baseline clearly, where before none did convincingly.
`positive_contrast` shows the largest correction (40% → 70%) — it was previously *below* chance.

**A second finding changed along with the headline number.** Before the fix, `negative_contrast` and
`positive_contrast` (identical images, only instruction wording differs) disagreed on 5/10 tasks —
read at the time as "Qwen is prompt-sensitive but not in a way that tracks correctness" (§8.1). After
the fix, **the two conditions agree on all 10/10 tasks** (identical target, identical parsed answer,
identical correctness per task id). With clean markers, Qwen's answer is invariant to which of the
two phrasings it's given — the earlier phrasing-sensitivity finding was very likely an artifact of
the model reacting to noisy/ambiguous marker placement, not a genuine wording effect. Similarly, the
old "Qwen never answered '1' across all 10 `hardneg` queries" finding (read as a pure
numeric/positional bias) softens: post-fix it answers "1" once (task 1, still incorrect there, but no
longer a hard zero) — consistent with the fix removing a source of noise rather than a source of
genuine task difficulty.

**Revised conclusion.** With accurate, non-occluding markers, the bowl-pointing referring expression
IS resolvable well above chance by a capable general-purpose VLM. §8.1's "not resolvable in
principle" conclusion was itself a probe-rendering artifact, not evidence about the language. This
reopens (but does not resolve) the question §8.1 was trying to close: findings 1-11's outcome-level
interpretation (finding 12's caveat) still cannot be corroborated by directly interrogating OpenVLA's
internal grounding (§8's action-token-collapse dead end is untouched by this fix — it's a separate,
model-family-specific problem), but the *"maybe the language itself is just ambiguous"* alternative
raised by §8.1 is now weaker than it looked: a capable VLM resolves it fine (70%) once the stimulus
itself isn't broken.

Corrected images (same filenames, overwritten in place):
`openvla/experiments/figures/probe_bowl_pointing/`. Re-run structured output (same path/filename as
§8.1, overwritten): `openvla/experiments/logs/probe_bowl_pointing_qwen/probe_bowl_pointing_qwen.jsonl`
(gitignored, local only). Code fix: `openvla/experiments/robot/libero/bowl_pointing_common.py`
(uncommitted in the `openvla` fork as of this write-up — commit separately there per repo
convention).

### 8.3 `default` (no-distractor-mention) baselines added, 2-bowl and 3-bowl (2026-08-26)

**Motivation.** §8.1/§8.2 only ever asked Qwen to resolve the referring expression under
distractor-mention phrasing (`negative_contrast`, `positive_contrast`, `hardneg` — every instruction
either negates or names the distractor). Missing: how does Qwen do on the *same* 2-bowl scenes under
LIBERO's own native task language, which describes only the target and never mentions the distractor
at all (this is Split 1's `default` condition, `eval_registry.CONDITIONS["default"] = None` — "use
the task's own language"). This is the natural comparison point for the distractor-mention
conditions, so it was added to the probe.

**Method.** Added `"default": ("libero_spatial", None)` and, after a follow-up request to extend the
same comparison to the 3-bowl scene, `"hardneg_default": ("libero_spatial_3bowl_hardneg", None)` to
`bowl_pointing_common.CONDITION_SUITES`. `render_and_annotate()` now also returns LIBERO's native
`task.language` string (previously discarded); when a condition's instruction dict is `None`, both
probe scripts use that string instead of a custom phrasing — same convention
`eval_registry.CONDITIONS` already uses for real evals (confirmed `task.language` is target-only and
identical in form across both suites — the 3-bowl scene's extra distractor doesn't change LIBERO's own
description of the task). Re-ran the full battery (now 5 conditions x 10 tasks = 50 queries) in one
invocation so all five conditions' results live in the same run/file.

**Result: both `default` baselines score lower than their distractor-mention counterpart.**

| Condition | Scene | Accuracy | Chance level |
|---|---|---|---|
| `default` (no distractor mention) | `libero_spatial` (2 bowls) | **5/10 (50%)** | 50% |
| `negative_contrast` | `libero_spatial` (2 bowls) | 7/10 (70%) | 50% |
| `positive_contrast` | `libero_spatial` (2 bowls) | 7/10 (70%) | 50% |
| `hardneg_default` (no distractor mention) | `libero_spatial_3bowl_hardneg` (3 bowls) | **6/10 (60%)** | 33% |
| `hardneg` | `libero_spatial_3bowl_hardneg` (3 bowls) | 7/10 (70%) | 33% |

`negative_contrast` and `positive_contrast` remain in 10/10 agreement with each other (see §8.2).
`default` disagrees with both on exactly 2 of the 10 tasks (task ids 4 and 5, both target=bowl "1"):
given only the target's own location ("pick up the black bowl inside the top drawer..." / "...on top
of the ramekin...") Qwen picks the wrong bowl on both, but once the instruction also states where the
*other* bowl is (either condition), it gets both right. All other 8 tasks are identical across all
three 2-bowl conditions regardless of phrasing.

The 3-bowl pair shows the same direction but a much smaller gap: `hardneg_default` differs from
`hardneg` on exactly **one** task (task id 5, target=bowl "3", "on top of the ramekin") — mentioning
the distractor family flips that single task from wrong (guesses the numerically-common "2") to
right; every other task agrees between the two conditions. Task id 0 ("between the plate and the
ramekin") is the single hardest case in the whole battery — the only task wrong in **all 5**
conditions across both scenes, regardless of phrasing; task id 2 ("at the center of the table") is
wrong in both 3-bowl conditions but correct in all three 2-bowl conditions, so the extra distractor
bowl (not the phrasing) looks specifically responsible for that one. Both 3-bowl conditions clear the
33% chance baseline comfortably even without any distractor mention, unlike the 2-bowl case where
`default` lands exactly on chance.

**Reading.** For this scene/model, naming the distractor family — whether negated
(`negative_contrast`), stated positively (`positive_contrast`), or as a closer 3-bowl hard-negative
(`hardneg`) — is not the source of difficulty findings 1/5/9/10 attribute to it for OpenVLA; if
anything, for Qwen it's a mildly *disambiguating* cue in both scenes (worth 2/10 tasks at 2 bowls,
1/10 at 3 bowls) rather than a source of confusion. The 3-bowl scene being resolvable well above
chance even with zero distractor mention (`hardneg_default`, 60% vs. 33% chance) also weakens a
"more distractors = harder to ground for any model" reading — the extra bowl doesn't come close to
erasing Qwen's signal the way it erases OpenVLA's task success (findings 2, 7-8). This sharpens
finding 14's point further: the OpenVLA distractor-mention collapse documented in findings 1-11
cannot be explained by "the distractor mention itself makes the scene harder to ground" in any
general sense — a capable VLM grounds the same mentions at least as accurately as no mention at all,
in both the 2-bowl and 3-bowl scenes. That still doesn't prove OpenVLA's action-decoding head reacts
the same way (§8's dead end for that model family stands untouched), but it further narrows what's
left of the "maybe the phrasing/clutter is just inherently harder" reading of findings 1-11.

Updated structured output (same file, now with all 5 conditions):
`openvla/experiments/logs/probe_bowl_pointing_qwen/probe_bowl_pointing_qwen.jsonl`. New images:
`openvla/experiments/figures/probe_bowl_pointing/libero_spatial--default--t{0-9}.png` and
`libero_spatial_3bowl_hardneg--hardneg_default--t{0-9}.png`. Code change: `bowl_pointing_common.py`,
`probe_bowl_pointing.py`, `probe_bowl_pointing_qwen.py` (`openvla` commit `1b27db3`).

### 8.4 Per-task render + marker table (2026-08-26)

Every rendered scene actually fed to Qwen, with the Set-of-Mark markers baked in (see §8.2 for how
they're now computed) and each condition's `target→answer` verdict, so the numbers in §8.1-8.3's
tables can be checked against the actual stimulus per task instead of just the aggregate score.
`target→answer` reads as the ground-truth marker number, then Qwen's parsed answer (✓ = correct).

**2-bowl scenes (`libero_spatial`) — `default`, `negative_contrast`, `positive_contrast` show the
identical image per task (render is cached per suite/task_id); only the instruction differs:**

| id | task | scene (markers baked in) | `default` | `negative_contrast` | `positive_contrast` |
|--:|---|---|---|---|---|
| 0 | between the plate and the ramekin | ![](openvla/experiments/figures/probe_bowl_pointing/libero_spatial--negative_contrast--t0.png) | 1→2 ✗ | 1→2 ✗ | 1→2 ✗ |
| 1 | next to the ramekin | ![](openvla/experiments/figures/probe_bowl_pointing/libero_spatial--negative_contrast--t1.png) | 2→2 ✓ | 2→2 ✓ | 2→2 ✓ |
| 2 | from table center | ![](openvla/experiments/figures/probe_bowl_pointing/libero_spatial--negative_contrast--t2.png) | 1→1 ✓ | 1→1 ✓ | 1→1 ✓ |
| 3 | on the cookie box | ![](openvla/experiments/figures/probe_bowl_pointing/libero_spatial--negative_contrast--t3.png) | 2→2 ✓ | 2→2 ✓ | 2→2 ✓ |
| 4 | in the top drawer of the wooden cabinet | ![](openvla/experiments/figures/probe_bowl_pointing/libero_spatial--negative_contrast--t4.png) | 1→2 ✗ | 1→1 ✓ | 1→1 ✓ |
| 5 | on the ramekin | ![](openvla/experiments/figures/probe_bowl_pointing/libero_spatial--negative_contrast--t5.png) | 1→2 ✗ | 1→1 ✓ | 1→1 ✓ |
| 6 | next to the cookie box | ![](openvla/experiments/figures/probe_bowl_pointing/libero_spatial--negative_contrast--t6.png) | 1→1 ✓ | 1→1 ✓ | 1→1 ✓ |
| 7 | on the stove | ![](openvla/experiments/figures/probe_bowl_pointing/libero_spatial--negative_contrast--t7.png) | 1→1 ✓ | 1→1 ✓ | 1→1 ✓ |
| 8 | next to the plate | ![](openvla/experiments/figures/probe_bowl_pointing/libero_spatial--negative_contrast--t8.png) | 2→1 ✗ | 2→1 ✗ | 2→1 ✗ |
| 9 | on the wooden cabinet | ![](openvla/experiments/figures/probe_bowl_pointing/libero_spatial--negative_contrast--t9.png) | 1→2 ✗ | 1→2 ✗ | 1→2 ✗ |

**3-bowl scenes (`libero_spatial_3bowl_hardneg`) — `hardneg_default` and `hardneg` show the identical
image per task; only the instruction differs:**

| id | task | scene (markers baked in) | `hardneg_default` | `hardneg` |
|--:|---|---|---|---|
| 0 | between the plate and the ramekin | ![](openvla/experiments/figures/probe_bowl_pointing/libero_spatial_3bowl_hardneg--hardneg--t0.png) | 1→2 ✗ | 1→2 ✗ |
| 1 | next to the ramekin | ![](openvla/experiments/figures/probe_bowl_pointing/libero_spatial_3bowl_hardneg--hardneg--t1.png) | 3→2 ✗ | 3→1 ✗ |
| 2 | from table center | ![](openvla/experiments/figures/probe_bowl_pointing/libero_spatial_3bowl_hardneg--hardneg--t2.png) | 2→3 ✗ | 2→3 ✗ |
| 3 | on the cookie box | ![](openvla/experiments/figures/probe_bowl_pointing/libero_spatial_3bowl_hardneg--hardneg--t3.png) | 2→2 ✓ | 2→2 ✓ |
| 4 | in the top drawer of the wooden cabinet | ![](openvla/experiments/figures/probe_bowl_pointing/libero_spatial_3bowl_hardneg--hardneg--t4.png) | 2→2 ✓ | 2→2 ✓ |
| 5 | on the ramekin | ![](openvla/experiments/figures/probe_bowl_pointing/libero_spatial_3bowl_hardneg--hardneg--t5.png) | 3→2 ✗ | 3→3 ✓ |
| 6 | next to the cookie box | ![](openvla/experiments/figures/probe_bowl_pointing/libero_spatial_3bowl_hardneg--hardneg--t6.png) | 2→2 ✓ | 2→2 ✓ |
| 7 | on the stove | ![](openvla/experiments/figures/probe_bowl_pointing/libero_spatial_3bowl_hardneg--hardneg--t7.png) | 2→2 ✓ | 2→2 ✓ |
| 8 | next to the plate | ![](openvla/experiments/figures/probe_bowl_pointing/libero_spatial_3bowl_hardneg--hardneg--t8.png) | 2→2 ✓ | 2→2 ✓ |
| 9 | on the wooden cabinet | ![](openvla/experiments/figures/probe_bowl_pointing/libero_spatial_3bowl_hardneg--hardneg--t9.png) | 3→3 ✓ | 3→3 ✓ |

Marker color key: **1** = red, **2** = green, **3** = blue (which bowl gets which number is shuffled
per task; color always maps to the same digit). Source images:
`openvla/experiments/figures/probe_bowl_pointing/` (`openvla` commit `1b27db3`).
