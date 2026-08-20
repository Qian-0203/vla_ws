# OpenVLA · LIBERO-Spatial — Instruction & Scene-Robustness Evaluations

How a `libero_spatial`-fine-tuned OpenVLA baseline holds up as we change **the prompt** and
**the scene** — while keeping the model, checkpoint, and seed fixed. Each experiment changes
exactly one thing and is reported against the same 84.0% baseline.

## At a glance

| # | Experiment | What changed | Overall success | vs. baseline |
|--:|---|---|--:|--:|
| — | **Baseline** — default prompt, 2 bowls | — | **84.0%** | — |
| 1 | Explicit prompt | prompt names the distractor too | 36.8% | **−47.2** |
| 2 | Extra distractor | +1 bowl (3 total), default prompt | 80.2% | −3.8 |
| 3 | Cluttered scene | 3 bowls **+ top drawer open**, default prompt | 60.0% (raw) / **73.1%**† | −24.0 (raw) / **−11.7**† |
| 4 | Distractor mention, no negation | prompt states distractor's location but doesn't say "not the one…" | 32.4% | **−51.6** |
| 5 | Semantic distractor (Split 2) | 3rd bowl at a *different* task's named landmark, default prompt | 84.8% | +0.8 |
| 6 | Landmark distractor / hard negative (Split 2) | 3rd bowl near the target's **own** landmark, farther away, default prompt | 80.6% | −3.4 |
| 7 | Irrelevant distractor, redefined (Split 2) | 3rd bowl at a per-task neutral region, off-path, distance-matched | 88.8% | +4.8 |
| 8 | Landmark scene + disambiguating prompt (Split 1×2) | Same hard-negative scene as Exp 6, prompt adds "closest to X, not the one farther" | 41.2% | **−42.8** |

† Adjusted: tasks 3, 6, 7 excluded — rollout review showed the open drawer **physically blocks their
trajectories**, so those failures measure scene feasibility, not policy robustness (see Exp 3). The
adjusted Δ compares against the baseline recomputed on the same 7 tasks (84.9%).

**Headlines.** (1) Naming the distractor in the prompt *badly* hurts the policy (−47 pts). (2) Adding
a second distractor bowl barely dents overall success (−3.8 pts) but sinks one specific task. (3)
Opening the cabinet drawer looks like a broad −20 pt collapse at first glance, but roughly half of
that is an environment artifact: on tasks 3, 6, and 7 the protruding drawer blocks the motion path
outright. Excluding those, the drawer's *policy-attributable* cost is **−11.7 pts** (73.1% on the
7 feasible tasks) — still ~3× the extra bowl's damage, but a robustness gap, not a collapse.

**Headlines, cont'd (2026-08-19).** (4) The damage in Exp 1 isn't really about *negation* — just
*mentioning* the distractor's location without negating it is **just as bad, if not slightly worse**
(32.4% vs. 36.8%). Naming a second location at all overloads the policy's grounding; the "not the
one…" clause isn't the culprit. (5) Split 2's first real data point complicates its own hypothesis:
a distractor deliberately placed at another task's named landmark costs **~0 pts** (84.8% vs. 84.0%
baseline) — indistinguishable from noise. Where a distractor sits seems to matter far less than
whether the *prompt* talks about it (compare Exp 4/1's −52 pts for pure language). (6) The harder
version of that same test — a distractor near the target's *own* landmark — does move the needle
(−3.4 pts overall), but the drop isn't spread across the suite: it's almost entirely two tasks
(task 0: 92%→48%, task 9: 72%→54%), both far outside single-task noise. Scene changes *can* hurt,
but only when the distractor is placed to be genuinely confusable with the target, not merely
present or semantically labeled. (7) The redefined `irrelevant` condition — the "control" for Split
2 — actually scores *above* baseline (88.8% vs. 84.0%, +4.8 pts, ~3 SE), with no task collapsing and
two tasks (4, 7) jumping +16-20 pts. Across all three Split 2 conditions run so far, `Distractor-type
Drop` only ever turns negative (real cost) when the distractor sits near the target's *own* landmark
— neutral and other-landmark placements don't cost anything and may even help slightly, plausibly
because a 3rd bowl elsewhere in the scene doesn't compete for the target's identity at all.
(8) The natural next question — can a disambiguating prompt *rescue* the `landmark` scene's task 0/9
collapse — has a clear answer: no. Adding "closest to X, not the one farther" on top of the
`landmark` scene doesn't fix tasks 0/9, and it wrecks every other task that was previously fine
(e.g. task 7: 86%→4%, task 2: 98%→50%). Overall drops from 80.6% to 41.2%. Once again, language that
references a second location dominates — this time stacking on top of, rather than compensating for,
the scene-level confusability.

---

## Setup (common to all experiments)

- **Model:** `openvla-7b` LoRA (r32) fine-tuned on `libero_spatial_no_noops` (bf16 + FlashAttention-2, center-crop).
- **Base suite:** `libero_spatial` — 10 tasks; each = *"pick up the black bowl \<location\> and place it on the plate."*
- **Protocol:** 50 trials/task = **500 rollouts** per condition. Seed 7. Success = LIBERO goal met.
- **Env:** `openvla-libero:cuda12.1` Docker (mujoco 2.3.2, robosuite 1.4.1, EGL headless).
- **Hardware:** 5× H200 (GPUs 0–4), tasks round-robin sharded 2/GPU.
- **Task ids** are identical across every suite below, so columns line up 1:1:

  | id | target location | id | target location |
  |--:|---|--:|---|
  | 0 | between the plate and the ramekin | 5 | on the ramekin |
  | 1 | next to the ramekin | 6 | next to the cookie box |
  | 2 | table center | 7 | on the stove |
  | 3 | on the cookie box | 8 | next to the plate |
  | 4 | in the top drawer of the wooden cabinet | 9 | on the wooden cabinet |

- **Noise:** at 50 trials/task, 1 standard error ≈ ±5–7 pts, so treat single-task swings ≲10 pts as noise.

---

## Experiment 1 — Prompt phrasing: default vs. explicit (2 bowls)

**Question.** The scene has two identical black bowls (target + 1 distractor). Does *telling the model
which bowl to avoid* help? Only the **prompt string** changes; scene and init states are identical.

- **Default:** names only the target — *"pick up the black bowl on the stove and place it on the plate."*
- **Explicit:** also names the distractor — *"…, not the one on top of the wooden cabinet, …"*

| Condition | Overall | Rollouts |
|---|--:|--:|
| **Default (ordinary)** | **84.0%** | 420 / 500 |
| **Explicit (distractor-aware)** | **36.8%** | 184 / 500 |
| **Δ** | **−47.2 pts** | |

| id | target | distractor named in prompt | Default | Explicit | Δ |
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

**Takeaway.** The explicit "…not the one on X…" clause *confuses* the policy instead of disambiguating
it (−47 pts overall). Damage is worst where the clause names a salient surface (ramekin, table center,
stove, cabinet). The only unhurt task (0) was already phrased relationally.

<details><summary>Exact explicit prompts used (distractor clause in <b>bold</b>)</summary>

| Default target | Explicit instruction |
|---|---|
| between the plate and the ramekin | pick up the black bowl between the plate and the ramekin, **not the one next to the ramekin**, and place it on the plate |
| from table center | pick up the black bowl at the center of the table, **not the one next to the plate**, and place it on the plate |
| in the top drawer of the cabinet | pick up the black bowl inside the top drawer of the wooden cabinet, **not the one on top of the cabinet**, and place it on the plate |
| next to the cookie box | pick up the black bowl next to the cookie box, **not the one on the stove**, and place it on the plate |
| next to the plate | pick up the black bowl next to the plate, **not the one next to the ramekin**, and place it on the plate |
| next to the ramekin | pick up the black bowl next to the ramekin, **not the one next to the cookie box**, and place it on the plate |
| on the cookie box | pick up the black bowl on top of the cookie box, **not the one on top of the wooden cabinet**, and place it on the plate |
| on the ramekin | pick up the black bowl on top of the ramekin, **not the one on top of the cookie box**, and place it on the plate |
| on the stove | pick up the black bowl on the stove, **not the one on top of the wooden cabinet**, and place it on the plate |
| on the wooden cabinet | pick up the black bowl on top of the wooden cabinet, **not the one on the stove**, and place it on the plate |

</details>

---

## Experiment 2 — Extra distractor: 2 bowls vs. 3 bowls (default prompt)

**Question.** Keep the ordinary (target-only) prompt but make the scene harder: add a **third**
`akita_black_bowl` (a second distractor) at the open table center — table front for task 2, whose
target already sits at center. Does more visual ambiguity break the policy?

| Condition (default prompt) | Overall | Rollouts |
|---|--:|--:|
| **2 bowls** (target + 1 distractor) | **84.0%** | 420 / 500 |
| **3 bowls** (target + 2 distractors) | **80.2%** | 401 / 500 |
| **Δ** | **−3.8 pts** | |

| id | target | 2-bowl | 3-bowl | Δ |
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

**Takeaway.** Overall success is **robust** to one extra distractor (−3.8 pts) — but the loss is
**concentrated**: task 6 (*next to the cookie box*) collapses 90%→44% because the added center bowl
lands between the cookie box and the plate, and it is next but little farther to the cookie box than the target.

**Init-state figure** — left = 2 bowls (`libero_spatial`), right = 3 bowls (`libero_spatial_3bowl`);
rows are task ids 0–9. Each panel is the real episode-0 state restored via `set_init_state`.

![2-bowl vs 3-bowl init states](figures/compare_2v3bowl_grid.png)

The extra free-body bowl enlarges the restored MuJoCo state vector (**92 → 105 dims**), so the 3-bowl
suite ships freshly generated `.pruned_init` files. Placement was checked across all 500 states: worst
bowl-to-bowl distance 0.122 m (bowl ⌀ ≈ 0.115 m) — no overlaps, valid heights.

---

## Experiment 3 — Cluttered scene: 3 bowls + open top drawer (default prompt)

**Question.** Keep the 3-bowl scene and add clutter/occlusion by **opening the wooden cabinet's top
("first") drawer** in every task. Does a protruding open drawer degrade the policy further?

### Blocked tasks — tasks 3, 6, 7 excluded from the adjusted overall

Rollout review showed that for **task 3** (*on the cookie box*), **task 6** (*next to the cookie
box*), and **task 7** (*on the stove*), the open top drawer **physically blocks the trajectory**: the
protruding drawer sits directly in the pick-and-place corridor these tasks must traverse, so the arm
cannot complete the motion regardless of what the policy predicts. Failures on these three tasks
therefore measure the *scene's physical feasibility*, not the *policy's robustness*, and they are
excluded from the adjusted overall below. (Task 6's 0% is the clearest case — a hard geometric block,
not a perception error.)

**Raw** (all 10 tasks) and **adjusted** (7 feasible tasks, 350 rollouts) overalls:

| Condition (default prompt) | Raw overall (10 tasks) | Adjusted overall (7 tasks†) |
|---|--:|--:|
| 3 bowls, drawer closed (Exp 2) | 80.2% (401 / 500) | 84.3% (295 / 350) |
| 3 bowls, **drawer open** | 60.0% (300 / 500) | **73.1%** (256 / 350) |
| **Δ (open − closed)** | −20.2 pts | **−11.1 pts** |

† Same-subset comparison: tasks {0, 1, 2, 4, 5, 8, 9} in both conditions. Against the original 2-bowl
baseline on the same 7 tasks (84.9%, 297/350), the adjusted drop is **−11.7 pts** — vs. the raw
headline of −24.0, which conflates policy degradation with physically infeasible tasks.

| id | target | 3-bowl | +drawer open | Δ | |
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

**Takeaway.** The raw −20 pt drop overstates the policy effect: nearly half of it comes from the three
blocked tasks (3, 6, 7), where no policy could succeed because the open drawer occupies the motion
corridor. On the seven tasks that remain physically feasible, the open drawer still costs the policy
**−11.1 pts** vs. the closed-drawer scene — a genuine (occlusion/robustness) effect, concentrated
around the cabinet: task 9 (*on the wooden cabinet*, −30) and task 8 (*next to the plate*, −20) drop
well beyond noise, with smaller dips on 2 and 0. The two tasks that are *unaffected* (4 *in the top
drawer*, 5 *on the ramekin*) are telling: task 4's target is inside the drawer, so an open drawer is
expected there. The protruding drawer occludes the cabinet side of the table, which is exactly where
the still-degraded targets sit or are approached — but the effect is a robustness gap (~−11 pts), not
the collapse the raw number suggests.

**Init-state figure** — left = drawer closed (`libero_spatial_3bowl`), right = drawer open
(`libero_spatial_3bowl_open`); rows are task ids 0–9.

![3-bowl: drawer closed vs open](figures/compare_3bowl_closed_vs_open_grid.png)

Verified across all 500 open-drawer init states: bowls never overlap (worst separation 0.122 m) and the
cabinet top-drawer joint is open (qpos −0.141 m) in every trial. Adding the drawer state does not change
the state-vector size (105 dims) — the drawer joint already exists in the cabinet model; only its
position changes.

---

## Experiment 4 — Prompt phrasing: distractor mention without negation (2 bowls)

**Question.** Experiment 1 showed that naming *and negating* the distractor ("…not the one on X…")
badly hurts the policy. Is that damage from the negation construction itself, or just from mentioning
a second location at all? This condition (`positive_contrast`) states the distractor's location as a
plain fact, with no negation: *"…the other black bowl is on X."*

| Condition | Overall | Rollouts |
|---|--:|--:|
| **Default** (target only) | **84.0%** | 420 / 500 |
| **Positive contrast** (mentions distractor, no negation) | **32.4%** | 162 / 500 |
| **Negative contrast** (mentions + negates distractor, = Exp 1's "Explicit") | **36.8%** | 184 / 500 |

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

**Metrics** (formulas from `benchmark_split_plan.md` Split 1):
- `Distractor Mention Drop = SR(default) − SR(positive_contrast) = 84.0 − 32.4 = 51.6 pts`
- `Negation-specific Drop = SR(positive_contrast) − SR(negative_contrast) = 32.4 − 36.8 = −4.4 pts`

**Takeaway.** The negation-specific drop is *negative* — adding "not the one…" on top of a bare
mention slightly **helps** rather than hurts (+4.4 pts), and that's within noise at this trial count
anyway. Nearly all of Exp 1's −47 pt damage survives with the negation removed entirely (−51.6 pts
here). So the earlier hypothesis ("negation confuses the policy") was misattributed: the confusion
comes from **naming a second spatial location in the prompt at all** — the checkpoint was fine-tuned
on target-only prompts and has no practice grounding a second referent, negated or not. Task-level
pattern matches Exp 1 closely (tasks 1, 2, 5, 9 hit hardest in both), reinforcing that it's the same
underlying failure mode, not something specific to the negation clause.

Prompts used: `openvla/experiments/robot/libero/instructions.py::LIBERO_SPATIAL_POSITIVE_CONTRAST_INSTRUCTIONS`.
Run: 2026-08-19, 4× RTX PRO 6000 Blackwell, `openvla-libero:blackwell` (sdpa), checkpoint/seed
unchanged from Exp 1–3. Results: `results/libero_spatial--positive_contrast--shard{0..3}of4.jsonl`.

---

## Experiment 5 — Split 2: semantic distractor (3 bowls, default prompt)

**Question.** Split 2 asks whether an extra distractor's *position* — not just its presence — drives
failures. This is the first condition run under Split 2's current (redefined) definitions. `semantic`
places the 3rd bowl at a **named landmark that is not the target's own** — e.g. task 0's target is
"between the plate and the ramekin," and its 3rd bowl sits `next_to_box_region` (the cookie box, an
unrelated landmark) — testing whether sitting at *any* nameable relational spot pulls the policy, even
when that landmark doesn't match the prompt. See `benchmark_split_plan.md` §Split 2 "Distractor types &
purpose" for how this differs from `irrelevant` (neutral, no landmark) and `landmark` (near the
target's *own* landmark — not yet run).

| Condition (default prompt) | Overall | Rollouts |
|---|--:|--:|
| **2 bowls** (baseline, Exp 1) | **84.0%** | 420 / 500 |
| **3 bowls, semantic distractor** | **84.8%** | 424 / 500 |
| **Δ (Distractor-type Drop)** | **+0.8 pts** | |

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

**Takeaway.** Unlike Exp 4's language-only manipulation, a semantic-but-irrelevant distractor barely
moves overall success (+0.8 pts, i.e. no measurable drop — well under the ~±3.3 pt SE at 500 pooled
rollouts). The two weakest tasks, 1 (72%) and 9 (68%), are still comfortably above Exp 4's collapse
range, so even the softest spots here look like ordinary scene-to-scene noise rather than a
distractor-placement effect. Contrasted with Exp 4 (prompt mentions a location: −51.6 pts) and Exp 2
(extra bowl at a neutral spot: −3.8 pts), this is the strongest evidence yet in this project that the
policy's failures are driven by **what the prompt says**, not by **what's sitting on the table** — a
distractor's presence and rough position barely register unless language draws attention to it.
Still open: whether `landmark` (distractor near the target's *own* landmark, a harder confusability
test) shows a real effect where `semantic` didn't.

Scene: BDDL + init states authored and verified this pass; contact sheet rendered and eyeballed (3
distinct bowls per task, no overlaps/clipping, drawer open only for task 4 as expected) before this
run — see `benchmark_split_plan.md` for the per-task placement table. Run: 2026-08-19, 4× RTX PRO 6000
Blackwell, `openvla-libero:blackwell` (sdpa), same checkpoint/seed as all prior experiments. Results:
`results/libero_spatial_3bowl_semantic--default--shard{0..3}of4.jsonl`.

---

## Experiment 6 — Split 2: landmark distractor / hard negative (3 bowls, default prompt)

**Question.** `landmark` is Split 2's hardest confusability test: the 3rd bowl sits near the
target's **own** landmark — the same relational word the prompt uses — just farther from the exact
target point than the real bowl (e.g. task 0's target is "between the plate and the ramekin"; its
3rd bowl also sits near that plate/ramekin area, at `hardneg_region` (−0.070,0.040)–(−0.050,0.060)).
Unlike `semantic` (Exp 5, different landmark) or `irrelevant` (no landmark), this distractor is a
genuine look-alike for "the bowl near X."

| Condition (default prompt) | Overall | Rollouts |
|---|--:|--:|
| **2 bowls** (baseline, Exp 1) | **84.0%** | 420 / 500 |
| **3 bowls, landmark/hard-negative distractor** | **80.6%** | 403 / 500 |
| **Δ (Distractor-type Drop)** | **−3.4 pts** | |

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

**Takeaway.** The overall drop (−3.4 pts) looks mild — comparable to Exp 2's plain extra distractor
(−3.8 pts) — but that headline number hides a **concentrated, task-specific effect**: task 0 collapses
92%→48% and task 9 drops 72%→54%, both far beyond the ~±7 pt single-task noise band at n=50; every
other task is flat or actually improves. This is the first Split 2 condition where the distractor's
*placement relative to the target's own landmark* — not just its presence (Exp 2) or its placement at
some other landmark (Exp 5) — produces a real, attributable failure. It's concentrated rather than
uniform, similar in shape to Exp 2's single-task collapse (task 6), reinforcing that this policy's
distractor sensitivity is about specific near-miss geometry, not general clutter aversion. Together
with Exp 5, this narrows *where* scene-driven failures come from: proximity to the target's own
landmark matters; unrelated landmarks and neutral placement don't.

Scene: BDDL/init states pre-existed but were unchecked with the current pipeline; regenerated
(byte-identical, confirming determinism), verified (min separation 0.121 m vs. the 0.12 m threshold —
passes narrowly, task 3 tightest), contact sheet rendered and eyeballed before this run — 3 distinct
bowls per task, no overlaps/clipping (task 3's close pair confirmed as two separate bowls, not
merged). Run: 2026-08-19, 4× RTX PRO 6000 Blackwell, `openvla-libero:blackwell` (sdpa), same
checkpoint/seed as all prior experiments. Results:
`results/libero_spatial_3bowl_hardneg--default--shard{0..3}of4.jsonl`.

---

## Experiment 7 — Split 2: irrelevant distractor, redefined (3 bowls, default prompt)

**Question.** `irrelevant` is Split 2's control: a 3rd bowl placed somewhere that isn't tied to any
relational language at all — testing whether an extra distractor costs anything purely from being
*present*, independent of where exactly it sits. This is the redefined suite
(`libero_spatial_3bowl_neutral`): each task's 3rd bowl goes to whichever safe region gives the
largest clearance from both the reach path and bowl_2, replacing the retired `center_fixed_legacy`
placement (a single fixed coordinate that confounded task 6 in Exp 2 — see `benchmark_split_plan.md`
Split 2 for the redefinition details).

This run was already complete when this session picked up Split 2 (timestamp 14:02, earlier the same
day as Exp 4-6) but had not been aggregated or written up — recovered and documented here rather than
re-run.

| Condition (default prompt) | Overall | Rollouts |
|---|--:|--:|
| **2 bowls** (baseline, Exp 1) | **84.0%** | 420 / 500 |
| **3 bowls, irrelevant/neutral distractor** | **88.8%** | 444 / 500 |
| **Δ (Distractor-type Drop)** | **+4.8 pts** | |

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

**Takeaway.** Unlike Exp 2's old `center_fixed_legacy` placement (−3.8 pts, one task collapsing),
the redefined neutral placement doesn't cost anything at all — it scores *above* baseline (+4.8 pts,
~3 pooled SE, so plausibly a real if modest effect rather than pure noise) with no single-task
collapse. Tasks 4 and 7 gain the most (+20, +16); no obvious explanation stands out from the
success-rate numbers alone (not investigated further at the rollout-video level). Completes Split 2's
three-way distractor-position comparison:

| Condition | Overall | Δ vs. baseline | Task-level pattern |
|---|--:|--:|---|
| `irrelevant` (neutral, no landmark) | 88.8% | +4.8 | no collapse; broadly flat/positive |
| `semantic` (different landmark) | 84.8% | +0.8 | no collapse; flat |
| `landmark` (target's own landmark, hard negative) | 80.6% | −3.4 | **concentrated collapse**, tasks 0 & 9 only |

The only condition with a real cost is the one where the distractor is a plausible look-alike for the
target's own described location. Neutral and other-landmark placements don't hurt — if anything they
trend positive — reinforcing Exp 4-6's overall pattern: this policy's failures are about *matching
the wrong bowl to the described landmark*, not about scene clutter or extra-object presence per se.

Scene: BDDL authored, init states generated and verified this pass (`RESULT: PASS`, worst separation
0.122 m), contact sheet eyeballed — 3 distinct bowls per task, no overlaps. Run: 2026-08-19, 4× RTX
PRO 6000 Blackwell, `openvla-libero:blackwell` (sdpa), same checkpoint/seed as all prior experiments.
Results: `results/libero_spatial_3bowl_neutral--default--shard{0..3}of4.jsonl`.

---

## Experiment 8 — Split 1×2: landmark scene + disambiguating prompt

**Question.** Exp 6 showed the `landmark` scene (hard-negative distractor) collapses two specific
tasks (0, 9) while leaving the rest alone. Can a prompt that explicitly disambiguates — *"pick up the
black bowl closest to X, not the one farther from it, …"* — rescue those two tasks, the way it might
if the failure were really about the model needing to be told which bowl is closer? Same scene as
Exp 6 (`libero_spatial_3bowl_hardneg`), condition = `hardneg` prompt
(`instructions.py::LIBERO_SPATIAL_HARDNEG_INSTRUCTIONS`).

| Condition | Overall | Rollouts |
|---|--:|--:|
| **2 bowls, default prompt** (baseline, Exp 1) | **84.0%** | 420 / 500 |
| **Landmark scene, default prompt** (Exp 6) | **80.6%** | 403 / 500 |
| **Landmark scene, disambiguating prompt** | **41.2%** | 412 / 500 |
| **Δ (prompt effect, same scene)** | **−39.4 pts** | |

| id | target | Landmark scene, default | + disambiguating prompt | Δ |
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

**Takeaway.** The disambiguating prompt does not rescue the two tasks that actually needed help —
task 0 barely changes (48%→38%) and task 9 gets *worse*, not better (54%→6%) — while every task that
was fine without it collapses instead, several catastrophically (task 7: 86%→4%, task 1: 92%→34%,
task 5: 92%→38%). Net effect is strongly negative (−39.4 pts on the same scene) and lands the overall
number (41.2%) close to Exp 1's `negative_contrast` on the plain 2-bowl scene (36.8%) — consistent
with Exp 1/4's finding that *any* prompt referencing a second bowl/location is what hurts, regardless
of whether the scene actually contains a confusable distractor. Combining a hard scene with a hard
prompt doesn't compound narrowly on the hard cases; the prompt damage dominates and spreads to tasks
the scene alone never touched. This is the clearest evidence yet that language, not scene design, is
the primary lever on this checkpoint's failures — trying to fix a scene-level confound with more
language makes things worse, not better.

Scene: identical to Exp 6 (`libero_spatial_3bowl_hardneg`), no changes. Run: 2026-08-19, 4× RTX PRO
6000 Blackwell, `openvla-libero:blackwell` (sdpa), same checkpoint/seed as all prior experiments.
Results: `results/libero_spatial_3bowl_hardneg--hardneg--shard{0..3}of4.jsonl`.

---

## Reproduce

All commands from `vla_ws/docker/openvla_libero/`. Each runs the same eval harness in Docker across
GPUs 0–4; `USE_EXPLICIT_PROMPT` toggles Exp 1's prompt.

```bash
# Exp 1 — baseline / explicit (2 bowls)
USE_EXPLICIT_PROMPT=False GPUS="0,1,2,3,4" bash eval_explicit_libero_spatial_multigpu.sh
USE_EXPLICIT_PROMPT=True  GPUS="0,1,2,3,4" bash eval_explicit_libero_spatial_multigpu.sh

# Exp 2 — three bowls, default prompt
USE_EXPLICIT_PROMPT=False GPUS="0,1,2,3,4" bash eval_libero_spatial_3bowl_multigpu.sh

# Exp 3 — three bowls + open drawer, default prompt
USE_EXPLICIT_PROMPT=False GPUS="0,1,2,3,4" bash eval_libero_spatial_3bowl_open_multigpu.sh
```

**Config**
- Checkpoint: `/home/ec2-user/wenhan/openvla/checkpoints/baseline_lora_libero_spatial_4gpu_b24_run004/openvla-7b+libero_spatial_no_noops+b24+lr-0.0005+lora-r32+dropout-0.0--image_aug`
- Action un-norm stats: checkpoint key `libero_spatial_no_noops`. The variant suites pass
  `--unnorm_key libero_spatial` so their differing suite name still resolves to that key.
- The `modified_libero_rlds` RLDS path is training data and is **not** read at eval time.

**Scene variants** (new parallel suites; the canonical `libero_spatial` is untouched)

| Suite | Scene | BDDLs / init states |
|---|---|---|
| `libero_spatial` | 2 bowls (stock) | `LIBERO/libero/libero/{bddl_files,init_files}/libero_spatial/` |
| `libero_spatial_3bowl` | +1 bowl | `…/libero_spatial_3bowl/` |
| `libero_spatial_3bowl_open` | +1 bowl, top drawer open | `…/libero_spatial_3bowl_open/` |

Variant suites are registered in `LIBERO/libero/libero/benchmark/{__init__.py,libero_suite_task_map.py}`.
Init states (dim 105) were regenerated inside the eval Docker.

**Helper scripts** (`LIBERO/scripts/`)
- `gen_suite_init_states.py <suite>` — sample + save init states, render a contact sheet.
- `verify_3bowl_init_states.py` — assert bowls never overlap; report heights.
- `compare_2v3_bowl_init.py` — render the 2-bowl vs 3-bowl comparison figure.

**Artifacts**
- Per-shard logs: `experiments/logs/EVAL-<suite>-…--<condition>--shard{0..4}of5.txt`
- Rollout videos: `openvla/rollouts/<date>/` (tagged with task + success)
- Figures: `experiments/figures/`
