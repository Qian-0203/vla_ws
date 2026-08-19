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
whether the *prompt* talks about it (compare Exp 4/1's −52 pts for pure language).

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

**Metrics** (formulas from `benchmark_split.md` Split 1):
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
when that landmark doesn't match the prompt. See `benchmark_split.md` §Split 2 "Distractor types &
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
run — see `benchmark_split.md` for the per-task placement table. Run: 2026-08-19, 4× RTX PRO 6000
Blackwell, `openvla-libero:blackwell` (sdpa), same checkpoint/seed as all prior experiments. Results:
`results/libero_spatial_3bowl_semantic--default--shard{0..3}of4.jsonl`.

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
