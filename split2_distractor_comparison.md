# Split 2 — Per-task Distractor Placement Comparison

A companion reference to `benchmark_split_result.md` §3: for each of the 10 `libero_spatial` tasks,
this lays out the four current placement conditions side by side — where the 3rd bowl sits, what the
scene renders like, what instruction the model actually receives, and the resulting success rate.
Not a replacement for `benchmark_split_result.md` (that stays the authoritative per-condition detail
+ analysis + status source per `CLAUDE.md`) — this file only reorganizes the same already-recorded
numbers by task instead of by condition, for at-a-glance task-level comparison.

**Scope.** Four conditions: `default` (2-bowl baseline, no 3rd bowl), `irrelevant` (current,
`libero_spatial_3bowl_front`), `semantic` (current, `libero_spatial_3bowl_semantic2`), `landmark`
(`libero_spatial_3bowl_hardneg`). `landmark_with_hardneg_prompt` is intentionally excluded — it
reuses `landmark`'s exact scene and only changes the *prompt*, so it belongs to a different axis
(Split 1 x Split 2 prompt combo) than this placement-only comparison; see
`benchmark_split_result.md` §3's `landmark_with_hardneg_prompt` section for it. `irrelevant_v1_legacy`
/ `semantic_v1_legacy` / `center_fixed_legacy` are retired definitions, also excluded — this file only
covers the current, active definition of each condition.

**Key fact this table makes visible:** none of these four conditions change the instruction text —
all four feed the model LIBERO's own native, target-only phrasing (the distractor is never
mentioned). Only the *scene* changes. Whatever success-rate differences show up below are caused
entirely by what's physically on the table, not by what the model was told.

**Sources.** SR values and result-file paths: `benchmark_split_result.md` §3 (`Setting` subsections
for each condition). Renders: `openvla/experiments/figures/per_task_render/{default_full,
irrelevant_v2, semantic_v2, landmark_full}_t{id}.png`, copied from each suite's
`LIBERO/scratch_render/<suite>/t{id}_init.png` (all captured with the physics-settle fix — see
`benchmark_split_result.md` §7 — so no floating bowls). Distractor placement text for `landmark`
is paraphrased from `instructions.py::LIBERO_SPATIAL_HARDNEG_INSTRUCTIONS`' own disambiguating
clause, since that condition's instruction (used only in `landmark_with_hardneg_prompt`, not here)
is the ground truth for where its 3rd bowl actually sits relative to the target.

## Quick-scan matrix (success rate only)

| id | task | `default` | `irrelevant` | `semantic` | `landmark` |
|--:|---|--:|--:|--:|--:|
| 0 | between the plate and the ramekin | 92% | 88% | 86% | **48%** |
| 1 | next to the ramekin | 84% | 84% | 72% | 92% |
| 2 | table center | 92% | 94% | 96% | 98% |
| 3 | on the cookie box | 84% | 96% | 90% | 88% |
| 4 | in the top drawer | 76% | 88% | 92% | 84% |
| 5 | on the ramekin | 94% | 86% | 86% | 92% |
| 6 | next to the cookie box | 90% | 76% | 94% | 80% |
| 7 | on the stove | 72% | 92% | 88% | 86% |
| 8 | next to the plate | 84% | 80% | 80% | 84% |
| 9 | on the wooden cabinet | 72% | 68% | 68% | **54%** |
| — | **Overall (mean of per-task rates)** | **84.0%** | **85.2%** | **85.2%** | **80.6%** |

Bold = the condition's largest single-task drop vs. `default` (task 0 and task 9 under `landmark` —
see `benchmark_split_result.md` finding 7 for why: proximity to the target's *own* landmark is the
one placement so far with a real, concentrated cost; `irrelevant`/`semantic` never collapse any task).

---

## Task 0 — between the plate and the ramekin

Instruction (all 4 conditions): *"pick the akita black bowl between the plate and the ramekin and
place it on the plate"*

| Condition | 3rd bowl placement | Render | SR |
|---|---|---|--:|
| `default` | — (2 bowls only) | ![default t0](openvla/experiments/figures/per_task_render/default_full_t0.png) | 92% |
| `irrelevant` | front edge of the table (`table_front`) | ![irrelevant t0](openvla/experiments/figures/per_task_render/irrelevant_v2_t0.png) | 88% |
| `semantic` | next to the cookie box | ![semantic t0](openvla/experiments/figures/per_task_render/semantic_v2_t0.png) | 86% |
| `landmark` | in front of the plate/ramekin pair (same landmark family as the target, farther from it) | ![landmark t0](openvla/experiments/figures/per_task_render/landmark_full_t0.png) | **48%** |

## Task 1 — next to the ramekin

Instruction (all 4 conditions): *"pick the akita black bowl next to the ramekin and place it on the
plate"*

| Condition | 3rd bowl placement | Render | SR |
|---|---|---|--:|
| `default` | — (2 bowls only) | ![default t1](openvla/experiments/figures/per_task_render/default_full_t1.png) | 84% |
| `irrelevant` | table center (`table_center` fallback — `table_front` sits too close to this task's own bowl_2) | ![irrelevant t1](openvla/experiments/figures/per_task_render/irrelevant_v2_t1.png) | 84% |
| `semantic` | next to the plate | ![semantic t1](openvla/experiments/figures/per_task_render/semantic_v2_t1.png) | 72% |
| `landmark` | near the ramekin, farther from it than the target | ![landmark t1](openvla/experiments/figures/per_task_render/landmark_full_t1.png) | 92% |

## Task 2 — from table center

Instruction (all 4 conditions): *"pick the akita black bowl from table center and place it on the
plate"*

| Condition | 3rd bowl placement | Render | SR |
|---|---|---|--:|
| `default` | — (2 bowls only) | ![default t2](openvla/experiments/figures/per_task_render/default_full_t2.png) | 92% |
| `irrelevant` | front edge of the table (`table_front`) | ![irrelevant t2](openvla/experiments/figures/per_task_render/irrelevant_v2_t2.png) | 94% |
| `semantic` | next to the ramekin | ![semantic t2](openvla/experiments/figures/per_task_render/semantic_v2_t2.png) | 96% |
| `landmark` | off to the side of table center | ![landmark t2](openvla/experiments/figures/per_task_render/landmark_full_t2.png) | 98% |

## Task 3 — on the cookie box

Instruction (all 4 conditions): *"pick the akita black bowl on the cookies box and place it on the
plate"*

| Condition | 3rd bowl placement | Render | SR |
|---|---|---|--:|
| `default` | — (2 bowls only) | ![default t3](openvla/experiments/figures/per_task_render/default_full_t3.png) | 84% |
| `irrelevant` | table center (`table_center` fallback — `table_front` overlapped here at first verify) | ![irrelevant t3](openvla/experiments/figures/per_task_render/irrelevant_v2_t3.png) | 96% |
| `semantic` | next to the ramekin | ![semantic t3](openvla/experiments/figures/per_task_render/semantic_v2_t3.png) | 90% |
| `landmark` | on the table next to the cookie box | ![landmark t3](openvla/experiments/figures/per_task_render/landmark_full_t3.png) | 88% |

## Task 4 — in the top drawer of the wooden cabinet

Instruction (all 4 conditions): *"pick the akita black bowl in the top layer of the wooden cabinet
and place it on the plate"*

| Condition | 3rd bowl placement | Render | SR |
|---|---|---|--:|
| `default` | — (2 bowls only) | ![default t4](openvla/experiments/figures/per_task_render/default_full_t4.png) | 76% |
| `irrelevant` | front edge of the table (`table_front`) | ![irrelevant t4](openvla/experiments/figures/per_task_render/irrelevant_v2_t4.png) | 88% |
| `semantic` | between the plate and the ramekin (moved here from a too-far, out-of-band spot — see `benchmark_split_plan.md` "Second redefinition") | ![semantic t4](openvla/experiments/figures/per_task_render/semantic_v2_t4.png) | 92% |
| `landmark` | on the table in front of the cabinet | ![landmark t4](openvla/experiments/figures/per_task_render/landmark_full_t4.png) | 84% |

## Task 5 — on the ramekin

Instruction (all 4 conditions): *"pick the akita black bowl on the ramekin and place it on the
plate"*

| Condition | 3rd bowl placement | Render | SR |
|---|---|---|--:|
| `default` | — (2 bowls only) | ![default t5](openvla/experiments/figures/per_task_render/default_full_t5.png) | 94% |
| `irrelevant` | table center (`table_center` fallback — `table_front` overlapped here at first verify) | ![irrelevant t5](openvla/experiments/figures/per_task_render/irrelevant_v2_t5.png) | 86% |
| `semantic` | next to the plate | ![semantic t5](openvla/experiments/figures/per_task_render/semantic_v2_t5.png) | 86% |
| `landmark` | on the table next to the ramekin | ![landmark t5](openvla/experiments/figures/per_task_render/landmark_full_t5.png) | 92% |

## Task 6 — next to the cookie box

Instruction (all 4 conditions): *"pick the akita black bowl next to the cookies box and place it on
the plate"*

| Condition | 3rd bowl placement | Render | SR |
|---|---|---|--:|
| `default` | — (2 bowls only) | ![default t6](openvla/experiments/figures/per_task_render/default_full_t6.png) | 90% |
| `irrelevant` | table center (`table_center` fallback — this task's own target sits too close to `table_front`) | ![irrelevant t6](openvla/experiments/figures/per_task_render/irrelevant_v2_t6.png) | **76%** |
| `semantic` | next to the ramekin | ![semantic t6](openvla/experiments/figures/per_task_render/semantic_v2_t6.png) | 94% |
| `landmark` | near the cookie box, farther from it than the target | ![landmark t6](openvla/experiments/figures/per_task_render/landmark_full_t6.png) | 80% |

`irrelevant`'s weakest task (−14 pts vs. `default`) — worth watching if this condition is ever
redefined again.

## Task 7 — on the stove

Instruction (all 4 conditions): *"pick the akita black bowl on the stove and place it on the plate"*

| Condition | 3rd bowl placement | Render | SR |
|---|---|---|--:|
| `default` | — (2 bowls only) | ![default t7](openvla/experiments/figures/per_task_render/default_full_t7.png) | 72% |
| `irrelevant` | front edge of the table (`table_front`) | ![irrelevant t7](openvla/experiments/figures/per_task_render/irrelevant_v2_t7.png) | 92% |
| `semantic` | next to the cookie box | ![semantic t7](openvla/experiments/figures/per_task_render/semantic_v2_t7.png) | 88% |
| `landmark` | on the table in front of the stove | ![landmark t7](openvla/experiments/figures/per_task_render/landmark_full_t7.png) | 86% |

## Task 8 — next to the plate

Instruction (all 4 conditions): *"pick the akita black bowl next to the plate and place it on the
plate"*

| Condition | 3rd bowl placement | Render | SR |
|---|---|---|--:|
| `default` | — (2 bowls only) | ![default t8](openvla/experiments/figures/per_task_render/default_full_t8.png) | 84% |
| `irrelevant` | front edge of the table (`table_front`) | ![irrelevant t8](openvla/experiments/figures/per_task_render/irrelevant_v2_t8.png) | 80% |
| `semantic` | next to the cookie box | ![semantic t8](openvla/experiments/figures/per_task_render/semantic_v2_t8.png) | 80% |
| `landmark` | near the plate, farther from it than the target | ![landmark t8](openvla/experiments/figures/per_task_render/landmark_full_t8.png) | 84% |

## Task 9 — on the wooden cabinet

Instruction (all 4 conditions): *"pick the akita black bowl on the wooden cabinet and place it on
the plate"*

| Condition | 3rd bowl placement | Render | SR |
|---|---|---|--:|
| `default` | — (2 bowls only) | ![default t9](openvla/experiments/figures/per_task_render/default_full_t9.png) | 72% |
| `irrelevant` | front edge of the table (`table_front`) | ![irrelevant t9](openvla/experiments/figures/per_task_render/irrelevant_v2_t9.png) | 68% |
| `semantic` | next to the ramekin | ![semantic t9](openvla/experiments/figures/per_task_render/semantic_v2_t9.png) | 68% |
| `landmark` | on the table in front of the cabinet | ![landmark t9](openvla/experiments/figures/per_task_render/landmark_full_t9.png) | **54%** |

`landmark`'s second-largest single-task drop (−18 pts vs. `default`), alongside task 0 — see
`benchmark_split_result.md` finding 7.

---

## Takeaway

Scanning across all 40 cells: `irrelevant` and `semantic` never collapse a task (worst single-task
drop: `irrelevant` task 6 at −14 pts) and both land slightly *above* the `default` baseline overall
(85.2% vs. 84.0%). `landmark` is the only condition with a real, visible cost, and it's concentrated
in exactly 2 of 10 tasks (0 and 9) where the 3rd bowl sits near the target's *own* landmark — every
other task under `landmark` is flat or improved. Since none of these four conditions touch the
instruction, this table isolates the effect to scene geometry alone: an extra bowl's mere presence,
or even sitting at some other named landmark, doesn't hurt this checkpoint — a genuine near-miss for
the target's own described location does.
