# RoboLab results

One section per run, newest last. Numbers come from `results/robolab/<run_id>/summary.csv`, built from
RoboLab's `episode_results.jsonl` by `robolab_lang/make_run_record.py`. Launch details are in
[`eval_log.md`](eval_log.md). Only runs with stage `main` go into result tables. Smoke, gate, screen and
exploratory runs are listed below and labelled. They are checks, not findings.

SR is RoboLab's own success flag. The 95% CI is Wilson.

## Main results

None yet.

## Checks (smoke and gate), π0.5-DROID, `default` instruction, Isaac Sim 5.0 / Isaac Lab 2.2.0, RoboLab `b6cfcf6`

| Run ID | Stage | Task | n | SR | 95% CI | mean score | episodes with `WRONG_OBJECT_GRABBED` |
|---|---|---|---:|---:|---|---:|---:|
| `20261021-1551_robolab_none_smoke_nj21` | smoke (no policy, install check) | BananaInBowlTask | 1 | — | — | — | — |
| `20261021-1552_robolab_pi05_smoke_nj21` | smoke | BananaInBowlTask | 1 | 0/1 | — | 0.00 | 0 |
| `20261021-1555_robolab_pi05_smoke_nj21` | smoke | BananaInBowlTask | 10 | 9/10 | 60–98% | 0.90 | 0 |
| `20261021-1600_robolab_pi05_gate_nj21` | gate A | BananaInBowlTask | 50 | 46/50 (92%) | 81–97% | 0.92 | 1 |
| `20261021-1600_robolab_pi05_gate_nj21` | gate A | RubiksCubeAndBananaTask | 50 | 36/50 (72%) | 58–83% | 0.84 | 6 |

Gate A was set at ≥ 85% per task. BananaInBowl passed. RubiksCubeAndBanana was accepted by the user on
2026-10-23 because its failures are grasp failures (see `eval_log.md`). The leaderboard publishes no
per-task numbers, so neither row can be compared with the board.

### Outcome fields: what they mean and their limits

Each row of `episodes.jsonl` carries, from RoboLab's HDF5:

- `lifted_objects` / `first_lifted`: objects whose root rose more than 3 cm.
- `in_container`: object centroid inside a container's bbox footprint at the last step.
- `pair_*`: mirror-pair labels for the native pair tasks.

`in_container` is looser than RoboLab's success predicate, which also requires contact and an open
gripper. In the gate run, every successful episode has its objects in the bowl. Eight failed episodes
also do: the object ended over the bowl but was still held, or rested on the rim at timeout.

## Screens (not results), π0.5-DROID, Isaac Sim 5.0 / Isaac Lab 2.2.0, RoboLab `b6cfcf6`

### Option A: `20261028-0848_robolab_pi05_screen_aws-h200_{default,blank,no_target}` — screen

13 task files × 3 instruction keys × 20 episodes = 780 episodes (`NUM_ENVS=10`, `NUM_RUNS=2`), all recorded. Run by
Wen-Han Hsieh on an AWS H200 with Apptainer, not the RTX + Docker setup of Gate A (`eval_log.md`). Records came through
[PR #9](https://github.com/Qian-0203/vla_ws/pull/9) (merged 2026-10-31) and were checked against each
run's `SHA256SUMS`. **Screen: each cell repeats one fixed object layout 20 times (below), so the Wilson CI understates
the uncertainty. Not a result table.**

SR is RoboLab's success flag for the task class. Overall: `default` 116/260 (44.6%), `blank` 8/260 (3.1%), `no_target`
89/260 (34.2%). The `no_target` total is not comparable with `default`: without the target word, the task's check
still requires one particular answer, so most `no_target` cells measure whether the policy happens to match that check.

| Task | `default` instruction | `default` | 95% CI | `blank` | `no_target` | `no_target` instruction |
|---|---|---:|---|---:|---:|---|
| MustardInLeftBinTask | Put the mustard in the left bin | 8/20 | 22–61% | 2/20 | 3/20 | Put the mustard in the bin |
| MustardInRightBinTask | Put the mustard in the right bin | 8/20 | 22–61% | 0/20 | 0/20 | Put the mustard in the bin |
| BowlStackingLeftOnRightTask | Stack the left bowl on the right bowl | 4/20 | 8–42% | 2/20 | 12/20 | Stack one bowl on the other bowl |
| BowlStackingRightOnLeftTask | Stack the right bowl on the left bowl | 11/20 | 34–74% | 4/20 | 10/20 | Stack one bowl on the other bowl |
| SmallerObjectButterInBinTask | Place the smaller object in the grey bin. | 18/20 | 70–97% | 0/20 | 16/20 | Place the object in the grey bin. |
| LargerObjectRaisinBoxInBinTask | Place the larger object in the grey bin. | 0/20 | 0–16% | 0/20 | 3/20 | Place the object in the grey bin. |
| RubiksCubeOrBananaTask | Put the cube or the banana in the bowl | 20/20 | 84–100% | 0/20 | 18/20 | Put the object in the bowl |
| RubiksCubeAndBananaTask | Put the cube and the banana in the bowl | 18/20 | 70–97% | 0/20 | 14/20 | Put the objects in the bowl |
| BananaThenRubiksCubeTask | put the banana then the cube in the bowl | 13/20 | 43–82% | 0/20 | 1/20 | Put the objects in the bowl, one then the other |
| RubiksCubeThenBananaTask | Put the cube then the banana in the bowl | 0/20 | 0–16% | 0/20 | 12/20 | Put the objects in the bowl, one then the other |
| RubiksCubeBehindBowlTask | Put the rubiks cube behind the bowl | 14/20 | 48–85% | 0/20 | 0/20 | Put the rubiks cube near the bowl |
| RubiksCubeLeftOfBowlTask | Put the rubiks cube to the left of the bowl | 2/20 | 3–30% | 0/20 | 0/20 | Put the rubiks cube near the bowl |
| RubiksCubeInFrontOfBowlTask | Put the rubiks cube in front of the bowl | 0/20 | 0–16% | 0/20 | 0/20 | Put the rubiks cube near the bowl |

What the which-object fields show (`first_lifted`, `in_container`; limits as described above):

- **Size words.** "Smaller" and "larger" both lift the butter first (19/20 each). "Place the object in the grey bin."
  also lifts the butter first (36/40 across the two task checks). "Smaller" therefore succeeds without evidence that
  the word is read. Fixed layout: this cannot separate the butter's position from its identity.
- **Order words.** "Banana then cube" and "cube then banana" lift the banana first (20/20, 16/20). "Put the objects in
  the bowl, one then the other" lifts the cube first (36/40), and `blank` lifts the banana (74/140 across the seven
  tasks of this scene; the cube once). The first pick changes with the sentence but does not follow the stated order.
  Without the order words, the official cube-then-banana check goes from 0/20 to 12/20.
- **Side words.** Told "right bin", the mustard reaches the right bin 8/20 and the left bin 6/20. Told "left bin": left
  8/20, right 0/20. With no side word: left 5/40, right 0/40. The right bin is reached only when the sentence says
  so, which is evidence that the word has an effect, together with a pull toward the left bin.
- **`blank` (".").** At least one object is lifted in 123/260 episodes, and few transfers finish. Unlike RoboCasa365,
  where the empty instruction stops the robot, this token leaves the robot moving. The two blank tokens differ
  (`""` there, `"."` here), so the two boards are not directly comparable on this condition.
- **`no_target` on the three spatial tasks.** All three use the same sentence, "Put the rubiks cube near the bowl",
  while each check requires one specific relation, so their 0/20 scores say nothing about spatial words.

Sync: records in `results/robolab/<run_id>/` on this branch. HF and the laptop have no copy of the raw runs, and the PR's tgz link is still an
unfilled placeholder (`TGZ_LINE`), so rollout videos are not available for checking yet.

### Language tests: `20261104-0833_robolab_pi05_ext-{cube,mustard,size}_berkeley3_<key>` — screen

π0.5-DROID on three original RoboLab scenes, one sentence per key (`robolab_lang/tasks/`). 20 episodes per task and key,
one fixed layout per cell (gap G1). "Requested" = the asked-for object (or bin) is in the container and the other is not;
"other" = the other object or bin received it. RoboLab's own success for the cube/banana task accepts either object, so
it is not used here. Defaults are from the Option A screen above.

**Cube or banana into the bowl** (per key, n = 20)

| Key | Sentence | Requested | Other |
|---|---|--:|--:|
| switch_cube | Put the cube in the bowl | 20 | 0 |
| switch_banana | Put the banana in the bowl | 20 | 0 |
| negation_not_banana | Put the object that is not the banana in the bowl | 1 | **17** |
| negation_not_cube | Put the object that is not the cube in the bowl | 0 | **20** |
| distractor_cube | Put the cube in the bowl, not the banana | 20 | 0 |
| distractor_banana | Put the banana in the bowl, not the cube | 20 | 0 |

**Larger / smaller object** (target: raisin box / butter; n = 20 each)

| Key | Larger: requested / butter | Smaller: requested / other |
|---|--:|--:|
| default (Option A) | 0 / — (butter lifted first 19/20) | 18 / — |
| negation ("not the butter" / "not the raisin box") | 1 / 16 | 12 / 1 |
| distractor_mention | 0 / 17 | 16 / 0 |
| irrelevant_content | 1 / 17 | 17 / 0 |

**Mustard into the left or right bin** (n = 20 per bin; default 8 + 8 = 16/40)

| Key | Left | Right | Total | Wrong bin |
|---|--:|--:|--:|--:|
| negation ("the bin that is not on the right/left") | 3 | 0 | 3 | 7 |
| distractor_mention | 6 | 9 | 15 | 2 |
| irrelevant_content | 7 | 3 | 10 | 4 |
| paraphrase_common ("container") | 5 | 18 | 23 | 0 |
| paraphrase_rare ("receptacle") | 4 | 8 | 12 | 0 |
| paraphrase_functional | 3 | 5 | 8 | 0 |
| paraphrase_zh | 0 | 0 | **0** | 0 |
| spatial_robot ("on the robot's left") | 9 | 14 | 23 | 0 |
| spatial_image ("on the left side of the image") | 6 | 4 | 10 | 3 |

Reading (two-sided Fisher tests against the default; optimistic, because each cell repeats one layout):

- **Clear:** negation in the cube scene (the excluded object is delivered 37/40); the Chinese sentence (0/40 vs 16/40,
  p < 0.001); the two wordings of the same bin (robot frame 23/40 vs image frame 10/40, p = 0.006). *(Did not replicate with random
  layouts, 2026-11-15: 12/40 vs 10/40, p = 0.80; see the A1 section below.)*
- **Not clear:** the English rewordings (8–23/40, p = 0.09–0.48), filler (10/40, p = 0.23) and distractor mention
  (15/40) against the default; neither spatial wording differs clearly from the default (p = 0.18 and 0.23).
- **Switch is followed** in the cube/banana scene, which has no habitual answer. The size task picks the butter
  whatever the size word says; that is a test of size words, not of switching.
- **Mustard negation is confounded** with the pull toward the robot's left bin.
- The policy's camera sees left and right as the robot does (checked on one frame), so "the robot's left" and "the left
  side of the image" name the same bin; the viewer videos are mirrored.

### Scene changes, 2026-11-15: `20261115-1617_robolab_pi05_ext-*_berkeley3_<key>` — screen

π0.5-DROID, same stack and scoring as the language tests above; 20 episodes per task and key. Three changes, decided in
the scene assessment of 2026-11-15: **A1** reruns the main keys with RoboLab's ±10 cm initial-pose randomization
(`*Rand10Task`), so every episode has its own layout (closes gap G1 for these keys); **A3** tries the bowl-stacking
scene as a second left/right task; **A4** swaps the butter and the raisin box (each keeps its own height and
orientation) to separate an object habit from a position habit.

**A1, cube or banana into the bowl** (n = 20 per key; fixed-layout numbers from the section above)

| Key | Sentence | Requested | Other (excluded) | Fixed layout: requested / other |
|---|---|--:|--:|---|
| default | Put the cube or the banana in the bowl | cube 8, banana 12 | — | — |
| switch_cube | Put the cube in the bowl | 19 | 0 | 20 / 0 |
| switch_banana | Put the banana in the bowl | 20 | 0 | 20 / 0 |
| negation_not_banana | Put the object that is not the banana in the bowl | 0 | **19** | 1 / 17 |
| negation_not_cube | Put the object that is not the cube in the bowl | 2 | **16** | 0 / 20 |

**A1, mustard into the left or right bin** (n = 20 per bin; requested bin)

| Key | Left | Right | Total | Wrong bin | Fixed layout total |
|---|--:|--:|--:|--:|--:|
| default | 6 | 7 | 13 (success 10) | 0 | 16 |
| negation | 1 | 0 | 1 | 10 | 3 |
| paraphrase_zh | 0 | 0 | **0** | 0 | 0 |
| spatial_robot | 5 | 7 | 12 | 1 | 23 |
| spatial_image | 5 | 5 | 10 | 2 | 10 |

The default total counts the mustard in the requested bin (13); RoboLab's success also needs it released upright, 3 + 7
= 10/40. Fisher tests below use success.

**A1, larger / smaller object** (n = 20 each; requested / other delivered)

| Key | Larger (raisin box) | Smaller (butter) |
|---|---|---|
| default ("Place the larger/smaller object in the grey bin.") | 2 / 12 | 16 / 0 |
| negation ("…not the butter" / "…not the raisin box") | 1 / 16 | 15 / 3 |

**A4, swapped positions** (n = 20 each; requested / other delivered; first object lifted)

| Key | Sentence | Larger: raisin box / butter | Smaller: butter / raisin box | First lifted |
|---|---|---|---|---|
| default | Place the larger/smaller object in the grey bin. | 0 / 14 | 18 / 0 | butter 40/40 |
| no_target | Place the object in the grey bin. | 0 / 18 | 18 / 0 | butter 40/40 |

**A3, bowl stacking** (default only): "Stack the left bowl on the right bowl" 2/20, "Stack the right bowl on the left
bowl" 7/20; 9/40 in total, below the 50% bar for adding language keys.

Reading (two-sided Fisher tests; n = 40 per comparison unless stated):

- **Holds with random layouts:** cube negation (the excluded object is delivered 35/40, fixed layout 37/40); the switch
  (39/40); the Chinese sentence (0/40).
- **Does not replicate:** robot frame vs image frame (12/40 vs 10/40, p = 0.80; fixed layout 23 vs 10, p = 0.006). The
  fixed-layout gap was a property of that one layout; the claim is withdrawn.
- **Mustard default drops** to 10/40 (fixed 16/40, p = 0.23), below 50%, so the mustard keys other than Chinese are weak
  evidence either way.
- **Object habit, not position habit:** after the swap the butter is still lifted first in every episode, and the
  larger-object sentence delivers the butter as often as a sentence that names no object (14/20 vs 18/20, p = 0.24). The
  size task therefore measures this butter preference, not the understanding of size words.
- **Bowl stacking is not usable** as a language task for π0.5 (9/40).

Sync: records in `results/robolab/<run_id>/` on this branch; laptop copy verified. HF upload pending approval.
