# RoboLab language stress test: feasibility and design (phase 1)

Drafted 2026-10-21. Phase 1 only: desk research and design, no installation and no GPU use. Nothing in
this file has been run on Isaac Sim yet; every cost below is an estimate and every RoboLab fact is
quoted from the sources in §9 (RoboLab `main` at `b6cfcf6`, 2026-10-18; leaderboard JSON v55,
generated 2026-10-19).

**Goal.** Run the LIBERO-Spatial language stress tests on NVIDIA RoboLab leaderboard models to find
their language weaknesses. We are not training a model or competing for a rank. A parallel session
does the same on RoboCasa365; both use the freeze record's hypotheses P1–P7
(`git show 618b407:docs/libero_spatial_freeze.md`, §6) as the common spec, so condition keys and
outcome fields in §4–§5 are meant to be shared.

Launches go to [`eval_log.md`](eval_log.md) in this directory (decision D4 of the freeze record), not
to `docs/eval_log.md`.

## 0. Summary

| Question | Answer |
|---|---|
| Can we run RoboLab? | Yes on `berkeley` / `berkeley-2` (RTX PRO 6000 Blackwell is Isaac Sim's "ideal" GPU; driver 580.178.04 is in the tested R580 branch). Not on dgx1 (A100 has no RT cores; Isaac Sim lists it as unsupported), but dgx1 could host a policy server. Not on the laptop (8 GB < 16 GB minimum). |
| Which models first? | π0.5-DROID (required), π0-FAST-DROID, Cosmos3-Nano-Policy-DROID, GR00T N1.6-DROID; DreamZero-DROID in batch 2 (needs two GPUs). |
| Mirror pairs? | RoboLab-120 has two strict pairs (identical objects, only the sentence picks the target): `MustardInLeftBin/RightBin` and `BowlStackingLeftOnRight/RightOnLeft`. Not enough for P2, so we add three scene families (12 scene × target cells) built with `/robolab-scenegen`. |
| Which-object logging? | Offline from RoboLab's own per-step HDF5 (object poses, EE pose, bounding boxes) plus its `WRONG_OBJECT_GRABBED` events; no RoboLab fork needed. |
| First GPU ask | Smoke + gate A + language pilot on the native pairs, π0.5 only: about 10 GPU-hours. Gate B (full RoboLab-120 `default`, reproduces 28.0%) adds about 36 GPU-hours. |

## 1. Hardware feasibility

### 1.1 Requirements (sources §9)

- RoboLab: Isaac Sim 5.0 + Isaac Lab 2.2.0 (`--extra isaac50`, default) or Isaac Sim 5.1 + Isaac Lab
  2.3.2.post1 (`isaac51`); Python 3.11; Ubuntu 22.04+; about 8 GB disk (7 GB assets); "NVIDIA RTX GPU
  required, recommend 48GB+ VRAM"; "30 GPU hours / 100 tasks" (benchmark page: about 40 GPU-hours
  for 120 tasks × 10 episodes).
- PhysX differs between 5.0 and 5.1, so "results are best compared against runs on the same stack".
  Every leaderboard row that states its stack used **Isaac Sim 5.0.0 / Isaac Lab 2.2.0**; the π0.5
  row was run at RoboLab `v0.1.0` (commit `0954575`, 2026-05-09). We therefore use `isaac50`.
- Isaac Sim 5.1 requirements: minimum RTX 4080, ideal **RTX PRO 6000 Blackwell**; 16 GB VRAM minimum,
  48 GB ideal; tested Linux driver **580.65.06**; "GPUs without RT Cores (A100, H100) are not
  supported". The container page says 5.1.0 is "no longer supported" upstream, which matters only if
  we hit a bug that needs a fix.
- Forum reports tie Blackwell start-up crashes in Isaac Sim 5.1 to the **595** driver branch; NVIDIA
  staff point to R580 as the validated branch. Do not upgrade the servers to 595.

### 1.2 Machines

| Machine | Verdict | Notes |
|---|---|---|
| `berkeley`, `berkeley-2` (4× RTX PRO 6000 Blackwell Server Edition, 98 GB, driver 580.178.04, compute cap 12.0) | **Run here** | Same R580 branch as the tested 580.65.06. Isaac Sim needs Vulkan, not only EGL: these are GCP compute-only driver images, and our MuJoCo pitfall (missing `libnvidia-gl-*-server`) applies to Vulkan too. Check `vulkaninfo --summary` / the Isaac Sim compatibility checker inside the container before anything else. Both hosts are shared (`vaclis` used about 83 GB per GPU on `berkeley-2`; on 2026-10-10 only GPU 3 of `berkeley` had room), so free VRAM, not the GPU, is the binding constraint. |
| dgx1 (8× A100 80 GB, Ken Zheng's machine) | **No simulation**; possible policy host | No RT cores → Isaac Sim unsupported. The server–client split means a policy server (e.g. DreamZero, 2×80 GB) could run there with the simulator on `berkeley`, if network latency is acceptable and Ken agrees. Not planned for batch 1. |
| Laptop (RTX 5060 Laptop 8 GB, driver 580.178.04, Ubuntu 24.04) | **No eval** | Below the 16 GB minimum. Usable for scene text editing, `frame_check.py`, vocabulary counts and dashboard viewing. It already holds a local `isaac-lab-base` image (Isaac Lab 2.1.1 on Isaac Sim 5.1 assets, built for another project), which says nothing about RoboLab's 2.2.0 stack. |

### 1.3 Installation path (to be approved, §8)

Docker, matching how we run LIBERO:

1. `docker pull nvcr.io/nvidia/isaac-lab:2.2.0` (public NGC image; tens of GB) and build RoboLab's
   `docker/Dockerfile` on top (copies `assets/`, about 6.5 GB via git-lfs; pins `openpi-client` to
   `xuningy/openpi@aa64205`). Pin the RoboLab commit we build from in `eval_log.md`.
2. Run the container with `--gpus '"device=N"'` (one GPU per simulator; Isaac Sim otherwise sees all
   GPUs), `-e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES`, `--headless` always (known VRAM leak in GUI
   mode), cache volumes for `~/.cache/ov` and the Kit shader cache (first start compiles shaders).
3. Policy servers run in their own containers / `uv` envs (RoboLab says not to share the venv). For
   the π family use the **`xuningy/openpi` fork** (it has the `*_droid_jointpos` configs; upstream
   openpi, which we have at `~/openpi`, does not).
4. VRAM budget per GPU: simulator at `--num-envs 10–25` (L40 measurements: 60–100 envs fit 48 GB for
   every RoboLab-120 task, so 10–25 envs should need well under 20 GB; measure in the smoke test) plus
   the policy server. RoboLab's π0.5 example sets `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`, which on a
   98 GB card reserves 49 GB; use `0.15` or `XLA_PYTHON_CLIENT_PREALLOCATE=false` on shared hosts.
5. Existing LIBERO containers can share a GPU with Isaac Sim (nothing is exclusive), but a co-resident
   LIBERO job steals both VRAM and step time; for timing numbers run alone.

## 2. Model inventory

Leaderboard RoboLab-120 rows (SR on the `default` instruction, 10 episodes per task unless noted);
"vague / specific" from the same JSON. All rows that state it ran Isaac Sim 5.0.0.

| # | Policy | SR default (vague / specific) | Weights | RoboLab client | Size, VRAM | Sim data in training |
|--:|---|---|---|---|---|---|
| 1 | Lakeside | 51.0 (25 eps/task) | closed | — | 5B, 27 GB | other sim (1.6%) |
| 2 | FLUX 3 Action | 42.9 | listed "open", no link | — | 7B, 69 GB | — |
| 3 | Cosmic | 42.3 | closed | — | ~350M, 10 GB, RGB-D | — |
| 4 | HiDream-O1-Embodied | 39.9 | closed | — | 6B | RoboLab in fine-tuning |
| 5 | Atomic-WAM | 39.6 (vague 31.6) | open (HF dataset repo) | via cosmos-framework | 16.2B, 40 GB | **RoboLab in fine-tuning** |
| 6 | OASIS WAM | 39.0 | closed | — | 16B + VLM | none stated |
| 7 | **Cosmos3-Nano-Policy** | 36.8 (20.6 / 39.7) | `nvidia/Cosmos3-Nano-Policy-DROID`, 32.9 GB, OpenMDW-1.1 | `cosmos3` | 16B, 40 GB | none stated |
| 8 | Phoenix | 34.4 | closed | — | ~250M, RGB-D | other sim |
| 9 | BiMind v0.1 | 33.3 (30.2 / 30.6) | closed | — | 7B | RoboLab in training + fine-tuning |
| 10 | VoLo | 28.2 (29.5 / 31.4) | agent around π0.5 | `volo` | 80 GB | other sim |
| 11 | **π0.5** | 28.0 (15.2 / 28.1) | `gs://openpi-assets-simeval/pi05_droid_jointpos`, 12.4 GB | `pi0_family` | 3.3B, >8 GB | none |
| 12 | **DreamZero** | 25.7 (14.9 / 23.9) | `GEAR-Dreams/DreamZero-DROID`, 64.8 GB, CC-BY-NC-4.0 | `dreamzero` | 14B, 2×80 GB | none stated |
| 13 | Cosmos3-Edge-Policy | 22.9 (15.4 / 28.8) | `nvidia/Cosmos3-Edge-Policy-DROID`, 9.2 GB, OpenMDW-1.1 | `cosmos3` (assumed) | 4B, 9.2 GB | none stated |
| 14 | **π0-FAST** | 15.5 (9.2 / 14.9) | `gs://openpi-assets-simeval/pi0_fast_droid_jointpos`, 10.8 GB | `pi0_family` | 3B, >8 GB | none |
| 15 | **GR00T N1.6** | 7.2 (5.4 / 5.3) | `nvidia/GR00T-N1.6-DROID`, 6.6 GB | `gr00t` | 3B, 24 GB | none |
| 16 | π0 | 5.0 | `gs://openpi-assets-simeval/pi0_droid_jointpos`, 12.0 GB | `pi0_family` | 3.3B | none |
| 17 | paligemma-binning | 3.4 | same bucket | `pi0_family` | 3B | none |

Also available but not on the board: `nvidia/GR00T-N1.7-DROID` (6.9 GB, NVIDIA Open Model License; needs
gated `nvidia/Cosmos-Reason2-2B`), and π0.5 variants in the same bucket whose names suggest training
ablations (`pi05_droid_jointpos_nocotrain` 24.9 GB, `pi05_droid_jointpos_sim10`,
`droid_pi05_jointpos_with_web_and_sim` 99.5 GB each). Their training recipes are not documented in
RoboLab or the fork; treat them as exploratory until the authors confirm what they are.

Licences: openpi code is Apache-2.0 (the fork's `LICENSE`); check the openpi README's terms for the
checkpoints before redistributing anything derived. DreamZero weights and the `/robolab-scenegen`
skill are CC-BY-NC-4.0 (research use is fine). RoboLab itself is Apache-2.0.

**Leaderboard SR is the `default` instruction only.** The vague → default gap is already a language
effect: π0.5 loses 12.8 points, Cosmos3-Nano 16.2, DreamZero 10.8, while BiMind (RoboLab data in
training) and VoLo (agent) lose about 3. That is a cheap first finding to cite, but it mixes
instruction quality with grounding; the conditions below separate them.

### 2.1 Recommended first batch

| Model | Why |
|---|---|
| **π0.5-DROID** (required) | Same architecture as our `pi05_libero` (openpi `Pi0Config(pi05=True)`, both from `pi05_base`), different fine-tuning data (DROID joint position vs. LIBERO). Directly tests whether LIBERO findings H1–H3 come from the architecture or from LIBERO's data. Cheapest server, known stack. |
| **π0-FAST-DROID** | Same DROID data and openpi stack, different backbone use and action tokenisation; same server code, so almost free once π0.5 runs. Separates "π-family on DROID" from "π0.5 specifically". Low SR (15.5%) is a floor risk: keep it only if it passes the per-model gate (§5.2). |
| **Cosmos3-Nano-Policy-DROID** | Highest open-weight model with a shipped client; a world-action model (video backbone), so a different architecture family; largest vague→default drop on the board (16.2). 40 GB server fits next to the simulator on one 98 GB GPU. |
| **GR00T N1.6-DROID** | NVIDIA's VLA on the board with open weights and a client. Its board SR (7.2%) makes a floor effect likely; run it through the gate only, and switch to N1.7-DROID (validated client, not on the board) if it fails. |
| DreamZero-DROID (batch 2) | Second world-action model, 25.7%, but needs two GPUs per server; schedule when two GPUs are free. |

Not in batch 1: Atomic-WAM (fine-tuned on RoboLab data, so the scenes are in-distribution; useful later
as an "in-distribution" contrast), Cosmos3-Edge (keep as a cheaper Cosmos substitute), closed models.

## 3. Mirror-pair scenes (P1, P2)

### 3.1 What RoboLab-120 already has

From the scene each task imports (47 scenes, 120 tasks) and the scene USDA files:

| Class | Tasks | Scene | Use |
|---|---|---|---|
| **Strict mirror pair** (identical objects, only the sentence picks the target) | `MustardInLeftBinTask` / `MustardInRightBinTask` | `two_bin.usda`: one mustard at y = 0, two `grey_bin.usd` at y = ±0.29 | destination choice; symmetric |
| **Strict mirror pair** | `BowlStackingLeftOnRightTask` / `BowlStackingRightOnLeftTask` | `bowls_2_table.usda`: two `ycb/bowl.usd`, at (0.49, −0.19) and (0.34, 0.00) | object choice; **asymmetric layout** (one bowl is nearer the robot) |
| Same scene, relation picks the goal | `RubiksCube{LeftOf,Behind,InFrontOf}BowlTask` (`RightOf` uses another scene) | `rubiks_cube_banana_bowl.usda` | spatial goal wording (P7), not a target-identity test |
| Same scene, attribute picks the target (objects differ) | `BigPumpkin`/`SmallPumpkin`, `LargerObjectRaisinBox`/`SmallerObjectButter`, `FoodPacking{1,2,3}Boxes`/`Cans`, `FruitsOnion`/`FruitsOnionToPlate`, `ClampInRightBin`/`HammersInLeftBin` | various | layout-prior check (a policy that ignores the sentence can pass at most one of each group), but objects are visually distinct, so not a clean language test |

Two strict pairs is LIBERO's L5 problem again (one pair there). They are worth running because they are
official, but P2 needs more.

Two facts from the scenes matter for the design:

- **Every RoboLab-120 task starts from one fixed layout.** No benchmark task attaches a pose
  randomisation event; the 10 episodes per task differ only by policy sampling and physics.
  `robolab/tasks/randomize_initial_pose/` (±10/20/30 cm variants of two tasks) is separate.
- **Robot-frame "left" is not image "left" in the policy camera.** `frame_check.py` (§6) projects
  the scenes into `over_shoulder_left_camera` (mounted at the robot's left, looking right-and-down),
  letterboxed to 224×224 as the π client sends it. Mustard bins: robot-left bin is image-left (u 105
  vs 143), consistent. Bowl stacking: the two bowls land in the **same image column** (u 140 vs 141)
  and differ only in height (v 113 vs 128), so "left bowl" has no image-left counterpart. In
  `rubiks_cube_banana_bowl`, "bowl left of banana" in the robot frame is not image-left either.
  DROID's own instructions use "left/right" thousands of times with no stated frame (§4.3).

### 3.2 Scene families to generate

Three families, each a set of USDA scenes built with `/robolab-scenegen` into our repo (RoboLab is a
dependency, not forked; `docs/task.md` supports external scenes and tasks). Objects come from RoboLab's
catalog; each family uses *two identical instances* of the target class so that only the instruction
can pick one.

| Family | Scene | Variants | Targets per variant | What it tests |
|---|---|---|---|---|
| **F1 twin bins + landmarks** (destination) | `two_bin` geometry: object to move at the centre, two identical `grey_bin` at y = ±0.29; landmark A (e.g. a mug) beside one bin, landmark B (e.g. a banana) beside the other | `s0` landmark A on the left, `s1` swapped | 2 (bin next to A, bin next to B) | P1/P2 landmark grounding; side and landmark decorrelated by the swap |
| **F2 twin bowls + plate** (object, LIBERO analogue) | two identical `ycb/bowl` at equal distance from the robot (same x, y = ±0.18), a plate centred in front, landmark A / B beside each bowl | `s0`, `s1` swapped | 2 | the LIBERO tasks 7/9 design with many more instances; negation and contrast (P3, P5) |
| **F3 depth pair** (frame) | two identical bowls on the robot's centre line, one near (x ≈ 0.35) and one far (x ≈ 0.60), plus a plate | `s0`, `s1` (landmarks swapped between near and far) | 2 | P7: near/far ("closer to you") vs. robot left/right vs. image-frame words; the image-vs-robot conflict made deliberate |

That is 3 families × 2 variants × 2 targets = **12 cells**, plus the 4 native cells = 16. Each scene also
gets a small pose randomisation (±2 cm, ±10° yaw on the movable objects, `RandomizeInitPoseUniform`)
so a cell is not one frozen layout; official tasks run unrandomised when we reproduce the board.

Scene acceptance (before any policy run): RoboLab's scenegen stability check (300 settle steps,
< 2 cm drift), `frame_check.py` on the final USDA, one rendered frame from `over_shoulder_left_camera`
saved under `docs/robolab/figures/`, and the two twin objects confirmed identical (same payload asset,
same scale) by text diff of the USDA.

### 3.3 Which-object outcome logging

RoboLab already records, per step, every rigid object's root pose, the EE pose, oriented bounding
boxes, subtask status, and per episode `events` (incl. `WRONG_OBJECT_GRABBED` with the object names)
in `output/<run>/<env>/run_<i>.hdf5` and `episode_results.jsonl`. We compute LIBERO's outcome fields
offline from those files (`robolab_lang/outcomes.py`, to be written against the first smoke output):

| Field | Definition (same thresholds as the LIBERO `RolloutOutcomeTracker`) |
|---|---|
| `target_object`, `distractor_objects` | from the task's instruction key (written by our task subclass into a sidecar JSON per condition) |
| `first_object_approached`, `_step` | first twin whose centroid comes within 8 cm of the EE position |
| `first_object_grasped`, `first_grasp_step` | first twin lifted > 3 cm above its settled height (cross-checked with RoboLab's contact-based grasp events) |
| `delivered` | for destination tasks: which container holds the object at episode end (bbox containment, RoboLab's `object_in_container` geometry); for object tasks: which twin ends on the goal |
| `outcome` | one of target delivered / wrong twin delivered / wrong twin grasped / target grasped only / no grasp |
| `min_dist_by_object`, `approach_step_by_object`, `grasp_step_by_object` | per twin |

RoboLab's success term only ends an episode on the *target* goal, so a wrong-target delivery runs to
timeout. We keep that (it preserves the official success semantics) and pay the time.

## 4. Conditions (P1–P7 → instruction keys)

### 4.1 Mechanism

A RoboLab task's `instruction` is a dict; `--instruction-type <key>` exposes only that field, and every
shipped text client sends `default` if present, else the first exposed field. So each condition is one
key, selected per run, and needs no client change. We do not edit RoboLab task files: for each task we
use, `robolab_lang/tasks/` defines a subclass that copies the official `default` / `vague` / `specific`
strings unchanged, adds our keys, and sets `task_name` to the official name so RoboLab's analysis
tools group the runs. `episode_results.jsonl` records `instruction`, `instruction_fields` and
`instruction_fields_used` per episode, which is our audit trail that the intended key was sent.

### 4.2 Keys (proposed shared spec with the RoboCasa session)

Notation: target T, the other twin D, landmark L_T beside T, L_D beside D. Examples are for F1 with the
mustard ("Put the mustard in the bin next to the mug"); final texts are frozen per task in
`robolab_lang/tasks/` before the run.

| Key | P | Example | Measures |
|---|---|---|---|
| `default`, `vague`, `specific` | — | official strings | baseline; the board's own language axis |
| `p1_target_swap` | P1 | "…in the bin next to the banana" (= the twin cell's `default`) | discrimination index (below) |
| `p1_no_location` | P1 | "Put the mustard in the bin" | layout prior: which twin is chosen with no cue |
| `p3_neg_not` | P3 | "Put the mustard in the bin that is not next to the banana" | negated-twin deliveries |
| `p3_neg_except` | P3 | "Put the mustard in any bin except the one next to the banana" | idem, other surface form |
| `p3_neg_other` | P3 | "Put the mustard in the other bin, not the one next to the banana" | idem |
| `p5_contrast` | P5 | "Put the mustard in the bin next to the mug, not the bin next to the banana" | drop vs `default`; commit-to-D vs commit-to-nothing |
| `p5_len_ctrl` | P5 | "Put the mustard in the bin next to the mug, and then let go of it gently" | length-matched content-free control |
| `p4_syn_common` | P4 | "…in the container next to the cup" (DROID-frequent synonyms) | lexical ladder rung 1 |
| `p4_syn_rare` | P4 | "…in the receptacle beside the mug" (absent from DROID) | rung 2 |
| `p4_functional` | P4 | "…in the thing you throw rubbish into, next to the mug" | rung 3 |
| `p4_zh` | P4 | 「把芥末醬放進馬克杯旁邊的桶子裡」 | rung 4 |
| `p6_freq_relation` | P6 | "…in the bin near the mug" (relation word frequent in DROID) | familiar relation template |
| `p6_novel_relation` | P6 | "…in the bin alongside the mug" (relation word rare or absent in DROID) | unseen wording, same referent |
| `p7_robot_frame` | P7 | "…in the bin on the robot's left" | robot-frame relation |
| `p7_image_frame` | P7 | "…in the bin on the left side of the camera view" | image-frame relation |
| `p7_near_far` | P7 | F3: "Put the bowl closest to you on the plate" | depth relation |

P6 only partly transfers: in LIBERO the familiar template ("next to X") was bound to other *training
tasks*; DROID is too diverse for task-bound templates, so here P6 compares frequent vs. unseen relation
wording for the same referent. A task-bound version needs a model fine-tuned on a narrow task set (e.g.
Atomic-WAM or BiMind, RoboLab data in training); note this when comparing with RoboCasa.

Native pairs use the same keys with side words in place of landmarks (`MustardInLeftBin`: `p1_no_location`
= "Put the mustard in the bin", `p3_neg_not` = "Put the mustard in the bin that is not on the right").

Metrics per (model, family, key):

- **Success rate**, mean over cells (LIBERO convention: mean of per-cell rates).
- **Discrimination index** D = P(deliver T | instruction names T) − P(deliver T | instruction names D),
  computed per scene from the `default` and `p1_target_swap` runs: 1 = reads the sentence, 0 = ignores
  it, both directions collapse to a side bias. This is P2's "language score on mirror pairs", reported
  next to the overall mean, never folded into it.
- **Side bias** from `p1_no_location`: share of deliveries to the left twin.
- **Negated-twin rate** (P3): deliveries to D over all rollouts under `p3_*`.
- **Commit profile** (P5): of failures, share that grasped D vs. grasped nothing.

### 4.3 Relation to the official vague / default / specific

The official axis varies how much is said about the same goal; it cannot separate "did not understand"
from "understood a different goal", and on the 47 scenes most targets are unique, so layout alone can
solve many tasks. Our keys keep the goal fixed and vary only the words that pick between identical
objects, with outcome logging, so every failure is attributed. We run the three official keys on the 4
native cells as the bridge: if a model's vague→default gap on those cells matches its board-wide gap,
the cells are representative of how it uses language on the board.

DROID's instruction corpus is public (31,307 distinct task strings in RoboLab's
`analysis/droid_dataset/tasks.jsonl`; episode-level annotations on HF `KarlP/droid`), so P4 and P6 can be
graded by measured frequency, which LIBERO could not do. Counts over distinct strings
(`robolab_lang/droid_vocab.py --ladder robolab_lang/ladders.txt`):

- Negation is essentially absent: "not" in 3 strings, "except" in 9, "instead of" / "rather than" in 0.
  P3 is therefore a pure zero-shot composition test for every DROID-trained model.
- "the other" occurs 231 times, "on the left" 947, "on the right" 992; "left bin" 7, "right bin" 17.
- Containers: box 2271, container 667, tray 583, bin 482, basket 477, tub 2, receptacle 0.
- RoboLab's own nouns are sometimes rare: "mustard" 4 (and partly as a colour), "rubik's cube" 64,
  "cube" 265. So RoboLab's `default` wording is not always the in-distribution rung; order rungs by the
  counts, not by intuition.

This applies to the DROID-only models (π family, GR00T, Cosmos3, DreamZero all report DROID fine-tuning
and no sim data). Their pre-training corpora (PaliGemma web data, Cosmos video captions) are not
public at the word level; say so where P4 is reported.

### 4.4 P7 check before any run

E1 in LIBERO was a direction word that did not match the camera. Here:

1. `frame_check.py` on every scene: robot-frame relation (RoboLab predicate convention: left = +y,
   behind = +x, 45° cone) vs. image-frame relation in the policy camera at 224×224; any key that uses a
   left/right/front/behind word must agree with the frame it claims, or the cell is dropped.
2. One real render per scene from `over_shoulder_left_camera` (RoboLab `run_empty.py --headless`
   records sensor video), checked by eye against the projected pixel table; this also validates the
   projection code once.
3. The check result (pass / dropped cells) is logged in `docs/robolab/eval_log.md` before the run.

## 5. Cost and pilot plan

### 5.1 Cost model

RoboLab's own timing (`robolab/tasks/_metadata/task_timing.json`, one batch per task, L40) totals 36.3
hours for 120 tasks; the short tasks we use take 2.9–3.9 minutes per batch of 10 episodes (e.g.
`MustardInLeftBin` 235 s, `BowlStackingLeftOnRight` 176 s, `BananaInBowl` 227 s), i.e. **about 0.4
GPU-minutes per episode for π0.5**. We assume the same on an RTX PRO 6000 (faster GPU, but shared) and
**3× for the 14–16B world-action models** (inference time unknown; measure in each model's smoke test
and re-plan).

| Block | Cells | Keys | Episodes / cell·key | Episodes | π0.5 GPU-h | Heavy model GPU-h |
|---|---:|---:|---:|---:|---:|---:|
| Native pairs, official keys | 4 | 3 | 30 | 360 | 2.4 | 7.2 |
| Native pairs, our keys (all §4.2 keys that apply) | 4 | 14 | 30 | 1,680 | 11 | 34 |
| Families F1–F3, core keys (`default`, swap, no_location, 3 neg, contrast, len_ctrl, 2 P4, 2 P7) | 12 | 12 | 20 | 2,880 | 19 | 58 |
| Families, full ladder (remaining P4/P6 keys) | 12 | 4 | 20 | 960 | 6.4 | 19 |
| **Per model, total** | | | | **5,880** | **≈ 39** | **≈ 118** |

Batch 1 (π0.5, π0-FAST, Cosmos3-Nano, GR00T N1.6) ≈ 39 + 39 + 118 + 39 ≈ **235 GPU-hours**, about 2.5 days
on four GPUs if the hosts are free. Per-cell CI at 20–30 episodes is ±18–22 points (as in LIBERO's
per-task numbers), so claims are made on pooled cells (240–360 rollouts per key and family).

### 5.2 Pilot (the first GPU request)

All π0.5 on one GPU of `berkeley` or `berkeley-2`, RoboLab pinned at the commit we build, `isaac50`.

| Step | What | Episodes | GPU-h | Pass criterion |
|---|---|---:|---:|---|
| 0 | Install check: Isaac Sim compatibility checker in the container; `run_empty.py --headless`; `pytest tests/` | — | 0.5 | checker PASSED, one episode runs |
| 1 | Smoke: `BananaInBowlTask`, 1 env, then 10 envs | 11 | 0.3 | episode completes, HDF5 written, VRAM recorded |
| 2 | **Gate A**: `BananaInBowlTask` and `RubiksCubeAndBananaTask`, `default`, 50 episodes each | 100 | 1 | ≥ 85% each (paper Table II: π0.5 90–100% under nominal-like lighting on these two tasks) |
| 3 | **Gate B** (recommended, the only check against a published number): RoboLab-120 `default`, 10 episodes per task, 4 GPUs in parallel | 1,200 | ≈ 36 (≈ 9 h wall) | overall SR within the board's 95% CI for π0.5 (28.0%, 336/1200; Wilson ≈ 25.5–30.6%) |
| 4 | Frame check + native-pair language pilot: 4 native cells × official 3 keys + 6 core keys (`p1_*`, `p3_neg_not`, `p5_contrast`, `p5_len_ctrl`) × 20 episodes | 720 | ≈ 5 | none (pilot); decide whether the native pairs have enough base SR to be informative |
| 5 | `outcomes.py` written and checked against the step-4 HDF5 (by-eye on 10 videos) | — | 0 | outcome labels match videos |

Steps 0–2 and 4: **≈ 7 GPU-hours** (≈ 10 with retries). Step 3 adds ≈ 36. If Gate A fails, stop and
debug (driver, Vulkan, stack version, image transform) before anything else. If Gate B fails while Gate A
passes, suspect stack drift (5.0 vs 5.1, RoboLab `v0.1.0` → current) and check against the `v0.1.0`
tag before trusting any language number.

Per-model gate for the other batch-1 models: smoke + `default` on the 4 native cells and Gate A tasks
(≈ 140 episodes). A model whose SR on the native cells' `default` is below 30% is not run on the full
condition set (floor effect: a language drop cannot be measured below that).

## 6. Code in this repo

`robolab_lang/` (new, outside the LIBERO tree):

| File | Status |
|---|---|
| `frame_check.py` | written; pure numpy projection of USDA object positions into the policy camera, robot vs. image relation per pair. Output in §3.1 came from it |
| `droid_vocab.py`, `ladders.txt` | written; word/phrase counts over the DROID corpus; counts in §4.3 came from it |
| `tasks/` | to write: subclasses of the official tasks with our instruction keys; family task files for F1–F3 |
| `scenes/` | to generate with `/robolab-scenegen` (phase 2) |
| `outcomes.py` | to write against the first real HDF5 (pilot step 5) |
| `run_robolab.sh` | to write: docker launcher (one GPU per simulator, cache mounts, `--instruction-type`, `--output-folder-name` encoding model / key / note) |

## 7. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| Floor effects: most board SRs are 5–37% | language drops not measurable for weak models | per-model gate (§5.2); twin tasks are simple pick-and-place |
| Two strict native pairs only | P2 rests on our generated scenes | scene acceptance checks (§3.2); report native and generated cells separately |
| Stack drift (board used Isaac Sim 5.0 and older RoboLab) | numbers not comparable to the board | `isaac50`, gate B, pin commit |
| GCP compute-only driver lacks Vulkan/GL libs | Isaac Sim does not start | install `libnvidia-gl-580-server`, regenerate CDI (same fix as MuJoCo EGL) |
| Driver upgrade to 595 on the hosts | known Blackwell crashes | stay on R580 |
| Shared GPUs (`vaclis` ≈ 83 GB/GPU on `berkeley-2`) | OOM or slow runs | check `nvidia-smi` first; small `--num-envs`; JAX memory fraction 0.15 |
| Frame ambiguity (robot vs. image vs. DROID annotators' viewpoint) | E1 again | §4.4 checks; landmark keys as the frame-free reference |
| Policy sees a 224×224 letterboxed 16:9 image (objects ~15–40 px apart) | small-object failures read as language failures | outcome logging separates "approached the wrong twin" from "approached nothing" |
| Wrong-target episodes run to timeout | cost | budgeted (§5.1) |
| Licences: DreamZero, scenegen skill CC-BY-NC | no commercial use | research only; note in any release |
| Coordination with RoboCasa | keys diverge | §4.2 keys proposed as the shared spec; changes go to both plans |

## 8. Needs approval before phase 2

Nothing below has been done.

1. **Installs on one server** (`berkeley-2` preferred, or whichever has a free GPU): check / install
   `libnvidia-gl-580-server` and regenerate the CDI spec (sudo); `docker pull nvcr.io/nvidia/isaac-lab:2.2.0`
   and build the RoboLab image (≈ 6.5 GB assets via git-lfs); clone `xuningy/openpi` and `uv sync`.
2. **Downloads**: `pi05_droid_jointpos` (12.4 GB) and `pi0_fast_droid_jointpos` (10.8 GB) for the pilot;
   later `Cosmos3-Nano-Policy-DROID` (32.9 GB), `GR00T-N1.6-DROID` (6.6 GB), `DreamZero-DROID` (64.8 GB).
3. **GPU runs**: pilot steps 0–2 and 4 (≈ 7–10 GPU-h, π0.5 only); gate B (≈ 36 GPU-h) as a separate
   decision.

## 9. Sources

- RoboLab repo `NVlabs/RoboLab` @ `b6cfcf6` (2026-10-18): `README.md`, `docker/`, `docs/{task,subtask,event_tracking,data,analysis,environment_run,environment_registration,camera,frames,benchmark,statistical_significance,env_vram_size_guide,known_issues}.md`,
  `robolab/tasks/benchmark/*.py`, `assets/scenes/*.usda`, `robolab/variations/camera.py`,
  `robolab/core/utils/geometry_utils.py`, `robolab/tasks/_metadata/task_timing.json`,
  `analysis/droid_dataset/`, `policies/*/README.md`, `skills/robolab-scenegen/SKILL.md`.
- Leaderboard: <https://research.nvidia.com/labs/srl/projects/robolab/leaderboard.html> and its
  `static/data/leaderboard.json` (v55, 2026-10-19).
- Paper: RoboLab, arXiv 2604.09860 v4 (HTML), §IV and Table II.
- openpi fork `xuningy/openpi` @ `aa64205` (`src/openpi/training/config.py`); GCS bucket
  `openpi-assets-simeval` listing; HF model API for the NVIDIA / GEAR checkpoints.
- Isaac Sim 5.1 requirements and container pages (docs.isaacsim.omniverse.nvidia.com); NVIDIA developer
  forum threads on Isaac Sim 5.1 with 595-branch drivers on Blackwell.
