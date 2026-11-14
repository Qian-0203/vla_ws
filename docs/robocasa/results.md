# RoboCasa365 results

One table per run, newest last. Every number traces to `results/robocasa/<run_id>/summary.csv` (small files, in the
repo), whose full run directory is backed up at `~/vla_results_backup/robocasa/<run_id>/` on the laptop and, once
synced, at HF `Qian0203/vla_ws-eval-results` under `robocasa/<run_id>/`. Launch details: `docs/robocasa/eval_log.md`.

Run ID: `YYYYMMDD-HHMM_robocasa_<model>_<stage>_<host>`. Smoke and screen runs are labelled and never enter the main
result tables (§ Main results below stays empty until a `gate`-passed `main` run exists).

## Main results

None yet.

## Gate runs

### `20261023-0927_robocasa_xr1_gate_berkeley2` — gate, **passed**

Xiaomi-Robotics-1 (HF `3a6d029`, bf16, sdpa), upstream client, published seeds, `berkeley-2` GPU 0. Rule: per task
within 14 pts and mean within 7 pts of the published results (Xiaomi-Robotics-1 `summary.json`).

| Task | n | SR | Published | Diff (pts) | Episode agreement |
|---|--:|--:|--:|--:|--:|
| NavigateKitchen | 50 | 88% | 92% | −4 | 0.88 |
| OpenStandMixerHead | 50 | 94% | 98% | −4 | 0.96 |
| PickPlaceCounterToCabinet | 50 | 88% | 86% | +2 | 0.90 |
| TurnOffStove | 50 | 58% | 60% | −2 | 0.78 |
| **Mean** | 200 | **82.0%** | **84.0%** | **−2.0** | |

Instructions and scene hashes reconstructed after the run (upstream client); no outcome fields. Sync: HF, repo,
laptop — all three. An earlier attempt, `20261023-0915_robocasa_xr1_gate_berkeley2`, aborted before any episode (port held
by a leftover process).

### `20261104-1530_robocasa_pi05_gate_berkeley3` — gate, **passed**

π0.5 (`robocasa/robocasa365_checkpoints` `pi05_pretrain_human300/multitask_learning/75000`), our client
`robocasa_lang/run_lang_eval.py --policy pi05`, Xiaomi-Robotics-1's seed rule, `berkeley-3` GPUs 0–2. π0.5 publishes
only its Atomic-Seen mean, so the rule is: mean over the 18 Atomic-Seen tasks (first 10 seeds each) within 7 pts of 39.6.

| Task | n | SR (first 10) | | Task | n | SR (first 10) |
|---|--:|--:|---|---|--:|--:|
| CloseBlenderLid | 10 | 0% | | PickPlaceCounterToCabinet | 50 | 80% (56% on 50) |
| CloseFridge | 10 | 40% | | PickPlaceCounterToStove | 10 | 70% |
| CloseToasterOvenDoor | 10 | 10% | | PickPlaceDrawerToCounter | 10 | 40% |
| CoffeeSetupMug | 10 | 30% | | PickPlaceSinkToCounter | 10 | 100% |
| NavigateKitchen | 50 | 0% (0% on 50) | | PickPlaceToasterToCounter | 10 | 0% |
| OpenCabinet | 10 | 50% | | SlideDishwasherRack | 10 | 50% |
| OpenDrawer | 10 | 60% | | TurnOffStove | 50 | 40% (18% on 50) |
| OpenStandMixerHead | 50 | 70% (72% on 50) | | TurnOnElectricKettle | 10 | 10% |
| | | | | TurnOnMicrowave | 10 | 10% |
| | | | | TurnOnSinkFaucet | 10 | 50% |
| **Mean (18 tasks)** | 180 | **39.4%** (published 39.6%) | | | | |

Only PickPlaceCounterToCabinet (28/50) and OpenStandMixerHead (36/50) reach the 50% default floor for language cells.
Sync: repo, laptop backup (checked against `SHA256SUMS`) and HF `robocasa/<run_id>/`, 2026-11-09.

## Screens, smoke and exploratory runs (not results)

### `20261023-1002_robocasa_xr1_screen_berkeley2` — screen (20 episodes per cell)

Xiaomi-Robotics-1 (HF `3a6d029`, bf16, sdpa), our client `robocasa_lang/run_lang_eval.py` (upstream EvalClient
unchanged), first 20 published seeds per task, `berkeley-2` GPU 0. **Screen: ±10–20 pts per cell; not a result
table.** "Paired default" = the gate run `20261023-0927_robocasa_xr1_gate_berkeley2` on the same seeds whose scene hash matches (count in brackets);
"Drop" = paired default − cell, in points. The stove-pair cells have one extra lit burner, so they are compared with
each other, not with the default.

| Cell | Env (slot) | Condition | SR | Paired default | Drop | Outcomes |
|---|---|---|--:|--:|--:|---|
| nav_blank | NavigateKitchen | blank | 0% | 84.2% (19) | 84.2 | none reached 20 |
| nav_no_target | NavigateKitchen | no_target | 40% | 84.2% (19) | 47.4 | named reached 8, none 12 |
| nav_target_swap | NavigateKitchenPair (b) | default (other destination) | 90% | 84.2% (19) | −5.3 | named 18, other 1, none 1 |
| nav_positive_contrast | NavigateKitchenPair (a) | positive_contrast | 70% | 84.2% (19) | 15.8 | named 14, **other 3**, none 3 |
| nav_length_control_positive | NavigateKitchenPair (a) | length_control_positive | 80% | 83.3% (18) | 5.6 | named 16, none 4 |
| nav_paraphrase_syntactic | NavigateKitchen | paraphrase_syntactic | 65% | 84.2% (19) | 21.1 | named 13, none 7 |
| pnp_blank | PickPlaceCounterToCabinet | blank | 0% | 85.0% (20) | 85.0 | target grasped only 9, none 11 |
| pnp_no_target | PickPlaceCounterToCabinet | no_target | 85% | 85.0% (20) | 0.0 | target delivered 17 |
| pnp_target_swap | PnPCounterToCabinetSwap | target_swap (names the distractor) | 30% | 80.0% (20) | 50.0 | named (distractor) delivered 7, **original target delivered 8**, original grasped 4, none 1 |
| pnp_negation_only | PickPlaceCounterToCabinet | negation_only | 65% | 85.0% (20) | 20.0 | target delivered 13, negated delivered 1, none 5 |
| pnp_positive_contrast | PickPlaceCounterToCabinet | positive_contrast | 85% | 85.0% (20) | 0.0 | target delivered 17 |
| pnp_length_control_positive | PickPlaceCounterToCabinet | length_control_positive | 80% | 85.0% (20) | 5.0 | target delivered 16 |
| pnp_paraphrase_syntactic | PickPlaceCounterToCabinet | paraphrase_syntactic | 90% | 85.0% (20) | −5.0 | target delivered 18 |
| stove_blank | TurnOffStove | blank | 0% | 65.0% (20) | 65.0 | burner left on 20 |
| stove_no_target | TurnOffStove | no_target | 30% | 65.0% (20) | 35.0 | off 6, on 14 |
| stove_paraphrase_syntactic | TurnOffStove | paraphrase_syntactic | 45% | 65.0% (20) | 20.0 | off 9, on 11 |
| stovepair_a | TurnOffStovePair (a) | default | 60% | — | — | named off only 12, **other off only 3**, none 5 |
| stovepair_b | TurnOffStovePair (b) | default | 80% | — | — | named off only 16, other off only 1, none 3 |
| mixer_blank | OpenStandMixerHead | blank | 0% | 94.4% (18) | 94.4 | — |
| mixer_paraphrase_syntactic | OpenStandMixerHead | paraphrase_syntactic | 100% | 94.1% (17) | −5.9 | — |

In `pnp_target_swap` the outcome labels are relative to the *named* object: "named delivered" = the distractor the
instruction asked for (success); "original target delivered" = the object the default instruction names, which
always sits nearer the robot (48/50 scenes, plan §3.4).

Sync: HF `robocasa/20261023-1002_robocasa_xr1_screen_berkeley2/` (418 files incl. 400 videos, checked against `SHA256SUMS`), repo
`results/robocasa/20261023-1002_robocasa_xr1_screen_berkeley2/` (`run.json`, `episodes.jsonl`, `summary.csv`, `screen_report.json`, `skipped.jsonl`,
`cells.jsonl`), laptop backup.

### `20261021-1636_robocasa_xr1_smoke_berkeley2` — smoke

Xiaomi-Robotics-1 (HF `3a6d029`, bf16, sdpa), upstream client, `berkeley-2` GPU 0. Seeds 15 and 16 (upstream seed rule
with `num_trials=2`), **not** the published seeds, so not comparable with the leaderboard.

| Task | Condition | n | Successes | SR |
|---|---|--:|--:|--:|
| NavigateKitchen | default | 2 | 1 | 50% |

Instruction and scene hash were reconstructed afterwards on the laptop (not logged by the upstream client); no
outcome fields. Sync status (2026-10-23): HF `robocasa/20261021-1636_robocasa_xr1_smoke_berkeley2/` (12 files incl. videos),
repo `results/robocasa/…`, laptop backup — all three.

### `20261107-2012_robocasa_pi05_screen_berkeley3` — screen (50 episodes per cell)

π0.5, as the gate run above; same 50 seeds as each task's default (defaults from the gate run). "Requested" = the
object the sentence asks for is in the cabinet and the other is not; "other" = the other object is in the cabinet.
Per-cell counts: `screen_report.json`.

| Cell | Condition | Example instruction (seed 407 scene) | n | Requested | Other | First grasp: requested / other / none |
|---|---|---|--:|--:|--:|---|
| default (gate run) | default | Pick the bread from the counter and place it in the cabinet. | 50 | 28 | 0 | 36 / 0 / 14 |
| pnp_target_swap | target_swap | Pick the juice from the counter and place it in the cabinet. | 40 | 9 | **16** | 11 / 19 / 10 |
| pnp_negation_only | negation_only | Pick the object that is not the juice and place it in the cabinet. | 50 | 16 | 2 | 31 / 3 / 16 |
| pnp_positive_contrast | positive_contrast | Pick the bread, not the juice, from the counter and place it in the cabinet. | 50 | 18 | 1 | 35 / 1 / 14 |
| pnp_length_control_positive | length_control_positive | Pick the bread, slowly and carefully, from the counter and place it in the cabinet. | 50 | 9 | 0 | 33 / 0 / 17 |
| pnp_paraphrase_syntactic | paraphrase_syntactic | From the counter, pick the bread and place it in the cabinet. | 50 | 31 | 0 | 39 / 0 / 11 |
| pnp_verb_synonym | verb_synonym | Fetch the bread off the counter and stow it in the cabinet. | 50 | 7 | 0 | 33 / 0 / 17 |

| Cell | Instruction | n | Success |
|---|---|--:|--:|
| default (gate run) | Open the stand mixer head. | 50 | 36 |
| mixer_paraphrase_syntactic | Open the head of the stand mixer. | 50 | 19 |

In `pnp_target_swap`, "requested" is the object the default sentence does *not* name (`distr_counter`) and "other" is
the original target; 10 of the 50 seeds have no usable second object and are skipped (`skipped.jsonl`).
Sync: repo, laptop backup (checked against `SHA256SUMS`) and HF `robocasa/<run_id>/`, 2026-11-09.

### `20261104-1528_robocasa_pi05_smoke_berkeley3` — smoke

π0.5 install check on `berkeley-3`: PickPlaceCounterToCabinet seed 407 and NavigateKitchen seed 207, both failed to the
horizon; pipeline and records complete.

### `20261114-1319_robocasa_pi05_screen_berkeley3` — screen (negation of the usual object)

π0.5, as above. Same 50 seeds as the default; 10 skipped (no usable second object). "Requested" = the counter
distractor; "usual" = the object the default sentence names.

| Cell | Condition | Example instruction (seed 407 scene) | n | Requested | Usual (negated) | First grasp: requested / usual / none |
|---|---|---|--:|--:|--:|---|
| pnp_negation_swap | negation_swap | Pick the object that is not the bread and place it in the cabinet. | 40 | **0** | **22** | 0 / 26 / 14 |

With `pnp_negation_only` (the answer is the usual object) π0.5 scores 16/50; with the usual object negated it never
delivers the requested object. Negation failure is visible only when ignoring "not" leads to the wrong object.
