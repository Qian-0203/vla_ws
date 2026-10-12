# Data status, 2026-10-13

An inventory of what has been run, what is in the tables, and what is not. No new numbers and no
interpretation: every figure here comes from `docs/result_tables/` (built from raw JSONL) or
`docs/benchmark_split_result.md`, which stay the authoritative sources. Launch details are in `docs/eval_log.md`.

## 1. What the tables cover

`docs/result_tables/` (built 2026-10-13, success counted within 230 environment steps): 10 checkpoints,
26 condition rows, 92,600 rollouts. Seed 7, 50 trials per task, one run per condition.

| Checkpoint | Conditions | Rollouts | Not run |
|---|--:|--:|---|
| ours (OpenVLA LoRA r32) | 21 | 9,000 | `region_surface`, `surface_landmark`, `target_cue_region_*_img` |
| official OpenVLA | 22 | 8,600 | `target_cue_region_*_img`, `closed_drawer_control` |
| QwenVLA (run by Ken Zheng) | 20 | 7,600 | `target_cue_region_*_img`; `negation_only` / `no_location` / `target_swap` were run but are not in the tables (§2) |
| `pi05_libero` | 22 | 8,600 | `target_cue_region_*_img`, `closed_drawer_control` |
| `pi0_libero` | 25 | 9,800 | `closed_drawer_control` |
| OpenVLA-OFT (spatial) | 25 | 9,800 | `closed_drawer_control` |
| VLA-Adapter Spatial | 25 | 9,800 | `closed_drawer_control` |
| VLA-Adapter Spatial-Pro | 25 | 9,800 | `closed_drawer_control` |
| X-VLA (Libero) | 25 | 9,800 | `closed_drawer_control` |
| GR00T N1.5 (spatial) | 25 | 9,800 | `closed_drawer_control` |

`no_location` and `target_swap` are instruction-file conditions (`screening/language_stress/candidates/`),
not `SPLITS` entries.

## 2. Data that exists but is not in the tables

| Item | Rollouts | Status | Where it is described |
|---|--:|---|---|
| `grounding/landmark_exchange` (official OpenVLA, ours, pi05; and inside the gate of the six public checkpoints) | 600 + 6 × 200 | Exploratory; left out of the official data by the author's decision of 2026-10-12 | result doc §14, plan Split 5 |
| `target_cue_region`, `_v2`, `_v3` (ours, official, pi05, QwenVLA) | 4 × 3 × 400 | Withdrawn: direction words used swapped axes | result doc Erratum E1 |
| `pi0_fast_libero` gate | 500 | Excluded: gate 16.0% (80/500), cause not resolved | eval_log 2026-10-12 → 2026-10-13 |
| QwenVLA `negation_only` / `target_swap` / `no_location` (2026-10-11, Ken's package, which-bowl logged): 1.8 / 50.2 / 27.6% | 3 × 500 | Valid runs; recorded in the docs on 2026-10-13, after the tables were built | result doc §11 addendum, eval_log |
| Language-stress screens (5 trials/task) | — | Screening only, not benchmark numbers | result doc §12 |

## 3. Runs that were repeated, and how far apart they are

- **Our LoRA r32, 11 conditions** (original raw JSONL lost; re-run 2026-10-11). The tables use the re-run (`*`).
  Six conditions match the recorded per-task rates exactly; the other five differ by 0.4–3.2 points (e.g.
  `default` 84.4 vs 84.0, `negative_contrast` 40.0 vs 36.8). Region v1 was not re-run (withdrawn).
- **pi05 with which-bowl logging** (2026-10-11): `target_swap` identical to the first run (56.8);
  `no_location` 80.4 vs 81.4; `negation_only` 43.4 vs 44.8. The tables carry the first run; the which-bowl
  counts come from the re-run (result doc §12, batch E re-run).
- **Our LoRA r32 `negation_only` with which-bowl logging**: identical per (task, episode) to the earlier
  success-only run (500/500).
- **pi0 `no_location`, duplicate copy** (`~/vla_ws_lang2`, two shards, 42 lost-server rows): excluded; the
  complete copy in `~/vla_ws_lang` is used.
- **VLA-Adapter**: 158 rows written by orphaned clients (`num_steps == 10`) were removed and re-run; originals
  kept in `results/_garbage_rows_backup_2026-10-12` on `berkeley`.

## 4. Which-bowl (outcome) logging coverage

Rows carry `bowls_on_plate`, `first_bowl_approached`, `first_bowl_grasped`, etc. for the runs below; earlier
runs are success-only.

- In `docs/result_tables/outcomes_2026-10-13.csv` (the run each table cell uses): all 25 conditions of the six
  public checkpoints; official OpenVLA `negation_only`, `no_location`, `target_swap`; ours `no_location`,
  `target_swap` and the 11 re-run conditions.
- Logged but not in that CSV (the tables use the earlier success-only run of the same condition): pi05
  `target_swap` / `no_location` / `negation_only` re-run, ours `negation_only` (`--whichbowl`). Their counts are in
  result doc §12 (batch E re-run) and §2 (`negation_only` which-bowl run).
- QwenVLA `negation_only` / `target_swap` / `no_location`: logged, counts in result doc §11 addendum; not in the CSV.

## 5. Where the raw data is

- HF dataset `Qian0203/vla_ws-eval-results` (private): server results since 2026-09-30, QwenVLA raw (uploaded
  2026-10-12). Open issue: the mirror is flat, and the `vla_ws_lang2` upload overwrote pi0 `no_location`
  shards 8/9 with the duplicate copy; the clean copy is on the servers and the laptop.
- Laptop: `~/vla_results_backup/raw_data_library_2026-10-12/` (manifest, coverage, checksums) and
  `~/vla_results_backup/libero_ckpt_campaign_2026-10-12/` (campaign raw and the `full_data_v3` package).
- Landmark-exchange raw: `~/vla_results_backup/landmark_exchange_2026-10-12/` (not in the HF mirror).
- Not recoverable: raw JSONL of region v1 for our LoRA (withdrawn condition).

## 6. Records merged into `main` on 2026-10-13

| Branch | Content |
|---|---|
| `claude/libero-finetuned-ckpt-benchmark-4f7040` | Public-checkpoint campaign: code (policy servers, queues, Dockerfiles), eval_log entry, result doc §13, `docs/result_tables/`, README table |
| `claude/pi-0-5-which-bowl-logging-42a54a` (incl. `claude/epic-goldstine-92757c`) | pi05 which-bowl re-run; OpenVLA `negation_only` with which-bowl logging; language-weakness doc updates |
| `claude/vigorous-kepler-3c375e` | Batch F (`target_swap` / `no_location` on the six public checkpoints): eval_log entry, result doc §12 batch F |
| `claude/target-swap-bowl-logging-b269ec` | Landmark-exchange probe (exploratory): plan Split 5, eval_log entries, result doc §14 |
| (this branch) | QwenVLA three-condition package recorded (eval_log entry, result doc §11 addendum); this file |

Submodule pointers: openvla `77e1579` (fork branch `xvla-client`, contains `main` plus the OFT, raw-obs,
landmark-exchange and X-VLA commits), LIBERO `63b26c5` (fork branch `landmark-exchange-suite`).

## 7. Open items recorded but not done

- Not started: NORA-Long, InternVLA-M1, MolmoAct.
- Corrected region cues (`*_img`) have no run on ours, official OpenVLA, pi05 or QwenVLA.
- `docs/result_tables/` does not yet include QwenVLA's three language conditions (rebuild with the 2026-10-11 package).
- Server housekeeping noted at 2026-10-13 01:18 CST: four idle pi0 servers on `berkeley-2` (ports 8700–8703),
  scratch directories on `berkeley-2` (`~/vla_ws_oft`, `~/oft_official`, `~/oft_test`).
