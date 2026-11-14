# RoboCasa365 eval launch log

Chronological record of every RoboCasa365 GPU run: what ran, in what order, on what hardware and image, with which
checkpoint and robocasa commit, and where the outputs landed. This log is separate from the LIBERO log
(`docs/eval_log.md`, closed for new platforms by freeze decision D4).

**Rules.** Append-only: never edit or delete an entry; corrections go in a new dated entry. Add an entry after
every real run (gate, screen or full condition), including failed or aborted ones. Each entry records: date;
machine and GPUs; sim image and robocasa / robosuite / MuJoCo versions; model, checkpoint (HF repo + revision) and
server image; seeds; conditions and tasks with rollouts per cell; one-line headline per cell; results-file paths;
whether a scene-hash check passed against `default`. Label screens (fewer than 50 episodes per cell) and exploratory
probes as such; they never enter result tables.

Design and condition definitions: `docs/robocasa/plan.md`. Machine and environment setup lives in the
"Servers & environments" section at the end, which is maintained in place (like the LIBERO log's reference section).

---

## Still queued (none run yet)

- Pilot done (smoke, gate, screen; entries below). Next: the main experiment (plan §4.0, 50 episodes per cell) awaits
  approval.

---

<!-- entries below, oldest first -->

## 2026-10-21 — Smoke: Xiaomi-Robotics-1, NavigateKitchen × 2 (smoke, not a result)

- **Machine:** `berkeley-2` (`…-s4mp`), GPU 0 only (`CUDA_VISIBLE_DEVICES=0`, `MUJOCO_EGL_DEVICE_ID=0`). GPUs 1–3 held by
  another user's vLLM throughout.
- **Stack:** sim venv `~/vla_robocasa/.venv` (robocasa `4f8a298`, robosuite `5ce6643`, MuJoCo 3.3.1, transformers
  4.57.1 for the client's processor); server venv `~/vla_robocasa/.venv-xr1` (Python 3.12, torch 2.8.0+cu128,
  transformers 4.57.1, no flash-attn).
- **Model:** `XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365` @ `3a6d029`, bf16, **sdpa** via
  `robocasa_lang/servers/xr1_server.py` (wraps the upstream `deploy/server.py` unchanged).
- **Client:** upstream `eval_robocasa365/entry.py` (Xiaomi-Robotics-1 @ `0dd7aef`), `--split pretrain --task-set target50
  --task-name NavigateKitchen --num-trials 2 --seed 7`, replan 16, obs history 4 / interval 2, crop 0.95 (defaults).
- **Attempts, in order:**
  1. 15:24 UTC: server refused to load: transformers raised "MiBoTForActionGeneration does not support ... sdpa"
     (the top-level class does not declare `_supports_sdpa`). The client then waited on the socket; stopped by hand.
     The wrapper now declares `_supports_sdpa` on `MiBoTForActionGeneration` and `DiT` (the VLM already declares it;
     the DiT always calls `F.scaled_dot_product_attention`, so only the VLM's attention changes).
  2. 16:35 UTC: aborted by the pre-launch guard: GPU 0 had 47.9 GB in use by another user's process (gone a minute
     later). The guard now checks free memory (abort below 30 GB free) instead of used memory.
  3. 16:36 UTC: ran. Server up in 30 s. 2 episodes in 36 s of client time (55 s wall incl. env build).
- **Headline:** 1/2 (seed 15: failure at the 450-step horizon, 20.7 s; seed 16: success at step 226). **Not comparable
  with the published results:** the upstream client seeds episode *i* of task *k* as `7 + k × num_trials + i`, so
  `--num-trials 2` gave seeds 15 and 16, not the published 207 and 208. A gate must use `num_trials=50` (or our client
  with explicit seeds).
- **Measured:** about 0.045 s per env step all-in (simulation, 3-camera render, a policy query every 16 steps, one
  client); GPU 0 peak 12.8 GB total (server process 8.5 GB), 42% utilisation with one client.
- **Outputs:** `berkeley-2:~/vla_robocasa/runs/smoke_xr1_20261021T163629Z/` (`server.log`, `client.log`, `gpu0.csv`,
  `results/smoke/{summary.json,NavigateKitchen/stats.json,*.mp4}`); failed attempts in `…T152438Z/`, `…T163525Z/`.

## 2026-10-23 — Records: smoke backfilled to the run-record format; HF sync installed

- Adopted the run-record format of `docs/results_storage.md` (run ID `YYYYMMDD-HHMM_robocasa_<model>_<stage>_<host>`;
  `run.json`, `episodes.jsonl`, `summary.csv` per run; HF + repo + laptop). Tool: `robocasa_lang/runrecord.py`.
- The 2026-10-21 smoke is now run `20261021-1636_robocasa_xr1_smoke_berkeley2` (NavigateKitchen/default 1/2, seeds 15/16,
  smoke, not comparable). Its instructions and scene hashes were reconstructed on the laptop from the same seeds and
  robocasa commit (the upstream client logs neither); no outcome fields. Server dir
  `berkeley-2:~/vla_robocasa/runs/20261021-1636_robocasa_xr1_smoke_berkeley2/` (`raw/` = the original
  `smoke_xr1_20261021T163629Z/`), with `SHA256SUMS` (verified on the server).
- HF sync on `berkeley-2` (user approved 2026-10-23): `~/vla_robocasa/sync_robocasa_hf.sh` uploads every run-ID directory
  under `~/vla_robocasa/runs/` to `Qian0203/vla_ws-eval-results` at `robocasa/<run_id>/`; cron `*/30`, log
  `~/logs/sync_robocasa_hf.log`; existing cron lines untouched. Smoke uploaded 09:14 UTC (12 files, checked by listing).
- **Leftover process found:** the smoke's server (pid 3484216) had kept running on GPU 0 since 2026-10-21 16:37 UTC
  (~41 h, ~10.8 GB, port 10086): the smoke script's `kill $SRV` hit the subshell, not the python process. Killed
  2026-10-23 09:26 UTC. Launchers now `exec` the servers and refuse to start when a port is taken.

## 2026-10-23 — Gate: Xiaomi-Robotics-1, 4 tasks × 50 published seeds — **passed**

- **Run:** `20261023-0927_robocasa_xr1_gate_berkeley2` (stage gate). Approval: user, in chat, 2026-10-23 (gate + screen).
- **Machine:** `berkeley-2` GPU 0; 3 server/client pairs, then 2 for the continuation; ~34 GB on GPU 0.
- **Stack / model:** as the smoke (robocasa `4f8a298`, robosuite `5ce6643`, MuJoCo 3.3.1; HF `3a6d029`, bf16, sdpa via
  `robocasa_lang/servers/xr1_server.py`). **Client:** upstream `eval_robocasa365/entry.py` @ `0dd7aef`, unmodified,
  `--num-trials 50 --seed 7` (published seeds: Nav 207–256, Mixer 357–406, PnPCab 407–456, Stove 707–756).
- **Launch order:**
  0. 09:15 UTC, run `20261023-0915_robocasa_xr1_gate_berkeley2`: aborted before any episode. Port 10086 was held by the
     leftover smoke server (see the records entry above). No episodes; `run.json` + `ABORTED.txt` kept.
  1. 09:27–09:45 UTC: TurnOffStove (w0) and PickPlaceCounterToCabinet (w1). Worker 2 (NavigateKitchen +
     OpenStandMixerHead) failed at start: the launcher gave both names to one `--task-name` (upstream uses
     `action=append`); no episodes (`raw/client_w2_argparse_error.log`).
  2. 09:47–09:56 UTC: NavigateKitchen (w2) and OpenStandMixerHead (w3) via `run_gate_continue.sh`, same run directory.
- **Result (gate rule: per task within 14 pts, mean within 7 pts of published):**

  | Task | SR | Published | Diff | Episode agreement |
  |---|--:|--:|--:|--:|
  | NavigateKitchen | 88% (44/50) | 92% | −4 | 0.88 |
  | OpenStandMixerHead | 94% (47/50) | 98% | −4 | 0.96 |
  | PickPlaceCounterToCabinet | 88% (44/50) | 86% | +2 | 0.90 |
  | TurnOffStove | 58% (29/50) | 60% | −2 | 0.78 |
  | **Mean** | **82.0%** | **84.0%** | **−2.0** | |

  The sdpa server with the upstream client reproduces the published numbers; the screen may run.
- **Records:** instruction and scene hash reconstructed after the run on the server CPU, same seed order, one env per
  task (`robocasa_lang/reconstruct_scenes.py`; the upstream client logs neither); outcome fields null for this run.
  Seed 207's scene hash equals the laptop's for the same seed (cross-machine determinism).
- **Files:** server `~/vla_robocasa/runs/20261023-0927_robocasa_xr1_gate_berkeley2/` (229 files + `SHA256SUMS`, 105 MB incl.
  200 videos); HF `robocasa/20261023-0927_robocasa_xr1_gate_berkeley2/` (230 files; 4 intermediate `rec_*.jsonl` captured by
  the 10:00 cron mid-postprocessing were removed in a follow-up HF commit); laptop backup verified against
  `SHA256SUMS`; repo `results/robocasa/20261023-0927_robocasa_xr1_gate_berkeley2/` (`run.json`, `episodes.jsonl`,
  `summary.csv`, `gate.json`).

## 2026-10-23 — Screen: Xiaomi-Robotics-1, 20 language cells × 20 episodes (screen, not a result)

- **Run:** `20261023-1002_robocasa_xr1_screen_berkeley2` (stage screen). Approval: user, in chat, 2026-10-23, conditional on the gate, which passed
  (`20261023-0927_robocasa_xr1_gate_berkeley2`).
- **Machine / stack / model:** as the gate (`berkeley-2` GPU 0, 3 server/client pairs; HF `3a6d029`, sdpa).
- **Client:** `robocasa_lang/run_lang_eval.py` (vla_ws, uncommitted at launch; copy on the server), importing the upstream
  `EvalClient` and observation helpers from `eval_robocasa365/entry.py` @ `0dd7aef` unchanged; instruction from the cell's
  condition (`robocasa_lang/conditions.py`); env classes from `robocasa_lang/envs.py` where the success check changes
  (`PnPCounterToCabinetSwap`, `NavigateKitchenPair`, `TurnOffStovePair`); scene hash and outcome fields logged per episode.
- **Cells:** `robocasa_lang/screen_cells.py` (plan §4.0/§5.3): blank ×4 tasks, no_target ×3, target_swap (PnP on graspable
  distractors; Nav via the pair's slot b), stove pair a/b, PnP negation_only, contrast + length control (PnP, Nav),
  paraphrase_syntactic ×4. Seeds: first 20 published seeds per task; PnP swap skipped 417, 419, 425 (non-graspable
  distractor).
- **Ran:** 10:03–10:50 UTC, no errors, 400 episodes.
- **Headlines (20 per cell, ±10–20 pts):** blank 0% on all four tasks (default 65–94%); PnP no_target 85% = default, but
  PnP target_swap 30% with the *original* (nearer) object delivered in 8/20; Nav target_swap 90% and Nav no_target 40%;
  stove pair a/b 60%/80% with the wrong burner turned off in 3 and 1; PnP negation_only 65% (negated object delivered
  once); Nav positive_contrast 70% (3 went to the named distractor) vs its length control 80%; paraphrase costs 21 (Nav) and
  20 (Stove) pts, nothing on PnP or the mixer. Full table: `docs/robocasa/results.md`.
- **Files:** server `~/vla_robocasa/runs/20261023-1002_robocasa_xr1_screen_berkeley2/` (417 files + `SHA256SUMS`, 306 MB); HF `robocasa/20261023-1002_robocasa_xr1_screen_berkeley2/` (418 files,
  identical to the local list); laptop backup; repo `results/robocasa/20261023-1002_robocasa_xr1_screen_berkeley2/`.

## 2026-11-04 — Setup and smoke: π0.5 on `berkeley-3`, 2 episodes (smoke, not a result)

- **Why π0.5:** meeting 2026-10-28 asked for one model family across LIBERO-Spatial, RoboLab and RoboCasa365. RoboCasa's
  own π0.5 checkpoint, not the DROID one used on RoboLab.
- **Machine:** `berkeley-3` (AWS 8× H200, no docker, no sudo). Everything under `/mnt/dt-vol-00494956a3ff3a063-20000G/qian/robocasa_ws/` on the persistent volume.
- **Stack:** sim venv `.venv-sim` (uv, Python 3.11: robocasa `4f8a298`, robosuite `5ce6643`, MuJoCo 3.3.1, numpy 2.2.5,
  openpi-client); kitchen assets 23 GB (`download_kitchen_assets --type all`). Server venv = the openpi fork
  `robocasa-benchmark/openpi` @ `5a6beda` (`uv sync --no-install-package av`: PyAV has no wheel here and is only used to
  load training data; plus robocasa/robosuite `--no-deps` and numpy 2.2.5, which robocasa asserts).
- **Model:** HF `robocasa/robocasa365_checkpoints`, `pi05_pretrain_human300/multitask_learning/75000`, `params/` + `assets/`
  only (12 GB; the training state is not needed). Config `pi05_pretrain_human300`. Served by
  `robocasa_lang/servers/pi05_server.py`, which clears the config's `data_dirs` so that the norm stats come from the
  checkpoint's `assets/norm_stats.json` instead of the Human300 datasets (not downloaded). CPU-only check before any GPU
  use: loads in 9 s, one inference returns a 50 × 12 action chunk.
- **Client:** `robocasa_lang/run_lang_eval.py --policy pi05` (`robocasa_lang/pi05_policy.py`: three cameras padded to
  224, state = eef pos + eef rot + base pos + base rot + gripper qpos, replan every 5 steps, as the fork's
  `examples/robocasa/main.py`). Launcher `robocasa_lang/run_pi05_b3.sh`.
- **Run:** `20261104-1528_robocasa_pi05_smoke_berkeley3`, PickPlaceCounterToCabinet seed 407 and NavigateKitchen seed 207,
  GPUs 0–1. Approval: user, in chat, 2026-11-04 (steps 0–2: smoke, gate, defaults).
- **Result:** pipeline passed (both episodes logged with scene hash and outcome fields); both failed to the horizon
  (PnP reached the bread within 2.3 cm, no grasp; Nav never came closer than 2.98 m).
- **Files:** server `/mnt/dt-vol-00494956a3ff3a063-20000G/qian/robocasa_ws/runs/20261104-1528_robocasa_pi05_smoke_berkeley3/`; repo
  `results/robocasa/20261104-1528_robocasa_pi05_smoke_berkeley3/`. Not yet on HF or the laptop.

## 2026-11-04 — Gate: π0.5, 18 Atomic-Seen tasks × 10 episodes (+ defaults to 50 on 4 tasks) — **passed**

- **Run:** `20261104-1530_robocasa_pi05_gate_berkeley3` (stage gate). Approval: as the smoke entry above.
- **Machine:** `berkeley-3` GPUs 0–2, one server per GPU (ports 8800–8802, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.2`, ~29 GB
  each), 9 client workers (3 per GPU), one cell per job. GPUs 0–2 also carried another user's jobs (~89 GB each).
- **Gate rule:** π0.5 publishes only its Atomic-Seen mean (39.6%, robocasa 1.0.0; no per-task numbers in the leaderboard
  repo or the robocasa docs), so the gate is the mean of the 18 Atomic-Seen tasks over each task's first 10 seeds, within
  7 pts of 39.6. Seeds follow Xiaomi-Robotics-1's published rule, 7 + 50 × task index + episode, so the four screen tasks
  share seeds with the XR-1 runs (`robocasa_lang/pi05_gate_cells.py`).
- **Ran:** 15:30–16:07 UTC, no errors, 340 episodes, none skipped.
- **Result:** mean **39.4%** vs published 39.6% → **passed** (`robocasa_lang/pi05_gate_report.py`). Defaults on 50
  episodes: PickPlaceCounterToCabinet 28/50, OpenStandMixerHead 36/50, TurnOffStove 9/50, NavigateKitchen 0/50. Only the
  first two reach the 50% floor and are used for the language cells.
- **Files:** server `/mnt/dt-vol-00494956a3ff3a063-20000G/qian/robocasa_ws/runs/20261104-1530_robocasa_pi05_gate_berkeley3/` (390 files + `SHA256SUMS`, 111 MB incl. 340
  videos; checksums verified); repo `results/robocasa/20261104-1530_robocasa_pi05_gate_berkeley3/` (`run.json`,
  `episodes.jsonl`, `summary.csv`, `cells.jsonl`, `gate.json`). Not yet on HF or the laptop.

## 2026-11-07 — Screen: π0.5, 7 language cells × 50 episodes (screen, not a result)

- **Run:** `20261107-2012_robocasa_pi05_screen_berkeley3` (stage screen). Approval: user, in chat, 2026-11-08 (Taiwan
  time; "run it, then shut the servers down").
- **Machine / stack / model:** as the gate; 9 workers; servers stopped by the launcher right after the run (20:54 UTC).
- **Cells:** `robocasa_lang/pi05_screen_cells.py`, each on the default's 50 seeds (split into halves _a/_b): PnP
  target_swap (`PnPCounterToCabinetSwap`, require_valid: 10 seeds skipped, no usable second object), negation_only,
  positive_contrast, length_control_positive, paraphrase_syntactic, verb_synonym; mixer paraphrase_syntactic.
- **Ran:** 20:12–20:54 UTC, no errors, 340 episodes + 10 skipped.
- **Headlines (vs default 28/50 on the same seeds; requested / other object delivered):** target_swap 9/40 vs the
  original object 16/40; negation_only 16/50, negated object 2/50; positive_contrast 18/50 (other 1); length control
  ("slowly and carefully") 9/50; paraphrase_syntactic 31/50; verb_synonym ("fetch … stow") 7/50; mixer paraphrase 19/50
  (default 36/50). The object choice rarely flips; rewording costs success. Full table: `docs/robocasa/results.md`.
- **Files:** server `/mnt/dt-vol-00494956a3ff3a063-20000G/qian/robocasa_ws/runs/20261107-2012_robocasa_pi05_screen_berkeley3/` (375 files, 126 MB incl. videos;
  `SHA256SUMS` verified; `screen_report.json` and `skipped.jsonl` were derived afterwards and are not in it); repo
  `results/robocasa/20261107-2012_robocasa_pi05_screen_berkeley3/`. Not yet on HF or the laptop.

## 2026-11-09 — Records: π0.5 runs backed up to HF and the laptop

- `20261104-1528_robocasa_pi05_smoke_berkeley3` (11 files), `20261104-1530_robocasa_pi05_gate_berkeley3` (390 files, 111 MB)
  and `20261107-2012_robocasa_pi05_screen_berkeley3` (375 files, 126 MB): copied from `berkeley-3` to the laptop
  (`~/vla_results_backup/robocasa/<run_id>/`, each verified against its `SHA256SUMS`), then uploaded from the laptop to HF
  `Qian0203/vla_ws-eval-results` (dataset), `robocasa/<run_id>/`, one commit per run; remote file counts equal the local
  ones. Approval: user, in chat, 2026-11-09. This supersedes "Not yet on HF or the laptop" in the three entries above.

## 2026-11-14 — Screen: π0.5, negation of the usual object (1 cell × 50 seeds, screen, not a result)

- **Run:** `20261114-1319_robocasa_pi05_screen_berkeley3` (stage screen). Approval: user, in chat, 2026-11-14.
- **Why:** in `pnp_negation_only` ("…not the {distractor}…") the answer is the object π0.5 takes anyway (nearer the
  robot in 48/50 scenes), so ignoring "not" still succeeds (16/50, negated object 2/50). The new condition
  `negation_swap` negates the usual object instead ("Pick the object that is not the {target} and place it in the
  cabinet."), scored on the distractor by `PnPCounterToCabinetSwap`, as LIBERO's `negation_swap`.
- **Machine / stack / model:** as the 2026-11-04 gate; one server on GPU 0, 2 workers; server stopped after the run.
  Cells: `robocasa_lang/pi05_screen_cells.py` (`pnp_negation_swap`), same 50 seeds as the default; require_valid
  skipped 10 (same seeds as `pnp_target_swap`).
- **Ran:** 13:19–13:40 UTC, no errors, 40 episodes + 10 skipped.
- **Headline:** requested object delivered **0/40**; the negated (usual) object delivered **22/40**; first grasp: usual
  26, requested 0, none 14. Matches LIBERO π0.5 `negation_swap` (0/50).
- **Files:** server `/mnt/dt-vol-00494956a3ff3a063-20000G/qian/robocasa_ws/runs/20261114-1319_robocasa_pi05_screen_berkeley3/` (`SHA256SUMS` verified); repo
  `results/robocasa/20261114-1319_robocasa_pi05_screen_berkeley3/`. Not yet on HF or the laptop.

---

## Servers & environments (maintained in place)

| Machine | GPUs used | Simulator env | What is there | Used for |
|---|---|---|---|---|
| Laptop (RTX 5060 Laptop 8 GB, 24 threads) | EGL rendering only | `~/vla_robocasa/.venv` (uv, Python 3.11): robocasa `4f8a298` (1.0.1), robosuite `5ce6643` (1.5.2), MuJoCo 3.3.1, numpy 2.2.5; lerobot pulled torch 2.7.1+cu126 | kitchen assets 23 GB (`download_kitchen_assets --type all`) | simulator-only checks: `docs/robocasa/scene_probe_2026-10-21/`, `pair_check_2026-10-21/`, `stove_direction_2026-10-21/`. No policy |
| `berkeley-2` = GCP `g4-zzst-1016-mig-uswest3-s4mp` (34.185.64.167), boots from the old `berkeley-2` disk | GPU 0 only (GPUs 1–3: another user's vLLM, 65–85 GB each, 2026-10-21) | `~/vla_robocasa/.venv`, same pins as the laptop (setup script `~/vla_robocasa/setup_robocasa.sh`, log `setup.log`) | kitchen assets 23 GB; `~/vla_robocasa/checkpoints/Xiaomi-Robotics-1-RoboCasa365` (HF revision `3a6d029`, 21 files, sizes checked against the HF listing, 9.5 GiB); `~/vla_robocasa/Xiaomi-Robotics-1` @ `0dd7aef` | RoboCasa pilot host (plan §5.3). Set up 2026-10-21 15:04–15:08 UTC, downloads only; no GPU used. Root disk 92% full after setup (328 GB free) |
| `berkeley-3` = AWS `ip-172-31-64-36` (18.88.135.19), 8× H200 143 GB, no docker / sudo | GPUs 0–2 (shared with another user's jobs) | `/mnt/dt-vol-00494956a3ff3a063-20000G/qian/robocasa_ws/.venv-sim` (uv, Python 3.11), same pins as `berkeley-2` | kitchen assets 23 GB; π0.5 checkpoint `pi05_pretrain_human300/multitask_learning/75000` (params + assets, 12 GB); openpi fork @ `5a6beda` with its own `.venv` | π0.5 smoke, gate and screen, 2026-11-04 → 11-07. Set up 2026-11-04 (downloads and CPU only) |

Notes:
- The Xiaomi-Robotics-1 checkpoint loads with `trust_remote_code=True` (`modeling_mibot.py`, `processing_mibot.py`,
  `configuration_mibot.py` ship in the HF repo); that is how the authors' own server loads it.
- `~/.local/bin/uv` exists on `berkeley-2` but is not on `PATH`; the setup script prepends it.
- Inside `~/vla_robocasa`, run Python from another directory (e.g. `/tmp`): the `robosuite/` checkout there shadows
  the installed package as a namespace import.
