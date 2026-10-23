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

- Pilot, Xiaomi-Robotics-1: language screen `20261023-1002_robocasa_xr1_screen_berkeley2` (20 cells × 20) running since
  2026-10-23 10:03 UTC; smoke and gate done (entries below).

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

---

## Servers & environments (maintained in place)

| Machine | GPUs used | Simulator env | What is there | Used for |
|---|---|---|---|---|
| Laptop (RTX 5060 Laptop 8 GB, 24 threads) | EGL rendering only | `~/vla_robocasa/.venv` (uv, Python 3.11): robocasa `4f8a298` (1.0.1), robosuite `5ce6643` (1.5.2), MuJoCo 3.3.1, numpy 2.2.5; lerobot pulled torch 2.7.1+cu126 | kitchen assets 23 GB (`download_kitchen_assets --type all`) | simulator-only checks: `docs/robocasa/scene_probe_2026-10-21/`, `pair_check_2026-10-21/`, `stove_direction_2026-10-21/`. No policy |
| `berkeley-2` = GCP `g4-zzst-1016-mig-uswest3-s4mp` (34.185.64.167), boots from the old `berkeley-2` disk | GPU 0 only (GPUs 1–3: another user's vLLM, 65–85 GB each, 2026-10-21) | `~/vla_robocasa/.venv`, same pins as the laptop (setup script `~/vla_robocasa/setup_robocasa.sh`, log `setup.log`) | kitchen assets 23 GB; `~/vla_robocasa/checkpoints/Xiaomi-Robotics-1-RoboCasa365` (HF revision `3a6d029`, 21 files, sizes checked against the HF listing, 9.5 GiB); `~/vla_robocasa/Xiaomi-Robotics-1` @ `0dd7aef` | RoboCasa pilot host (plan §5.3). Set up 2026-10-21 15:04–15:08 UTC, downloads only; no GPU used. Root disk 92% full after setup (328 GB free) |

Notes:
- The Xiaomi-Robotics-1 checkpoint loads with `trust_remote_code=True` (`modeling_mibot.py`, `processing_mibot.py`,
  `configuration_mibot.py` ship in the HF repo); that is how the authors' own server loads it.
- `~/.local/bin/uv` exists on `berkeley-2` but is not on `PATH`; the setup script prepends it.
- Inside `~/vla_robocasa`, run Python from another directory (e.g. `/tmp`): the `robosuite/` checkout there shadows
  the installed package as a namespace import.
