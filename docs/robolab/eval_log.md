# RoboLab launch log

Append-only record of what ran on RoboLab, in what order, on which machine and stack, and where the
outputs landed. Same rules as the LIBERO `docs/eval_log.md` (freeze record decision D4): never edit or
delete an entry; corrections are new dated entries. Design and conditions live in [`plan.md`](plan.md).

Each entry records: date; machine and GPU index; RoboLab commit and Isaac Sim / Isaac Lab version
(`isaac50` unless stated); image tag; policy server (repo, commit, checkpoint path); tasks, instruction
keys, `--num-envs`, episodes; `frame_check.py` result for any spatial key; output folder; one-line
headline; GPU-hours.

## Still queued

- Phase 2 pilot, π0.5-DROID (plan §5.2 steps 0–5): awaiting approval for installs, downloads and GPU time.
- Gate B, π0.5-DROID, RoboLab-120 `default` × 10 episodes: separate approval.

## Entries

### 2026-10-21 — phase 1 (no runs)

Feasibility and design only; nothing installed, no GPU used. RoboLab source read at `NVlabs/RoboLab`
`b6cfcf6` (shallow clone and scene USDA files in a local scratch directory, not in this repo).
`robolab_lang/frame_check.py` run on `two_bin`, `bowls_2_table`, `rubiks_cube_banana_bowl`,
`butter_raisin_box_grey_bin` (CPU, projection only); `robolab_lang/droid_vocab.py` run on RoboLab's
`analysis/droid_dataset/tasks.jsonl`. Results in `plan.md` §3.1 and §4.3.

### 2026-10-21 — preliminary analysis without simulation; servers unavailable

`berkeley` (34.174.233.202) now presents a different host key and rejects our SSH key; `berkeley-2`
(34.106.148.66) times out on port 22. Both appear to have been replaced or released, so the approved
pilot (plan §5.2) did not start. `VMRL-SERVER` was inspected read-only (1× RTX 5080 16 GB shared with a
running Isaac Sim 5.1 session, root disk 99% full) and judged unsuitable for the pilot as it stands.
Instead `robolab_lang/prelim_analysis.py` was run on the laptop (CPU) against the leaderboard JSON v55
(`docs/robolab/leaderboard_v55.json`) and RoboLab `b6cfcf6`; results and figure in
`preliminary_analysis.md` (findings R1–R8). No rollouts, no GPU.

### 2026-10-21 — render only, laptop (no policy)

Laptop RTX 5060 8 GB, driver 580.178.04; local image `isaac-lab-base:latest` (Isaac Sim 5.1.0, Isaac Lab
2.3.2); RoboLab `b6cfcf6` with all assets (git LFS) in a local scratch checkout, missing pip packages
(tyro, json_numpy, msgpack-numpy, pandas 2.2, pyzmq) in a separate mounted user-base, not in the image.
Container ran as root (the image's `/isaac-sim` is not readable otherwise) and chowned every mounted
directory back to the user on exit. `robolab_lang/render_policy_views.py`: `create_env` with
`num_envs=1`, reset, 15 hold steps, save `over_shoulder_left_camera` and `wrist_cam`. Tasks:
MustardInLeftBinTask, BowlStackingLeftOnRightTask, FoodPackingByColorTask, BananaInBowlTask,
FruitsMovingTask. First multi-task process hit `CUDA error 2` on FruitsMovingTask; rerun alone, fine.
Output: `docs/robolab/figures/renders/*.jpg`. Confirms R7 and the colours for R8
(`preliminary_analysis.md` §E). No inference, no eval. Isaac Sim 5.1 is not the leaderboard's 5.0 stack;
renders only, not for numbers.

### 2026-10-21 — pilot steps 0–2, π0.5-DROID, `g4-r1sc-1019-mig-uswest3-nj21` GPU 0

Host `g4-r1sc-1019-mig-uswest3-nj21` (34.106.249.51; boots from the old `berkeley` disk), 4× RTX PRO 6000
Blackwell, driver 580.178.04; GPUs 2–3 held by another user's job, GPU 0 used alone. Workspace
`~/robolab_ws/`.

Installed (user approved): `git-lfs` 3.6.1 binary in `~/.local/bin` (plus `git lfs install` in the user's
git config); RoboLab clone at `b6cfcf6` with all assets (5.8 GB); `nvcr.io/nvidia/isaac-lab:2.2.0`;
image `robolab:b6cfcf6` built from RoboLab's `docker/Dockerfile` (Isaac Sim 5.0.0-rc.45, Isaac Lab
2.2.0, `openpi-client` from `xuningy/openpi@aa64205`), then `robolab:b6cfcf6-ffmpeg` (adds Ubuntu
`ffmpeg` 4.4.2, needed for videos); `xuningy/openpi@aa64205` at `~/robolab_ws/openpi_robolab` (`uv sync`);
checkpoint `gs://openpi-assets-simeval/pi05_droid_jointpos` in `~/.cache/openpi/`.

Policy server: `robolab_lang/serve_pi05.sh` (config `pi05_droid_jointpos`, port 8765,
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.2`, 20 GB). Client: `robolab_lang/run_robolab.sh` → `run_pi0.py`
(RoboLab's pi0-family client and eval loop, `default` instruction, subtask tracking on, videos on).
Container runs as root and chowns mounts back on exit. Warp prints a harmless
`cuDeviceGetUuid` entry-point warning at start.

| Step | Run (output under `~/robolab_ws/output/`) | Episodes | Result |
|---|---|---:|---|
| 0 | `examples/run_empty.py --headless --task BananaInBowlTask` (`step0_run_empty.log`) | 1 | env builds and steps on Isaac Sim 5.0; pass |
| 1 | `smoke1_pi05`: BananaInBowlTask, 1 env | 1 | 0/1 (banana dropped 3×); pipeline, HDF5 and videos OK; 5.2 it/s, inference 20 ms avg |
| 1 | `smoke10_pi05`: BananaInBowlTask, 10 envs | 10 | 9/10; GPU 0 peak 34 GB incl. 20 GB server; 219 s wall (0.37 GPU-min/episode) |
| 2 | `gateA_pi05`: BananaInBowlTask + RubiksCubeAndBananaTask, 25 envs × 2 runs each | 100 | **BananaInBowl 46/50 (92%) pass; RubiksCubeAndBanana 36/50 (72%) below the 85% gate** |

RubiksCubeAndBanana failures (14): 9 never complete the cube grasp (`object_grabbed(rubiks_cube)` not
satisfied; `TARGET_OBJECT_DROPPED` in all 14, `GRIPPER_FULLY_CLOSED` 11), 4 place the banana but drop
or misplace the cube, 1 places the cube only. Run 0 16/25, run 1 20/25. These are grasp failures, not
wrong-object choices (`WRONG_OBJECT_GRABBED` in 3). The 85% threshold was our reading of the paper's
Table II (variation rows averaged over this task and BananaInBowl), not a published nominal per-task
number. Per plan §5.2 the language pilot (step 4) was not started; policy server stopped, GPU 0 released.
GPU time ≈ 1 GPU-hour.

### 2026-10-23 — Gate A relaxed by the user; run records back-filled

**Gate decision.** On 2026-10-23 the user relaxed the gate criterion (decision relayed to this session by
the session "語言實驗測試平台遷移"): Gate A counts as passed. The 85% threshold was our own reading of the
paper's Table II (variation rows averaged over two tasks), not a published per-task number.
BananaInBowl 46/50 (92%) passes; RubiksCubeAndBanana 36/50 (72%) is accepted because its 14 failures are
mostly grasp failures (cube never held, dropped, or gripper closed empty); only 3 of the 14 failed episodes
show `WRONG_OBJECT_GRABBED` (6 of all 50 episodes do), which is not a language failure. Gate B was not run.
The language pilot (redesigned: original shared scenes, paired instructions) needs separate approval.

**Run records.** Adopted the results-storage convention (run ID `YYYYMMDD-HHMM_robolab_<model>_<stage>_<host>`;
`run.json`, `episodes.jsonl`, `summary.csv` per run). The four runs of 2026-10-21 were moved on the server
from `~/robolab_ws/output/<name>/` to `~/robolab_ws/runs/<run_id>/robolab_output/`, and their records were
written afterwards (`run.json` says so) with `robolab_lang/make_run_record.py`:

| Run ID | Was | Stage | Episodes | Raw size |
|---|---|---|---:|---:|
| `20261021-1551_robolab_none_smoke_nj21` | `run_empty_env` + `step0_run_empty.log` (no policy) | smoke | 1 | < 1 MB |
| `20261021-1552_robolab_pi05_smoke_nj21` | `smoke1_pi05` | smoke | 1 | 21 MB |
| `20261021-1555_robolab_pi05_smoke_nj21` | `smoke10_pi05` | smoke | 10 | 68 MB |
| `20261021-1600_robolab_pi05_gate_nj21` | `gateA_pi05` | gate | 100 | 1,097 MB |

Copies: full runs (HDF5, videos, logs, `SHA256SUMS`) in the laptop's `~/vla_results_backup/robolab/`;
small files in `results/robolab/<run_id>/`. Not yet on Hugging Face and no RoboLab cron on the server
(both need the user's approval). Results in [`results.md`](results.md).
