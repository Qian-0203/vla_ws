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

### 2026-10-23 — runs mirrored to Hugging Face; RoboLab sync cron on nj21

User approved (2026-10-23, in this session): HF upload of the four existing runs, a sync cron on
34.106.249.51, and commits/pushes of results records to `results-log/robolab`.

- HF: `~/robolab_ws/runs/` uploaded to `Qian0203/vla_ws-eval-results` under `robolab/<run_id>/` (whole run
  directories: records, `SHA256SUMS`, RoboLab output with HDF5 and videos; 366 files, 1.2 GB). File counts on
  HF match the server per run (8 / 10 / 37 / 311).
- Cron: `*/30 * * * * bash $HOME/robolab_ws/robolab_lang/sync_robolab_hf.sh >> $HOME/logs/sync_robolab_hf.log`
  added to qian's crontab on nj21; the two existing sync lines unchanged (previous crontab saved in
  `~/robolab_ws/logs/crontab.before.*`).
- Laptop: full copy in `~/vla_results_backup/robolab/` verified against each run's `SHA256SUMS` (all four pass).

### 2026-10-28 — Option A language screen, π0.5-DROID, on an AWS H200 (run by Wen-Han Hsieh)

Our two cloud servers were unreachable, so the user handed the run kit (`robolab_lang/`, `docs/robolab/run_option_a.md`
on branch `robolab/option-a-runkit`) to Wen-Han Hsieh, whose agent ran it and returned the records in
[PR #9](https://github.com/Qian-0203/vla_ws/pull/9). Times are from each `run.json`.

| Run ID | Key | Episodes | Started (UTC) | Records written (UTC) | Raw size |
|---|---|---:|---|---|---:|
| `20261028-0842_robolab_pi05_smoke_aws-h200_default` | `default` | 1 | 2026-10-28 08:43 (per PR) | 08:47 (per PR) | — |
| `20261028-0848_robolab_pi05_screen_aws-h200_default` | `default` | 260 | 08:48 | 12:31 | 2.9 GB |
| `20261028-0848_robolab_pi05_screen_aws-h200_blank` | `blank` | 260 | 12:32 | 15:56 | 3.8 GB |
| `20261028-0848_robolab_pi05_screen_aws-h200_no_target` | `no_target` | 260 | 15:57 | 19:41 | 3.2 GB |

Hardware and stack: GPU 0 of 8 × NVIDIA H200 143 GB, driver 580.126.09, `NUM_ENVS=10`, `NUM_RUNS=2`. **Differs from Gate
A:** an H200 with Apptainer 1.5.5 instead of an RTX card with Docker (that account has no Docker group or sudo). Same
Isaac Sim 5.0, Isaac Lab 2.2.0, RoboLab `b6cfcf6` and openpi `aa64205`; the Dockerfile's contents were reproduced with bind
mounts and `pip --target`. Image `sif-sha256:2ae6ebb7…`. The PR reports no interruptions or reruns. Gate A was not
repeated on this machine; `RubiksCubeAndBanana` `default` scored 18/20 here, 36/50 in Gate A.

Checked on 2026-10-31: 260 rows per run, and `run.json`, `episodes.jsonl` and `summary.csv` match each run's `SHA256SUMS`.
Not done: PR #9 is unmerged (commit `eec958b`); the tgz link in the PR is a placeholder (`TGZ_LINE`), so the raw HDF5 files
and videos, and the Apptainer scripts packed with them, have not been received; nothing on HF or the laptop.
Results in [`results.md`](results.md).

### 2026-10-31 — PR #9 merged

The user approved the merge. The Option A records are now in `results/robolab/` on this branch (merge commit `29ecbbd`).
The tgz link is still missing, and there is still no copy on HF or the laptop.

### 2026-11-04 — Language tests on the original scenes, π0.5-DROID, on `berkeley-3` (18 runs, 600 episodes)

Recorded on 2026-11-15; the runs themselves are from 2026-11-04 (times below from the GPU job logs, UTC).

- **Machine:** `berkeley-3` (ip-172-31-64-36), the same 8 × H200 host as Option A (`aws-h200`), GPUs 0–2; driver 580.126.09.
  No Docker or sudo: Wen-Han Hsieh's Apptainer setup was copied to our folder on the persistent volume
  (`isaac-lab-2.2.0.sif`, image `sif-sha256:2ae6ebb7…`, bind mounts for RoboLab and site-packages); the openpi venv was
  rebuilt with uv in our folder. Same Isaac Sim 5.0 / Isaac Lab 2.2.0, RoboLab `b6cfcf6`, openpi `aa64205`,
  `pi05_droid_jointpos`.
- **Launcher:** `/mnt/dt-vol-00494956a3ff3a063-20000G/qian/robolab_ws/paper_tests_gpu.sh` (calls `robolab_lang/run_paper_tests.sh` → `run_option_a.sh`, Apptainer
  variant), one policy server per GPU (ports 8765–8767), `NUM_ENVS=5`, `NUM_RUNS=4`, `HOST_TAG=berkeley3`. Task files:
  `robolab_lang/tasks/Lang*Task.py` at `e80d6ca` (branch `robolab/option-a-runkit`). Approval: user, in chat, 2026-11-04.
- **Smoke (before 08:33):** `smoke_robolab_pi05_ext-mustard_berkeley3_paraphrase_zh` (2 episodes, the Chinese sentence)
  and `smoke_robolab_pi05_ext-cube-rand10_berkeley3_switch_cube` (1 episode, ±10 cm variant); both completed with the
  `requested_delivered` / `other_delivered` columns. Their run IDs use `RUN_PREFIX=smoke` instead of a timestamp.
- **Launch order:**

  | GPU | Group | Keys | Ran (UTC) |
  |---|---|---|---|
  | 0 | mustard | negation, distractor_mention, irrelevant_content, paraphrase_common, paraphrase_rare | 08:33–13:22 |
  | 1 | mustard | paraphrase_functional, paraphrase_zh, spatial_robot, spatial_image | 08:33–12:23 |
  | 1 | cube | switch_cube, switch_banana | 12:23–12:57 |
  | 2 | cube | negation_not_banana, negation_not_cube, distractor_cube, distractor_banana | 08:33–09:54 |
  | 2 | size | negation, distractor_mention, irrelevant_content | 09:54–12:16 |

- **Checks:** all 18 runs exit 0; 20 episodes per task and key (cube 20, mustard and size 40 per key); each run's
  `SHA256SUMS` verified on the server and, for the copied files, again on 2026-11-15. Fixed initial poses (gap G1): every
  cell repeats one layout.
- **Headline:** negation is followed as a pointer in the cube scene ("not the banana" → banana 17/20; "not the cube" →
  cube 20/20); the switch requests are followed (40/40); the Chinese sentence never gets the mustard into a bin (0/40).
  Full tables: [`results.md`](results.md).
- **Files:** server `/mnt/dt-vol-00494956a3ff3a063-20000G/qian/robolab_ws/runs/` (18 runs + 2 smoke, about 6.1 GB incl. videos); repo `results/robolab/<run_id>/` (`run.json`,
  `episodes.jsonl`, `summary.csv`, `SHA256SUMS`). Not yet on HF or the laptop.

### 2026-11-15 — Records: the 2026-11-04 runs backed up to HF and the laptop

- The 18 runs `20261104-0833_robolab_pi05_ext-*` and the two smoke runs (incl. raw videos): copied to the laptop
  (`~/vla_results_backup/robolab/`, every run checked against its `SHA256SUMS`) and uploaded to HF
  `Qian0203/vla_ws-eval-results`, `robolab/<run_id>/`; remote file count equals the local one for all 20. Approval: user,
  in chat, 2026-11-15. Supersedes "Not yet on HF or the laptop" in the entry above.

### 2026-11-15 — Scene changes A1, A3, A4: random layouts, bowl stacking, swapped size scene, π0.5-DROID, on `berkeley-3` (15 runs, 520 episodes)

- **Machine and stack:** as the 2026-11-04 entry (`berkeley-3`, Apptainer, RoboLab `b6cfcf6`, openpi `aa64205`,
  `pi05_droid_jointpos`, `NUM_ENVS=5`, `NUM_RUNS=4`, 20 episodes per task and key). GPUs 0–2, one policy server each
  (ports 8765–8767). `RUN_PREFIX=20261115-1617`. Approval: user, in chat, 2026-11-15 ("A1, A2, A4 and A3 default").
- **Code:** branch `robolab/option-a-runkit`, committed after the runs together with these records (swapped scene
  `robolab_lang/scenes/butter_raisin_box_grey_bin_swapped.usda` from `scenes/make_swapped_scene.py`, tasks
  `Lang{LargerObjectRaisinBox,SmallerObjectButter}InBinSwappedTask.py`, `make_run_record.py` base-task mapping);
  the server copies matched the committed files (md5 checked 2026-11-16).
- **Launch order:**

  | GPU | Change | Group / tasks | Keys | Ran (UTC) |
  |---|---|---|---|---|
  | 0 | A1 (`VARIANT=rand10`, ±10 cm initial poses) | mustard | default, negation, paraphrase_zh, spatial_robot, spatial_image | 16:17–19:32 |
  | 1 | A1 (`VARIANT=rand10`) | cube, then size | cube: default, switch ×2, negation ×2; size: default, negation | 16:17–18:43 |
  | 2 | A3 (bowl stacking, original scene) | `LangBowlStacking{LeftOnRight,RightOnLeft}Task` via `run_option_a.sh`, `STAGE=ext-bowls` | default | 16:21–17:14 |
  | 2 | A4 (butter and raisin box positions swapped) | `Lang*SwappedTask` via `run_option_a.sh`, `STAGE=ext-size-swapped` | default, no_target | 16:34–18:21 |

  A1 ran through `paper_tests_gpu.sh` (`VARIANT=rand10 ONLY_KEYS=...`); the policy servers were stopped by
  `logs/a_finisher.sh` at 19:33. GPU 2's policy server was shared by A3 and A4; the RoboCasa A2 run used its own server on
  the same GPU.
- **Smoke:** `smoke_robolab_pi05_ext-size-swapped_berkeley3_default` (1 episode, larger-object sentence): objects start
  at the swapped positions (0.0 cm drift over 20 steps, checked separately), butter delivered.
- **Checks:** all 15 runs exit 0, 20 episodes per task and key; `SHA256SUMS` verified on the server and on the laptop
  copy (2026-11-16).
- **Headline:** with random layouts the cube negation failure (excluded object 35/40) and the switch (39/40) hold, the
  Chinese sentence stays at 0/40, but the robot-frame vs image-frame gap does not (12 vs 10 of 40). In the swapped
  scene π0.5 still takes the butter (default 14/20 butter on the larger-object task; with no object named, 36/40). Bowl
  stacking default 9/40, below the 50% bar, so no language keys follow. Tables: [`results.md`](results.md).
- **Files:** server `/mnt/dt-vol-00494956a3ff3a063-20000G/qian/robolab_ws/runs/20261115-1617_*` (15 runs, 5.1 GB incl.
  videos) and the smoke run; laptop `~/vla_results_backup/robolab/` (verified); repo `results/robolab/<run_id>/`
  (`run.json`, `episodes.jsonl`, `summary.csv`, `SHA256SUMS`). Not yet on HF.
