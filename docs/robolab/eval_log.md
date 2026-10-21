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
