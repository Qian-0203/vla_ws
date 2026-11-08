#!/usr/bin/env python3
"""Run records for RoboCasa runs: run.json, episodes.jsonl, summary.csv (format agreed 2026-10-23).

Run ID: YYYYMMDD-HHMM_robocasa_<model>_<stage>_<host>, stage in smoke / gate / screen / main / exploratory.

  episodes.jsonl  one line per episode: run_id, task, condition, instruction, seed, episode, success, num_steps,
                  scene_hash, and the outcome fields (approached / grasped / delivered object or fixture; null when
                  the client did not log them)
  summary.csv     task, condition, n, successes, sr

Subcommand `from-upstream` converts the upstream Xiaomi-Robotics-1 client's per-task stats.json files (which have no
instruction, scene hash or outcome fields; those are written as null) into episodes.jsonl + summary.csv.

Usage:
  python runrecord.py from-upstream --run-id <id> --results <dir with <Task>/stats.json> --out <run dir> \
      [--condition default] [--instruction-from-log client.log]
  python runrecord.py summarize --episodes <episodes.jsonl> --out <summary.csv>
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import OrderedDict

STAGES = ("smoke", "gate", "screen", "main", "exploratory")
RUN_ID = re.compile(r"^\d{8}-\d{4}_robocasa_[a-z0-9.\-]+_(%s)_[a-z0-9\-]+$" % "|".join(STAGES))
OUTCOME_FIELDS = ("target", "distractors", "first_approached", "first_approached_step", "first_grasped",
                  "first_grasp_step", "delivered")


def check_run_id(run_id):
    if not RUN_ID.match(run_id):
        raise SystemExit(f"bad run id {run_id!r}: want YYYYMMDD-HHMM_robocasa_<model>_<stage>_<host>")
    return run_id


def summarize(rows, out_csv):
    """Per task x condition (and per cell when rows carry one: several cells can share task and condition, e.g. the
    two slots of a pair). Rows without `success` (skipped seeds) are left out."""
    rows = [r for r in rows if "success" in r]
    with_cell = any("cell" in r for r in rows)
    cells = OrderedDict()
    for r in rows:
        key = (r["task"], r.get("cell", ""), r["condition"])
        n, s = cells.get(key, (0, 0))
        cells[key] = (n + 1, s + int(bool(r["success"])))
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "cell", "condition", "n", "successes", "sr"] if with_cell else
                   ["task", "condition", "n", "successes", "sr"])
        for (task, cell, cond), (n, s) in cells.items():
            w.writerow(([task, cell, cond] if with_cell else [task, cond]) + [n, s, round(s / n, 4)])
    return {(t, c or cond): v for (t, c, cond), v in cells.items()}


def from_upstream(a):
    run_id = check_run_id(a.run_id)
    rows = []
    for f in sorted(glob.glob(os.path.join(a.results, "*", "stats.json"))):
        d = json.load(open(f))
        for e in d["episodes"]:
            row = dict(run_id=run_id, task=d["env_name"], condition=a.condition, instruction=None, seed=e["seed"],
                       episode=e["episode"], global_episode_index=e.get("global_episode_index"),
                       success=bool(e["success"]), num_steps=e.get("steps"), horizon=d.get("horizon"),
                       scene_hash=None)
            row.update({k: None for k in OUTCOME_FIELDS})
            rows.append(row)
    if a.instruction_from_log:
        # the upstream client logs "EPISODE_START ... instr='...'" only in some versions; fill when present
        pat = re.compile(r"task=(\S+) ep=(\d+) seed=(\d+).*instr=['\"](.*)['\"]")
        found = {}
        for line in open(a.instruction_from_log, errors="replace"):
            m = pat.search(line)
            if m:
                found[(m.group(1), int(m.group(3)))] = m.group(4)
        for r in rows:
            r["instruction"] = found.get((r["task"], r["seed"]))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "episodes.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    cells = summarize(rows, os.path.join(a.out, "summary.csv"))
    print(json.dumps({f"{t}/{c}": f"{s}/{n}" for (t, c), (n, s) in cells.items()}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("from-upstream")
    u.add_argument("--run-id", required=True)
    u.add_argument("--results", required=True)
    u.add_argument("--out", required=True)
    u.add_argument("--condition", default="default")
    u.add_argument("--instruction-from-log")
    s = sub.add_parser("summarize")
    s.add_argument("--episodes", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "from-upstream":
        from_upstream(a)
    else:
        summarize([json.loads(l) for l in open(a.episodes)], a.out)


if __name__ == "__main__":
    main()
