#!/usr/bin/env python3
"""Gate check for π0.5 on RoboCasa365 against its published Atomic-Seen mean (39.6%, no per-task numbers published).

Gate: mean over the 18 Atomic-Seen tasks of each task's success on its first 10 seeds, within 7 pts of 39.6.
Also lists every task's default SR on all its episodes and which screen tasks keep default >= 50%.
Usage: python pi05_gate_report.py --episodes <episodes.jsonl> [--out gate.json]
"""
import argparse
import json
from collections import defaultdict

from pi05_gate_cells import FAMILY, GATE_N, TARGET50_ATOMIC_SEEN, seeds

PUBLISHED_MEAN = 39.6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.episodes) if '"success"' in l]
    by_task = defaultdict(dict)
    for r in rows:
        by_task[r["task"]][r["seed"]] = bool(r["success"])
    out = {"tasks": {}}
    gate_srs = []
    for t in TARGET50_ATOMIC_SEEN:
        got = by_task.get(t, {})
        first = [got[s] for s in seeds(t, GATE_N) if s in got]
        g = 100 * sum(first) / len(first) if first else None
        allv = list(got.values())
        out["tasks"][t] = dict(n_gate=len(first), gate_sr=g and round(g, 1), n_all=len(allv),
                               sr_all=round(100 * sum(allv) / len(allv), 1) if allv else None,
                               screen_task=t in FAMILY)
        if len(first) == GATE_N:
            gate_srs.append(g)
    complete = len(gate_srs) == len(TARGET50_ATOMIC_SEEN)
    m = sum(gate_srs) / len(gate_srs) if gate_srs else None
    out.update(mean_sr=m and round(m, 1), published_mean_sr=PUBLISHED_MEAN, tasks_complete=len(gate_srs),
               gate_pass=complete and abs(m - PUBLISHED_MEAN) <= 7,
               screen_tasks_kept=[t for t in FAMILY if (out["tasks"][t]["sr_all"] or 0) >= 50])
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
