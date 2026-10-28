#!/usr/bin/env python3
"""Write the per-run record files for one RoboLab run directory (results-storage convention, 2026-10-23).

    runs/<run_id>/run.json          metadata (written before a run by the launcher; completed here)
    runs/<run_id>/episodes.jsonl    one line per episode, with which-object outcome fields
    runs/<run_id>/summary.csv       per task x instruction key: n, successes, SR, Wilson CI, mean score
    runs/<run_id>/robolab_output/   RoboLab's own output folder (episode_results.jsonl, HDF5, videos, logs)

Outcome fields per episode, from RoboLab's HDF5 (states/rigid_object/*/root_pose, bbox/bbox_mm/*):
`lifted_objects` (root z > 3 cm above its first value at any step), `first_lifted`, `first_lift_step`,
`in_container` ({object: container} at the last step: object centroid inside a container's bbox
footprint and above its bottom), plus the mirror-pair labels of outcomes.py when the task has a spec.
`requested_delivered` / `other_delivered`: the requested target, or the other one (the excluded object
under negation, the non-requested one under switch), ends in the container. These are the paper's
violation counts; for LangRubiksCubeOrBananaTask they replace RoboLab's `success` (any object counts).

Usage (Python with h5py + numpy, e.g. the RoboLab image's):
    python make_run_record.py <run dir> [--meta key=value ...]
`--meta` entries are merged into run.json (JSON values allowed: --meta 'num_envs=25').
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from datetime import datetime, timezone

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from outcomes import OBJECT_REQUESTS, SPECS, label_dest, label_stack  # noqa: E402

LIFT = 0.03
CONTAINER = re.compile(r"bowl|bin|plate|crate|basket|pot|mug|box_|storage_box|pail|shelf", re.I)


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def base_task(name):
    """Pose-randomized variants (e.g. LangMustardInLeftBinRand10Task) share their base task's outcome spec."""
    return name.replace("Rand10Task", "Task")


def generic_outcome(g):
    objs = [k for k in g["states/rigid_object"].keys() if k != "table"]
    lifted, first, first_step = [], None, None
    for o in objs:
        z = g[f"states/rigid_object/{o}/root_pose"][:, 2]
        up = np.where(z - z[0] > LIFT)[0]
        if len(up):
            lifted.append(o)
            if first_step is None or up[0] < first_step:
                first, first_step = o, int(up[0])
    in_container = {}
    if "bbox" in g:
        for o in objs:
            p = g[f"bbox/centroid/{o}"][-1].astype(float)
            for c in objs:
                if c == o or not CONTAINER.search(c):
                    continue
                b = g[f"bbox/bbox_mm/{c}"][-1].astype(float) / 1000.0
                if (b[:, 0].min() <= p[0] <= b[:, 0].max() and b[:, 1].min() <= p[1] <= b[:, 1].max()
                        and p[2] >= b[:, 2].min()):
                    in_container[o] = c
    return dict(lifted_objects=sorted(lifted), first_lifted=first, first_lift_step=first_step,
                in_container=in_container)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--meta", action="append", default=[])
    args = ap.parse_args(argv)
    run_dir = os.path.abspath(args.run_dir)
    run_id = os.path.basename(run_dir)
    out = os.path.join(run_dir, "robolab_output")

    meta_path = os.path.join(run_dir, "run.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {"run_id": run_id}
    for kv in args.meta:
        k, v = kv.split("=", 1)
        try:
            meta[k] = json.loads(v)
        except json.JSONDecodeError:
            meta[k] = v

    res_file = os.path.join(out, "episode_results.jsonl")
    eps = [json.loads(l) for l in open(res_file) if l.strip()] if os.path.exists(res_file) else []
    rows = []
    for ep in eps:
        key = (ep.get("instruction_fields_used") or ep.get("instruction_fields") or ["default"])[0]
        row = dict(run_id=run_id, task=ep["env_name"], task_name=ep.get("task_name"), instruction_key=key,
                   instruction=ep.get("instruction"), run=ep.get("run"), env_id=ep.get("env_id"),
                   episode=ep.get("episode"), success=bool(ep.get("success")), score=ep.get("score"),
                   num_steps=ep.get("episode_step"), reason=ep.get("reason"), events=ep.get("events") or {})
        h5 = os.path.join(out, ep["env_name"], f"run_{ep.get('run', 0)}.hdf5")
        if os.path.exists(h5):
            with h5py.File(h5, "r") as fh:
                g = fh["data"].get(f"demo_{ep.get('env_id', 0)}")
                if g is not None:
                    row.update(generic_outcome(g))
                    spec = SPECS.get(base_task(ep["env_name"]))
                    if spec:
                        lab = label_dest(g, spec) if spec["kind"] == "dest" else label_stack(g, spec)
                        row.update({f"pair_{k}": v for k, v in lab.items()},
                                   pair_target=spec["target"], pair_other=spec["other"])
        req = OBJECT_REQUESTS.get((base_task(row["task"]), key)) or OBJECT_REQUESTS.get(base_task(row["task"]))
        if req and "in_container" in row:
            obj, other, cont = req
            inside = {o for o, c in row["in_container"].items() if c == cont}
            row.update(requested=obj, other=other, requested_delivered=obj in inside and other not in inside,
                       other_delivered=other in inside)
        elif "pair_delivered" in row:
            row.update(requested=row["pair_target"], other=row["pair_other"],
                       requested_delivered=row["pair_delivered"] == row["pair_target"],
                       other_delivered=row["pair_delivered"] == row["pair_other"])
        rows.append(row)

    with open(os.path.join(run_dir, "episodes.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    groups = {}
    for r in rows:
        groups.setdefault((r["task"], r["instruction_key"]), []).append(r)
    with open(os.path.join(run_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "task", "instruction_key", "n", "successes", "sr", "sr_ci_lo", "sr_ci_hi",
                    "mean_score", "wrong_object_grabbed_eps", "requested_delivered", "other_delivered"])
        for (task, key), rs in sorted(groups.items()):
            n, k = len(rs), sum(r["success"] for r in rs)
            lo, hi = wilson(k, n)
            sc = [r["score"] for r in rs if r["score"] is not None]
            wrong = sum(1 for r in rs if r["events"].get("WRONG_OBJECT_GRABBED"))
            w.writerow([run_id, task, key, n, k, f"{k / n:.4f}", f"{lo:.4f}", f"{hi:.4f}",
                        f"{sum(sc) / len(sc):.4f}" if sc else "", wrong,
                        sum(1 for r in rs if r.get("requested_delivered")) if any("requested" in r for r in rs) else "",
                        sum(1 for r in rs if r.get("other_delivered")) if any("requested" in r for r in rs) else ""])

    raw_bytes = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(out) for fn in fns) \
        if os.path.isdir(out) else 0
    meta.update(episodes_recorded=len(rows), raw_output_dir=out, raw_output_bytes=raw_bytes,
                record_written_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print(f"{run_id}: {len(rows)} episodes, {len(groups)} task x key cells, raw {raw_bytes / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
