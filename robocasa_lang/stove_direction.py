#!/usr/bin/env python3
"""Do the stove's burner names (front_left, rear_right, ...) match what the robot sees? (plan §3.5 / §4 P7)

TurnOffStove's instruction names a burner in the stove's own frame. For each seed this script reads every burner's
world position (site `burner_on_<loc>`) and its knob's position (anchor of joint `knob_<loc>_joint`), and checks, for
every pair of burners whose names differ in a direction word:

  left/right : the burner named "left" is to the robot's left (larger y in the base frame, x forward / y left) and
               to the image left (smaller u) in robot0_agentview_left / _right
  front/rear : the burner named "front" is nearer the robot (smaller x in the base frame) and lower in the image
               (larger v)

Writes <out>/stove_direction.jsonl (one row per seed) and prints agreement rates. No policy.
Usage: python robocasa_lang/stove_direction.py --out <dir> [--seeds 707-756] [--extra 0-149]
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_probe import kitchen_of, project, to_base, robot_base  # noqa: E402

CAMS = ("robot0_agentview_left", "robot0_agentview_right")


def words(loc):
    w = set(loc.split("_"))
    lr = "left" if "left" in w else "right" if "right" in w else None
    fr = "front" if "front" in w else "rear" if ("rear" in w or "back" in w) else None
    return lr, fr


def parse_range(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def burner_points(k):
    st = k.stove
    pts = {}
    for loc, site in st.burner_sites.items():
        if site is None:
            continue
        p = np.array(k.sim.data.get_site_xpos(site.get("name")))
        knob = None
        jn = f"{st.naming_prefix}knob_{loc}_joint"
        try:
            jid = k.sim.model.joint_name2id(jn)
            knob = np.array(k.sim.data.xanchor[jid])  # several knob joints can share one body; use the joint anchor
        except Exception:
            pass
        pts[loc] = dict(burner=p, knob=knob)
    return pts


def check_pairs(k, pts):
    res = {"lr": [], "fr": []}
    info = {}
    for loc, d in pts.items():
        b = to_base(k, d["burner"])
        info[loc] = dict(base_xy=[round(float(b[0]), 3), round(float(b[1]), 3)],
                         px={c[7:]: project(k, d["burner"], cam=c) for c in CAMS},
                         knob_base_xy=None if d["knob"] is None else
                         [round(float(x), 3) for x in to_base(k, d["knob"])[:2]])
    for a, b in itertools.combinations(sorted(pts), 2):
        (lra, fra), (lrb, frb) = words(a), words(b)
        ia, ib = info[a], info[b]
        if lra and lrb and lra != lrb:
            left, right = (ia, ib) if lra == "left" else (ib, ia)
            row = dict(pair=[a, b], robot=left["base_xy"][1] > right["base_xy"][1])
            for c in CAMS:
                pl, pr = left["px"][c[7:]], right["px"][c[7:]]
                row[c[7:]] = None if pl is None or pr is None else bool(pl[0] < pr[0])
            if left["knob_base_xy"] and right["knob_base_xy"]:
                row["knob_robot"] = left["knob_base_xy"][1] > right["knob_base_xy"][1]
            res["lr"].append(row)
        if fra and frb and fra != frb:
            front, rear = (ia, ib) if fra == "front" else (ib, ia)
            row = dict(pair=[a, b], robot=front["base_xy"][0] < rear["base_xy"][0])
            for c in CAMS:
                pf, pr = front["px"][c[7:]], rear["px"][c[7:]]
                row[c[7:]] = None if pf is None or pr is None else bool(pf[1] > pr[1])
            res["fr"].append(row)
    return res, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="707-756")
    ap.add_argument("--extra", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import gymnasium as gym
    import robocasa  # noqa: F401
    import robocasa.wrappers.gym_wrapper  # noqa: F401
    env = gym.make("robocasa/TurnOffStove", split="pretrain", seed=7)
    rows = []
    for seed in parse_range(a.seeds) + parse_range(a.extra):
        env.reset(seed=seed)
        k = kitchen_of(env)
        pts = burner_points(k)
        res, info = check_pairs(k, pts)
        _, mat = robot_base(k)
        stove_yaw_in_base = None
        try:
            smat = np.array(k.sim.data.body_xmat[k.sim.model.body_name2id(k.stove.root_body)]).reshape(3, 3)
            rel = mat.T @ smat
            stove_yaw_in_base = round(float(np.degrees(np.arctan2(rel[1, 0], rel[0, 0]))), 1)
        except Exception:
            pass
        rows.append(dict(seed=seed, layout=k.layout_id, style=k.style_id, stove=type(k.stove).__name__,
                         knob=k.knob, stove_yaw_in_base_deg=stove_yaw_in_base, burners=info, **res))
        print(seed, k.layout_id, len(res["lr"]), len(res["fr"]), flush=True)
    env.close()
    with open(os.path.join(a.out, "stove_direction.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    summary = {}
    for kind in ("lr", "fr"):
        for field in ("robot", "agentview_left", "agentview_right", "knob_robot"):
            vals = [p.get(field) for r in rows for p in r[kind] if p.get(field) is not None]
            if vals:
                summary[f"{kind}_{field}"] = f"{sum(vals)}/{len(vals)}"
        scenes_bad = [r["seed"] for r in rows if any(p["robot"] is False for p in r[kind])]
        summary[f"{kind}_scenes_with_robot_frame_disagreement"] = scenes_bad
    summary["n_scenes"] = len(rows)
    summary["n_layouts"] = len({r["layout"] for r in rows})
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("SUMMARY", json.dumps(summary))


if __name__ == "__main__":
    main()
