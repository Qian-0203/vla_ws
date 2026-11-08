#!/usr/bin/env python3
"""CPU/EGL checks for the mirror-pair classes in robocasa_lang/envs.py (plan §3.2). No policy, no model.

For each pair class and each seed:
  1. scene identity: slot a and slot b (two env instances, same seed) give the same scene hash, and the same
     instruction except for the named candidate; for NavigateKitchenPair / TurnOffStovePair, slot a's hash also
     equals the parent task's hash (from the 2026-10-21 scene probe).
  2. validity: pair_valid() (distinct words for the two candidates / a second destination exists / two knobs)
     and, for the stove, both knobs lit at reset.
  3. scripted oracle on slot a: put the *named* candidate in its goal state -> _check_success() must be True;
     reset, put the *other* candidate in its goal state -> must be False; reset, both -> must be False.
     PnP: teleport the object into the cabinet interior and let it settle. Stove: set the knob to off.
     Navigation: teleport the robot base to the destination's placement pose.

Usage: python robocasa_lang/pair_check.py --seeds robocasa_lang/pilot_seeds.json --out <dir> [--n 20] [--classes ...]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_probe import kitchen_of, scene_hash, save_sheet, project, body_pos, visibility  # noqa: E402

PARENT = {"PnPCounterToCabPair": "PickPlaceCounterToCabinet", "TurnOffStovePair": "TurnOffStove",
          "NavigateKitchenPair": "NavigateKitchen"}


def settle(k, n=20):
    act = np.zeros(k.action_spec[0].shape)
    for _ in range(n):
        k.step(act)


def goal_pnp(k, name):
    obj = k.objects[name]
    sites = k.cab.get_int_sites(relative=False)
    p0, px, py, pz = (np.array(p) for p in next(iter(sites.values())))
    center = p0 + 0.5 * (px - p0) + 0.5 * (py - p0) + np.array([0, 0, 0.03])
    # spread two objects apart along the cabinet's x axis so "both" does not stack them
    shift = 0.25 * (px - p0) * (-1 if name.endswith("a") else 1)
    quat = k.sim.data.get_joint_qpos(obj.joints[0])[3:]
    k.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([center + shift, quat]))
    k.sim.forward()


def goal_stove(k, knob):
    k.stove.set_knob_state(mode="off", knob=knob, env=k, rng=np.random.default_rng(0))
    k.sim.forward()


def goal_nav(k, slot):
    import robocasa.utils.env_utils as EnvUtils
    pos, ori = k.pair_poses[slot]
    yaw_addr = k.sim.model.get_joint_qpos_addr("mobilebase0_joint_mobile_yaw")
    rid = k.sim.model.body_name2id("mobilebase0_base")
    cmd = np.array(pos, dtype=float)
    # the base body is offset from the slide joints, so after the yaw is set the body lands off target;
    # correct the commanded position by the measured error a few times
    for _ in range(5):
        EnvUtils.set_robot_to_position(k, cmd)
        k.sim.data.qpos[yaw_addr] = ori[2] - k.init_robot_base_ori_anchor[2]
        k.sim.forward()
        err = np.array(pos[:2]) - np.array(k.sim.data.body_xpos[rid][:2])
        if np.linalg.norm(err) < 0.01:
            break
        cmd[:2] += err


def oracle(env, seed, cls):
    """Returns dict(target, other, both) -> _check_success() after putting those candidates in the goal state."""
    res = {}
    for case in ("target", "other", "both"):
        env.reset(seed=seed)
        k = kitchen_of(env)
        names = {"target": [k.pair_target], "other": [k.pair_other], "both": [k.pair_target, k.pair_other]}[case]
        for n in names:
            if cls == "PnPCounterToCabPair":
                goal_pnp(k, n)
            elif cls == "TurnOffStovePair":
                goal_stove(k, n)
            else:
                goal_nav(k, n)
        if cls == "PnPCounterToCabPair":
            settle(k)
        res[case] = bool(k._check_success())
        res[f"{case}_outcome"] = {str(a): bool(b) for a, b in k.pair_outcome().items()}
    return res


def check_class(gym, cls, seeds, n, out, parent_hashes):
    envs = {s: gym.make(f"robocasa/{cls}", split="pretrain", seed=7, target_slot=s) for s in ("a", "b")}
    rows, frames = [], []
    for seed in seeds[:n]:
        row = dict(cls=cls, seed=seed)
        for s, env in envs.items():
            obs, _ = env.reset(seed=seed)
            k = kitchen_of(env)
            row[f"hash_{s}"] = scene_hash(k)
            row[f"lang_{s}"] = str(obs["annotation.human.task_description"])
            row["valid"] = bool(k.pair_valid())
            row["pair"] = k.get_ep_meta().get("pair")
            if cls == "TurnOffStovePair":
                row[f"lit_{s}"] = [kk for kk in k.pair_knobs.values() if k.stove.is_burner_on(env=k, burner_loc=kk)]
            if cls == "PnPCounterToCabPair" and s == "a":
                pa, pb = body_pos(k, "obj_a"), body_pos(k, "obj_b")
                row["ab_dist_m"] = float(np.linalg.norm(pa[:2] - pb[:2]))
                row["px_a"], row["px_b"] = project(k, pa), project(k, pb)
                row["visibility"] = {cam: {n: visibility(k, n, cam) for n in ("obj_a", "obj_b")}
                                     for cam in ("robot0_agentview_left", "robot0_agentview_right")}
                frames.append((seed, obs["video.robot0_agentview_left"], row["px_a"], row["px_b"]))
        row["same_scene"] = row["hash_a"] == row["hash_b"]
        if seed in parent_hashes:
            row["slot_a_equals_parent"] = row["hash_a"] == parent_hashes[seed]
        if row["valid"]:
            row["oracle"] = oracle(envs["a"], seed, cls)
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ("seed", "same_scene", "slot_a_equals_parent", "valid")}
                         | {"oracle": {c: row.get("oracle", {}).get(c) for c in ("target", "other", "both")}}),
              flush=True)
    for env in envs.values():
        env.close()
    with open(os.path.join(out, f"{cls}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    if frames:
        save_sheet(frames, os.path.join(out, f"{cls}_first_frames.png"))
    valid = [r for r in rows if r["valid"]]
    return dict(cls=cls, n=len(rows), same_scene=sum(r["same_scene"] for r in rows),
                slot_a_equals_parent=sum(bool(r.get("slot_a_equals_parent")) for r in rows) if parent_hashes else None,
                valid=len(valid),
                oracle_ok=sum(r["oracle"]["target"] and not r["oracle"]["other"] and not r["oracle"]["both"]
                              for r in valid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--classes", default=",".join(PARENT))
    ap.add_argument("--parent-probe", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                                           "docs", "robocasa", "scene_probe_2026-10-21"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import gymnasium as gym
    import envs  # noqa: F401  (registers the pair classes and their gym ids)
    spec = json.load(open(a.seeds))
    summary = []
    for cls in a.classes.split(","):
        parent = PARENT[cls]
        seeds = [e["seed"] for e in spec["tasks"][parent]["episodes"]]
        ph = {}
        pp = os.path.join(a.parent_probe, f"{parent}.jsonl")
        if cls != "PnPCounterToCabPair" and os.path.exists(pp):
            ph = {r["seed"]: r["hash"] for r in map(json.loads, open(pp))}
        r = check_class(gym, cls, seeds, a.n, a.out, ph)
        print("SUMMARY", json.dumps(r), flush=True)
        summary.append(r)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
