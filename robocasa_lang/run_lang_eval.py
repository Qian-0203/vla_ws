#!/usr/bin/env python3
"""Language-condition client for Xiaomi-Robotics-1 on RoboCasa365 (plan §4.0, §5.3 step 4).

The policy side is the upstream client, imported unchanged from the Xiaomi-Robotics-1 checkout: EvalClient (processor,
chat template, crop, state padding), the observation history, observation_to_state and robocasa's convert_action. The
episode loop is the upstream evaluate_task loop. Added on top:

  - the instruction comes from the cell's condition (robocasa_lang/conditions.py), built from the scene's own slots
    after reset; `default` keeps the env's instruction, `blank` sends ""
  - the env class can be a robocasa task or one of robocasa_lang/envs.py (retargeted success, pairs)
  - scene hash after reset; which-object / which-fixture outcome logging every step (plan §3.3)
  - one row per episode appended to <run>/episodes.jsonl (resumable: (cell, seed) rows already present are skipped)

A cell is one line of the --cells JSON file:
  {"cell": "pnp_blank", "env": "PickPlaceCounterToCabinet", "env_kwargs": {}, "horizon_task": "PickPlaceCounterToCabinet",
   "family": "pnp", "condition": "blank", "seeds": [407, ...], "n": 20, "require_valid": false}
`require_valid` skips seeds whose env reports pair_valid() == False and moves on to the next seed in the list.

Usage (sim venv): python run_lang_eval.py --xr1-repo <checkout> --model-path <ckpt> --port 10086 --run-dir <dir>
                   --run-id <id> --cells <cells.json> [--cell-names a,b]
       π0.5:      python run_lang_eval.py --policy pi05 --port 8000 --run-dir <dir> --run-id <id> --cells <cells.json>
                   (pi05_policy.py; observation packing as robocasa-benchmark/openpi examples/robocasa/main.py)
"""
import argparse
import collections
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

APPROACH_M = 0.08
LIFT_M = 0.03
MIXER_PARAPHRASE = "Open the head of the stand mixer."  # paraphrase_syntactic for the constant-instruction control


def load_upstream(repo):
    spec = importlib.util.spec_from_file_location("xr1_entry", os.path.join(repo, "eval_robocasa365", "entry.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Outcome:
    """Per-episode which-candidate record. Families: pnp (objects), nav (destinations), stove (knobs), other."""

    def __init__(self, k, family):
        import robocasa.utils.object_utils as OU
        self.k, self.family, self.OU = k, family, OU
        self.rows = {}
        if family == "pnp":
            names = [c["name"] for c in k.object_cfgs if c["name"] in k.obj_body_id]
            self.cands = [n for n in ("obj", "distr_counter", "obj_a", "obj_b") if n in names]
            self.settle = {n: float(k.sim.data.body_xpos[k.obj_body_id[n]][2]) for n in self.cands}
            self.min_d = {n: float("inf") for n in self.cands}
            self.approach = {n: None for n in self.cands}
            self.grasp = {n: None for n in self.cands}
        elif family == "nav":
            self.poses = dict(getattr(k, "pair_poses", {"a": (k.target_pos, k.target_ori)}))
            self.reached = {s: None for s in self.poses}
            self.min_d = {s: float("inf") for s in self.poses}
        elif family == "stove":
            self.knobs = dict(getattr(k, "pair_knobs", {"a": k.knob}))
            self.lit0 = {kn: k.stove.is_burner_on(env=k, burner_loc=kn) for kn in k.stove.knob_joints
                         if k.stove.knob_joints[kn] is not None}
            self.off_step = {kn: None for kn in self.lit0}

    def update(self, t):
        k = self.k
        if self.family == "pnp":
            eef = np.array(k.sim.data.site_xpos[k.robots[0].eef_site_id["right"]])
            for n in self.cands:
                p = np.array(k.sim.data.body_xpos[k.obj_body_id[n]])
                d = float(np.linalg.norm(eef - p))
                self.min_d[n] = min(self.min_d[n], d)
                if d < APPROACH_M and self.approach[n] is None:
                    self.approach[n] = t
                if self.grasp[n] is None and p[2] - self.settle[n] > LIFT_M and self.OU.check_obj_grasped(k, n):
                    self.grasp[n] = t
        elif self.family == "nav":
            from robocasa.utils.env_utils import T
            rid = k.sim.model.body_name2id("mobilebase0_base")
            bp = np.array(k.sim.data.body_xpos[rid])
            yaw = T.mat2euler(np.array(k.sim.data.body_xmat[rid]).reshape(3, 3))[2]
            for s, (pos, ori) in self.poses.items():
                d = float(np.linalg.norm(np.array(pos[:2]) - bp[:2]))
                self.min_d[s] = min(self.min_d[s], d)
                if self.reached[s] is None and d <= 0.20 and np.cos(ori[2] - yaw) >= 0.98:
                    self.reached[s] = t
        elif self.family == "stove":
            for kn, lit in self.lit0.items():
                if lit and self.off_step[kn] is None and not k.stove.is_burner_on(env=k, burner_loc=kn):
                    self.off_step[kn] = t

    @staticmethod
    def _first(steps):
        hits = sorted((v, n) for n, v in steps.items() if v is not None)
        return (hits[0][1], hits[0][0]) if hits else (None, None)

    def result(self):
        k = self.k
        if self.family == "pnp":
            fa, fas = self._first(self.approach)
            fg, fgs = self._first(self.grasp)
            delivered = [n for n in self.cands if self.OU.obj_inside_of(k, n, k.cab)
                         and self.OU.gripper_obj_far(k, obj_name=n)] if hasattr(k, "cab") else None
            return dict(candidates={n: k.get_obj_lang(n) for n in self.cands}, first_approached=fa,
                        first_approached_step=fas, first_grasped=fg, first_grasp_step=fgs, delivered=delivered,
                        min_dist_by_obj={n: round(v, 3) for n, v in self.min_d.items()},
                        approach_step_by_obj=self.approach, grasp_step_by_obj=self.grasp)
        if self.family == "nav":
            fr, frs = self._first(self.reached)
            fx = getattr(k, "pair_fixtures", {"a": k.target_fixture})
            return dict(candidates={s: fx[s].nat_lang for s in self.poses}, first_reached=fr, first_reached_step=frs,
                        reached_step_by_dest=self.reached, min_base_dist_by_dest={s: round(v, 3) for s, v in
                                                                                  self.min_d.items()})
        if self.family == "stove":
            fo, fos = self._first(self.off_step)
            lit_end = [kn for kn in self.lit0 if k.stove.is_burner_on(env=k, burner_loc=kn)]
            return dict(candidates=self.knobs, lit_at_start=[kn for kn, v in self.lit0.items() if v],
                        lit_at_end=lit_end, first_turned_off=fo, first_turned_off_step=fos,
                        off_step_by_knob=self.off_step)
        return {}


def instruction_for(cell, k, env_instruction):
    import conditions as C
    cond, fam = cell["condition"], cell["family"]
    if cond == "default":
        return env_instruction
    if cond == "blank":
        return ""
    if fam == "other":
        if cond == "paraphrase_syntactic" and cell["horizon_task"] == "OpenStandMixerHead":
            return MIXER_PARAPHRASE
        raise ValueError(f"no rule for {cell['cell']}")
    if fam == "pnp":
        names = [c["name"] for c in k.object_cfgs]
        t, d = ("obj_a", "obj_b") if "obj_a" in names else ("obj", "distr_counter")
        if getattr(k, "target_slot", "a") == "b":
            t, d = d, t
        return C.make("pnp", cond, target=k.get_obj_lang(t), distractor=k.get_obj_lang(d), src="counter",
                      dst="cabinet")
    if fam == "nav":
        fx = getattr(k, "pair_fixtures", {"a": k.target_fixture})
        slot = getattr(k, "target_slot", "a")
        other = "b" if slot == "a" else "a"
        return C.make("nav", cond, target=fx[slot].nat_lang,
                      distractor=fx[other].nat_lang if other in fx else None)
    if fam == "stove":
        kn = dict(getattr(k, "pair_knobs", {"a": k.knob}))
        slot = getattr(k, "target_slot", "a")
        other = "b" if slot == "a" else "a"
        return C.make("stove", cond, target=kn[slot], distractor=kn.get(other))
    raise ValueError(fam)


def run_cell(cell, args, up, client, gym, get_task_horizon, convert_action, done, out_f):
    from scene_probe import kitchen_of, scene_hash
    env = gym.make(f"robocasa/{cell['env']}", split="pretrain", seed=7, **cell.get("env_kwargs", {}))
    horizon = get_task_horizon(cell["horizon_task"])
    n_done = sum(1 for (c, _s) in done if c == cell["cell"])
    vid_dir = os.path.join(args.run_dir, "raw", "videos", cell["cell"])
    os.makedirs(vid_dir, exist_ok=True)
    try:
        for seed in cell["seeds"]:
            if n_done >= cell["n"]:
                break
            if (cell["cell"], seed) in done:
                continue
            obs, _ = env.reset(seed=seed)
            k = kitchen_of(env)
            if cell.get("require_valid") and hasattr(k, "pair_valid") and not k.pair_valid():
                out_f.write(json.dumps(dict(run_id=args.run_id, cell=cell["cell"], seed=seed, skipped="pair_invalid"))
                            + "\n")
                out_f.flush()
                continue
            try:
                instruction = instruction_for(cell, k, str(obs["annotation.human.task_description"]))
            except ValueError as e:  # e.g. target and distractor named by the same words
                out_f.write(json.dumps(dict(run_id=args.run_id, cell=cell["cell"], seed=seed, skipped=str(e))) + "\n")
                out_f.flush()
                continue
            sh = scene_hash(k)
            tracker = Outcome(k, cell["family"])
            qlen = (args.obs_history - 1) * args.obs_interval + 1
            iq = {key: collections.deque(maxlen=qlen) for key in up.CAMERA_KEYS}
            sq = collections.deque(maxlen=qlen)
            for key, im in up.collect_images(obs).items():
                iq[key].append(im)
            sq.append(up.observation_to_state(obs))
            plan, frames, success, steps, t0 = collections.deque(), [up.make_video_frame(obs)], False, 0, time.time()
            while steps < horizon:
                tracker.update(steps)
                if not plan:
                    states = up.sample_history(sq, args.obs_history, args.obs_interval)
                    images = {key: up.sample_history(q, args.obs_history, args.obs_interval) for key, q in iq.items()}
                    plan.extend(client.infer(states, images, instruction)[: args.replan_steps])
                obs, _, d, tr, info = env.step(convert_action(np.asarray(plan.popleft(), dtype=np.float32)))
                steps += 1
                for key, im in up.collect_images(obs).items():
                    iq[key].append(im)
                sq.append(up.observation_to_state(obs))
                success = bool(info.get("success", False))
                if steps % 2 == 0 or success or d or tr:
                    frames.append(up.make_video_frame(obs))
                if success or d or tr:
                    break
            tracker.update(steps)
            status = "success" if success else "failure"
            vid = os.path.join(vid_dir, f"seed_{seed}_{status}.mp4")
            import imageio.v2 as imageio
            imageio.mimsave(vid, frames, fps=20)
            row = dict(run_id=args.run_id, task=cell["horizon_task"], env=cell["env"],
                       env_kwargs=cell.get("env_kwargs", {}), cell=cell["cell"], condition=cell["condition"],
                       instruction=instruction, seed=seed, episode=n_done, success=success, num_steps=steps,
                       horizon=horizon, scene_hash=sh, wall_s=round(time.time() - t0, 1),
                       video=os.path.relpath(vid, args.run_dir), **{"outcome": tracker.result()})
            out_f.write(json.dumps(row, default=str) + "\n")
            out_f.flush()
            done.add((cell["cell"], seed))
            n_done += 1
            print(f"[{cell['cell']}] seed {seed} {status} steps={steps} ({n_done}/{cell['n']})", flush=True)
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=("xr1", "pi05"), default="xr1")
    ap.add_argument("--xr1-repo")
    ap.add_argument("--model-path")
    ap.add_argument("--port", type=int, default=10086)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--cell-names", default="")
    ap.add_argument("--episodes-file", default="episodes.jsonl")
    ap.add_argument("--obs-history", type=int, default=4)
    ap.add_argument("--obs-interval", type=int, default=2)
    ap.add_argument("--replan-steps", type=int, default=None, help="default: 16 (xr1), 5 (pi05)")
    ap.add_argument("--crop-ratio", type=float, default=0.95)
    a = ap.parse_args()
    if a.policy == "pi05":
        import pi05_policy as up
        a.obs_history, a.obs_interval = 1, 1
        a.replan_steps = a.replan_steps or up.REPLAN_STEPS
    else:
        if not (a.xr1_repo and a.model_path):
            ap.error("--xr1-repo and --model-path are required for --policy xr1")
        up = load_upstream(a.xr1_repo)
        a.replan_steps = a.replan_steps or 16
    import gymnasium as gym
    import robocasa  # noqa: F401
    import robocasa.wrappers.gym_wrapper  # noqa: F401
    import envs  # noqa: F401  (pair / swap classes)
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    from robocasa.utils.env_utils import convert_action
    cells = [json.loads(l) for l in open(a.cells) if l.strip()]
    if a.cell_names:
        want = a.cell_names.split(",")
        cells = [c for c in cells if c["cell"] in want]
    path = os.path.join(a.run_dir, a.episodes_file)
    done = set()
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            if "success" in r:
                done.add((r["cell"], r["seed"]))
    if a.policy == "pi05":
        client = up.Client(host="127.0.0.1", port=a.port)
    else:
        client = up.EvalClient(model_path=a.model_path, host="localhost", port=a.port, robot_type=up.ROBOT_TYPE,
                               crop_ratio=a.crop_ratio)
    try:
        with open(path, "a") as out_f:
            for cell in cells:
                run_cell(cell, a, up, client, gym, get_task_horizon, convert_action, done, out_f)
    finally:
        client.close()
    print("CLIENT_DONE", flush=True)


if __name__ == "__main__":
    main()
