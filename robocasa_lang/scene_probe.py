#!/usr/bin/env python3
"""Simulator-only scene checks for the RoboCasa365 pilot tasks (plan §2.2, §3, §5.1). No policy, no model.

For each task and each published seed, builds the env exactly as the leaderboard clients do
(`gym.make("robocasa/<Task>", split="pretrain", seed=7)`, then `env.reset(seed=s)`) and records:

  - scene hash: object names + categories + initial positions (1 mm), fixture refs, layout/style, robot base pose
  - the instruction (`annotation.human.task_description`)
  - task-specific layout facts (target vs distractor geometry, navigation distance, stove knob states)
  - the pixel column of target / distractor in the upright `robot0_agentview_left` image (P7 direction check)
  - reset time and step time (zero action, cameras on)

Determinism checks (the precondition for paired conditions):
  A  same env, same seed twice in a row                         -> hash must match
  B  a fresh env, seed s directly                               -> hash must match A's (history independence)

Writes <out>/<task>.jsonl (one row per seed), <out>/<task>_first_frames.png (contact sheet with projected points)
and prints a summary. Usage:
  python robocasa_lang/scene_probe.py --seeds robocasa_lang/pilot_seeds.json --out <dir> [--n 50] [--tasks A,B]
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np

CAM = "robot0_agentview_left"


def kitchen_of(env):
    e = env.unwrapped
    return e.env if hasattr(e, "env") else e


def body_pos(k, name):
    return np.array(k.sim.data.body_xpos[k.obj_body_id[name]], dtype=float)


def robot_base(k):
    bid = k.sim.model.body_name2id("mobilebase0_base")
    return np.array(k.sim.data.body_xpos[bid]), np.array(k.sim.data.body_xmat[bid]).reshape(3, 3)


def to_base(k, p):
    """World point -> robot-base frame (x forward, y left, z up)."""
    pos, mat = robot_base(k)
    return mat.T @ (np.asarray(p) - pos)


def project(k, p, width=256, height=256, cam=CAM):
    """Pixel (u, v) of world point p in the upright image of `cam` (u grows to the right as a person sees it)."""
    cid = k.sim.model.camera_name2id(cam)
    cpos = np.array(k.sim.data.cam_xpos[cid])
    cmat = np.array(k.sim.data.cam_xmat[cid]).reshape(3, 3)
    pc = cmat.T @ (np.asarray(p) - cpos)  # MuJoCo camera: x right, y up, looks along -z
    if pc[2] >= 0:
        return None
    fy = (height / 2) / np.tan(np.deg2rad(k.sim.model.cam_fovy[cid]) / 2)
    return float(width / 2 + fy * pc[0] / -pc[2]), float(height / 2 - fy * pc[1] / -pc[2])


_RENDERERS = {}


def object_mask(k, name, cam=CAM, width=256, height=256):
    """Boolean (H, W) mask of the pixels where object `name`'s geoms are the visible surface in `cam`.

    Uses MuJoCo segmentation rendering, so occlusion by the arm or a door and leaving the frame are both handled.
    Rows run top-down (upright image, as `project` assumes).
    """
    import mujoco
    m, d = k.sim.model._model, k.sim.data._data
    # RoboCasa builds a new MjModel at every reset; keep a reference to the model the renderer was made for
    # (an id() key can be reused by the next model once the old one is freed)
    if _RENDERERS.get("model") is not m or _RENDERERS.get("size") != (width, height):
        if "renderer" in _RENDERERS:
            _RENDERERS["renderer"].close()
        r = mujoco.Renderer(m, height=height, width=width)
        r.enable_segmentation_rendering()
        _RENDERERS.update(model=m, size=(width, height), renderer=r)
    r = _RENDERERS["renderer"]
    r.update_scene(d, camera=cam)
    # Renderer.render() in segmentation mode can raise IndexError on RoboCasa kitchens: some pixels decode to an id
    # past the scene's geom list (seed 409 of PnPCounterToCabPair). Same steps as render(), but ids that match no
    # scene geom count as background.
    scn = r._scene
    flags = scn.flags.copy()
    scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = True
    scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = True
    r._gl_context.make_current()
    mujoco.mjr_render(r._rect, scn, r._mjr_context)
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(rgb, None, r._rect, r._mjr_context)
    np.copyto(scn.flags, flags)
    rgb = np.flipud(rgb).astype(np.int64)
    segid = rgb[..., 0] + rgb[..., 1] * 256 + rgb[..., 2] * 65536 - 1  # -1 = background
    root = k.obj_body_id[name]
    wanted = set(np.where(m.body_rootid[m.geom_bodyid] == root)[0].tolist())
    hit = [g.segid for g in scn.geoms[:scn.ngeom]
           if g.segid != -1 and g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid in wanted]
    return np.isin(segid, hit)


def visibility(k, name, cam=CAM, width=256, height=256, min_px=15):
    """(visible, n_pixels, centroid_uv) of object `name` in `cam`; visible = at least `min_px` pixels."""
    mask = object_mask(k, name, cam, width, height)
    n = int(mask.sum())
    if n == 0:
        return False, 0, None
    vs, us = np.nonzero(mask)
    return n >= min_px, n, (float(us.mean()), float(vs.mean()))


def scene_hash(k):
    items = []
    for cfg in k.object_cfgs:
        n = cfg["name"]
        if n in k.obj_body_id:
            items.append((n, cfg.get("info", {}).get("cat"), tuple(np.round(body_pos(k, n), 3))))
    meta = k.get_ep_meta()
    base_pos, _ = robot_base(k)
    blob = json.dumps(dict(objs=sorted(items, key=str), refs=meta.get("fixture_refs"), layout=meta.get("layout_id"),
                           style=meta.get("style_id"), base=list(np.round(base_pos, 3))), sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def cat(k, name):
    for cfg in k.object_cfgs:
        if cfg["name"] == name:
            return cfg.get("info", {}).get("cat")
    return None


def task_facts(task, k):
    f = {}
    if task.startswith("PickPlace"):
        names = [c["name"] for c in k.object_cfgs if c["name"] in k.obj_body_id]
        f["objects"] = {n: cat(k, n) for n in names}
        if "obj" in names and "distr_counter" in names:
            t, d = to_base(k, body_pos(k, "obj")), to_base(k, body_pos(k, "distr_counter"))
            f["target_base_xy"], f["distr_base_xy"] = list(np.round(t[:2], 3)), list(np.round(d[:2], 3))
            f["target_nearer_robot"] = bool(np.linalg.norm(t[:2]) < np.linalg.norm(d[:2]))
            f["same_category"] = cat(k, "obj") == cat(k, "distr_counter")
            from robocasa.models.objects.kitchen_objects import OBJ_CATEGORIES
            # OBJ_CATEGORIES[cat] maps registry type (objaverse / aigen / ...) -> ObjCat
            f["distr_graspable"] = any(getattr(o, "graspable", False)
                                       for o in OBJ_CATEGORIES.get(cat(k, "distr_counter"), {}).values())
            pt, pd = project(k, body_pos(k, "obj")), project(k, body_pos(k, "distr_counter"))
            f["target_px"], f["distr_px"] = pt, pd
    elif task == "NavigateKitchen":
        src, tgt = k.src_fixture, k.target_fixture
        f["src"], f["target"], f["target_lang"] = src.name, tgt.name, tgt.nat_lang
        base_pos, _ = robot_base(k)
        f["dist_to_target_m"] = float(np.linalg.norm(np.array(k.target_pos[:2]) - base_pos[:2]))
        tb = to_base(k, np.array(list(k.target_pos[:2]) + [base_pos[2]]))
        f["target_bearing_deg"] = float(np.degrees(np.arctan2(tb[1], tb[0])))  # 0 = straight ahead, + = left
        f["n_fixtures_by_lang"] = len({getattr(x, "nat_lang", None) for x in k.fixtures.values()})
    elif task == "TurnOffStove":
        st = k.stove
        f["knob"], f["cookware_burner"] = k.knob, k.cookware_burner
        try:
            f["knob_states"] = {kk: float(v) for kk, v in st.get_knobs_state(env=k).items()}
            f["lit_knobs"] = [kk for kk in f["knob_states"] if st.is_burner_on(env=k, burner_loc=kk)]
        except Exception as e:  # API differs across versions; keep the row
            f["knob_states_error"] = repr(e)
        f["valid_knobs"] = [kk for kk, v in st.knob_joints.items() if v is not None]
    return f


def first_frame(obs):
    for key in ("video.robot0_agentview_left", "robot0_agentview_left_image"):
        if key in obs:
            return np.asarray(obs[key])
    return None


def probe_task(gym, task, seeds, n, out):
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    env = gym.make(f"robocasa/{task}", split="pretrain", seed=7)
    make_s = time.time() - t0
    rows, frames = [], []
    for s in seeds[:n]:
        t0 = time.time()
        obs, _ = env.reset(seed=s)
        reset_s = time.time() - t0
        k = kitchen_of(env)
        h1, lang = scene_hash(k), str(obs.get("annotation.human.task_description", ""))
        facts = task_facts(task, k)
        frame = first_frame(obs)
        obs, _ = env.reset(seed=s)  # check A: same env, same seed again
        h2 = scene_hash(kitchen_of(env))
        rows.append(dict(task=task, seed=s, hash=h1, hash_repeat=h2, lang=lang, reset_s=round(reset_s, 2), **facts))
        if frame is not None:
            frames.append((s, frame, facts.get("target_px"), facts.get("distr_px")))
    # step timing on the last scene: zero action through the raw kitchen env (cameras rendered as configured)
    k = kitchen_of(env)
    act = np.zeros(k.action_spec[0].shape)
    t0 = time.time()
    for _ in range(50):
        k.step(act)
    step_s = (time.time() - t0) / 50
    env.close()
    # check B: a fresh env, first seed directly
    env2 = gym.make(f"robocasa/{task}", split="pretrain", seed=7)
    env2.reset(seed=seeds[0])
    hB = scene_hash(kitchen_of(env2))
    env2.close()
    with open(os.path.join(out, f"{task}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    save_sheet(frames, os.path.join(out, f"{task}_first_frames.png"))
    return dict(task=task, n=len(rows), make_s=round(make_s, 1),
                reset_s_mean=round(float(np.mean([r["reset_s"] for r in rows])), 2),
                step_s=round(step_s, 4),
                repeat_hash_ok=sum(r["hash"] == r["hash_repeat"] for r in rows),
                fresh_env_hash_ok=hB == rows[0]["hash"],
                distinct_hashes=len({r["hash"] for r in rows}))


def save_sheet(frames, path, cols=10):
    if not frames:
        return
    from PIL import Image, ImageDraw
    h, w = frames[0][1].shape[:2]
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h))
    for i, (s, img, pt, pd) in enumerate(frames):
        im = Image.fromarray(np.asarray(img).astype(np.uint8))
        dr = ImageDraw.Draw(im)
        sx, sy = w / 256, h / 256
        for p, col in ((pt, (0, 255, 0)), (pd, (255, 0, 0))):  # green = target, red = distractor
            if p:
                dr.ellipse([p[0] * sx - 4, p[1] * sy - 4, p[0] * sx + 4, p[1] * sy + 4], outline=col, width=2)
        dr.text((2, 2), str(s), fill=(255, 255, 0))
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--tasks", default="")
    a = ap.parse_args()
    import gymnasium as gym
    import robocasa  # noqa: F401  (registers robocasa/<Task> gym ids)
    import robocasa.wrappers.gym_wrapper  # noqa: F401
    spec = json.load(open(a.seeds))
    tasks = a.tasks.split(",") if a.tasks else list(spec["tasks"])
    summary = []
    for t in tasks:
        seeds = [e["seed"] for e in spec["tasks"][t]["episodes"]]
        r = probe_task(gym, t, seeds, a.n, a.out)
        print(json.dumps(r))
        summary.append(r)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
