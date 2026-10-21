#!/usr/bin/env python3
"""Pre-render check of spatial wording against the policy camera (P7 / LIBERO erratum E1).

RoboLab's spatial predicates (`left_of`, `right_of`, `in_front_of`, `behind`) are defined in the
robot-root frame: left = +y, right = -y, behind = +x (away from the robot), in front = -x.
The DROID policies see the scene through `over_shoulder_left_camera`, which sits at the robot's
left and looks right-and-down, so "left of" in the robot frame need not be "left" in the image.

This script projects scene objects into that camera with a pinhole model (no Isaac Sim needed),
then for every object pair prints the robot-frame relation and the image-frame relation and flags
pairs where a word would mean different things in the two frames. Use it on every scene before
writing a spatial instruction; confirm the result on one real render before a GPU run.

Usage:
    python3 robolab_lang/frame_check.py <scene.usda> [<scene.usda> ...] [--objects a,b,c]
    python3 robolab_lang/frame_check.py --point bin_L=0.52,0.29,0.0 --point bin_R=0.52,-0.29,0.0

Object positions are read from top-level prims that carry a `payload`/`references` arc and an
`xformOp:translate` in the USDA text. This is a text parse, not a USD composition: nested or
rotated parents are not handled, so check the printed table against the scene image.
"""

from __future__ import annotations

import argparse
import re
import sys

import numpy as np

# RoboLab robolab/variations/camera.py, OverShoulderLeftCameraCfg (DROID exterior_image_1_left).
# Quaternion is (w, x, y, z); convention "opengl" (camera looks along -z, +y up, +x right).
CAMERAS = {
    "over_shoulder_left": dict(
        pos=(0.05, 0.57, 0.66), rot=(-0.393, -0.195, 0.399, 0.805),
        focal=2.1, h_ap=5.376, v_ap=3.024, width=1280, height=720,
    ),
}
# The pi0-family client sends this camera as resize_with_pad(img, 224, 224).
POLICY_SIZE = 224
# Image-frame relations closer than this many policy pixels are reported as "~" (ambiguous).
MIN_PX = 8

FIXTURES = {"table", "franka_table", "GroundPlane", "visuals"}


def quat_to_rot(q):
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def project(cam: dict, p) -> tuple[float, float] | None:
    """Return (u, v) in the camera's native pixels, or None if behind the camera."""
    pc = quat_to_rot(cam["rot"]).T @ (np.asarray(p, dtype=float) - np.asarray(cam["pos"]))
    if pc[2] >= 0:
        return None
    u = cam["focal"] * pc[0] / -pc[2] / cam["h_ap"] * cam["width"] + cam["width"] / 2
    v = cam["height"] / 2 - cam["focal"] * pc[1] / -pc[2] / cam["v_ap"] * cam["height"]
    return u, v


def to_policy_px(cam: dict, uv):
    """Map native pixels to the 224x224 letterboxed image the policy receives."""
    s = POLICY_SIZE / max(cam["width"], cam["height"])
    pad_y = (POLICY_SIZE - cam["height"] * s) / 2
    pad_x = (POLICY_SIZE - cam["width"] * s) / 2
    return uv[0] * s + pad_x, uv[1] * s + pad_y


def parse_usda(path: str) -> dict[str, tuple[float, float, float]]:
    text = open(path).read()
    out = {}
    # Top-level children of the root Xform are indented by 4 spaces.
    for m in re.finditer(r'\n    def (?:Xform )?"(\w+)"\s*\((.*?)\n    \)\s*\{(.*?)\n    \}', text, re.S):
        name, meta, body = m.groups()
        if name in FIXTURES or not re.search(r"(payload|references)\s*=\s*@", meta + body):
            continue
        t = re.search(r"xformOp:translate\s*=\s*\(([^)]*)\)", body)
        if t:
            out[name] = tuple(float(v) for v in t.group(1).split(","))
    return out


def robot_relation(a, b, cone_deg=45.0):
    """Relation of a with respect to b, using RoboLab's vector + cone rule (robot frame)."""
    d = np.array([a[0] - b[0], a[1] - b[1]])
    n = np.linalg.norm(d)
    if n < 1e-6:
        return "-", "-"
    c = np.cos(np.deg2rad(cone_deg))
    lr = "left_of" if d[1] / n >= c else "right_of" if -d[1] / n >= c else "-"
    fb = "behind" if d[0] / n >= c else "in_front_of" if -d[0] / n >= c else "-"
    return lr, fb


def image_relation(ua, ub):
    du, dv = ua[0] - ub[0], ua[1] - ub[1]
    lr = "~" if abs(du) < MIN_PX else ("img_left" if du < 0 else "img_right")
    ud = "~" if abs(dv) < MIN_PX else ("img_above" if dv < 0 else "img_below")
    return lr, ud


def report(objects: dict, cam_name: str) -> int:
    cam = CAMERAS[cam_name]
    print(f"camera: {cam_name}; policy image {POLICY_SIZE}x{POLICY_SIZE} (letterboxed)")
    print(f"{'object':28s} {'x':>7s} {'y':>7s}   {'u224':>6s} {'v224':>6s}")
    px = {}
    for name, p in objects.items():
        uv = project(cam, p)
        if uv is None:
            print(f"{name:28s} behind camera")
            continue
        px[name] = to_policy_px(cam, uv)
        print(f"{name:28s} {p[0]:7.3f} {p[1]:7.3f}   {px[name][0]:6.1f} {px[name][1]:6.1f}")
    print()
    conflicts = 0
    names = [n for n in objects if n in px]
    print(f"{'A rel B':45s} {'robot frame':24s} {'image frame':24s} flag")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rlr, rfb = robot_relation(objects[a], objects[b])
            ilr, iud = image_relation(px[a], px[b])
            flags = []
            if rlr == "left_of" and ilr != "img_left":
                flags.append("robot-left is not image-left")
            if rlr == "right_of" and ilr != "img_right":
                flags.append("robot-right is not image-right")
            if rlr == "-" and ilr != "~":
                flags.append(f"image says {ilr}, robot frame has no left/right")
            conflicts += bool(flags)
            print(f"{a + ' rel ' + b:45s} {rlr + ',' + rfb:24s} {ilr + ',' + iud:24s} {'; '.join(flags)}")
    print(f"\n{conflicts} pair(s) where left/right wording differs between robot and image frame")
    return conflicts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("--objects", help="comma-separated subset of prim names to keep")
    ap.add_argument("--point", action="append", default=[], help="name=x,y,z (robot-root frame, metres)")
    ap.add_argument("--camera", default="over_shoulder_left", choices=sorted(CAMERAS))
    args = ap.parse_args(argv)
    if not args.scenes and not args.point:
        ap.error("give a scene file or at least one --point")
    for scene in args.scenes:
        objs = parse_usda(scene)
        if args.objects:
            keep = args.objects.split(",")
            objs = {k: v for k, v in objs.items() if k in keep}
        print(f"=== {scene}")
        report(objs, args.camera)
        print()
    if args.point:
        objs = {}
        for spec in args.point:
            name, xyz = spec.split("=")
            objs[name] = tuple(float(v) for v in xyz.split(","))
        print("=== --point")
        report(objs, args.camera)
    return 0


if __name__ == "__main__":
    sys.exit(main())
