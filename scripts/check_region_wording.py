#!/usr/bin/env python3
"""Check the target_cue_region wording dicts against where each target bowl really sits.

Positions are the target bowl's pixel centre in the 256x256 agentview render of init state 0
(LIBERO/scratch_render/libero_spatial/t<k>_init.png; the image the policy sees: top = back/toward the
robot, bottom = front/toward the camera, left/right as seen). They were read off the render by eye
(+-5 px) and are the ground truth for the left/right/front/back words. Table frame: +x = front,
+y = image-right (docs/benchmark_split_result.md, Erratum E1).

A phrase passes when it (a) names the correct side whenever the bowl is clearly left/right of the
table center-line and (b) contains no left/right/front/back word that contradicts the position.
Depth words may be omitted. Usage: python3 scripts/check_region_wording.py   (no GPU, no simulator)
"""
import importlib.util
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "ins", os.path.join(ROOT, "openvla/experiments/robot/libero/instructions.py"))
ins = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ins)

U_CENTER = 131          # image column of the table center-line (y = 0)
SIDE_BAND = 20          # +-px around the center-line counted as "center"
BACK_V, FRONT_V = 150, 190   # image rows: above = back, below = front, between = middle
TARGET_PX = {0: (192, 147), 1: (218, 125), 3: (136, 175), 5: (187, 116),
             6: (102, 201), 7: (92, 108), 8: (227, 165), 9: (20, 111)}
NAMES = {0: "between_the_plate_and_the_ramekin", 1: "next_to_the_ramekin", 3: "on_the_cookie_box",
         5: "on_the_ramekin", 6: "next_to_the_cookie_box", 7: "on_the_stove", 8: "next_to_the_plate",
         9: "on_the_wooden_cabinet"}
DICTS = [
    ("region (v1) OLD", ins.LIBERO_SPATIAL_TARGET_CUE_REGION_INSTRUCTIONS),
    ("region_v2 OLD", ins.LIBERO_SPATIAL_TARGET_CUE_REGION_V2_INSTRUCTIONS),
    ("region_v3 OLD", ins.LIBERO_SPATIAL_TARGET_CUE_REGION_V3_INSTRUCTIONS),
    ("region_img (v1) NEW", ins.LIBERO_SPATIAL_TARGET_CUE_REGION_IMG_INSTRUCTIONS),
    ("region_v2_img NEW", ins.LIBERO_SPATIAL_TARGET_CUE_REGION_V2_IMG_INSTRUCTIONS),
    ("region_v3_img NEW", ins.LIBERO_SPATIAL_TARGET_CUE_REGION_V3_IMG_INSTRUCTIONS),
]


def truth(task):
    u, v = TARGET_PX[task]
    side = "right" if u > U_CENTER + SIDE_BAND else "left" if u < U_CENTER - SIDE_BAND else "center"
    depth = "back" if v < BACK_V else "front" if v > FRONT_V else "mid"
    return side, depth


def judge(task, text):
    side, depth = truth(task)
    # drop the neutral depth phrases that mention both words, and the trailing action clause
    t = re.sub(r"midway between front and back|midway back", "", text.lower().split(" and place it")[0])
    words = set(re.findall(r"\b(left|right|front|back)\b", t))
    problems = []
    if side in ("left", "right"):
        if side not in words:
            problems.append(f"missing '{side}'")
        if ({"left", "right"} - {side}) & words:
            problems.append(f"says {sorted(({'left', 'right'} - {side}) & words)}, bowl is {side}")
    else:
        if {"left", "right"} & words and "just" not in t and "slightly" not in t and "little" not in t:
            problems.append("side word but bowl is near the center-line")
    if depth == "back" and "front" in words:
        problems.append("says front, bowl is at the back")
    if depth == "front" and "back" in words:
        problems.append("says back, bowl is at the front")
    return problems


bad_total = 0
for label, d in DICTS:
    bad = []
    for task in sorted(TARGET_PX):
        text = d[f"pick_up_the_black_bowl_{NAMES[task]}_and_place_it_on_the_plate"]
        p = judge(task, text)
        if p:
            bad.append((task, p))
    new = label.endswith("NEW")
    print(f"{label:22s} {len(TARGET_PX) - len(bad)}/{len(TARGET_PX)} tasks OK"
          + ("" if not bad else "   FAIL: " + ", ".join(f"t{t}" for t, _ in bad)))
    if new:
        bad_total += len(bad)
        for t, p in bad:
            print(f"    task {t}: {p}")
sys.exit(1 if bad_total else 0)
