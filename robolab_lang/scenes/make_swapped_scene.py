#!/usr/bin/env python3
"""Build butter_raisin_box_grey_bin_swapped.usda from RoboLab's butter_raisin_box_grey_bin.usda (A4, 2026-11-16).

In the official scene the butter sits nearer the robot (x 0.30, y 0.05) and the raisin box farther and to the robot's
left (x 0.40, y 0.25). pi0.5-DROID lifts the butter first whatever the size word says, so "size word", "nearer
object" and "this object" cannot be told apart. The swapped scene exchanges only the two objects' x and y; each keeps
its own height and orientation, so each rests as in the original. Asset and material paths are rewritten to the RoboLab
checkout's path inside the container (/workspace/robolab), because this file lives outside RoboLab's scene folder.
The official scene is not modified.
Usage: python make_swapped_scene.py <RoboLab>/assets/scenes/butter_raisin_box_grey_bin.usda <out.usda>
"""
import re
import sys

src, out = sys.argv[1:]
s = open(src).read()
TR = re.compile(r'(def "(butter|raisin_box)" \(.*?\n        double3 xformOp:translate = \()([^)]*)(\))', re.S)
pos = {m.group(2): [float(v) for v in m.group(3).split(",")] for m in TR.finditer(s)}
assert set(pos) == {"butter", "raisin_box"}, pos
new = {"butter": pos["raisin_box"][:2] + pos["butter"][2:], "raisin_box": pos["butter"][:2] + pos["raisin_box"][2:]}
s = TR.sub(lambda m: m.group(1) + ", ".join(repr(v) for v in new[m.group(2)]) + m.group(4), s)
s = s.replace("@../", "@/workspace/robolab/assets/")  # objects, fixtures and materials
assert "@../" not in s
open(out, "w").write(s)
for k in ("butter", "raisin_box"):
    print(f"{k}: {pos[k]} -> {new[k]}")
