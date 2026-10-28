"""LangMustardInRightBinTask with RoboLab's official 10 cm initial-pose randomization (gap G1).

Without it every RoboLab episode starts from the same object poses. The event is the one RoboLab uses in
robolab/tasks/randomize_initial_pose/*_uniform_10cm.py (uniform x, y in ±10 cm, collision check, all other
assets reset to default). Instructions, scene, terminations and subtasks are those of LangMustardInRightBinTask.
"""

import importlib.util
import os
from dataclasses import dataclass

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from robolab.core.events.reset_pose import reset_pose_uniform

_spec = importlib.util.spec_from_file_location(
    "_lang_LangMustardInRightBinTask", os.path.join(os.path.dirname(os.path.abspath(__file__)), "LangMustardInRightBinTask.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


@configclass
class _RandomizeInitPose10cm:
    randomize_init_pose = EventTerm(
        func=reset_pose_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": ['mustard'],
            "use_collision_check": True,
            "reset_to_default_otherwise": True,
        },
    )


@dataclass
class LangMustardInRightBinRand10Task(_base.LangMustardInRightBinTask):
    events = _RandomizeInitPose10cm
