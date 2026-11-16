"""LangLargerObjectRaisinBoxInBinTask on a copy of its scene with the butter and raisin box positions swapped (A4, gap G3).

Scene: /lang/scenes/butter_raisin_box_grey_bin_swapped.usda, built by scenes/make_swapped_scene.py from RoboLab's
butter_raisin_box_grey_bin.usda (only the two objects' x, y exchanged). Instructions, terminations and subtasks are
those of LangLargerObjectRaisinBoxInBinTask.
"""

import importlib.util
import os
from dataclasses import dataclass

from robolab.core.scenes.utils import import_scene

_spec = importlib.util.spec_from_file_location(
    "_lang_LangLargerObjectRaisinBoxInBinTask", os.path.join(os.path.dirname(os.path.abspath(__file__)), "LangLargerObjectRaisinBoxInBinTask.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

_SCENE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scenes",
                      "butter_raisin_box_grey_bin_swapped.usda")


@dataclass
class LangLargerObjectRaisinBoxInBinSwappedTask(_base.LangLargerObjectRaisinBoxInBinTask):
    scene = import_scene(_SCENE, _base.LangLargerObjectRaisinBoxInBinTask.contact_object_list)
