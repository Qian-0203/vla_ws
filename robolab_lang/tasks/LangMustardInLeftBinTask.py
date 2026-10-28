"""MustardInLeftBinTask with the language-stress instruction keys (robolab_lang, plan §4.2).

Scene, terminations and subtasks are the official ones, unchanged; the official default / vague /
specific strings are copied from the base class. Select a key with --instruction-type <key>.
"""

from dataclasses import dataclass

import robolab.tasks.benchmark.mustard_in_left_bin as _official


@dataclass
class LangMustardInLeftBinTask(_official.MustardInLeftBinTask):
    instruction = {
        **_official.MustardInLeftBinTask.instruction,
        "blank": ".",
        "no_target": "Put the mustard in the bin",
        "p1_no_location": "Put the mustard in the bin",
        "p3_neg_not": "Put the mustard in the bin that is not on the right",
        "p5_contrast": "Put the mustard in the left bin, not the right one",
        "p5_len_ctrl": "Put the mustard in the left bin, and let go gently",
        "p4_syn_common": "Put the mustard in the left container",
        "p4_syn_rare": "Place the mustard in the left receptacle",
        "p4_functional": "Put the mustard in the left thing you throw trash into",
        "p4_zh": "把芥末醬放進左邊的桶子裡",
        "p7_robot_frame": "Put the mustard in the bin on the robot's left",
        "p7_image_frame": "Put the mustard in the bin on the left side of the image",
    }
