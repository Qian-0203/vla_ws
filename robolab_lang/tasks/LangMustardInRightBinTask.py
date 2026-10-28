"""MustardInRightBinTask with the language-stress instruction keys (robolab_lang, plan §4.2).

Scene, terminations and subtasks are the official ones, unchanged; the official default / vague /
specific strings are copied from the base class. Select a key with --instruction-type <key>.
"""

from dataclasses import dataclass

import robolab.tasks.benchmark.mustard_in_right_bin as _official


@dataclass
class LangMustardInRightBinTask(_official.MustardInRightBinTask):
    instruction = {
        **_official.MustardInRightBinTask.instruction,
        "blank": ".",
        "no_target": "Put the mustard in the bin",
        "p1_no_location": "Put the mustard in the bin",
        "p3_neg_not": "Put the mustard in the bin that is not on the left",
        "p5_contrast": "Put the mustard in the right bin, not the left one",
        "p5_len_ctrl": "Put the mustard in the right bin, and let go gently",
        "p4_syn_common": "Put the mustard in the right container",
        "p4_syn_rare": "Place the mustard in the right receptacle",
        "p4_functional": "Put the mustard in the right thing you throw trash into",
        "p4_zh": "把芥末醬放進右邊的桶子裡",
        "p7_robot_frame": "Put the mustard in the bin on the robot's right",
        "p7_image_frame": "Put the mustard in the bin on the right side of the image",
    }
