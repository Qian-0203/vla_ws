"""MustardInRightBinTask with the language-stress instruction keys (robolab_lang; condition names follow the ARR draft's tests).

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
        "negation": "Put the mustard in the bin that is not on the left",
        "distractor_mention": "Put the mustard in the right bin, not the left one",
        "irrelevant_content": "Put the mustard in the right bin, and let go gently",
        "paraphrase_common": "Put the mustard in the right container",
        "paraphrase_rare": "Place the mustard in the right receptacle",
        "paraphrase_functional": "Put the mustard in the right thing you throw trash into",
        "paraphrase_zh": "把芥末醬放進右邊的桶子裡",
        "spatial_robot": "Put the mustard in the bin on the robot's right",
        "spatial_image": "Put the mustard in the bin on the right side of the image",
    }
