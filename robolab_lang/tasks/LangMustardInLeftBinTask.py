"""MustardInLeftBinTask with the language-stress instruction keys (robolab_lang; condition names follow the ARR draft's tests).

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
        "negation": "Put the mustard in the bin that is not on the right",
        "distractor_mention": "Put the mustard in the left bin, not the right one",
        "irrelevant_content": "Put the mustard in the left bin, and let go gently",
        "paraphrase_common": "Put the mustard in the left container",
        "paraphrase_rare": "Place the mustard in the left receptacle",
        "paraphrase_functional": "Put the mustard in the left thing you throw trash into",
        "paraphrase_zh": "把芥末醬放進左邊的桶子裡",
        "spatial_robot": "Put the mustard in the bin on the robot's left",
        "spatial_image": "Put the mustard in the bin on the left side of the image",
    }
