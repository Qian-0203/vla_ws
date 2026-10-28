"""BowlStackingLeftOnRightTask with the language-stress instruction keys (robolab_lang, plan §4.2).

Scene, terminations and subtasks are the official ones, unchanged; the official default / vague /
specific strings are copied from the base class. Select a key with --instruction-type <key>.
"""

from dataclasses import dataclass

import robolab.tasks.benchmark.bowl_stacking_left_on_right as _official


@dataclass
class LangBowlStackingLeftOnRightTask(_official.BowlStackingLeftOnRightTask):
    instruction = {
        **_official.BowlStackingLeftOnRightTask.instruction,
        "blank": ".",
        "no_target": "Stack one bowl on the other bowl",
        "p1_no_location": "Stack one bowl on the other bowl",
        "p3_neg_not": "Stack the bowl that is not on the right on the other bowl",
        "p5_contrast": "Stack the left bowl, not the right one, on the right bowl",
        "p5_len_ctrl": "Stack the left bowl, slowly and very carefully, on the right bowl",
    }
