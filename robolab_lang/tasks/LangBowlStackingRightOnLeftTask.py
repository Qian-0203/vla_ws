"""BowlStackingRightOnLeftTask with the language-stress instruction keys (robolab_lang, plan §4.2).

Scene, terminations and subtasks are the official ones, unchanged; the official default / vague /
specific strings are copied from the base class. Select a key with --instruction-type <key>.
"""

from dataclasses import dataclass

import robolab.tasks.benchmark.bowl_stacking_right_on_left as _official


@dataclass
class LangBowlStackingRightOnLeftTask(_official.BowlStackingRightOnLeftTask):
    instruction = {
        **_official.BowlStackingRightOnLeftTask.instruction,
        "blank": ".",
        "no_target": "Stack one bowl on the other bowl",
        "p1_no_location": "Stack one bowl on the other bowl",
        "p3_neg_not": "Stack the bowl that is not on the left on the other bowl",
        "p5_contrast": "Stack the right bowl, not the left one, on the left bowl",
        "p5_len_ctrl": "Stack the right bowl, slowly and very carefully, on the left bowl",
    }
