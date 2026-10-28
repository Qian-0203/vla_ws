"""RubiksCubeOrBananaTask with the language-stress instruction keys (robolab_lang, plan §4.2).

Scene, terminations and subtasks are the official ones, unchanged; the official default / vague /
specific strings are copied from the base class. Select a key with --instruction-type <key>.
`blank` is "." because RoboLab rejects empty instructions; `no_target` drops the words that pick the goal.
"""

from dataclasses import dataclass

import robolab.tasks.benchmark.rubiks_cube_or_banana_task as _official


@dataclass
class LangRubiksCubeOrBananaTask(_official.RubiksCubeOrBananaTask):
    instruction = {
        **_official.RubiksCubeOrBananaTask.instruction,
        "blank": ".",
        "no_target": "Put the object in the bowl",
    }
