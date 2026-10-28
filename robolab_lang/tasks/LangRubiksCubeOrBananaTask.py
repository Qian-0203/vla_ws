"""RubiksCubeOrBananaTask with the language-stress instruction keys (robolab_lang; condition names follow the ARR draft's tests).

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
        "switch_cube": "Put the cube in the bowl",
        "switch_banana": "Put the banana in the bowl",
        "negation_not_banana": "Put the object that is not the banana in the bowl",
        "negation_not_cube": "Put the object that is not the cube in the bowl",
        "distractor_cube": "Put the cube in the bowl, not the banana",
        "distractor_banana": "Put the banana in the bowl, not the cube",
    }
