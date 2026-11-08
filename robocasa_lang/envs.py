"""Mirror-pair task classes for the RoboCasa365 language stress test (docs/robocasa/plan.md §3.2).

A mirror pair is one scene with two candidate targets; `target_slot` ("a" or "b") picks which one the instruction
names and the success check scores. Rules shared by all three classes:

  - For a given reset seed, slots a and b build the *same* scene. Everything that differs between the slots happens
    after the scene is built (instruction text, success check).
  - Extra random choices (second burner, alternative destination) use an auxiliary RNG derived from the state of
    env.rng at a fixed point; env.rng itself is not consumed. So in NavigateKitchenPair and TurnOffStovePair,
    slot a's scene equals the parent task's scene for the same seed (TurnOffStovePair adds one lit burner).
  - The parent's own attributes (`knob`, `target_fixture`, `target_pos`) are left as the parent set them; the pair
    state lives in `pair_*` attributes.
  - Success is stricter than the parent's: the named candidate must reach the goal state AND the other must not
    (otherwise "do both" would score on both slots). `pair_outcome()` reports both candidates separately.

Never edits robocasa itself (canonical tasks stay untouched, as with LIBERO's BDDL files). Importing this module
registers the classes with robosuite (EnvMeta) and creates gym ids `robocasa/<ClassName>`; pass
`target_slot="a"|"b"` through `gym.make(...)`.
"""
import hashlib
import json

import numpy as np
import robocasa.utils.env_utils as EnvUtils
import robocasa.utils.object_utils as OU
from robocasa.environments.kitchen.atomic.kitchen_navigate import NavigateKitchen
from robocasa.environments.kitchen.atomic.kitchen_pick_place import PickPlaceCounterToCabinet
from robocasa.environments.kitchen.atomic.kitchen_stove import TurnOffStove
from robocasa.environments.kitchen.kitchen import Kitchen

SLOTS = ("a", "b")


def aux_rng(env):
    """RNG seeded from env.rng's current state; reading the state does not advance env.rng."""
    state = json.dumps(env.rng.bit_generator.state, sort_keys=True, default=str)
    return np.random.default_rng(int(hashlib.sha1(state.encode()).hexdigest()[:15], 16))


def _check_slot(slot):
    if slot not in SLOTS:
        raise ValueError(f"target_slot must be one of {SLOTS}, got {slot!r}")
    return slot


class PnPCounterToCabPair(PickPlaceCounterToCabinet):
    """Two graspable objects with the same counter placement (the parent's target placement), plus the parent's
    in-cabinet distractor. The instruction names `obj_a` or `obj_b`."""

    def __init__(self, target_slot="a", *args, **kwargs):
        self.target_slot = _check_slot(target_slot)
        super().__init__(*args, **kwargs)

    def _get_obj_cfgs(self):
        placement = dict(fixture=self.counter, sample_region_kwargs=dict(ref=self.cab), size=(0.60, 0.30),
                         pos=("ref", -1.0), offset=(0.0, 0.10))
        cfgs = [dict(name=f"obj_{s}", obj_groups=self.obj_groups, exclude_obj_groups=self.exclude_obj_groups,
                     graspable=True, placement=dict(placement)) for s in SLOTS]
        cfgs.append(dict(name="distr_cab", obj_groups="all",
                         placement=dict(fixture=self.cab, size=(1.0, 0.20), pos=(None, 1.0), offset=(0.0, 0.0))))
        return cfgs

    @property
    def pair_target(self):
        return f"obj_{self.target_slot}"

    @property
    def pair_other(self):
        return f"obj_{'b' if self.target_slot == 'a' else 'a'}"

    def get_ep_meta(self):
        ep_meta = Kitchen.get_ep_meta(self)  # skip the parent, which reads an object named "obj"
        ep_meta["lang"] = (f"Pick the {self.get_obj_lang(self.pair_target)} from the counter "
                           f"and place it in the cabinet.")
        ep_meta["pair"] = dict(slot=self.target_slot, target=self.pair_target, other=self.pair_other,
                               langs={s: self.get_obj_lang(f"obj_{s}") for s in SLOTS})
        return ep_meta

    def _delivered(self, name):
        return bool(OU.obj_inside_of(self, name, self.cab) and OU.gripper_obj_far(self, obj_name=name))

    def pair_outcome(self):
        return {name: self._delivered(name) for name in (f"obj_{s}" for s in SLOTS)}

    def pair_valid(self):
        """False when the two candidates would be named by the same words (the pair cannot be told apart)."""
        return self.get_obj_lang("obj_a") != self.get_obj_lang("obj_b")

    def _check_success(self):
        out = self.pair_outcome()
        return out[self.pair_target] and not out[self.pair_other]


class PnPCounterToCabinetSwap(PickPlaceCounterToCabinet):
    """The parent's original scene and instruction slots, with success retargeted to the counter distractor
    (`distr_counter`): main-experiment `target_swap` for PickPlaceCounterToCabinet (plan §4.0). The instruction is
    supplied by the client (conditions.py `target_swap`); get_ep_meta keeps the parent's. Valid only on seeds whose
    distractor is a graspable category with words different from the target's (`pair_valid`)."""

    def _delivered(self, name):
        return bool(OU.obj_inside_of(self, name, self.cab) and OU.gripper_obj_far(self, obj_name=name))

    def pair_outcome(self):
        return {name: self._delivered(name) for name in ("obj", "distr_counter")}

    def pair_valid(self):
        from robocasa.models.objects.kitchen_objects import OBJ_CATEGORIES
        cat = next(c["info"]["cat"] for c in self.object_cfgs if c["name"] == "distr_counter")
        graspable = any(getattr(o, "graspable", False) for o in OBJ_CATEGORIES.get(cat, {}).values())
        return graspable and self.get_obj_lang("obj") != self.get_obj_lang("distr_counter")

    def _check_success(self):
        out = self.pair_outcome()
        return out["distr_counter"] and not out["obj"]


class TurnOffStovePair(TurnOffStove):
    """The parent's knob plus a second knob, both lit. The instruction names `knob_a` (the parent's) or `knob_b`."""

    def __init__(self, target_slot="a", *args, **kwargs):
        self.target_slot = _check_slot(target_slot)
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        rng = aux_rng(self)
        valid = sorted(k for k, v in self.stove.knob_joints.items() if v is not None and k != self.knob)
        self.pair_knobs = {"a": self.knob, "b": str(rng.choice(valid))}
        self._pair_seed = int(rng.integers(2**31))

    def _setup_scene(self):
        super()._setup_scene()  # lights knob_a (= self.knob) with env.rng, as the parent does
        self.stove.set_knob_state(mode="on", knob=self.pair_knobs["b"], env=self,
                                  rng=np.random.default_rng(self._pair_seed))

    @property
    def pair_target(self):
        return self.pair_knobs[self.target_slot]

    @property
    def pair_other(self):
        return self.pair_knobs["b" if self.target_slot == "a" else "a"]

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"Turn off the {self.pair_target.replace('_', ' ')} burner of the stove."
        ep_meta["pair"] = dict(slot=self.target_slot, target=self.pair_target, other=self.pair_other,
                               knobs=dict(self.pair_knobs))
        return ep_meta

    def pair_outcome(self):
        """True = burner is off (the goal state for this task)."""
        return {k: not self.stove.is_burner_on(env=self, burner_loc=k) for k in self.pair_knobs.values()}

    def pair_valid(self):
        return self.pair_knobs["a"] != self.pair_knobs["b"]

    def _check_success(self):
        out = self.pair_outcome()
        return out[self.pair_target] and not out[self.pair_other]


# NavigateKitchen._setup_kitchen_references's destination classes (robocasa 4f8a298); a class counts only if the
# kitchen has exactly one fixture of it, as in the parent.
NAV_TARGET_CLASSES = {
    "CoffeeMachine", "Toaster", "ToasterOven", "Stove", "Stovetop", "OpenCabinet", "Microwave", "Sink", "Hood",
    "Oven", "FridgeFrenchDoor", "FridgeSideBySide", "FridgeBottomFreezer", "Dishwasher", "ElectricKettle",
    "StandMixer",
}


class NavigateKitchenPair(NavigateKitchen):
    """The parent's start and destination, plus an alternative destination drawn with the parent's own filter
    (unique fixture class, not the start, more than 1 m from the start) and also more than 1 m from the parent's
    destination, with a different `nat_lang`. Slot a = the parent's destination."""

    def __init__(self, target_slot="a", *args, **kwargs):
        self.target_slot = _check_slot(target_slot)
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        fixtures = list(self.fixtures.values())
        classes = [type(f).__name__ for f in fixtures]
        cands = [f for f in fixtures
                 if type(f).__name__ in NAV_TARGET_CLASSES and classes.count(type(f).__name__) == 1
                 and f is not self.src_fixture and f is not self.target_fixture
                 and f.nat_lang != self.target_fixture.nat_lang
                 and OU.fixture_pairwise_dist(self.src_fixture, f) > 1.0
                 and OU.fixture_pairwise_dist(self.target_fixture, f) > 1.0]
        cands.sort(key=lambda f: f.name)
        self.pair_fixtures = {"a": self.target_fixture}
        self.pair_poses = {"a": (self.target_pos, self.target_ori)}  # the parent's own destination pose
        if cands:
            self.pair_fixtures["b"] = cands[int(aux_rng(self).integers(len(cands)))]
            # compute_robot_base_placement_pose can draw from env.rng; restore its state so that the rest of
            # the scene (robot spawn, objects) is the parent's (checked: seed 208 moved the base by 9 cm before)
            state = self.rng.bit_generator.state
            self.pair_poses["b"] = EnvUtils.compute_robot_base_placement_pose(self, self.pair_fixtures["b"])
            self.rng.bit_generator.state = state

    @property
    def pair_target(self):
        return self.target_slot

    @property
    def pair_other(self):
        return "b" if self.target_slot == "a" else "a"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        target = self.pair_fixtures.get(self.target_slot)
        ep_meta["lang"] = f"Navigate to the {target.nat_lang}." if target is not None else ""
        ep_meta["pair"] = dict(slot=self.target_slot,
                               fixtures={s: (f.name, f.nat_lang) for s, f in self.pair_fixtures.items()})
        return ep_meta

    def _arrived(self, slot):
        # the parent's _check_success, with the destination pose of `slot`
        pos, ori = self.pair_poses[slot]
        robot_id = self.sim.model.body_name2id("mobilebase0_base")
        base_pos = np.array(self.sim.data.body_xpos[robot_id])
        base_ori = EnvUtils.T.mat2euler(np.array(self.sim.data.body_xmat[robot_id]).reshape((3, 3)))
        return bool(np.linalg.norm(pos[:2] - base_pos[:2]) <= 0.20 and np.cos(ori[2] - base_ori[2]) >= 0.98)

    def pair_outcome(self):
        return {s: self._arrived(s) for s in self.pair_poses}

    def pair_valid(self):
        return "b" in self.pair_fixtures

    def _check_success(self):
        if not self.pair_valid():
            return False
        out = self.pair_outcome()
        return out[self.pair_target] and not out[self.pair_other]


PAIR_CLASSES = (PnPCounterToCabPair, PnPCounterToCabinetSwap, TurnOffStovePair, NavigateKitchenPair)


def register_gym_ids():
    from robocasa.wrappers.gym_wrapper import create_robocasa_gym_env_class
    import gymnasium as gym
    for cls in PAIR_CLASSES:
        if f"robocasa/{cls.__name__}" not in gym.registry:
            create_robocasa_gym_env_class(cls.__name__)


register_gym_ids()
