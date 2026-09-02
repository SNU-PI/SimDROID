"""Phase A of the PhysicsGen redesign (2026-09-02): realistic workbench wrappers.

Same principle and saturation axis S as Bundle A, same conditioning contract
(five strictly pre-event 16 FPS frames, 21-frame horizon).  New: a P0 control
family with no decision variable (uniform rolling), and the P1/P2 scenes re-dressed
as a robotics workbench so the appearance domain is closer to what video world
models were trained on.  Physics, contact models, cameras and S grid are
unchanged, so Bundle A (toy) vs Phase A (workbench) is a pure appearance contrast.
"""

from __future__ import annotations

from core.threshold.rolling_hill import RollingHillWB
from core.threshold.two_ball import TwoBallWB, WallBounceWB
from core.threshold.kin_roll import KinRoll
from gen.bundle_a_spec import (BOUNDARY_TOL, CONDITION_INDICES, FUTURE_OFFSET,
                               ROLLOUT_FRAMES, S_GRID, in_map)

PRINCIPLE = "phase_a_workbench_direction_selection"

PHASE_A = {"hill_roll_wb": RollingHillWB, "two_ball_wb": TwoBallWB}
CONTROLS = {"kin_roll": (KinRoll, KinRoll.V_GRID)}
PRECHECKS = {"hill_roll_wb_pre": (RollingHillWB, dict(S=0.35)),
             "wall_bounce_wb_pre": (WallBounceWB, None)}

_CTX = ("A fixed camera on a robotics workbench observes a red ball rolling from "
        "left to right across a grey inspection table. A white robot arm stands "
        "idle in the background. ")
PROMPTS = {
    "kin_roll": _CTX + "The ball keeps its shape and moves under rolling contact and momentum.",
    "hill_roll_wb": (_CTX.replace("across a grey inspection table", "across a grey inspection table toward a low green rubber cable-cover ramp")
                     + "The ball keeps its shape and moves under gravity, rolling contact, and momentum."),
    "two_ball_wb": (_CTX.replace("across a grey inspection table", "across a grey low-friction inspection table toward a resting blue ball of the same material")
                    + "The balls collide head-on and continue with realistic contact physics, keeping their shapes and sizes."),
    "wall_bounce_wb_pre": (_CTX.replace("across a grey inspection table", "across a grey low-friction inspection table toward a rigid steel stopper block")
                           + "The ball bounces off the block and continues with realistic contact physics, keeping its shape."),
}
PROMPTS["hill_roll_wb_pre"] = PROMPTS["hill_roll_wb"]

ORACLE = {
    "hill_roll_wb": {
        1: "The red ball has enough energy: it rolls over the green ramp's crest and continues to the right.",
        0: "The red ball lacks energy: it rolls partway up the green ramp, slows, and returns to the left.",
    },
    "two_ball_wb": {
        1: "The blue ball is larger and heavier: after impact the red ball bounces back to the left while the blue ball moves right.",
        0: "The blue ball is smaller and lighter: after impact the red ball keeps moving right behind the blue ball.",
    },
}


def phase_specs():
    return {name: [cls.params_for_S(float(S)) for S in S_GRID]
            for name, cls in PHASE_A.items()}


__all__ = ["PRINCIPLE", "S_GRID", "BOUNDARY_TOL", "CONDITION_INDICES", "ROLLOUT_FRAMES",
           "FUTURE_OFFSET", "PHASE_A", "CONTROLS", "PRECHECKS", "PROMPTS", "ORACLE",
           "phase_specs", "in_map"]
