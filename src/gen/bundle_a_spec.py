"""Bundle A experimental design: one principle, one saturation axis S.

All three scenes share the dimensionless saturation S (S = 1 at the boundary)
and the same conditioning contract: five strictly pre-event 16 FPS frames.
The decision variable is statically visible geometry (hill height, ball
radius ratio, rod length); the initial motion inside each scene is fixed.

The map inclusion rule follows the spec: a grid point enters the map only if
MuJoCo decides the outcome within the 21-frame (1.3125 s) rollout horizon;
near-boundary points drop out by critical slowing down and are recorded.
"""

from __future__ import annotations

import numpy as np

from core.threshold.rolling_hill import RollingHill
from core.threshold.two_ball import TwoBall, WallBounce
from core.threshold.rod_pendulum import RodPendulum

PRINCIPLE = "bundle_a_direction_selection"
S_GRID = (0.50, 0.63, 0.79, 0.89, 0.95, 1.00, 1.05, 1.12, 1.26, 1.59, 2.00)
BOUNDARY_TOL = 0.04            # |S-1| <= tol: exempt from the outcome gate
CONDITION_INDICES = (0, 1, 2, 3, 4)
ROLLOUT_FRAMES = 21            # Cosmos horizon: 1.3125 s at 16 FPS
FUTURE_OFFSET = 5              # +0.3125 s reference endpoint (compat display)

BUNDLE_A = {"hill_roll": RollingHill, "two_ball": TwoBall,
            "pendulum_rod": RodPendulum}
PRECHECKS = {"hill_roll_pre": (RollingHill, dict(S=0.35)),
             "wall_bounce_pre": (WallBounce, None)}

PROMPTS = {
    "hill_roll": (
        "A fixed camera observes a red ball rolling from left to right on a flat "
        "dark floor toward a smooth white snow-covered hill. The ball keeps its "
        "shape and moves under gravity, rolling contact, and momentum."
    ),
    "two_ball": (
        "A fixed camera observes a red ball moving right on a flat dark floor "
        "toward a resting blue ball of the same material. The balls collide "
        "head-on and continue with realistic contact physics, keeping their "
        "shapes and sizes."
    ),
    "pendulum_rod": (
        "A fixed camera observes a red pendulum bob on a rigid grey rod swinging "
        "down past its lowest point. The pendulum keeps its shape and arm length "
        "and continues under gravity."
    ),
    "wall_bounce_pre": (
        "A fixed camera observes a red ball moving right on a flat dark floor "
        "toward a rigid dark wall. The ball bounces off the wall and continues "
        "with realistic contact physics, keeping its shape."
    ),
}
PROMPTS["hill_roll_pre"] = PROMPTS["hill_roll"]

# Oracle prompt templates per outcome, for the later prompt-contrast arm.
ORACLE = {
    "hill_roll": {
        1: "The red ball has enough energy: it rolls over the white hill's crest and continues to the right.",
        0: "The red ball lacks energy: it rolls partway up the white hill, slows, and returns to the left.",
    },
    "two_ball": {
        1: "The blue ball is larger and heavier: after impact the red ball bounces back to the left while the blue ball moves right.",
        0: "The blue ball is smaller and lighter: after impact the red ball keeps moving right behind the blue ball.",
    },
    "pendulum_rod": {
        1: "The rod is short enough: the pendulum swings over the top and completes the rotation.",
        0: "The rod is too long: the pendulum rises, stops before the top, and swings back.",
    },
}


def bundle_specs():
    """S grid -> per-scene parameter dicts (exact inversions)."""
    return {name: [cls.params_for_S(float(S)) for S in S_GRID]
            for name, cls in BUNDLE_A.items()}


def in_map(record):
    """Spec exclusion rule: decided within the 21-frame rollout horizon."""
    return bool(record["decided"]) and record["event_frame"] <= ROLLOUT_FRAMES - 1


__all__ = ["PRINCIPLE", "S_GRID", "BOUNDARY_TOL", "CONDITION_INDICES",
           "ROLLOUT_FRAMES", "FUTURE_OFFSET", "BUNDLE_A", "PRECHECKS",
           "PROMPTS", "ORACLE", "bundle_specs", "in_map"]
