"""Phase B of the PhysicsGen redesign: P1-B suspended payload (pendulum re-skin) and
P3-A support-edge departure, both on the workbench.

Same S axis, conditioning contract (five pre-event 16 FPS frames, 21-frame
horizon) and manifest schema as Phase A, so the Cosmos runner, the adjudicator
and the latent-track generators apply unchanged.  The wall-impact pre-check of
Phase B is the Phase A `wall_bounce_wb_pre` sample (already collected).
"""

from __future__ import annotations

from core.threshold.rod_pendulum import RodPendulumWB
from core.threshold.support_edge import SupportEdgeWB
from gen.bundle_a_spec import (BOUNDARY_TOL, CONDITION_INDICES, FUTURE_OFFSET,
                               ROLLOUT_FRAMES, S_GRID, in_map)

PRINCIPLE = "phase_b_workbench_direction_selection"

PHASE_B = {"pendulum_rod_wb": RodPendulumWB, "support_edge_wb": SupportEdgeWB}

_CTX = "A fixed camera on a robotics workbench observes "
PROMPTS = {
    "pendulum_rod_wb": (_CTX + "a red payload hanging from a rigid grey rod on a steel fixture post, "
                        "swinging down past its lowest point. A white robot arm stands idle in the "
                        "background. The pendulum keeps its shape and arm length and continues under gravity."),
    "support_edge_wb": (_CTX + "a red ball rolling from left to right along a raised steel fixture plate "
                        "whose right end drops off to the grey inspection table below. A white robot arm "
                        "stands idle in the background. The ball keeps its shape and moves under gravity, "
                        "rolling contact, and momentum."),
}

ORACLE = {
    "pendulum_rod_wb": {
        1: "The rod is short enough: the payload swings over the top and completes the rotation.",
        0: "The rod is too long: the payload rises, stops before the top, and swings back.",
    },
    "support_edge_wb": {
        1: "The plate ends before the horizon: the red ball rolls off the edge and falls onto the table below.",
        0: "The plate is long enough: the red ball keeps rolling along the plate and stays on it.",
    },
}


def phase_specs():
    return {name: [cls.params_for_S(float(S)) for S in S_GRID]
            for name, cls in PHASE_B.items()}


__all__ = ["PRINCIPLE", "S_GRID", "BOUNDARY_TOL", "CONDITION_INDICES", "ROLLOUT_FRAMES",
           "FUTURE_OFFSET", "PHASE_B", "PROMPTS", "ORACLE", "phase_specs", "in_map"]
