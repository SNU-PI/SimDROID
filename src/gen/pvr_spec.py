"""PVR PoC (predictive visual reliance) experimental design, 2026-09-06.

Three scenes share one structure: a RELEVANT edit changes the decision variable of the
on-path structure (hill height / target size / right-end distance), a DECOY edit changes
the same variable of a look-alike structure behind the ball (decoy hump / decoy ball /
left-end distance) that the simulator guarantees to be causally irrelevant, and EXTENSION
cells probe extreme, depth-ambiguous or appearance-only structure.  Every condition is
run with the same seeds (common random numbers).  Design: DESIGN_PVR_POC_2026-09-06.md.
"""

from __future__ import annotations

from core.threshold.hill_decoy import HillDecoy
from core.threshold.support_edge_decoy import SupportEdgeDecoy
from core.threshold.two_ball_decoy import TwoBallDecoy
from gen.bundle_a_spec import CONDITION_INDICES, ROLLOUT_FRAMES

PRINCIPLE = "pvr_predictive_visual_reliance"
SEEDS = list(range(1, 13))
CURRENT_IDX = CONDITION_INDICES[-1]
HORIZON = ROLLOUT_FRAMES            # 21 frames incl. the 5 conditioning frames

SCENES = {"hill": HillDecoy, "collide": TwoBallDecoy, "edge": SupportEdgeDecoy}
FAMILY = {"hill": "hill_roll_pvr", "collide": "two_ball_pvr", "edge": "support_edge_pvr"}

_CTX = "A fixed camera on a robotics workbench observes "
PROMPTS = {
    "hill": (_CTX + "a red ball rolling from left to right across a grey inspection table. Low green rubber "
             "cable-cover ramps lie on the table. A white robot arm stands idle in the background. The ball "
             "keeps its shape and moves under gravity, rolling contact, and momentum."),
    "collide": (_CTX + "a red ball moving right across a grey low-friction inspection table. Blue balls of the "
                "same material rest on the table. A white robot arm stands idle in the background. The balls "
                "keep their shapes and sizes and move with realistic contact physics."),
    "edge": (_CTX + "a red ball rolling from left to right along a raised steel fixture plate above a grey "
             "inspection table. A white robot arm stands idle in the background. The ball keeps its shape and "
             "moves under gravity, rolling contact, and momentum."),
}

# ---------------------------------------------------------------- Hill
_H_LOW = HillDecoy.params_for_S(2.0)["h"]        # 3.57 cm: crosses
_H_HIGH = HillDecoy.params_for_S(0.5)["h"]       # 14.27 cm: returns


def _hill(cid, params, group, expected, relevant, decoy, note=""):
    return dict(id=cid, params=params, group=group, expected=expected, relevant=relevant, decoy=decoy, note=note)


HILL = [
    _hill("A", HillDecoy.params_for_S(2.0, h_d=_H_LOW), "core", 1, dict(S=2.0, h=_H_LOW), dict(h_d=_H_LOW), "reference"),
    _hill("B", HillDecoy.params_for_S(0.5, h_d=_H_LOW), "core", 0, dict(S=0.5, h=_H_HIGH), dict(h_d=_H_LOW), "relevant edit A->B"),
    _hill("C", HillDecoy.params_for_S(2.0, h_d=_H_HIGH), "core", 1, dict(S=2.0, h=_H_LOW), dict(h_d=_H_HIGH), "decoy edit A->C"),
    _hill("D", HillDecoy.params_for_S(0.5, h_d=_H_HIGH), "core", 0, dict(S=0.5, h=_H_HIGH), dict(h_d=_H_HIGH), "relevant C->D, decoy B->D"),
    _hill("A0", HillDecoy.params_for_S(2.0, h_d=0.0), "nodecoy", 1, dict(S=2.0, h=_H_LOW), dict(h_d=0.0), "no decoy"),
    _hill("B0", HillDecoy.params_for_S(0.5, h_d=0.0), "nodecoy", 0, dict(S=0.5, h=_H_HIGH), dict(h_d=0.0), "no decoy"),
] + [
    _hill(f"L{i + 1}", HillDecoy.params_for_S(S, h_d=_H_LOW), "ladder", (1 if S > 1 else 0) if abs(S - 1) > 0.04 else None,
          dict(S=S, h=HillDecoy.params_for_S(S)["h"]), dict(h_d=_H_LOW), "height ladder")
    for i, S in enumerate((0.63, 0.79, 1.00, 1.26, 1.59))
] + [
    _hill("X20", HillDecoy.params_for_h(0.20, h_d=_H_LOW), "ext", 0, dict(h=0.20), dict(h_d=_H_LOW), "tall hump 20 cm (64 deg)"),
    _hill("X28", HillDecoy.params_for_h(0.28, h_d=_H_LOW), "ext", 0, dict(h=0.28), dict(h_d=_H_LOW), "tall hump 28 cm (71 deg)"),
    _hill("XW", dict(v0=HillDecoy.V0, h=0.0, h_d=_H_LOW, variant="wall"), "ext", 0, dict(wall=0.14), dict(h_d=_H_LOW), "vertical steel stopper"),
    _hill("XF", dict(v0=HillDecoy.V0, h=0.0, h_d=_H_LOW, variant="flat"), "ext", 1, dict(h=0.0), dict(h_d=_H_LOW), "flat: no obstacle"),
]

# ---------------------------------------------------------------- Collide
_R_LIGHT = TwoBallDecoy.params_for_S(0.5)["r2"]     # 3.17 cm
_R_HEAVY = TwoBallDecoy.params_for_S(2.0)["r2"]     # 5.04 cm


def _col(cid, params, group, expected, relevant, decoy, note=""):
    return dict(id=cid, params=params, group=group, expected=expected, relevant=relevant, decoy=decoy, note=note)


COLLIDE = [
    _col("A", TwoBallDecoy.params_for_S(0.5, r_d=_R_LIGHT), "core", 0, dict(S=0.5, r2=_R_LIGHT), dict(r_d=_R_LIGHT), "reference"),
    _col("B", TwoBallDecoy.params_for_S(2.0, r_d=_R_LIGHT), "core", 1, dict(S=2.0, r2=_R_HEAVY), dict(r_d=_R_LIGHT), "relevant edit A->B"),
    _col("C", TwoBallDecoy.params_for_S(0.5, r_d=_R_HEAVY), "core", 0, dict(S=0.5, r2=_R_LIGHT), dict(r_d=_R_HEAVY), "decoy edit A->C"),
    _col("D", TwoBallDecoy.params_for_S(2.0, r_d=_R_HEAVY), "core", 1, dict(S=2.0, r2=_R_HEAVY), dict(r_d=_R_HEAVY), "relevant C->D, decoy B->D"),
    _col("A0", TwoBallDecoy.params_for_S(0.5, r_d=0.0), "nodecoy", 0, dict(S=0.5, r2=_R_LIGHT), dict(r_d=0.0), "no decoy"),
    _col("B0", TwoBallDecoy.params_for_S(2.0, r_d=0.0), "nodecoy", 1, dict(S=2.0, r2=_R_HEAVY), dict(r_d=0.0), "no decoy"),
] + [
    _col(f"L{i + 1}", TwoBallDecoy.params_for_S(S, r_d=_R_LIGHT), "ladder", (1 if S > 1 else 0) if abs(S - 1) > 0.04 else None,
         dict(S=S, r2=TwoBallDecoy.params_for_S(S)["r2"]), dict(r_d=_R_LIGHT), "mass-ratio ladder")
    for i, S in enumerate((0.63, 0.79, 1.00, 1.26, 1.59))
] + [
    _col("XW", dict(r2=TwoBallDecoy.R1, v0=TwoBallDecoy.V0, r_d=_R_LIGHT, variant="wall", y_target=0.0), "ext", 1, dict(wall=True), dict(r_d=_R_LIGHT), "steel stopper (S -> inf)"),
    _col("XL", TwoBallDecoy.params_for_S(0.125, r_d=_R_LIGHT), "ext", 0, dict(S=0.125, r2=TwoBallDecoy.params_for_S(0.125)["r2"]), dict(r_d=_R_LIGHT), "very light target (2 cm)"),
    _col("XM", TwoBallDecoy.params_for_S(2.0, r_d=_R_LIGHT, variant="miss", y_target=0.12), "ext", 0, dict(S=2.0, y_target=0.12), dict(r_d=_R_LIGHT), "heavy target shifted off the path (miss)"),
    _col("XF", dict(r2=TwoBallDecoy.R1, v0=TwoBallDecoy.V0, r_d=_R_LIGHT, variant="none", y_target=0.0), "ext", 0, dict(target=None), dict(r_d=_R_LIGHT), "no target"),
]

# ---------------------------------------------------------------- Edge (timing scene)
_LEFT_FAR, _LEFT_NEAR, _LEFT_OFF = -1.08, -0.65, SupportEdgeDecoy.OFFSCREEN_LEFT


def _edge(cid, params, group, expected, relevant, decoy, note=""):
    params = dict(params)
    return dict(id=cid, params=params, group=group, expected=expected, relevant=relevant, decoy=decoy, note=note)


def _pt(t, v0=0.35, x_left=_LEFT_FAR, **kw):
    return SupportEdgeDecoy.params_for_t(t, v0=v0, x_left=x_left, **kw)


EDGE = [
    _edge("A", _pt(0.60), "core", 1, dict(t_edge=0.60, v0=0.35), dict(x_left=_LEFT_FAR), "reference: fall f12"),
    _edge("B", _pt(1.05), "core", 1, dict(t_edge=1.05, v0=0.35), dict(x_left=_LEFT_FAR), "relevant edit A->B: fall f19"),
    _edge("C", _pt(0.60, x_left=_LEFT_NEAR), "core", 1, dict(t_edge=0.60, v0=0.35), dict(x_left=_LEFT_NEAR), "decoy edit A->C"),
    _edge("D", _pt(1.05, x_left=_LEFT_NEAR), "core", 1, dict(t_edge=1.05, v0=0.35), dict(x_left=_LEFT_NEAR), "relevant C->D, decoy B->D"),
    _edge("A0", _pt(0.60, x_left=_LEFT_OFF), "nodecoy", 1, dict(t_edge=0.60, v0=0.35), dict(x_left=None), "no visible left end"),
    _edge("B0", _pt(1.05, x_left=_LEFT_OFF), "nodecoy", 1, dict(t_edge=1.05, v0=0.35), dict(x_left=None), "no visible left end"),
    _edge("T1", _pt(0.60, v0=0.25), "isotime", 1, dict(t_edge=0.60, v0=0.25), dict(x_left=_LEFT_FAR), "iso-time grid"),
    _edge("T2", _pt(0.75, v0=0.25), "isotime", 1, dict(t_edge=0.75, v0=0.25), dict(x_left=_LEFT_FAR), "iso-time grid"),
    _edge("T3", _pt(1.05, v0=0.25), "isotime", 1, dict(t_edge=1.05, v0=0.25), dict(x_left=_LEFT_FAR), "iso-time grid"),
    _edge("T4", _pt(0.60, v0=0.50), "isotime", 1, dict(t_edge=0.60, v0=0.50), dict(x_left=_LEFT_FAR), "iso-time grid"),
    _edge("T5", _pt(0.75, v0=0.50), "isotime", 1, dict(t_edge=0.75, v0=0.50), dict(x_left=_LEFT_FAR), "iso-time grid"),
    _edge("T6", _pt(1.05, v0=0.50), "isotime", 1, dict(t_edge=1.05, v0=0.50), dict(x_left=_LEFT_FAR), "iso-time grid"),
    _edge("T7", _pt(0.75), "isotime", 1, dict(t_edge=0.75, v0=0.35), dict(x_left=_LEFT_FAR), "iso-time grid midpoint"),
    _edge("N1", _pt(1.40), "hold", 0, dict(t_edge=1.40, v0=0.35), dict(x_left=_LEFT_FAR), "edge beyond the horizon: stays"),
    _edge("N2", _pt(1.40, v0=0.50), "hold", 0, dict(t_edge=1.40, v0=0.50), dict(x_left=_LEFT_FAR), "edge beyond the horizon: stays"),
    _edge("XJ", _pt(1.40, variant="joint", x_joint=-0.34), "ext", 0, dict(t_edge=1.40, joint_x=-0.34), dict(x_left=_LEFT_FAR), "panel joint line at the A-edge position, plate continues"),
    _edge("XS", _pt(0.60, variant="step", x_lower_end=0.20, step_drop=0.05), "ext", 1, dict(t_edge=0.60, step=0.05), dict(x_left=_LEFT_FAR), "step down 5 cm at the A-edge position"),
    _edge("XT", _pt(1.05, variant="tilt", tilt_deg=3.0), "ext", 1, dict(t_edge_uniform=1.05, tilt_deg=3.0), dict(x_left=_LEFT_FAR), "plate tilted 3 deg downhill: early fall"),
]

CONDITIONS = {"hill": HILL, "collide": COLLIDE, "edge": EDGE}

# Paired edits used by the analysis: (label, from, to, kind).
EDITS = {
    "hill": [("relevant A->B", "A", "B", "relevant"), ("relevant C->D", "C", "D", "relevant"),
             ("decoy A->C", "A", "C", "decoy"), ("decoy B->D", "B", "D", "decoy"),
             ("decoy off A0->A", "A0", "A", "decoy_presence"), ("decoy off B0->B", "B0", "B", "decoy_presence")],
    "collide": [("relevant A->B", "A", "B", "relevant"), ("relevant C->D", "C", "D", "relevant"),
                ("decoy A->C", "A", "C", "decoy"), ("decoy B->D", "B", "D", "decoy"),
                ("decoy off A0->A", "A0", "A", "decoy_presence"), ("decoy off B0->B", "B0", "B", "decoy_presence")],
    "edge": [("relevant A->B", "A", "B", "relevant"), ("relevant C->D", "C", "D", "relevant"),
             ("decoy A->C", "A", "C", "decoy"), ("decoy B->D", "B", "D", "decoy"),
             ("decoy off A0->A", "A0", "A", "decoy_presence"), ("decoy off B0->B", "B0", "B", "decoy_presence")],
}

__all__ = ["PRINCIPLE", "SEEDS", "CURRENT_IDX", "HORIZON", "SCENES", "FAMILY", "PROMPTS",
           "CONDITIONS", "EDITS", "HILL", "COLLIDE", "EDGE"]
