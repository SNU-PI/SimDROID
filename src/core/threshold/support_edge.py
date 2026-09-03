"""Phase B (P3-A): support-edge departure -- contact ends, free fall begins.

A ball rolls at a fixed speed along a raised fixture plate whose right end is
the only swept, statically visible decision variable.  Within the judged
horizon the ball either stays supported (S < 1) or passes the end of the plate
and drops onto the table below (S > 1).  Uniform motion (the history
extrapolator) never falls: it predicts a ball hovering past the end.

S = v0 * T_REF / d_edge with d_edge = x_edge - x_start.  T_REF is the time a
ball needs to reach the end so that its drop (centre half a radius below the
plate top) lands exactly on the last judged frame; it is calibrated against the
simulation in exp/verify_phase_b.py (mode {free -> free fall} is a solver-
insensitive transition, no critical slowing down).

Law readouts: uniform speed before the edge (rolling audit), then free fall
z(t) - z_edge = -g t^2 / 2 with x continuing at v0.
"""

import numpy as np
import mujoco

from core.threshold.base import Base, G, wrap
from core.threshold.rolling_hill import SCENERY
from core.threshold.workbench import wrap_wb


def plate_ticks(x_from, x_to, y, z, step=0.1):
    geoms = []
    n = int(np.floor((x_to - x_from) / step + 1e-9))
    for i in range(n + 1):
        x = x_from + i * step
        geoms.append(
            f'<geom name="ptick{i}" type="box" size="0.0035 0.0035 0.018" '
            f'pos="{x:.3f} {y} {z:.3f}" material="dark" contype="0" conaffinity="0"/>'
        )
    return "\n".join(geoms)


class SupportEdge(Base):
    """Contact-loss threshold; laws: uniform rolling, then projectile motion."""

    cam = "a_side"
    n_frames = 24
    settle_steps = 0
    capture_dt = 1 / 16
    style = "toy"              # "workbench": same physics, Phase B appearance
    BALL_R = 0.03
    V0 = 0.35                  # m/s, fixed (14.5 px/frame at 480x832)
    START_X = -0.55
    PLATE_LEFT = -1.10
    PLATE_TOP = 0.15           # drop height to the table
    PLATE_HALF_Y = 0.12
    HORIZON = 21               # judged frames (Cosmos clip: 5 context + 16 predicted)
    DROP = 0.5                 # fell := centre below PLATE_TOP + R * (1 - DROP)
    T_REF = 1.176              # s; see module docstring (bisection in verify_phase_b, 2026-09-03:
                               # boundary at S=0.978 with 1.15 -> fitted 1.1763)

    def __init__(self, p0=None):
        self.p = p0 or self.params_for_S(1.2)
        super().__init__()

    @classmethod
    def params_for_S(cls, S):
        return dict(v0=cls.V0, x_edge=float(cls.START_X + cls.V0 * cls.T_REF / S))

    @classmethod
    def S_of(cls, p):
        return p["v0"] * cls.T_REF / (p["x_edge"] - cls.START_X)

    @classmethod
    def margin_of(cls, p):
        return cls.S_of(p) - 1.0

    def xml(self):
        xe = self.p["x_edge"]
        cx = 0.5 * (self.PLATE_LEFT + xe)
        hx = 0.5 * (xe - self.PLATE_LEFT)
        hz = 0.5 * self.PLATE_TOP
        wb = self.style == "workbench"
        body = f"""
    <camera name="a_side" fovy="20" pos="0 -2.05 0.30" xyaxes="1 0 0 0 0.1 0.995"/>
    <camera name="vj_side" fovy="36" pos="-0.02 -2.05 0.30" xyaxes="1 0 0 0 0.1 0.995"/>
    {SCENERY}
    {plate_ticks(-0.6, xe - 0.02, y=0.10, z=self.PLATE_TOP + 0.018)}
    <geom name="plate" type="box" size="{hx:.4f} {self.PLATE_HALF_Y} {hz:.4f}" pos="{cx:.4f} 0 {hz:.4f}"
          material="{'steel' if wb else 'grey'}"/>
    <geom name="plate_lip" type="box" size="{hx:.4f} 0.006 0.004" pos="{cx:.4f} {self.PLATE_HALF_Y + 0.006:.3f} {self.PLATE_TOP - 0.004:.3f}"
          material="dark" contype="0" conaffinity="0"/>
    <body name="ball" pos="{self.START_X} 0 {self.PLATE_TOP + self.BALL_R + 0.0015}">
      <freejoint/>
      <geom name="ballg" type="sphere" size="{self.BALL_R}" material="red" mass="0.1"/>
    </body>
"""
        contact = """
  <contact>
    <pair geom1="ballg" geom2="plate" condim="3" friction="0.8 0.8 0.0001 0.00002 0.00002"
          solref="0.0005 1" solimp="0.995 0.999 0.0003"/>
    <pair geom1="ballg" geom2="floor" condim="3" friction="0.8 0.8 0.0001 0.00002 0.00002"
          solref="0.0005 1" solimp="0.995 0.999 0.0003"/>
  </contact>
"""
        if wb:
            xml = wrap_wb("support_edge", body, contact, tape_y=-0.17)
        else:
            xml = wrap("support_edge", body, contact)
        return xml.replace('timestep="0.001"', 'timestep="0.000125"')

    def set_params(self, p):
        self.p = p
        self.model = mujoco.MjModel.from_xml_string(self.xml())
        self.data = mujoco.MjData(self.model)
        self._r = None

    def init_state(self, p):
        self.data.qvel[0] = p["v0"]
        self.data.qvel[4] = p["v0"] / self.BALL_R      # rolling spin about +y

    def observe(self):
        return [self.data.qpos[0], self.data.qpos[2], self.data.qvel[0], self.data.qvel[2]]

    def labels(self, p, trace):
        x, z, vx = trace[:, 0], trace[:, 1], trace[:, 2]
        fell = z < self.PLATE_TOP + self.BALL_R * (1.0 - self.DROP)
        fell_h = fell[: self.HORIZON]
        past = x > p["x_edge"]
        edge_frame = int(np.argmax(past)) if past.any() else -1
        if fell_h.any():
            outcome, event = 1, int(np.argmax(fell_h))
        else:
            outcome, event = 0, self.HORIZON - 1
        pre = max(min(edge_frame if edge_frame > 0 else self.n_frames, 5), 1)
        speed_ratio = float(vx[pre - 1] / vx[0]) if vx[0] > 1e-9 else float("nan")
        return dict(margin=float(self.margin_of(p)), outcome=outcome,
                    event_frame=event, t_event_s=event * self.capture_dt,
                    decided=True, edge_frame=edge_frame,
                    speed_ratio=speed_ratio, energy_drift=float(speed_ratio - 1.0),
                    aux=float(z.min()), cond_end_x=float(x[min(4, len(x) - 1)]))

    @staticmethod
    def sample(rng):
        return SupportEdge.params_for_S(float(rng.uniform(0.6, 1.6)))


class SupportEdgeWB(SupportEdge):
    """Phase B: identical physics, workbench appearance (steel fixture plate)."""
    style = "workbench"
