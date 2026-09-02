"""Bundle A2: head-on collision of two equal-density balls.

Momentum conservation with near-elastic restitution: the incoming red ball
reverses iff m1 < e * m2.  Density is shared, so the mass ratio is carried by
the visible radius ratio alone: S = (r2 / r1)^3 with e ~= 1.

The floor is near-frictionless (repo precedent from collide.py), so the
approach is uniform motion, the impulse is exchanged only through the
ball-ball contact (condim=1, no tangential coupling), and the post-impact
velocities persist long enough to read momentum off the video.
"""

import numpy as np
import mujoco

from core.threshold.base import Base, wrap
from core.threshold.rolling_hill import SCENERY, tick_marks
from core.threshold.workbench import wrap_wb

RHO = 800.0


def sphere_mass(r):
    return RHO * 4.0 / 3.0 * np.pi * r ** 3


class TwoBall(Base):
    """Reversal threshold at the visible size ratio; laws: p conservation."""

    cam = "a2_side"
    n_frames = 30
    settle_steps = 0
    capture_dt = 1 / 16
    style = "toy"              # "workbench": same physics, Phase A appearance
    R1 = 0.040                 # red incoming ball, fixed
    V0 = 1.0
    START_X = -0.47            # blue rests at x = 0
    PAIR_STIFF = 30000         # near-elastic pair: solref="-stiff -damp"
    PAIR_DAMP = 0.2            # measured e_eff ~= 0.996 (verify_bundle_a)

    def __init__(self, p0=None):
        self.p = p0 or self.params_for_S(1.2)
        super().__init__()

    @classmethod
    def params_for_S(cls, S):
        return dict(r2=float(cls.R1 * S ** (1.0 / 3.0)), v0=cls.V0)

    @classmethod
    def S_of(cls, p):
        return (p["r2"] / cls.R1) ** 3

    @classmethod
    def margin_of(cls, p):
        return cls.S_of(p) - 1.0

    def xml(self):
        r2 = self.p["r2"]
        m1, m2 = sphere_mass(self.R1), sphere_mass(r2)
        body = f"""
    <camera name="a2_side" fovy="20" pos="-0.05 -1.5 0.24" xyaxes="1 0 0 0 0.1 0.995"/>
    {SCENERY.replace('-0.62 0.20', '-0.50 0.16')}
    {tick_marks(-0.5, 0.4, y=0.13)}
    <body name="ballA" pos="{self.START_X} 0 {self.R1 + 0.001}">
      <freejoint/>
      <geom name="gA" type="sphere" size="{self.R1}" material="red" mass="{m1:.6f}"/>
    </body>
    <body name="ballB" pos="0 0 {r2 + 0.001}">
      <freejoint/>
      <geom name="gB" type="sphere" size="{r2:.6f}" material="blue" mass="{m2:.6f}"/>
    </body>
"""
        contact = f"""
  <contact>
    <pair geom1="gA" geom2="floor" condim="3" friction="0.0002 0.0002 0.0001 0.0001 0.0001"
          solref="0.006 1" solimp="0.95 0.95 0.001"/>
    <pair geom1="gB" geom2="floor" condim="3" friction="0.0002 0.0002 0.0001 0.0001 0.0001"
          solref="0.006 1" solimp="0.95 0.95 0.001"/>
    <pair geom1="gA" geom2="gB" condim="1" solref="-{self.PAIR_STIFF} -{self.PAIR_DAMP:.2f}"
          solimp="0.95 0.95 0.001"/>
  </contact>
"""
        # timestep 0.5 ms keeps capture_dt / timestep integral (exact 16 FPS)
        return self._wrap("two_ball", body, contact).replace(
            'timestep="0.001"', 'timestep="0.0005"')

    def _wrap(self, name, body, contact):
        if self.style == "workbench":
            return wrap_wb(name, body, contact, tape_y=0.13, tape_half=0.45)
        return wrap(name, body, contact)

    def set_params(self, p):
        self.p = p
        self.model = mujoco.MjModel.from_xml_string(self.xml())
        self.data = mujoco.MjData(self.model)
        self._r = None

    def init_state(self, p):
        self.data.qvel[0] = p["v0"]

    def observe(self):
        return [self.data.qpos[0], self.data.qvel[0],
                self.data.qpos[7], self.data.qvel[6]]

    def labels(self, p, trace):
        xA, vA, xB, vB = trace[:, 0], trace[:, 1], trace[:, 2], trace[:, 3]
        hit = vB > 0.02
        contact = int(np.argmax(hit)) if hit.any() else self.n_frames - 1
        read = min(contact + 3, self.n_frames - 1)     # adjudication window
        outcome = int(vA[read] < -0.005)               # 1 = red reversed
        m1, m2 = sphere_mass(self.R1), sphere_mass(p["r2"])
        e_eff = float((vB[read] - vA[read]) / max(p["v0"], 1e-9))
        p_ratio = float((m1 * vA[read] + m2 * vB[read]) / (m1 * p["v0"]))
        ke_ratio = float((m1 * vA[read] ** 2 + m2 * vB[read] ** 2)
                         / (m1 * p["v0"] ** 2))
        return dict(margin=float(self.margin_of(p)), outcome=outcome,
                    event_frame=contact, t_event_s=contact * self.capture_dt,
                    decided=bool(hit.any()), read_frame=read,
                    e_eff=e_eff, momentum_ratio=p_ratio, ke_ratio=ke_ratio,
                    aux=float(vA[read]))

    @staticmethod
    def sample(rng):
        return dict(r2=float(rng.uniform(0.032, 0.050)), v0=1.0)


class WallBounce(TwoBall):
    """A2-0 pre-check: elastic wall bounce -- normal-velocity sign reversal
    with speed preserved.  Qualitative gate that the model can draw a
    direction flip at all."""

    n_frames = 24

    def __init__(self, p0=None):
        self.p = p0 or dict(v0=self.V0)
        Base.__init__(self)

    @classmethod
    def params_for_S(cls, S):
        raise NotImplementedError("wall bounce is a qualitative pre-check")

    def xml(self):
        m1 = sphere_mass(self.R1)
        body = f"""
    <camera name="a2_side" fovy="20" pos="-0.05 -1.5 0.24" xyaxes="1 0 0 0 0.1 0.995"/>
    {SCENERY.replace('-0.62 0.20', '-0.50 0.16')}
    {tick_marks(-0.5, 0.1, y=0.13)}
    <geom name="wall" type="box" size="0.02 0.12 0.10" pos="0.02 0 0.10" material="{'steel' if self.style == 'workbench' else 'dark'}"/>
    <body name="ballA" pos="{self.START_X} 0 {self.R1 + 0.001}">
      <freejoint/>
      <geom name="gA" type="sphere" size="{self.R1}" material="red" mass="{m1:.6f}"/>
    </body>
"""
        contact = f"""
  <contact>
    <pair geom1="gA" geom2="floor" condim="3" friction="0.0002 0.0002 0.0001 0.0001 0.0001"
          solref="0.006 1" solimp="0.95 0.95 0.001"/>
    <pair geom1="gA" geom2="wall" condim="1" solref="-{self.PAIR_STIFF} -{self.PAIR_DAMP:.2f}"
          solimp="0.95 0.95 0.001"/>
  </contact>
"""
        return self._wrap("wall_bounce", body, contact).replace(
            'timestep="0.001"', 'timestep="0.0005"')

    def observe(self):
        return [self.data.qpos[0], self.data.qvel[0], 0.0, 0.0]

    def labels(self, p, trace):
        vA = trace[:, 1]
        hit = vA < 0.5 * p["v0"]
        contact = int(np.argmax(hit)) if hit.any() else self.n_frames - 1
        read = min(contact + 3, self.n_frames - 1)
        return dict(margin=-1.0, outcome=int(vA[read] < -0.005),
                    event_frame=contact, t_event_s=contact * self.capture_dt,
                    decided=bool(hit.any()), read_frame=read,
                    e_eff=float(-vA[read] / p["v0"]),
                    momentum_ratio=float("nan"), ke_ratio=float(vA[read] ** 2),
                    aux=float(vA[read]))


class TwoBallWB(TwoBall):
    """Phase A2: identical physics, workbench appearance."""
    style = "workbench"


class WallBounceWB(WallBounce):
    """Phase A2-0 pre-check on the workbench: steel stopper block."""
    style = "workbench"
