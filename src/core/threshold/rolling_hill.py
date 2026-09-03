"""Bundle A1: a rolling ball against a smooth snow-white hill.

Work-energy theorem for rolling without slip: the ball crosses iff
(7/10) v0^2 >= g h, so S = 0.7 v0^2 / (g h) with v0 fixed and the hill
height h as the only swept, statically visible decision variable.

The profile is tangent-continuous, y(x) = h sin^2(pi (x - x0) / W), so there
is no impact loss at the base and no kink at the crest.  Mass cancels; the
friction coefficient only needs to sustain rolling (mu >= (2/7) tan(theta)).
"""

import numpy as np
import mujoco

from core.threshold.base import Base, G, wrap
from core.threshold.workbench import wrap_wb

SCENERY = """
    <geom name="refcube" type="box" size="0.025 0.025 0.025" pos="-0.62 0.20 0.025"
          material="grey" contype="0" conaffinity="0"/>
"""


def tick_marks(x_from, x_to, y=0.17, step=0.1):
    geoms = []
    n = int(round((x_to - x_from) / step))
    for i in range(n + 1):
        x = x_from + i * step
        geoms.append(
            f'<geom name="tick{i}" type="box" size="0.0035 0.0035 0.018" '
            f'pos="{x:.3f} {y} 0.018" material="dark" contype="0" conaffinity="0"/>'
        )
    return "\n".join(geoms)


class RollingHill(Base):
    """Conservative rolling threshold; every causal variable is visible."""

    cam = "a_side"
    n_frames = 40
    settle_steps = 0
    capture_dt = 1 / 16
    style = "toy"              # "workbench": same physics, Phase A appearance
    BALL_R = 0.03
    V0 = 1.0
    HILL_W = 0.30              # fixed width; h is the only decision variable
    BASE_X = -0.15             # hill spans [BASE_X, BASE_X + HILL_W]
    START_X = -0.55            # 0.40 m of flat approach before the base
    HF_HALF = 0.55             # hfield half-extent in x
    NC = 961                   # 1.1 mm cells: prism edges well below ball radius

    def __init__(self, p0=None):
        # Lazy: run()/set_params() builds the model; the hfield compile is slow
        # enough (tens of seconds) that the eager Base.__init__ build was pure waste.
        self.p = p0 or self.params_for_S(1.2)
        self.model = None
        self.data = None
        self._r = None

    @classmethod
    def params_for_S(cls, S):
        return dict(v0=cls.V0, h=float(0.7 * cls.V0 ** 2 / (G * S)))

    @staticmethod
    def S_of(p):
        return 0.7 * p["v0"] ** 2 / (G * p["h"])

    @classmethod
    def margin_of(cls, p):
        return cls.S_of(p) - 1.0

    def _profile(self):
        x = np.linspace(-self.HF_HALF, self.HF_HALF, self.NC)
        y = np.zeros_like(x)
        inside = (x >= self.BASE_X) & (x <= self.BASE_X + self.HILL_W)
        y[inside] = np.sin(np.pi * (x[inside] - self.BASE_X) / self.HILL_W) ** 2
        return np.tile(y, (2, 1))

    def xml(self):
        h = self.p["h"]
        hill_mat = "cover" if self.style == "workbench" else "snow"
        extra = (f'<hfield name="hillhf" nrow="2" ncol="{self.NC}" '
                 f'size="{self.HF_HALF} 0.14 {h} 0.001"/>'
                 '<material name="snow" rgba="0.92 0.93 0.95 1" specular="0.05" shininess="0.1"/>')
        body = f"""
    <camera name="a_side" fovy="20" pos="0 -2.05 0.30" xyaxes="1 0 0 0 0.1 0.995"/>
    <camera name="vj_side" fovy="36" pos="-0.02 -2.05 0.30" xyaxes="1 0 0 0 0.1 0.995"/>
    {SCENERY}
    {tick_marks(-0.6, 0.6)}
    <geom name="hillg" type="hfield" hfield="hillhf" pos="0 0 0.001" material="{hill_mat}"/>
    <body name="ball" pos="{self.START_X} 0 {self.BALL_R + 0.0015}">
      <freejoint/>
      <geom name="ballg" type="sphere" size="{self.BALL_R}" material="red" mass="0.1"/>
    </body>
"""
        # Stiff, near-lossless rolling contact; the small timestep and hard
        # solimp were tuned until the simulated boundary sits within 1.6% of
        # the closed form (see exp/verify_bundle_a.py bisection).
        contact = """
  <contact>
    <pair geom1="ballg" geom2="floor" condim="3" friction="0.8 0.8 0.0001 0.00002 0.00002"
          solref="0.0005 1" solimp="0.995 0.999 0.0003"/>
    <pair geom1="ballg" geom2="hillg" condim="3" friction="0.8 0.8 0.0001 0.00002 0.00002"
          solref="0.0005 1" solimp="0.995 0.999 0.0003"/>
  </contact>
"""
        wrapper = wrap_wb if self.style == "workbench" else wrap
        return wrapper("rolling_hill", body, contact, extra_asset=extra).replace(
            'timestep="0.001"', 'timestep="0.000125"')

    def set_params(self, p):
        self.p = p
        self.model = mujoco.MjModel.from_xml_string(self.xml())
        self.model.hfield_data[:] = self._profile().ravel()
        self.data = mujoco.MjData(self.model)
        self._r = None

    def init_state(self, p):
        self.data.qvel[0] = p["v0"]
        self.data.qvel[4] = p["v0"] / self.BALL_R      # rolling spin about +y

    def observe(self):
        v = self.data.qvel[:3]
        w = self.data.qvel[3:6]
        speed = float(np.linalg.norm(v))
        spin = float(np.linalg.norm(w))
        energy = 0.5 * speed ** 2 + 0.2 * self.BALL_R ** 2 * spin ** 2 \
            + G * self.data.qpos[2]
        return [self.data.qpos[0], self.data.qpos[2], self.data.qvel[0],
                speed, spin, energy]

    def labels(self, p, trace):
        x, vx = trace[:, 0], trace[:, 2]
        crest_x = self.BASE_X + self.HILL_W / 2
        crossed = x > crest_x + 0.03
        on_slope = x > self.BASE_X
        turned = on_slope & (vx <= 0.0)
        if crossed.any():
            outcome, event = 1, int(np.argmax(crossed))
        elif turned.any():
            outcome, event = 0, int(np.argmax(turned))
        else:
            outcome, event = 0, self.n_frames - 1
        speed, spin = trace[:, 3], trace[:, 4]
        slip = np.abs(speed - spin * self.BALL_R) / np.maximum(speed, 1e-3)
        energy = trace[:, 5]
        drift = float((energy[min(event + 2, len(energy) - 1)] - energy[0])
                      / max(energy[0], 1e-9))
        return dict(margin=float(self.margin_of(p)), outcome=outcome,
                    event_frame=event, t_event_s=event * self.capture_dt,
                    decided=bool(crossed.any() or turned.any()),
                    slip_max=float(slip[: event + 1].max()),
                    energy_drift=drift, aux=float(x.max()),
                    cond_end_x=float(x[min(4, len(x) - 1)]))

    @staticmethod
    def sample(rng):
        return dict(v0=1.0, h=float(rng.uniform(0.04, 0.14)))


class RollingHillWB(RollingHill):
    """Phase A1: identical physics, workbench appearance (green cable-cover hump)."""
    style = "workbench"
