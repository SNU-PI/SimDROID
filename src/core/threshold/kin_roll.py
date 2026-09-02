"""Phase A0 control: uniform rolling on the flat workbench (P0 kinematic continuation).

No decision variable and no event: the ball keeps rolling at its initial
speed, so a constant-velocity extrapolator is exactly right.  The scene exists
to separate video-generation failure (object loss, drift, stalls) from physics
failure in the P1/P2 scenes that share its appearance, contact model and camera.
"""

import numpy as np
import mujoco

from core.threshold.base import Base, G, wrap
from core.threshold.rolling_hill import SCENERY, tick_marks
from core.threshold.workbench import wrap_wb


class KinRoll(Base):
    cam = "a_side"
    n_frames = 24
    settle_steps = 0
    capture_dt = 1 / 16
    BALL_R = 0.03
    START_X = -0.55
    V_GRID = (0.30, 0.45, 0.60, 0.75, 0.90)      # m/s; 0.90 stays in frame for 21 frames
    style = "workbench"

    def __init__(self, p0=None):
        self.p = p0 or dict(v0=0.6)
        super().__init__()

    @staticmethod
    def S_of(p):
        return float("nan")

    @classmethod
    def margin_of(cls, p):
        return float("nan")

    def xml(self):
        body = f"""
    <camera name="a_side" fovy="20" pos="0 -2.05 0.30" xyaxes="1 0 0 0 0.1 0.995"/>
    {SCENERY}
    {tick_marks(-0.6, 0.6)}
    <body name="ball" pos="{self.START_X} 0 {self.BALL_R + 0.0015}">
      <freejoint/>
      <geom name="ballg" type="sphere" size="{self.BALL_R}" material="red" mass="0.1"/>
    </body>
"""
        contact = """
  <contact>
    <pair geom1="ballg" geom2="floor" condim="3" friction="0.8 0.8 0.0001 0.00002 0.00002"
          solref="0.0005 1" solimp="0.995 0.999 0.0003"/>
  </contact>
"""
        wrapper = wrap_wb if self.style == "workbench" else wrap
        return wrapper("kin_roll", body, contact).replace(
            'timestep="0.001"', 'timestep="0.000125"')

    def set_params(self, p):
        self.p = p
        self.model = mujoco.MjModel.from_xml_string(self.xml())
        self.data = mujoco.MjData(self.model)
        self._r = None

    def init_state(self, p):
        self.data.qvel[0] = p["v0"]
        self.data.qvel[4] = p["v0"] / self.BALL_R

    def observe(self):
        v = self.data.qvel[:3]
        w = self.data.qvel[3:6]
        speed = float(np.linalg.norm(v))
        spin = float(np.linalg.norm(w))
        energy = 0.5 * speed ** 2 + 0.2 * self.BALL_R ** 2 * spin ** 2 + G * self.data.qpos[2]
        return [self.data.qpos[0], self.data.qpos[2], self.data.qvel[0], speed, spin, energy]

    def labels(self, p, trace):
        x, vx = trace[:, 0], trace[:, 2]
        last = min(20, len(x) - 1)                      # end of the 21-frame rollout horizon
        speed_ratio = float(np.mean(vx[5:last + 1]) / p["v0"])
        outcome = int(vx[last] > 0.5 * p["v0"])          # 1 = keeps rolling right
        speed, spin = trace[:, 3], trace[:, 4]
        slip = np.abs(speed - spin * self.BALL_R) / np.maximum(speed, 1e-3)
        energy = trace[:, 5]
        drift = float((energy[last] - energy[0]) / max(energy[0], 1e-9))
        return dict(margin=float("nan"), outcome=outcome, event_frame=last,
                    t_event_s=last * self.capture_dt, decided=True,
                    slip_max=float(slip[: last + 1].max()), energy_drift=drift,
                    speed_ratio=speed_ratio, aux=float(x[last]),
                    cond_end_x=float(x[min(4, len(x) - 1)]))

    @staticmethod
    def sample(rng):
        return dict(v0=float(rng.uniform(0.3, 0.9)))
