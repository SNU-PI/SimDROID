"""Bundle A3: rigid rod pendulum, swing-over threshold set by arm length.

Same conservation law as the rolling hill (a gravity barrier), different
mechanism: the constraint is a rigid rod, the barrier is 2 g L, and the
decision variable is the statically visible arm length L.

S = KE_bottom / (m g 2L) = v0^2 (1 + 0.4 r^2/L^2) / (4 g L), v0 fixed.
L is inverted from S exactly (cubic), so the inertia correction of the finite
bob is part of the definition rather than an approximation error.

The sample starts on the left at the height holding fraction q of the bottom
kinetic energy, so the conditioning window shows the descent and the bottom
passage; the swing-over/turnaround event stays outside the window.
"""

import numpy as np
import mujoco

from core.threshold.base import Base, G, wrap
from core.threshold.workbench import wrap_wb


class RodPendulum(Base):
    """Swing-over threshold; laws: energy conservation on a rigid constraint."""

    cam = "a3_side"
    n_frames = 40
    settle_steps = 0
    capture_dt = 1 / 16
    style = "toy"              # "workbench": same physics, Phase B appearance (suspended payload)
    BOB_R = 0.035
    ROD_R = 0.012
    V0 = 3.6                    # bob speed at the lowest point, fixed
                                # (sets L_crit = v0^2/4g ~= 0.33 m; higher v0
                                # slows all angular rates, omega_b = 4gS/v0)
    PIVOT_Z = 0.80
    Q_START = 0.6               # start height = q * bottom-KE height
    Q_CAP = 0.95                # q_eff = min(Q_START, Q_CAP / S); the cap keeps
                                # fast (high-S) samples starting near the top so
                                # the swing-over event stays out of the window

    def __init__(self, p0=None):
        self.p = p0 or self.params_for_S(1.2)
        super().__init__()

    @classmethod
    def length_for_S(cls, S):
        # 4 g S L^3 - v0^2 L^2 - 0.4 v0^2 r^2 = 0, unique positive root.
        coeffs = [4 * G * S, -cls.V0 ** 2, 0.0, -0.4 * cls.V0 ** 2 * cls.BOB_R ** 2]
        roots = np.roots(coeffs)
        real = [float(r.real) for r in roots if abs(r.imag) < 1e-9 and r.real > 0]
        return max(real)

    @classmethod
    def params_for_S(cls, S):
        length = cls.length_for_S(float(S))
        q = min(cls.Q_START, cls.Q_CAP / float(S))
        theta0 = float(np.arccos(np.clip(1.0 - 2.0 * q * float(S), -1.0, 1.0)))
        return dict(v0=cls.V0, length=length, theta0=theta0, q=q)

    @classmethod
    def S_of(cls, p):
        arm = p["length"]
        return p["v0"] ** 2 * (1 + 0.4 * cls.BOB_R ** 2 / arm ** 2) / (4 * G * arm)

    @classmethod
    def margin_of(cls, p):
        return cls.S_of(p) - 1.0

    def xml(self):
        arm = self.p["length"]
        body = f"""
    <camera name="a3_side" fovy="24" pos="0 -3.3 0.62" xyaxes="1 0 0 0 0.06 0.998"/>
    <geom name="refcube" type="box" size="0.025 0.025 0.025" pos="-0.55 0.20 0.025"
          material="grey" contype="0" conaffinity="0"/>
    <geom name="post" type="box" size="0.022 0.022 {self.PIVOT_Z / 2 + 0.02:.3f}"
          pos="0 0.12 {self.PIVOT_Z / 2:.3f}" material="dark" contype="0" conaffinity="0"/>
    <geom name="axle" type="capsule" fromto="0 0.12 {self.PIVOT_Z} 0 -0.015 {self.PIVOT_Z}"
          size="0.009" material="dark" contype="0" conaffinity="0"/>
    <body name="pendulum" pos="0 0 {self.PIVOT_Z}">
      <joint name="hinge" type="hinge" axis="0 -1 0" damping="0"/>
      <geom name="rod" type="capsule" fromto="0 0 0 0 0 {-arm:.6f}"
            size="{self.ROD_R}" material="grey" mass="0.000001"
            contype="0" conaffinity="0"/>
      <geom name="bob" type="sphere" size="{self.BOB_R}" pos="0 0 {-arm:.6f}"
            material="red" mass="0.1" contype="0" conaffinity="0"/>
    </body>
"""
        # timestep 0.5 ms keeps capture_dt / timestep integral (exact 16 FPS)
        if self.style == "workbench":
            # Appearance only: steel post/axle; every added geom is collision-free.
            body = body.replace('name="post" type="box" size="0.022 0.022', 'name="post" type="box" size="0.022 0.022') \
                       .replace('material="dark" contype="0" conaffinity="0"/>\n    <geom name="axle"',
                                'material="steel" contype="0" conaffinity="0"/>\n    <geom name="axle"') \
                       .replace('size="0.009" material="dark"', 'size="0.009" material="steel"')
            return wrap_wb("rod_pendulum", body).replace(
                'timestep="0.001"', 'timestep="0.0005"')
        return wrap("rod_pendulum", body).replace(
            'timestep="0.001"', 'timestep="0.0005"')

    def set_params(self, p):
        self.p = p
        self.model = mujoco.MjModel.from_xml_string(self.xml())
        self.data = mujoco.MjData(self.model)
        self._r = None

    def init_state(self, p):
        arm = p["length"]
        omega_bottom = p["v0"] / arm
        inertia = arm ** 2 + 0.4 * self.BOB_R ** 2
        drop = 2 * G * arm * (1 - np.cos(p["theta0"])) / inertia
        self.data.qpos[0] = -p["theta0"]
        self.data.qvel[0] = float(np.sqrt(max(omega_bottom ** 2 - drop, 0.0)))

    def observe(self):
        arm = self.p["length"]
        theta, omega = self.data.qpos[0], self.data.qvel[0]
        inertia = arm ** 2 + 0.4 * self.BOB_R ** 2
        energy = 0.5 * inertia * omega ** 2 + G * arm * (1 - np.cos(theta))
        return [theta, omega, energy]

    def labels(self, p, trace):
        theta, omega = trace[:, 0], trace[:, 1]
        past_bottom = theta >= 0.0
        bottom = int(np.argmax(past_bottom)) if past_bottom.any() else -1
        rotated = theta >= np.pi
        turned = past_bottom & (omega <= 0.0)
        if rotated.any():
            outcome, event = 1, int(np.argmax(rotated))
        elif turned.any():
            outcome, event = 0, int(np.argmax(turned))
        else:
            outcome, event = 0, self.n_frames - 1
        energy = trace[:, 2]
        drift = float((energy[min(event + 2, len(energy) - 1)] - energy[0])
                      / max(energy[0], 1e-9))
        return dict(margin=float(self.margin_of(p)), outcome=outcome,
                    event_frame=event, t_event_s=event * self.capture_dt,
                    decided=bool(rotated.any() or turned.any()),
                    bottom_frame=bottom, energy_drift=drift,
                    aux=float(theta.max()))

    @staticmethod
    def sample(rng):
        return RodPendulum.params_for_S(float(rng.uniform(0.6, 1.6)))


class RodPendulumWB(RodPendulum):
    """Phase B (P1-B): identical physics, workbench appearance (payload on a steel post)."""
    style = "workbench"
