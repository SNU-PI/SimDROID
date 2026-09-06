"""PVR Predictive Hill: the Phase A rolling hill (physics unchanged) plus a decoy hump
behind the ball, a wider side camera, and extension obstacles.

Layout (x, metres): decoy hump [-0.78, -0.48], ball start -0.25, main hump [0.10, 0.40],
crest 0.25.  Both humps live in one height field, so the contact model is exactly the
Phase A one; the decoy is a real collider that the ball never reaches inside the
21-frame horizon (checked in exp/verify_pvr.py and by the generator gate).

Variants: "hump" (main hump of height h; S = 0.7 v0^2 / (g h)), "wall" (a steel stopper
block instead of the hump, S -> inf), "flat" (no obstacle on the path).
"""

import numpy as np

from core.threshold.base import G
from core.threshold.rolling_hill import RollingHill, tick_marks
from core.threshold.workbench import wrap_wb

SCENERY_PVR = ('<geom name="refcube" type="box" size="0.025 0.025 0.025" pos="0.72 0.20 0.025" '
               'material="grey" contype="0" conaffinity="0"/>')


class HillDecoy(RollingHill):
    """Predictive Hill with a decoy hump; workbench appearance."""

    cam = "pvr_side"
    style = "workbench"
    BASE_X = 0.10             # main hump spans [BASE_X, BASE_X + HILL_W]
    START_X = -0.25           # 0.35 m of flat approach (conditioning ends at x = 0.00)
    HF_HALF = 0.85
    NC = 1546                 # 1.1 mm cells across 1.70 m (same cell size as Phase A)
    DECOY_X0 = -0.78          # decoy hump spans [DECOY_X0, DECOY_X0 + HILL_W]
    CAM_XML = '<camera name="pvr_side" fovy="26" pos="0 -2.05 0.30" xyaxes="1 0 0 0 0.1 0.995"/>'
    WALL_X0, WALL_X1, WALL_H = 0.10, 0.16, 0.14
    PAIR_STIFF, PAIR_DAMP = 30000, 0.2          # near-elastic stopper pair (as WallBounce)

    def __init__(self, p0=None):
        self.p = p0 or self.params_for_S(1.2)
        self.model = None
        self.data = None
        self._r = None

    # ---- parameters -------------------------------------------------------
    @classmethod
    def params_for_S(cls, S, h_d=0.0, variant="hump"):
        return dict(v0=cls.V0, h=float(0.7 * cls.V0 ** 2 / (G * S)), h_d=float(h_d), variant=variant)

    @classmethod
    def params_for_h(cls, h, h_d=0.0, variant="hump"):
        return dict(v0=cls.V0, h=float(h), h_d=float(h_d), variant=variant)

    @staticmethod
    def S_of(p):
        if p.get("variant", "hump") != "hump" or float(p.get("h", 0.0)) <= 0.0:
            return float("nan")
        return 0.7 * p["v0"] ** 2 / (G * p["h"])

    @classmethod
    def margin_of(cls, p):
        return cls.S_of(p) - 1.0

    # ---- geometry ---------------------------------------------------------
    def _heights(self):
        variant = self.p.get("variant", "hump")
        h = float(self.p.get("h", 0.0)) if variant == "hump" else 0.0
        hd = float(self.p.get("h_d", 0.0))
        return h, hd, max(h, hd, 1e-3)

    def _profile(self):
        h, hd, top = self._heights()
        x = np.linspace(-self.HF_HALF, self.HF_HALF, self.NC)
        y = np.zeros_like(x)
        main = (x >= self.BASE_X) & (x <= self.BASE_X + self.HILL_W)
        y[main] = (h / top) * np.sin(np.pi * (x[main] - self.BASE_X) / self.HILL_W) ** 2
        dec = (x >= self.DECOY_X0) & (x <= self.DECOY_X0 + self.HILL_W)
        y[dec] = (hd / top) * np.sin(np.pi * (x[dec] - self.DECOY_X0) / self.HILL_W) ** 2
        return np.tile(y, (2, 1))

    def xml(self):
        h, hd, top = self._heights()
        variant = self.p.get("variant", "hump")
        extra = (f'<hfield name="hillhf" nrow="2" ncol="{self.NC}" '
                 f'size="{self.HF_HALF} 0.14 {top:.5f} 0.001"/>')
        wall = ""
        if variant == "wall":
            cx = 0.5 * (self.WALL_X0 + self.WALL_X1)
            hx = 0.5 * (self.WALL_X1 - self.WALL_X0)
            hz = 0.5 * self.WALL_H
            wall = (f'<geom name="wall" type="box" size="{hx:.3f} 0.12 {hz:.3f}" '
                    f'pos="{cx:.3f} 0 {hz:.3f}" material="steel"/>')
        body = f"""
    {self.CAM_XML}
    {SCENERY_PVR}
    {tick_marks(-0.8, 0.8)}
    <geom name="hillg" type="hfield" hfield="hillhf" pos="0 0 0.001" material="cover"/>
    {wall}
    <body name="ball" pos="{self.START_X} 0 {self.BALL_R + 0.0015}">
      <freejoint/>
      <geom name="ballg" type="sphere" size="{self.BALL_R}" material="red" mass="0.1"/>
    </body>
"""
        contact = """
  <contact>
    <pair geom1="ballg" geom2="floor" condim="3" friction="0.8 0.8 0.0001 0.00002 0.00002"
          solref="0.0005 1" solimp="0.995 0.999 0.0003"/>
    <pair geom1="ballg" geom2="hillg" condim="3" friction="0.8 0.8 0.0001 0.00002 0.00002"
          solref="0.0005 1" solimp="0.995 0.999 0.0003"/>
"""
        if variant == "wall":
            contact += (f'    <pair geom1="ballg" geom2="wall" condim="1" '
                        f'solref="-{self.PAIR_STIFF} -{self.PAIR_DAMP:.2f}" solimp="0.95 0.95 0.001"/>\n')
        contact += "  </contact>\n"
        return wrap_wb("hill_decoy", body, contact, extra_asset=extra, tape_half=0.85).replace(
            'timestep="0.001"', 'timestep="0.000125"')

    # ---- labels -----------------------------------------------------------
    def labels(self, p, trace):
        variant = p.get("variant", "hump")
        if variant != "wall":
            out = RollingHill.labels(self, p, trace)
        else:
            x, vx = trace[:, 0], trace[:, 2]
            turned = vx < 0.0
            event = int(np.argmax(turned)) if turned.any() else self.n_frames - 1
            speed, spin = trace[:, 3], trace[:, 4]
            slip = np.abs(speed - spin * self.BALL_R) / np.maximum(speed, 1e-3)
            energy = trace[:, 5]
            drift = float((energy[min(event + 2, len(energy) - 1)] - energy[0]) / max(energy[0], 1e-9))
            out = dict(margin=float("nan"), outcome=0, event_frame=event,
                       t_event_s=event * self.capture_dt, decided=bool(turned.any()),
                       slip_max=float(slip[: event + 1].max()), energy_drift=drift,
                       aux=float(x.max()), cond_end_x=float(x[min(4, len(x) - 1)]))
        x = trace[:, 0]
        out["x_min_horizon"] = float(x[:21].min())
        reach = self.DECOY_X0 + self.HILL_W + self.BALL_R
        touch = np.nonzero(x[:21] < reach)[0] if float(p.get("h_d", 0.0)) > 0.0 else []
        out["decoy_contact_frame"] = int(touch[0]) if len(touch) else -1
        out["decoy_clear"] = bool(out["decoy_contact_frame"] < 0)
        out["variant"] = variant
        return out

    @staticmethod
    def sample(rng):
        return HillDecoy.params_for_S(float(rng.uniform(0.5, 2.0)), h_d=0.036)
