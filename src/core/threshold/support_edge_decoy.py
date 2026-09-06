"""PVR Predictive Edge / Fall (timing scene): the Phase B support-edge plate (physics
unchanged), parameterised by (v0, x_edge) so that the ball reaches the plate's right
end at a chosen time t_edge = (x_edge - START_X) / v0.  The plate's LEFT end, behind the
ball, is the decoy support boundary (x_left; off-screen when absent).

Variants: "plain"; "joint" (a dark panel-joint line on the plate's front face at x_joint
while the plate itself continues to x_edge); "step" (the plate ends at x_edge and a lower
plate continues to x_lower_end, step_drop below); "tilt" (the plate is inclined downhill
by tilt_deg, so the ball accelerates and a constant-velocity extrapolation is late).
"""

import numpy as np

from core.threshold.rolling_hill import SCENERY
from core.threshold.support_edge import SupportEdge
from core.threshold.workbench import wrap_wb

PLATE_PAIR = ('<pair geom1="ballg" geom2="{g}" condim="3" friction="0.8 0.8 0.0001 0.00002 0.00002" '
              'solref="0.0005 1" solimp="0.995 0.999 0.0003"/>')


class SupportEdgeDecoy(SupportEdge):
    """Predictive Edge with a decoy (left) end; workbench appearance; timing readout."""

    cam = "pvr_side"
    style = "workbench"
    CAM_XML = '<camera name="pvr_side" fovy="26" pos="-0.30 -2.05 0.30" xyaxes="1 0 0 0 0.1 0.995"/>'
    OFFSCREEN_LEFT = -1.30

    def __init__(self, p0=None):
        self.p = p0 or self.params_for_t(0.75)
        super().__init__()

    # ---- parameters -------------------------------------------------------
    @classmethod
    def params_for_t(cls, t_edge, v0=0.35, x_left=-1.08, variant="plain", **kw):
        p = dict(v0=float(v0), x_edge=float(cls.START_X + v0 * t_edge), x_left=float(x_left),
                 variant=variant)
        p.update({k: float(v) for k, v in kw.items()})
        return p

    @classmethod
    def t_edge_of(cls, p):
        return (p["x_edge"] - cls.START_X) / p["v0"]

    # ---- geometry ---------------------------------------------------------
    def _tilt(self):
        if self.p.get("variant", "plain") != "tilt":
            return 0.0
        return float(np.radians(self.p.get("tilt_deg", 3.0)))

    def _plate_box(self):
        xl = float(self.p.get("x_left", self.OFFSCREEN_LEFT))
        xe = float(self.p["x_edge"])
        return 0.5 * (xl + xe), 0.5 * (xe - xl), 0.5 * self.PLATE_TOP

    def top_at(self, x):
        """Height of the plate's top surface at world x (tilt-aware; box rotated about y)."""
        cx, hx, hz = self._plate_box()
        th = self._tilt()
        return hz + hz * np.cos(th) - (x - cx - hz * np.sin(th)) * np.tan(th)

    def _ticks(self, x_from, x_to, step=0.1):
        geoms = []
        n = int(np.floor((x_to - x_from) / step + 1e-9))
        for i in range(n + 1):
            x = x_from + i * step
            z = self.top_at(x) + 0.018
            geoms.append(f'<geom name="ptick{i}" type="box" size="0.0035 0.0035 0.018" '
                         f'pos="{x:.3f} 0.10 {z:.4f}" material="dark" contype="0" conaffinity="0"/>')
        return "\n".join(geoms)

    def xml(self):
        p = self.p
        variant = p.get("variant", "plain")
        xl = float(p.get("x_left", self.OFFSCREEN_LEFT))
        xe = float(p["x_edge"])
        cx, hx, hz = self._plate_box()
        th = self._tilt()
        euler = f' euler="0 {th:.5f} 0"' if th else ""
        z_ball = self.top_at(self.START_X) + self.BALL_R + 0.0015
        lip = ""
        if not th:
            lip = (f'<geom name="plate_lip" type="box" size="{hx:.4f} 0.006 0.004" '
                   f'pos="{cx:.4f} {self.PLATE_HALF_Y + 0.006:.3f} {self.PLATE_TOP - 0.004:.3f}" '
                   f'material="dark" contype="0" conaffinity="0"/>')
        extra = ""
        pairs = [PLATE_PAIR.format(g="plate"), PLATE_PAIR.format(g="floor")]
        if variant == "joint":
            xj = float(p["x_joint"])
            extra += (f'<geom name="joint" type="box" size="0.004 0.002 {hz - 0.006:.4f}" '
                      f'pos="{xj:.4f} {-self.PLATE_HALF_Y - 0.002:.4f} {hz:.4f}" '
                      f'material="dark" contype="0" conaffinity="0"/>\n')
        if variant == "step":
            x2 = float(p.get("x_lower_end", 0.20))
            drop = float(p.get("step_drop", 0.05))
            hz2 = 0.5 * (self.PLATE_TOP - drop)
            cx2, hx2 = 0.5 * (xe + x2), 0.5 * (x2 - xe)
            extra += (f'<geom name="plate2" type="box" size="{hx2:.4f} {self.PLATE_HALF_Y} {hz2:.4f}" '
                      f'pos="{cx2:.4f} 0 {hz2:.4f}" material="steel"/>\n')
            pairs.append(PLATE_PAIR.format(g="plate2"))
        body = f"""
    {self.CAM_XML}
    {SCENERY}
    {self._ticks(max(xl + 0.05, -1.05), xe - 0.02)}
    <geom name="plate" type="box" size="{hx:.4f} {self.PLATE_HALF_Y} {hz:.4f}" pos="{cx:.4f} 0 {hz:.4f}"{euler} material="steel"/>
    {lip}
    {extra}
    <body name="ball" pos="{self.START_X} 0 {z_ball:.5f}">
      <freejoint/>
      <geom name="ballg" type="sphere" size="{self.BALL_R}" material="red" mass="0.1"/>
    </body>
"""
        contact = "\n  <contact>\n" + "\n".join("    " + q for q in pairs) + "\n  </contact>\n"
        return wrap_wb("support_edge_decoy", body, contact, tape_y=-0.17, tape_half=0.85).replace(
            'timestep="0.001"', 'timestep="0.000125"')

    # ---- labels -----------------------------------------------------------
    def labels(self, p, trace):
        x, z, vx = trace[:, 0], trace[:, 1], trace[:, 2]
        variant = p.get("variant", "plain")
        if variant == "tilt":
            z_ref = float(self.top_at(p["x_edge"])) + self.BALL_R      # rest height at the edge
        else:
            z_ref = self.PLATE_TOP + self.BALL_R
        fell = z < z_ref - self.BALL_R * self.DROP
        fell_h = fell[: self.HORIZON]
        past = x > p["x_edge"]
        edge_frame = int(np.argmax(past)) if past.any() else -1
        if fell_h.any():
            outcome, event = 1, int(np.argmax(fell_h))
        else:
            outcome, event = 0, self.HORIZON - 1
        pre = max(min(edge_frame if edge_frame > 0 else self.n_frames, 5), 1)
        speed_ratio = float(vx[pre - 1] / vx[0]) if vx[0] > 1e-9 else float("nan")
        return dict(margin=float(self.margin_of(p)), outcome=outcome, event_frame=event,
                    t_event_s=event * self.capture_dt, decided=True, edge_frame=edge_frame,
                    t_edge=float(self.t_edge_of(p)), speed_ratio=speed_ratio,
                    energy_drift=float(speed_ratio - 1.0), aux=float(z.min()),
                    cond_end_x=float(x[min(4, len(x) - 1)]), variant=variant,
                    x_min_horizon=float(x[:21].min()), decoy_clear=True, decoy_contact_frame=-1)

    @staticmethod
    def sample(rng):
        return SupportEdgeDecoy.params_for_t(float(rng.uniform(0.5, 1.1)))
