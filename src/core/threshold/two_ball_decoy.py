"""PVR Predictive Collide: the Phase A two-ball collision (physics unchanged) plus a decoy
blue ball resting behind the red ball, a wider side camera, and extension cells.

Layout (x, metres): decoy ball at -0.60, red ball start -0.47 (v0 = 1 m/s), target at 0.
Variants: "ball" (target of radius r2, S = (r2/r1)^3), "wall" (steel stopper instead of
the target, S -> inf), "miss" (target shifted to y_target so the red ball passes it
without contact), "none" (no target).  The decoy has radius r_d (0 = absent) and is a
real collider that the red ball never reaches inside the horizon (generator gate).
"""

import numpy as np
import mujoco

from core.threshold.base import Base
from core.threshold.rolling_hill import tick_marks
from core.threshold.two_ball import TwoBall, sphere_mass
from core.threshold.workbench import wrap_wb

SCENERY_PVR2 = ('<geom name="refcube" type="box" size="0.025 0.025 0.025" pos="-0.70 0.16 0.025" '
                'material="grey" contype="0" conaffinity="0"/>')
FLOOR_PAIR = ('<pair geom1="{g}" geom2="floor" condim="3" friction="0.0002 0.0002 0.0001 0.0001 0.0001" '
              'solref="0.006 1" solimp="0.95 0.95 0.001"/>')


class TwoBallDecoy(TwoBall):
    """Predictive Collide with a decoy ball; workbench appearance."""

    cam = "pvr2_side"
    style = "workbench"
    DECOY_X = -0.60
    CAM_XML = '<camera name="pvr2_side" fovy="26" pos="-0.15 -1.5 0.24" xyaxes="1 0 0 0 0.1 0.995"/>'

    def __init__(self, p0=None):
        self.p = p0 or self.params_for_S(1.2)
        Base.__init__(self)

    # ---- parameters -------------------------------------------------------
    @classmethod
    def params_for_S(cls, S, r_d=0.0, variant="ball", y_target=0.0):
        return dict(r2=float(cls.R1 * S ** (1.0 / 3.0)), v0=cls.V0, r_d=float(r_d),
                    variant=variant, y_target=float(y_target))

    @classmethod
    def S_of(cls, p):
        if p.get("variant", "ball") not in ("ball", "miss"):
            return float("nan")
        return (p["r2"] / cls.R1) ** 3

    # ---- geometry ---------------------------------------------------------
    def _elastic_pair(self, g1, g2):
        return (f'<pair geom1="{g1}" geom2="{g2}" condim="1" '
                f'solref="-{self.PAIR_STIFF} -{self.PAIR_DAMP:.2f}" solimp="0.95 0.95 0.001"/>')

    def xml(self):
        p = self.p
        variant = p.get("variant", "ball")
        r2 = float(p.get("r2", self.R1))
        r_d = float(p.get("r_d", 0.0))
        y_b = float(p.get("y_target", 0.0))
        m1 = sphere_mass(self.R1)
        body = f"""
    {self.CAM_XML}
    {SCENERY_PVR2}
    {tick_marks(-0.7, 0.4, y=0.13)}
    <body name="ballA" pos="{self.START_X} 0 {self.R1 + 0.001}">
      <freejoint/>
      <geom name="gA" type="sphere" size="{self.R1}" material="red" mass="{m1:.6f}"/>
    </body>
"""
        pairs = [FLOOR_PAIR.format(g="gA")]
        if variant in ("ball", "miss"):
            m2 = sphere_mass(r2)
            body += f"""
    <body name="ballB" pos="0 {y_b:.3f} {r2 + 0.001:.6f}">
      <freejoint/>
      <geom name="gB" type="sphere" size="{r2:.6f}" material="blue" mass="{m2:.6f}"/>
    </body>
"""
            pairs += [FLOOR_PAIR.format(g="gB"), self._elastic_pair("gA", "gB")]
        elif variant == "wall":
            body += '    <geom name="wall" type="box" size="0.02 0.12 0.10" pos="0.02 0 0.10" material="steel"/>\n'
            pairs.append(self._elastic_pair("gA", "wall"))
        if r_d > 0.0:
            md = sphere_mass(r_d)
            body += f"""
    <body name="ballD" pos="{self.DECOY_X} 0 {r_d + 0.001:.6f}">
      <freejoint/>
      <geom name="gD" type="sphere" size="{r_d:.6f}" material="blue" mass="{md:.6f}"/>
    </body>
"""
            pairs += [FLOOR_PAIR.format(g="gD"), self._elastic_pair("gA", "gD")]
        contact = "\n  <contact>\n" + "\n".join("    " + q for q in pairs) + "\n  </contact>\n"
        return wrap_wb("two_ball_decoy", body, contact, tape_y=0.13, tape_half=0.60).replace(
            'timestep="0.001"', 'timestep="0.0005"')

    # ---- state ------------------------------------------------------------
    def _addr(self, body_name):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            return None
        j = int(self.model.body_jntadr[bid])
        return int(self.model.jnt_qposadr[j]), int(self.model.jnt_dofadr[j])

    def observe(self):
        qa, da = self._addr("ballA")
        out = [float(self.data.qpos[qa]), float(self.data.qvel[da])]
        b = self._addr("ballB")
        if b is None:
            out += [0.0, 0.0]
        else:
            out += [float(self.data.qpos[b[0]]), float(self.data.qvel[b[1]])]
        return out

    # ---- labels -----------------------------------------------------------
    def labels(self, p, trace):
        variant = p.get("variant", "ball")
        xA, vA = trace[:, 0], trace[:, 1]
        if variant == "ball":
            out = TwoBall.labels(self, p, trace)
        elif variant == "wall":
            hit = vA < 0.5 * p["v0"]
            contact = int(np.argmax(hit)) if hit.any() else self.n_frames - 1
            read = min(contact + 3, self.n_frames - 1)
            out = dict(margin=float("nan"), outcome=int(vA[read] < -0.005), event_frame=contact,
                       t_event_s=contact * self.capture_dt, decided=bool(hit.any()), read_frame=read,
                       e_eff=float(-vA[read] / p["v0"]), momentum_ratio=float("nan"),
                       ke_ratio=float(vA[read] ** 2 / p["v0"] ** 2), aux=float(vA[read]))
        else:                                   # miss / none: no collision, keeps going
            passed = xA > 0.0
            contact = int(np.argmax(passed)) if passed.any() else self.n_frames - 1
            read = min(contact + 3, self.n_frames - 1)
            out = dict(margin=float("nan"), outcome=0, event_frame=contact,
                       t_event_s=contact * self.capture_dt, decided=True, read_frame=read,
                       e_eff=float("nan"), momentum_ratio=float("nan"),
                       ke_ratio=float(vA[read] ** 2 / p["v0"] ** 2), aux=float(vA[read]))
        r_d = float(p.get("r_d", 0.0))
        out["xA_min_horizon"] = float(xA[:21].min())
        touch = np.nonzero(xA[:21] < self.DECOY_X + r_d + self.R1)[0] if r_d > 0.0 else []
        out["decoy_contact_frame"] = int(touch[0]) if len(touch) else -1
        out["decoy_clear"] = bool(out["decoy_contact_frame"] < 0)
        out["v_post_ratio"] = float(vA[out["read_frame"]] / p["v0"])
        out["variant"] = variant
        return out

    @staticmethod
    def sample(rng):
        return TwoBallDecoy.params_for_S(float(rng.uniform(0.5, 2.0)), r_d=0.032)
