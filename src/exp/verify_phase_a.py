"""Phase A verification: the workbench dressing must not touch the physics.

For each (toy, workbench) scene pair this compiles both models at the same
parameters and compares everything the simulator can feel: options (timestep,
integrator, gravity, solver settings), explicit contact pairs, every geom that
can collide (type, size, position, mass, contype/conaffinity), and the height
field.  Then it runs the P0 control across its speed grid and reports the
uniform-motion audit (speed ratio, slip, energy drift), plus the outcome grid
of the two workbench threshold scenes for the map-inclusion table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import mujoco

from core.threshold.rolling_hill import RollingHill, RollingHillWB
from core.threshold.two_ball import TwoBall, TwoBallWB, WallBounce, WallBounceWB
from core.threshold.kin_roll import KinRoll
from gen.bundle_a_spec import S_GRID, BOUNDARY_TOL, in_map


def physics_signature(scene):
    m = scene.model
    sig = {"timestep": m.opt.timestep, "gravity": m.opt.gravity.tolist(),
           "integrator": int(m.opt.integrator), "cone": int(m.opt.cone),
           "impratio": m.opt.impratio, "iterations": m.opt.iterations,
           "tolerance": m.opt.tolerance, "noslip": m.opt.noslip_iterations}
    pairs = []
    for i in range(m.npair):
        pairs.append({"g1": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, m.pair_geom1[i]),
                      "g2": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, m.pair_geom2[i]),
                      "condim": int(m.pair_dim[i]), "friction": m.pair_friction[i].tolist(),
                      "solref": m.pair_solref[i].tolist(), "solimp": m.pair_solimp[i].tolist()})
    sig["pairs"] = sorted(pairs, key=lambda p: (p["g1"], p["g2"]))
    geoms = []
    for i in range(m.ngeom):
        if m.geom_contype[i] == 0 and m.geom_conaffinity[i] == 0:
            continue
        b = m.geom_bodyid[i]
        geoms.append({"name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i),
                      "type": int(m.geom_type[i]), "size": m.geom_size[i].tolist(),
                      "pos": m.geom_pos[i].tolist(), "contype": int(m.geom_contype[i]),
                      "conaffinity": int(m.geom_conaffinity[i]),
                      "body_mass": float(m.body_mass[b]), "body_inertia": m.body_inertia[b].tolist(),
                      "body_pos": m.body_pos[b].tolist()})
    sig["colliding_geoms"] = sorted(geoms, key=lambda g: g["name"])
    sig["hfield"] = m.hfield_data.tolist() if m.nhfield else None
    sig["hfield_size"] = m.hfield_size.tolist() if m.nhfield else None
    return sig


def compare(name, toy_cls, wb_cls, params):
    a, b = toy_cls(), wb_cls()
    a.set_params(params); b.set_params(params)
    sa, sb = physics_signature(a), physics_signature(b)
    same = json.dumps(sa, sort_keys=True) == json.dumps(sb, sort_keys=True)
    diff = [k for k in sa if json.dumps(sa[k], sort_keys=True) != json.dumps(sb[k], sort_keys=True)]
    print(f"{name}: physics identical = {same}" + ("" if same else f"  DIFF in {diff}"))
    return {"identical": same, "diff_keys": diff, "n_pairs": len(sa["pairs"]),
            "colliding_geoms": [g["name"] for g in sa["colliding_geoms"]]}


def main():
    out = {}
    out["hill_roll"] = compare("hill_roll", RollingHill, RollingHillWB, RollingHill.params_for_S(1.05))
    out["two_ball"] = compare("two_ball", TwoBall, TwoBallWB, TwoBall.params_for_S(1.26))
    out["wall_bounce"] = compare("wall_bounce", WallBounce, WallBounceWB, dict(v0=1.0))

    rows = []
    for v0 in KinRoll.V_GRID:
        res = KinRoll().run(dict(v0=v0), render=False)
        rows.append({"v0": v0, "speed_ratio": res["speed_ratio"], "slip_max": res["slip_max"],
                     "energy_drift": res["energy_drift"], "x_end": res["aux"], "outcome": res["outcome"]})
        print(f"kin_roll v0={v0:.2f}: speed_ratio={res['speed_ratio']:.4f} slip={res['slip_max']:.4f} "
              f"drift={res['energy_drift']:+.4f} x_end={res['aux']:+.3f}")
    out["kin_roll"] = rows

    for name, cls in (("hill_roll_wb", RollingHillWB), ("two_ball_wb", TwoBallWB)):
        grid = []
        for S in S_GRID:
            res = cls().run(cls.params_for_S(S), render=False)
            row = {"S": S, "outcome": int(res["outcome"]), "expected": int(S > 1.0),
                   "event_frame": int(res["event_frame"]), "decided": bool(res["decided"]),
                   "in_map": in_map(res), "boundary": abs(S - 1.0) <= BOUNDARY_TOL}
            row["gate_outcome"] = row["boundary"] or row["outcome"] == row["expected"]
            row["gate_leak"] = row["event_frame"] > 4
            grid.append(row)
        ok = all(r["gate_outcome"] and r["gate_leak"] and r["in_map"] for r in grid)
        print(f"{name}: grid gates all pass = {ok}; outcomes = {[r['outcome'] for r in grid]}; "
              f"events = {[r['event_frame'] for r in grid]}")
        out[name] = {"gates_pass": ok, "grid": grid}

    Path("artifacts/phase_a").mkdir(parents=True, exist_ok=True)
    Path("artifacts/phase_a/verify.json").write_text(json.dumps(out, indent=2))
    print("wrote artifacts/phase_a/verify.json")


if __name__ == "__main__":
    main()
