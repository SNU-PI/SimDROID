"""Phase B verification (no rendering).

1. pendulum_rod_wb: the workbench dressing must not touch the physics -- compare
   the simulator-visible signature of RodPendulum vs RodPendulumWB at three S.
2. support_edge_wb: outcome / event grid over S, the rolling audit before the
   edge (speed ratio), and a bisection on x_edge for the simulated boundary
   (drop event on the last judged frame) -> S_boundary_sim.  The scene's
   T_REF is right when S_boundary_sim == 1 within the boundary band (0.04).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.threshold.rod_pendulum import RodPendulum, RodPendulumWB
from core.threshold.support_edge import SupportEdge, SupportEdgeWB
from exp.verify_phase_a import physics_signature
from gen.bundle_a_spec import S_GRID, BOUNDARY_TOL, in_map


def pendulum_check():
    out = []
    for S in (0.6, 1.0, 1.6):
        p = RodPendulum.params_for_S(S)
        a, b = RodPendulum(p), RodPendulumWB(p)
        a.set_params(p); b.set_params(p)
        sa, sb = physics_signature(a), physics_signature(b)
        same = json.dumps(sa, sort_keys=True) == json.dumps(sb, sort_keys=True)
        ra, rb = a.run(p), b.run(p)
        out.append({"S": S, "identical_signature": same, "outcome_toy": ra["outcome"], "outcome_wb": rb["outcome"],
                    "event_toy": ra["event_frame"], "event_wb": rb["event_frame"],
                    "trace_max_abs_diff": float(np.abs(ra["trace"] - rb["trace"]).max())})
        print(f"pendulum S={S}: signature identical={same} outcome {ra['outcome']}/{rb['outcome']} "
              f"event {ra['event_frame']}/{rb['event_frame']} trace diff {out[-1]['trace_max_abs_diff']:.2e}")
    return out


def edge_grid(cls):
    rows = []
    for S in S_GRID:
        p = cls.params_for_S(S)
        r = cls(p).run(p)
        expected = int(S > 1.0)
        ok = abs(S - 1) <= BOUNDARY_TOL or r["outcome"] == expected
        rows.append({"S": S, "x_edge": p["x_edge"], "outcome": r["outcome"], "event_frame": r["event_frame"],
                     "edge_frame": r["edge_frame"], "speed_ratio": r["speed_ratio"], "in_map": in_map(r),
                     "gate_ok": ok, "z_min": r["aux"]})
        print(f"edge S={S:.2f}: x_edge={p['x_edge']:+.3f} y={r['outcome']} ev={r['event_frame']} "
              f"edge_fr={r['edge_frame']} speed_ratio={r['speed_ratio']:.4f} map={in_map(r)} gate={'ok' if ok else 'FAIL'}")
    return rows


def edge_bisection(cls, lo_S=0.8, hi_S=1.25, iters=14):
    """x_edge where the drop event first lands inside the judged horizon."""
    x_lo = cls.params_for_S(hi_S)["x_edge"]     # short plate -> falls (outcome 1)
    x_hi = cls.params_for_S(lo_S)["x_edge"]     # long plate -> stays (outcome 0)
    for _ in range(iters):
        mid = 0.5 * (x_lo + x_hi)
        p = dict(v0=cls.V0, x_edge=mid)
        r = cls(p).run(p)
        if r["outcome"] == 1:
            x_lo = mid
        else:
            x_hi = mid
    x_b = 0.5 * (x_lo + x_hi)
    S_b = cls.S_of(dict(v0=cls.V0, x_edge=x_b))
    t_ref_fit = (x_b - cls.START_X) / cls.V0
    print(f"edge boundary: x_edge*={x_b:+.4f} -> S_boundary_sim={S_b:.4f} (T_REF fit {t_ref_fit:.4f} s vs {cls.T_REF})")
    return {"x_edge_boundary": x_b, "S_boundary_sim": S_b, "T_REF_fit": t_ref_fit, "T_REF": cls.T_REF}


def main():
    out = {"pendulum": pendulum_check(), "support_edge": edge_grid(SupportEdgeWB),
           "support_edge_boundary": edge_bisection(SupportEdgeWB)}
    toy = SupportEdge(SupportEdge.params_for_S(1.2)); wb = SupportEdgeWB(SupportEdge.params_for_S(1.2))
    p = SupportEdge.params_for_S(1.2)
    toy.set_params(p); wb.set_params(p)
    out["support_edge_toy_vs_wb_identical"] = json.dumps(physics_signature(toy), sort_keys=True) == \
        json.dumps(physics_signature(wb), sort_keys=True)
    print("support_edge toy vs wb signature identical:", out["support_edge_toy_vs_wb_identical"])
    Path("artifacts/phase_b").mkdir(parents=True, exist_ok=True)
    Path("artifacts/phase_b/verify.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
