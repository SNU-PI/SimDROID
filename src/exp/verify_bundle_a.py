"""Pre-Cosmos MuJoCo verification for Bundle A (spec section 2.5 / 6).

Physics only, no rendering: grid gates, boundary bisection against the
closed form, energy audit, restitution/momentum audit, nuisance-parameter
outcome invariance, and the t_event map-inclusion table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core.threshold.rolling_hill import RollingHill
from core.threshold.two_ball import TwoBall, WallBounce
from core.threshold.rod_pendulum import RodPendulum
from gen.bundle_a_spec import (BOUNDARY_TOL, BUNDLE_A, ROLLOUT_FRAMES, S_GRID,
                               in_map)


def run_S(cls, S, **overrides):
    scene = cls()
    p = cls.params_for_S(S)
    p.update(overrides)
    return scene.run(p, render=False), p


def grid_table(name, cls):
    rows = []
    for S in S_GRID:
        out, p = run_S(cls, S)
        row = {"S": S, "outcome": int(out["outcome"]),
               "expected": int(S > 1.0), "event_frame": int(out["event_frame"]),
               "decided": bool(out["decided"]), "in_map": in_map(out),
               "boundary": abs(S - 1.0) <= BOUNDARY_TOL}
        for key in ("slip_max", "energy_drift", "e_eff", "momentum_ratio",
                    "ke_ratio", "bottom_frame", "read_frame", "aux"):
            if key in out:
                row[key] = float(out[key]) if not isinstance(out[key], bool) else out[key]
        row["gate_outcome"] = row["boundary"] or (row["outcome"] == row["expected"])
        row["gate_leak"] = row["event_frame"] > 4
        rows.append(row)
    return rows


def bisect_threshold(cls, key, lo_S, hi_S, tol=1e-3):
    """Bisect S until the simulated outcome flips; returns S* (sim boundary)."""
    def outcome_at(S):
        out, _ = run_S(cls, S)
        if not out["decided"]:
            # Undecided within the sim window counts as "not crossed".
            return 0
        return int(out["outcome"])

    lo, hi = lo_S, hi_S
    if outcome_at(lo) != 0 or outcome_at(hi) != 1:
        return {"error": f"bracket invalid for {key}", "lo": lo, "hi": hi}
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if outcome_at(mid) == 1:
            hi = mid
        else:
            lo = mid
    s_star = 0.5 * (lo + hi)
    return {"S_star_sim": s_star, "gap_percent": 100.0 * abs(s_star - 1.0),
            "grid_cell_percent": 5.0}


def nuisance_table(name, cls):
    """Outcome must be invariant to non-decision parameters near the boundary."""
    probes = [0.89, 1.12]
    cases = []
    for S in probes:
        base_out, p = run_S(cls, S)
        variations = {}
        if name == "hill_roll":
            for mu in (0.6, 1.0):
                variations[f"mu={mu}"] = _hill_with_mu(mu, p)
            variations["density=x2"] = _mass_scaled(cls, p, 2.0)
        elif name == "two_ball":
            for damp in (1.0, 5.0):
                variations[f"pair_damp={damp}"] = _twoball_with_damp(damp, p)
            variations["density=x2"] = _mass_scaled(cls, p, 2.0)
        elif name == "pendulum_rod":
            variations["density=x2"] = _mass_scaled(cls, p, 2.0)
            variations["q_start=0.45"] = _pendulum_with_q(0.45, p)
        cases.append({"S": S, "base_outcome": int(base_out["outcome"]),
                      "variations": {k: int(v) for k, v in variations.items()},
                      "invariant": all(int(v) == int(base_out["outcome"])
                                       for v in variations.values())})
    return cases


def _hill_with_mu(mu, p):
    class HillMu(RollingHill):
        def xml(self):
            return super().xml().replace('friction="0.8 0.8', f'friction="{mu} {mu}')
    out = HillMu().run(dict(p), render=False)
    return out["outcome"]


def _twoball_with_damp(damp, p):
    class BallDamp(TwoBall):
        PAIR_DAMP = damp
    out = BallDamp().run(dict(p), render=False)
    return out["outcome"]


def _pendulum_with_q(q, p):
    from core.threshold.rod_pendulum import RodPendulum as RP
    S = RP.S_of(p)
    q_eff = min(q, RP.Q_CAP / S)
    theta0 = float(np.arccos(np.clip(1.0 - 2.0 * q_eff * S, -1.0, 1.0)))
    p2 = dict(p, theta0=theta0, q=q_eff)
    out = RP().run(p2, render=False)
    return out["outcome"]


def _mass_scaled(cls, p, factor):
    class Scaled(cls):
        def xml(self):
            xml = super().xml()
            # double every explicit mass attribute
            import re
            return re.sub(r'mass="([0-9.eE+-]+)"',
                          lambda m: f'mass="{float(m.group(1)) * factor:.8f}"',
                          xml)
    out = Scaled().run(dict(p), render=False)
    return out["outcome"]


def timestep_check(name, cls):
    rows = []
    for S in (0.89, 1.12):
        class Fine(cls):
            def xml(self):
                return super().xml().replace('timestep="0.001"', 'timestep="0.0005"')
        base, p = run_S(cls, S)
        fine = Fine().run(dict(p), render=False)
        rows.append({"S": S, "outcome_dt1ms": int(base["outcome"]),
                     "outcome_dt05ms": int(fine["outcome"]),
                     "match": int(base["outcome"]) == int(fine["outcome"])})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/bundle_a/verify.json"))
    args = parser.parse_args()

    report = {}
    for name, cls in BUNDLE_A.items():
        grid = grid_table(name, cls)
        bisect = bisect_threshold(cls, name, 0.80, 1.25)
        report[name] = {
            "grid": grid,
            "bisection": bisect,
            "nuisance": nuisance_table(name, cls),
            "timestep": timestep_check(name, cls),
            "gates_pass": all(r["gate_outcome"] and r["gate_leak"] for r in grid),
        }
    pre_hill, _ = run_S(RollingHill, 0.35)
    wall = WallBounce().run(dict(v0=1.0), render=False)
    report["prechecks"] = {
        "hill_roll_S0.35": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                            for k, v in pre_hill.items() if k not in ("frames", "trace")},
        "wall_bounce": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                        for k, v in wall.items() if k not in ("frames", "trace")},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str))
    for name, item in report.items():
        if name == "prechecks":
            continue
        print(f"\n=== {name} (gates_pass={item['gates_pass']}) "
              f"bisect S*={item['bisection'].get('S_star_sim', 'ERR')}")
        for r in item["grid"]:
            print(f"  S={r['S']:.2f} y={r['outcome']} exp={r['expected']} "
                  f"ev={r['event_frame']:2d} map={int(r['in_map'])} "
                  + " ".join(f"{k}={r[k]:.4f}" for k in
                             ("slip_max", "energy_drift", "e_eff", "momentum_ratio")
                             if k in r))
    print("\nprechecks:", json.dumps(report["prechecks"], default=str)[:400])


if __name__ == "__main__":
    main()
