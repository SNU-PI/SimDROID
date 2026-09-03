"""Read the V-JEPA 2-AC latent track: gate, direction score, freeze baseline, curves.

Per sample x offset x step k (L1 mean over tokens and dims, layer-normed latents):
  dP  = d(z_pred_k, z_gt_k)        dK  = d(z_pred_k, z_kin_k)
  delta      = dK - dP             (> 0: the prediction sits closer to the physics future)
  sep        = d(z_gt_k, z_kin_k)  (how far apart the two candidate futures are)
  floor      = d(z_gt_k, z_jit_k)  (encoder response to a 1 px shift of the ball)
  motion     = d(z_gt_k, z_last)   (how much the physics future differs from the last context frame)
  delta_frz  = d(z_last, z_kin_k) - d(z_last, z_gt_k)   (freeze predictor baseline)
  track      = dP / d(z_last, z_gt_k)                   (< 1: prediction moved toward the physics future)
'local' variants restrict the token mean to patches whose pixels differ between the
P and K frames (the region where the futures actually disagree).
Gate: a (sample, step) is informative when sep > 3 * floor.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from gen.vjepa_spec import N_STEPS, SIZE, STEP, FPS

PATCH = 16
GRID = SIZE // PATCH


def l1(a, b, mask=None):
    d = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=-1)   # (N,)
    if mask is not None and mask.any():
        d = d[mask]
    return float(d.mean())


def patch_mask(frame_a, frame_b, thresh=8.0):
    diff = np.abs(frame_a.astype(np.int16) - frame_b.astype(np.int16)).sum(axis=-1)  # (H, W)
    per = diff.reshape(GRID, PATCH, GRID, PATCH).mean(axis=(1, 3))                  # (GRID, GRID)
    m = (per > thresh).reshape(-1)
    return m


def _median(xs):
    return float(np.median(xs)) if len(xs) else float("nan")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--gate", type=float, default=3.0, help="sep/floor ratio for an informative step")
    args = ap.parse_args()
    root = Path(args.root)
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    rows = [json.loads(l) for l in (root / "manifest.jsonl").read_text().splitlines() if l.strip()]
    recs = []
    for row in rows:
        sid = row["id"]
        inp = np.load(root / row["vj_input"])
        for oi, o in enumerate(inp["offsets"]):
            f = root / "latents" / f"{sid}_o{int(o)}.npz"
            if not f.exists():
                continue
            d = np.load(f)
            z_last = d["z_ctx"][-1]
            tidx = d["target_idx"]
            for k in range(N_STEPS):
                zp, zg, zk, zj = d["z_pred"][k], d["z_gt"][k], d["z_kin"][k], d["z_jit"][k]
                fg, fk = inp["frames_gt"][int(tidx[k])], inp["frames_kin"][oi][k]
                m = patch_mask(fg, fk)
                dP, dK = l1(zp, zg), l1(zp, zk)
                sep, floor = l1(zg, zk), l1(zg, zj)
                mot = l1(zg, z_last)
                rec = {
                    "id": sid, "family": row["family"], "kind": str(d["kind"]), "role": row.get("role"),
                    "S": row["S"], "outcome": int(d["outcome"]), "boundary": row.get("boundary", False),
                    "offset": int(o), "ctx_ok": bool(d["ctx_ok"]), "step": k + 1,
                    "t_s": round((int(tidx[k]) - int(d["ctx_idx"][0])) / FPS, 4),
                    "dP": dP, "dK": dK, "delta": dK - dP, "sep": sep, "floor": floor, "motion": mot,
                    "delta_norm": (dK - dP) / sep if sep > 0 else float("nan"),
                    "delta_frz": l1(z_last, zk) - l1(z_last, zg),
                    "track": dP / mot if mot > 0 else float("nan"),
                    "gate": bool(floor > 0 and sep > args.gate * floor),
                    "n_local": int(m.sum()),
                    "delta_local": (l1(zp, zk, m) - l1(zp, zg, m)) if m.any() else float("nan"),
                    "sep_local": l1(zg, zk, m) if m.any() else float("nan"),
                    "floor_local": l1(zg, zj, m) if m.any() else float("nan"),
                    "delta_frz_local": (l1(z_last, zk, m) - l1(z_last, zg, m)) if m.any() else float("nan"),
                }
                if "z_pred_b" in d.files:
                    zb = d["z_pred_b"][k]
                    rec["delta_b"] = l1(zb, zk) - l1(zb, zg)
                    rec["pose_gap"] = l1(zb, zp)
                recs.append(rec)
    if not recs:
        raise SystemExit("no latents found")
    keys = list(recs[0].keys())
    with open(out / "samples.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(recs)

    def agg(sel):
        n = len(sel)
        if n == 0:
            return {"n": 0}
        pos = sum(r["delta"] > 0 for r in sel)
        beat = sum((r["delta"] - r["delta_frz"]) > 0 for r in sel)
        loc = [r for r in sel if not math.isnan(r["delta_local"])]
        posl = sum(r["delta_local"] > 0 for r in loc)
        lo, hi = wilson(pos, n)
        return {
            "n": n, "p_physics": pos / n, "p_physics_ci": [lo, hi],
            "p_beats_freeze": beat / n,
            "p_physics_local": posl / len(loc) if loc else float("nan"), "n_local": len(loc),
            "delta_norm_mean": float(np.nanmean([r["delta_norm"] for r in sel])),
            "delta_frz_norm_mean": float(np.nanmean([r["delta_frz"] / r["sep"] for r in sel if r["sep"] > 0])),
            "track_mean": float(np.nanmean([r["track"] for r in sel])),
            "sep_over_floor_median": _median([r["sep"] / r["floor"] for r in sel if r["floor"] > 0]),
            "motion_over_floor_median": _median([r["motion"] / r["floor"] for r in sel if r["floor"] > 0]),
            "floor_zero_rate": float(np.mean([r["floor"] <= 0 for r in sel])),
            "dP_over_floor_median": _median([r["dP"] / r["floor"] for r in sel if r["floor"] > 0]),
        }

    summary = {"gate": args.gate, "families": {}}
    for fam in sorted({r["family"] for r in recs}):
        fr = [r for r in recs if r["family"] == fam and r["ctx_ok"]]
        entry = {"all_steps": agg(fr), "gated": agg([r for r in fr if r["gate"]]),
                 "last_step": agg([r for r in fr if r["step"] == N_STEPS]),
                 "last_step_gated": agg([r for r in fr if r["step"] == N_STEPS and r["gate"]]),
                 "gate_rate": float(np.mean([r["gate"] for r in fr])) if fr else float("nan"),
                 "by_step": {}, "by_S": {}}
        for k in range(1, N_STEPS + 1):
            entry["by_step"][k] = agg([r for r in fr if r["step"] == k])
        S_vals = sorted({r["S"] for r in fr if r["S"] is not None})
        for S in S_vals:
            sel = [r for r in fr if r["S"] == S and r["step"] == N_STEPS]
            entry["by_S"][str(S)] = agg(sel)
            entry["by_S"][str(S)]["gt_outcome"] = sel[0]["outcome"] if sel else None
        summary["families"][fam] = entry
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    # Curves: per family, by S (last step): normalized delta for predictor vs freeze, and gate ratio.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fams = [f for f in summary["families"] if summary["families"][f]["by_S"]]
        if fams:
            fig, axes = plt.subplots(2, len(fams), figsize=(5.2 * len(fams), 7.5), squeeze=False)
            for j, fam in enumerate(fams):
                byS = summary["families"][fam]["by_S"]
                S = [float(s) for s in byS]
                ax = axes[0, j]
                ax.axhline(0, color="k", lw=0.8)
                ax.plot(S, [byS[str(s)]["delta_norm_mean"] for s in byS], "o-", color="#c0392b", label="predictor Δ/sep")
                ax.plot(S, [byS[str(s)]["delta_frz_norm_mean"] for s in byS], "s--", color="#7f8c8d", label="freeze Δ/sep")
                ax.set_xscale("log"); ax.set_xlabel("S"); ax.set_title(f"{fam} (last step, t=1.25 s)")
                ax.set_ylabel("(d_kin - d_phys) / sep"); ax.axvline(1.0, color="k", ls="--", lw=0.8)
                ax.legend(fontsize=8); ax.grid(alpha=0.3)
                ax = axes[1, j]
                ax.plot(S, [byS[str(s)]["sep_over_floor_median"] for s in byS], "o-", color="#2c3e50", label="sep/floor")
                ax.plot(S, [byS[str(s)]["motion_over_floor_median"] for s in byS], "^-", color="#16a085", label="motion/floor")
                ax.axhline(args.gate, color="r", ls=":", label=f"gate {args.gate}")
                ax.set_xscale("log"); ax.set_xlabel("S"); ax.set_ylabel("ratio to 1 px floor"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out / "curves.png", dpi=110)
    except Exception as e:  # plotting must never block the numbers
        print("plot skipped:", e)

    brief = {f: {"gate_rate": round(v["gate_rate"], 3),
                 "track_all_steps": round(v["all_steps"].get("track_mean", float("nan")), 3),
                 "motion_over_floor": round(v["all_steps"].get("motion_over_floor_median", float("nan")), 2),
                 "dP_over_floor": round(v["all_steps"].get("dP_over_floor_median", float("nan")), 2),
                 "last_gated": {k: (round(x, 3) if isinstance(x, float) else x)
                                for k, x in v["last_step_gated"].items() if k in ("n", "p_physics", "p_beats_freeze", "p_physics_local", "delta_norm_mean", "delta_frz_norm_mean", "track_mean", "sep_over_floor_median")}}
             for f, v in summary["families"].items()}
    print(json.dumps(brief, indent=1))


if __name__ == "__main__":
    main()
