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


def cosv(a, b, mask=None):
    """Cosine between two (N, D) latent displacements, optionally restricted to masked tokens."""
    a = a.astype(np.float32); b = b.astype(np.float32)
    if mask is not None and mask.any():
        a, b = a[mask], b[mask]
    a, b = a.reshape(-1), b.reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


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
    ap.add_argument("--ctx-tag", default="", help="latent file suffix, e.g. c3 for 3-frame-context rollouts")
    args = ap.parse_args()
    root = Path(args.root)
    tag = f"_{args.ctx_tag}" if args.ctx_tag else ""
    out = root / ("analysis" + tag)
    out.mkdir(exist_ok=True)
    rows = [json.loads(l) for l in (root / "manifest.jsonl").read_text().splitlines() if l.strip()]
    recs = []
    for row in rows:
        sid = row["id"]
        inp = np.load(root / row["vj_input"])
        for oi, o in enumerate(inp["offsets"]):
            f = root / "latents" / f"{sid}_o{int(o)}{tag}.npz"
            if not f.exists():
                continue
            d = np.load(f)
            z_last = d["z_ctx"][-1]
            tidx = d["target_idx"]
            kin_off = int(d["kin_offset"]) if "kin_offset" in d.files else 0   # index shift into frames_kin
            for k in range(int(d["z_pred"].shape[0])):
                zp, zg, zk, zj = d["z_pred"][k], d["z_gt"][k], d["z_kin"][k], d["z_jit"][k]
                fg, fk = inp["frames_gt"][int(tidx[k])], inp["frames_kin"][oi][k + kin_off]
                m = patch_mask(fg, fk)
                dP, dK = l1(zp, zg), l1(zp, zk)
                sep, floor = l1(zg, zk), l1(zg, zj)
                mot = l1(zg, z_last)
                rec = {
                    "id": sid, "family": row["family"], "kind": str(d["kind"]), "role": row.get("role"),
                    "S": row["S"], "outcome": int(d["outcome"]), "boundary": row.get("boundary", False),
                    "offset": int(o), "ctx_ok": bool(d["ctx_ok"]), "step": k + 1 + kin_off,
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
                # Displacement direction: does the predicted change from the last context frame
                # point toward the physics future or the kinematic one?  Immune to the freeze
                # confound (freeze has zero displacement) and to the common drift magnitude.
                d_hat = zp.astype(np.float32) - z_last.astype(np.float32)
                d_P = zg.astype(np.float32) - z_last.astype(np.float32)
                d_K = zk.astype(np.float32) - z_last.astype(np.float32)
                rec.update({
                    "cos_P": cosv(d_hat, d_P), "cos_K": cosv(d_hat, d_K), "cos_PK": cosv(d_P, d_K),
                    "dcos": cosv(d_hat, d_P) - cosv(d_hat, d_K),
                    "dcos_oracle_P": 1.0 - cosv(d_P, d_K), "dcos_oracle_K": cosv(d_P, d_K) - 1.0,
                    "disp_ratio": float(np.abs(d_hat).mean() / max(np.abs(d_P).mean(), 1e-8)),
                    "cos_P_local": cosv(d_hat, d_P, m) if m.any() else float("nan"),
                    "cos_K_local": cosv(d_hat, d_K, m) if m.any() else float("nan"),
                    "dcos_local": (cosv(d_hat, d_P, m) - cosv(d_hat, d_K, m)) if m.any() else float("nan"),
                })
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
            "cos_P_mean": float(np.nanmean([r["cos_P"] for r in sel])),
            "cos_K_mean": float(np.nanmean([r["cos_K"] for r in sel])),
            "cos_PK_mean": float(np.nanmean([r["cos_PK"] for r in sel])),
            "dcos_mean": float(np.nanmean([r["dcos"] for r in sel])),
            "dcos_oracle_P_mean": float(np.nanmean([r["dcos_oracle_P"] for r in sel])),
            "p_dcos_pos": float(np.mean([r["dcos"] > 0 for r in sel if not math.isnan(r["dcos"])])) if any(not math.isnan(r["dcos"]) for r in sel) else float("nan"),
            "dcos_local_mean": float(np.nanmean([r["dcos_local"] for r in sel])) if any(not math.isnan(r["dcos_local"]) for r in sel) else float("nan"),
            "p_dcos_local_pos": float(np.mean([r["dcos_local"] > 0 for r in sel if not math.isnan(r["dcos_local"])])) if any(not math.isnan(r["dcos_local"]) for r in sel) else float("nan"),
            "disp_ratio_mean": float(np.nanmean([r["disp_ratio"] for r in sel])),
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
            fig, axes = plt.subplots(3, len(fams), figsize=(5.2 * len(fams), 11.0), squeeze=False)
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
                # Row 3: displacement direction (steps 1-3 pooled, gated): dcos vs the oracle scale.
                ax = axes[2, j]
                fr = [r for r in recs if r["family"] == fam and r["ctx_ok"] and r["gate"] and r["step"] <= 3 and r["S"] is not None]
                Sd = sorted({r["S"] for r in fr})
                dc = [float(np.nanmean([r["dcos"] for r in fr if r["S"] == s])) for s in Sd]
                dl = [float(np.nanmean([r["dcos_local"] for r in fr if r["S"] == s])) for s in Sd]
                orc = [float(np.nanmean([r["dcos_oracle_P"] for r in fr if r["S"] == s])) for s in Sd]
                ax.fill_between(Sd, [-o for o in orc], orc, color="#bdc3c7", alpha=0.35, label="oracle range (physics / kinematic)")
                ax.axhline(0, color="k", lw=0.8)
                ax.plot(Sd, dc, "o-", color="#c0392b", label="predictor dcos (all tokens)")
                ax.plot(Sd, dl, "^--", color="#8e44ad", label="predictor dcos (P≠K patches)")
                ax.set_xscale("log"); ax.set_xlabel("S"); ax.set_ylabel("cos(Δẑ, ΔP) − cos(Δẑ, ΔK)")
                ax.axvline(1.0, color="k", ls="--", lw=0.8); ax.set_title("direction of predicted change (steps 1–3)")
                ax.legend(fontsize=7); ax.grid(alpha=0.3)
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
    print("\n# displacement direction (ctx_ok rows): step | n gate% | cos_P cos_K dcos P(dcos>0) | local dcos P(>0) | disp_ratio | cos_PK")
    for fam in sorted({r["family"] for r in recs}):
        fr = [r for r in recs if r["family"] == fam and r["ctx_ok"]]
        print(f"== {fam}")
        for k in sorted({r["step"] for r in fr}):
            s = [r for r in fr if r["step"] == k]
            g = [r for r in s if r["gate"]] or s
            a = agg(g)
            print(f"  {k} | {len(s):3d} {100 * len([r for r in s if r['gate']]) / max(1, len(s)):4.0f}% | "
                  f"{a['cos_P_mean']:+.3f} {a['cos_K_mean']:+.3f} {a['dcos_mean']:+.3f} {a['p_dcos_pos']:.2f} | "
                  f"{a['dcos_local_mean']:+.3f} {a['p_dcos_local_pos']:.2f} | {a['disp_ratio_mean']:.2f} | {a['cos_PK_mean']:+.2f} | oracle ±{a['dcos_oracle_P_mean']:.2f}")
        Ss = sorted({r["S"] for r in fr if r["S"] is not None})
        if Ss:
            print("  by S (steps 1-3, gated): S n dcos P(>0) | local dcos P(>0) | gt")
            for S in Ss:
                g = [r for r in fr if r["S"] == S and r["step"] <= 3 and r["gate"]]
                if not g:
                    continue
                a = agg(g)
                print(f"   {S:5.2f} {len(g):2d} {a['dcos_mean']:+.3f} {a['p_dcos_pos']:.2f} | {a['dcos_local_mean']:+.3f} {a['p_dcos_local_pos']:.2f} | {g[0]['outcome']}")


if __name__ == "__main__":
    main()
