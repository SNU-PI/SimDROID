"""Pixel-space readout of decoded DiLA predictions: is the red ball still there, and where.

For every (sample, offset, 4 fps target step):
  area_ratio  red-mask area of the decoded prediction / area in the decoded last context frame
  present     area_ratio >= 0.25 (the ball survived the rollout+decode)
  err_P/err_K |x_pred - x_physics| / |x_pred - x_kinematic| in px (raw GT / counterfactual renders)
  closer_P    err_P < err_K, counted only where the two futures differ by > 2 px
  dir_agree   sign(x_pred - x_last) == sign(x_P - x_last), where the physics future moved > 2 px
The same quantities for the RAE reconstruction of the GT latents give the decoder ceiling
(present_rec, err_rec).  Summary per family x step -> summary.json, rows -> samples.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def red_mask(img):
    r, g, b = img[..., 0].astype(np.int16), img[..., 1].astype(np.int16), img[..., 2].astype(np.int16)
    return (r > 100) & (r - g > 45) & (r - b > 45)


def centroid(img):
    m = red_mask(img)
    n = int(m.sum())
    if n == 0:
        return 0, float("nan"), float("nan")
    ys, xs = np.nonzero(m)
    return n, float(xs.mean()), float(ys.mean())


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def _median(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.median(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ctx-tag", default="", help="latent file suffix without the leading underscore")
    args = ap.parse_args()
    root = Path(args.root)
    tag = f"_{args.ctx_tag}" if args.ctx_tag else ""
    out = root / ("pixels" + tag)
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
            if "rgb_pred" not in d.files:
                continue
            kin_off = int(d["kin_offset"]) if "kin_offset" in d.files else 0
            cidx, tidx = d["ctx_idx"], d["target_idx"]
            n_ref, x_ref, _ = centroid(d["rgb_ctx_recon"][0])
            n_last, x_last, _ = centroid(inp["frames_gt"][int(cidx[-1])])
            if n_ref == 0 or n_last == 0:
                continue
            for k in range(len(tidx)):
                fg = inp["frames_gt"][int(tidx[k])]
                fk = inp["frames_kin"][oi][k + kin_off]
                n_p, x_p, _ = centroid(fg)
                n_k, x_k, _ = centroid(fk)
                n_pred, x_pred, _ = centroid(d["rgb_pred"][k])
                n_rec, x_rec, _ = centroid(d["rgb_gt_recon"][k])
                area_ratio = n_pred / n_ref
                present = area_ratio >= 0.25
                rec = {"id": sid, "family": row["family"], "S": row["S"], "role": row.get("role"),
                       "boundary": row.get("boundary", False), "offset": int(o), "step": k + 1 + kin_off,
                       "ctx_ok": bool(d["ctx_ok"]), "gt_present": n_p > 0, "kin_present": n_k > 0,
                       "area_ratio": round(area_ratio, 3), "present": bool(present),
                       "rec_present": bool(n_rec / n_ref >= 0.25),
                       "err_rec": abs(x_rec - x_p) if (n_rec > 0 and n_p > 0) else float("nan"),
                       "err_P": abs(x_pred - x_p) if (present and n_p > 0) else float("nan"),
                       "err_K": abs(x_pred - x_k) if (present and n_k > 0) else float("nan"),
                       "closer_P": None, "dir_agree": None, "x_pred": x_pred, "x_P": x_p, "x_K": x_k, "x_last": x_last}
                if present and n_p > 0 and n_k > 0 and abs(x_p - x_k) > 2:
                    rec["closer_P"] = bool(abs(x_pred - x_p) < abs(x_pred - x_k))
                if present and n_p > 0 and abs(x_p - x_last) > 2:
                    rec["dir_agree"] = bool(np.sign(x_pred - x_last) == np.sign(x_p - x_last))
                recs.append(rec)
    with (out / "samples.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()) if recs else ["id"])
        w.writeheader()
        w.writerows(recs)
    summary = {"n_rows": len(recs), "families": {}}
    fams = sorted({r["family"] for r in recs})
    print(f"# pixel readout ({root.name}{tag}): family | step | n | present | area | err_P err_K | closer_P (n) | dir_agree (n) | rec_present err_rec")
    for fam in fams:
        summary["families"][fam] = {}
        for step in sorted({r["step"] for r in recs if r["family"] == fam}):
            sel = [r for r in recs if r["family"] == fam and r["step"] == step and r["ctx_ok"]]
            cp = [r["closer_P"] for r in sel if r["closer_P"] is not None]
            da = [r["dir_agree"] for r in sel if r["dir_agree"] is not None]
            e = {"n": len(sel), "present_rate": _mean([r["present"] for r in sel]),
                 "area_ratio_mean": _mean([r["area_ratio"] for r in sel]),
                 "err_P_median": _median([r["err_P"] for r in sel]), "err_K_median": _median([r["err_K"] for r in sel]),
                 "closer_P_rate": _mean(cp), "closer_P_n": len(cp),
                 "dir_agree_rate": _mean(da), "dir_agree_n": len(da),
                 "rec_present_rate": _mean([r["rec_present"] for r in sel]),
                 "err_rec_median": _median([r["err_rec"] for r in sel])}
            summary["families"][fam][str(step)] = e
            print(f"{fam:14s} | {step} | {e['n']:3d} | {e['present_rate']:.2f} | {e['area_ratio_mean']:.2f} | "
                  f"{e['err_P_median']:5.1f} {e['err_K_median']:5.1f} | {e['closer_P_rate']:.2f} ({e['closer_P_n']}) | "
                  f"{e['dir_agree_rate']:.2f} ({e['dir_agree_n']}) | {e['rec_present_rate']:.2f} {e['err_rec_median']:4.1f}")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
