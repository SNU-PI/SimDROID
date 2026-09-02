"""Adjudicate VERA DROID-planner rollouts on the Phase A canvases.

The canvases are time-stretched (100 Hz capture played at 15 fps) and the side
view is 192x128.  The frozen Bundle A adjudicators are resolution-agnostic
(all thresholds are in ball diameters) but were validated at 16 fps, so both
ground truth and generations are resampled to 16 fps-equivalent frames
(index = round(k * 100 / 16)) before adjudication.  The hill crest column is
measured from the green cable-cover mask of the GT side view.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import exp.analyze_bundle_a as A
from exp.analyze_bundle_a import adjudicate_kin, centroid_and_area, diameter_px, interp_gaps, kind_of, track

_ORIG_MASK = A.color_mask


def scaled_mask(image, color):
    """Bundle A colour masks with the blue sky-band exclusion scaled to the frame height."""
    x = np.asarray(image, dtype=np.float32)
    if color == "blue":
        r, g, b = x[..., 0], x[..., 1], x[..., 2]
        m = (b > 90) & (b > 1.35 * r) & (b - np.minimum(r, g) > 40)
        m[: int(140 * x.shape[0] / 480)] = False
        return m
    return _ORIG_MASK(image, color)


A.color_mask = scaled_mask          # track/diameter_px/centroid_and_area resolve it at call time


def resample16(frames, hz=100.0, fps=16.0):
    idx = np.round(np.arange(0, len(frames) * fps / hz) * hz / fps).astype(int)
    idx = idx[idx < len(frames)]
    return np.asarray(frames)[idx]


def green_cols(frame):
    x = np.asarray(frame, dtype=np.float32)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    m = (g > 90) & (g > 1.3 * r) & (g > 1.3 * b)
    cols = np.nonzero(m.any(axis=0))[0]
    return (int(cols.min()), int(cols.max())) if len(cols) else None


def reference(rec, gt_side):
    """Mirror of analyze_bundle_a.gt_reference for the 192x128 side view."""
    frames = resample16(gt_side)
    pts, areas = track(frames, "red")
    ref = {"n_frames": len(frames), "red_area": areas[0], "dia_red": diameter_px(frames[0], "red")}
    kind = kind_of(rec["family"])
    if kind == "hill_roll":
        cols = green_cols(frames[0])
        ref["crest_x"] = float(np.mean(cols)) if cols else 96.0
        ref["scale"] = ref["dia_red"] / 0.06
    elif kind == "two_ball":
        ref["dia_blue"] = diameter_px(frames[0], "blue")
        ref["blue_area"] = centroid_and_area(scaled_mask(frames[0], "blue"))[1]
        ref["scale"] = ref["dia_red"] / 0.08
    elif kind == "kin_roll":
        ref["scale"] = ref["dia_red"] / 0.06
    ref["gt_traj"] = interp_gaps(pts)
    ref["gt_frames"] = frames
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, default=Path("artifacts/phase_a/vera"))
    ap.add_argument("--rollouts", type=Path, default=Path("artifacts/phase_a/vera_out"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/phase_a/analysis_vera"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_meta = [json.loads(l) for l in (args.inputs / "manifest_vera.jsonl").read_text().splitlines() if l.strip()]
    rows, selftest = [], []
    for rec in rows_meta:
        z = np.load(args.inputs / f"{rec['id']}.npz")
        gt_side = z["side_view"]
        ref = reference(rec, gt_side)
        # the analyzer's blue mask ignores rows < 140 (sky band at 480p); at 128 px that kills the mask
        res_gt = adjudicate_lowres(rec, ref, ref["gt_frames"])
        selftest.append({"id": rec["id"], "gt_outcome": rec["outcome"], "adjudicated": res_gt["prediction"],
                         "match": res_gt["prediction"] == rec["outcome"]})
        for seed in args.seeds:
            p = args.rollouts / f"seed_{seed:02d}" / rec["id"] / "rollout.npz"
            if not p.exists():
                continue
            gen = np.load(p)["side_view"]
            res = adjudicate_lowres(rec, ref, resample16(gen))
            rows.append({"id": rec["id"], "family": rec["family"], "role": rec["role"], "S": rec["S"],
                         "v0": rec.get("v0"), "outcome": rec["outcome"], "seed": seed,
                         "prediction": res["prediction"], "correct": int(res["prediction"] == rec["outcome"]),
                         "valid": int(bool(res["valid"])), "valid_frac": res["valid_frac"],
                         "speed_ratio": res.get("speed_ratio"), "gt_err_px": res.get("gt_err_px"),
                         "momentum_ratio": res.get("momentum_ratio"), "law_slope": res.get("law_slope")})
    n_match = sum(s["match"] for s in selftest)
    print(f"GT self-test (192x128, 16fps-resampled): {n_match}/{len(selftest)}")
    for s in selftest:
        if not s["match"]:
            print("  MISMATCH:", s)
    (args.output_dir / "gt_selftest.json").write_text(json.dumps(selftest, indent=2))
    if rows:
        with (args.output_dir / "samples.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        summary = {}
        for fam in sorted({r["family"] for r in rows}):
            fr = [r for r in rows if r["family"] == fam]
            summary[fam] = {"n": len(fr), "accuracy": float(np.mean([r["correct"] for r in fr])),
                            "undecided": float(np.mean([r["prediction"] == -1 for r in fr])),
                            "valid_rate": float(np.mean([r["valid"] for r in fr])),
                            "valid_frac": float(np.mean([r["valid_frac"] for r in fr])),
                            "per_sample": [{"id": r["id"], "S": r["S"], "v0": r["v0"], "gt": r["outcome"],
                                            "pred": r["prediction"], "valid_frac": round(r["valid_frac"], 2)} for r in fr]}
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_sample"} for k, v in summary.items()}, indent=2))
        for fam, v in summary.items():
            print(fam, " ".join(f"{s['id'].split('_')[-1]}:{s['gt']}->{s['pred']}" for s in v["per_sample"]))


def adjudicate_lowres(rec, ref, frames):
    return A.adjudicate(rec, ref, frames)


if __name__ == "__main__":
    main()
