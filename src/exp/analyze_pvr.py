"""PVR PoC readout: frozen adjudicators + continuous event scores + paired counterfactual effects.

Per scene: (1) GT self-test -- the readout applied to the MuJoCo clips must reproduce every
manifest outcome (boundary cells exempt); (2) model rows -- one per condition x seed with
the binary prediction, the continuous score s_e, the event time and validity; (3) paired
effects for the design's edits (relevant vs decoy, same seeds = common random numbers) with
bootstrap CIs, the GT effect, normalised effect, direction agreement, and the Relevance
Index; (4) response curves along the decision variable; (5) result sheets.

Scores: hill  s_e = (max x - crest) / d_ball (crossing extent);
        collide s_e = v_post / v_pre of the red ball (theory (1-S)/(1+S));
        edge  s_e = fall onset frame (censored at the horizon), hover flag, drop depth.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exp.analyze_bundle_a import (adjudicate_edge, adjudicate_hill, adjudicate_two_ball, color_mask,
                                  centroid_and_area, diameter_px, fall_g_ratio, hill_energy_slope,
                                  interp_gaps, momentum_ratio, result_sheet, track, validity)
from exp.pvr_regions import CAMERAS, x_to_col
from gen.pvr_spec import EDITS, HORIZON, SEEDS

N_COND = 5
FPS = 16.0


# ---------------------------------------------------------------- blue target tracking
def blue_components(mask, gap=6, min_px=12):
    """Column runs of a mask separated by >= gap empty columns -> [(centroid xy, area)]."""
    cols = np.nonzero(mask.any(axis=0))[0]
    if len(cols) == 0:
        return []
    runs, start, prev = [], cols[0], cols[0]
    for c in cols[1:]:
        if c - prev > gap:
            runs.append((start, prev))
            start = c
        prev = c
    runs.append((start, prev))
    out = []
    for a, b in runs:
        sub = np.zeros_like(mask)
        sub[:, a:b + 1] = mask[:, a:b + 1]
        c, area = centroid_and_area(sub)
        if c is not None and area >= min_px:
            out.append((c, area))
    return out


def track_blue(frames, start_x, max_jump=90.0):
    """Track the blue component that starts nearest start_x (the target), ignoring the decoy."""
    points, areas, prev = [], [], None
    for frame in frames:
        comps = blue_components(color_mask(frame, "blue"))
        if not comps:
            points.append(None); areas.append(0); continue
        anchor = start_x if prev is None else prev[0]
        c, area = min(comps, key=lambda ca: abs(ca[0][0] - anchor))
        if prev is not None and abs(c[0] - prev[0]) > max_jump:
            points.append(None); areas.append(0); continue
        points.append(c); areas.append(area); prev = c
    return points, areas


# ---------------------------------------------------------------- references and readout
def gt_reference(record, root, scene):
    frames = imageio.mimread(root / record["gt_clip"], memtest=False)
    cam = CAMERAS[scene]
    red_pts, red_areas = track(frames, "red")
    traj = interp_gaps(red_pts)
    dia = diameter_px(frames[0], "red")
    ref = {"n_frames": len(frames), "red_area": red_areas[0], "dia_red": dia, "gt_frames": frames, "gt_traj": traj}
    p = record["params"]
    if scene == "hill":
        ref["crest_x"] = x_to_col(0.25, cam)
        ref["scale"] = dia / 0.06
    elif scene == "collide":
        ref["scale"] = dia / 0.08
        ref["target_x0"] = x_to_col(0.0, cam)
        if record["variant"] in ("ball", "miss"):
            comps = blue_components(color_mask(frames[0], "blue"))
            target = min(comps, key=lambda ca: abs(ca[0][0] - ref["target_x0"])) if comps else None
            ref["blue_area"] = target[1] if target else 0
            ref["dia_blue"] = float(2.0 * np.sqrt(ref["blue_area"] / np.pi)) if target else None
        else:
            ref["blue_area"], ref["dia_blue"] = 0, None
    elif scene == "edge":
        ref["y_ctx"] = float(np.nanmean(traj[:N_COND, 1]))
        ref["scale"] = dia / 0.06
        ref["edge_col"] = x_to_col(float(p["x_edge"]), cam)
    return ref


def turn_frame(x, dia):
    """First frame after the peak where x has dropped by 0.10 dia (a return), else None."""
    k = int(np.argmax(x))
    later = np.nonzero(x[k:] < x[k] - 0.10 * dia)[0]
    return int(k + later[0]) if len(later) else None


def adjudicate(scene, record, ref, frames):
    n = len(frames)
    dia = ref["dia_red"]
    red_pts, red_areas = track(frames, "red")
    frac, ok = validity(red_areas, ref["red_area"])
    traj = interp_gaps(red_pts)
    out = {"valid": bool(ok), "valid_frac": float(frac), "prediction": -1, "s_e": float("nan"),
           "event_gen": None}
    if traj is None:
        return out
    if scene == "hill":
        out["prediction"] = adjudicate_hill(traj, ref["crest_x"], dia)
        out["s_e"] = float((traj[N_COND:n, 0].max() - ref["crest_x"]) / dia)
        crossed = np.nonzero(traj[:, 0] > ref["crest_x"] + 0.75 * dia)[0]
        out["event_gen"] = int(crossed[0]) if len(crossed) else turn_frame(traj[:, 0], dia)
        out["law_slope"] = hill_energy_slope(traj, ref["scale"], n)
        out["x_final_dia"] = float((traj[-1, 0] - ref["crest_x"]) / dia)
    elif scene == "collide":
        v_pre = float(np.mean(np.diff(traj[: N_COND, 0])))
        if record["variant"] in ("ball", "miss") and ref.get("dia_blue"):
            blue_pts, blue_areas = track_blue(frames, ref["target_x0"])
            bfrac, bok = validity(blue_areas, ref["blue_area"])
            out["valid"] = bool(ok and bok); out["valid_frac"] = float(min(frac, bfrac))
            blue_traj = interp_gaps(blue_pts)
            pred, contact = adjudicate_two_ball(traj, blue_traj, dia, ref["dia_blue"])
            out["prediction"], out["event_gen"] = pred, contact
            if blue_traj is not None and contact is not None:
                out["momentum_ratio"] = momentum_ratio(traj, blue_traj, dia, ref["dia_blue"], contact, n)
        else:
            # wall / none / miss-without-target: contact = first frame the red ball slows below
            # 20 % of its approach speed while it is still visible (a ball leaving the frame
            # freezes the interpolated track and must not count as a stop).
            visible = [i for i, c in enumerate(red_pts) if c is not None]
            last_vis = visible[-1] if visible else 0
            vx = np.diff(traj[:, 0])
            slow = np.nonzero(vx[: max(last_vis - 1, 0)] < 0.2 * v_pre)[0]
            contact = int(slow[0]) if len(slow) else None
            out["event_gen"] = contact
            if contact is None or contact <= 1:
                out["prediction"] = -1 if record["variant"] == "wall" else 0
            else:
                read1 = min(contact + 5, n - 1)
                v_post = (traj[read1, 0] - traj[min(contact + 1, read1), 0]) / max(read1 - contact - 1, 1)
                out["prediction"] = 1 if v_post < -0.012 * dia else (0 if v_post > 0.012 * dia else -1)
        c = out["event_gen"]
        if c is not None and c > 1 and v_pre > 1e-6:
            r0, r1 = min(c, n - 2), min(c + 5, n - 1)
            v_post = (traj[r1, 0] - traj[r0, 0]) / max(r1 - r0, 1)
            out["s_e"] = float(v_post / v_pre)
        elif v_pre > 1e-6:
            # no contact drawn: read the speed where the collision would have been
            a, b = min(7, n - 2), min(12, n - 1)
            out["s_e"] = float((traj[b, 0] - traj[a, 0]) / max(b - a, 1) / v_pre)
        out["v_pre_px"] = v_pre
    elif scene == "edge":
        pred, event = adjudicate_edge(traj, ref["y_ctx"], dia)
        out["prediction"], out["event_gen"] = pred, event
        out["drop_dia"] = float((traj[-1, 1] - ref["y_ctx"]) / dia)
        past = traj[:, 0] > ref["edge_col"] + 0.5 * dia
        out["passed_edge"] = bool(past.any())
        out["hover"] = bool(past.any() and pred != 1)
        out["s_e"] = float(event) if event is not None else float(HORIZON)      # censored at the horizon
        out["fall_g_ratio"] = fall_g_ratio(traj, ref["y_ctx"], dia, ref["scale"])
    return out


# ---------------------------------------------------------------- statistics
def boot_ci(values, n_boot=4000, seed=0):
    v = np.asarray([x for x in values if x is not None and not np.isnan(x)], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), int(len(v))


def _p_event(rs):
    dec = [r for r in rs.values() if r["prediction"] >= 0]
    return float(np.mean([r["prediction"] == 1 for r in dec])) if dec else float("nan")


def paired_effect(rows_by_cond, c0, c1, key="s_e"):
    a, b = rows_by_cond.get(c0, {}), rows_by_cond.get(c1, {})
    seeds = sorted(set(a) & set(b))
    diffs = [b[s][key] - a[s][key] for s in seeds
             if b[s].get(key) is not None and a[s].get(key) is not None
             and not np.isnan(b[s][key]) and not np.isnan(a[s][key])]
    mean, lo, hi, n = boot_ci(diffs)
    sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else float("nan")
    d = mean / sd if sd and sd > 0 else float("nan")
    flips = [int(b[s]["prediction"] != a[s]["prediction"]) for s in seeds
             if b[s]["prediction"] >= 0 and a[s]["prediction"] >= 0]
    pa, pb = _p_event(a), _p_event(b)
    return {"from": c0, "to": c1, "n_pairs": n, "delta_mean": mean, "ci_lo": lo, "ci_hi": hi, "cohen_d": d,
            "flip_rate": float(np.mean(flips)) if flips else float("nan"),
            "p_event_from": pa, "p_event_to": pb, "delta_p_event": pb - pa}


def relevance_index(rows_by_cond, rel, dec, key="s_e"):
    """Per-seed |Delta relevant| - |Delta decoy|, bootstrap CI."""
    (r0, r1), (d0, d1) = rel, dec
    seeds = sorted(set(rows_by_cond.get(r0, {})) & set(rows_by_cond.get(r1, {}))
                   & set(rows_by_cond.get(d0, {})) & set(rows_by_cond.get(d1, {})))
    vals = []
    for s in seeds:
        a = abs(rows_by_cond[r1][s][key] - rows_by_cond[r0][s][key])
        b = abs(rows_by_cond[d1][s][key] - rows_by_cond[d0][s][key])
        if not (np.isnan(a) or np.isnan(b)):
            vals.append(a - b)
    mean, lo, hi, n = boot_ci(vals)
    return {"relevant": f"{r0}->{r1}", "decoy": f"{d0}->{d1}", "n": n, "ri_mean": mean, "ci_lo": lo, "ci_hi": hi}


def x_of(scene, record):
    """Decision-variable coordinate for the response curves."""
    p = record["params"]
    if scene == "hill":
        v = record.get("variant", "hump")
        return {"wall": 0.40, "flat": 0.0}.get(v, float(p.get("h", 0.0)))
    if scene == "collide":
        v = record.get("variant", "ball")
        if v == "wall":
            return 32.0
        if v == "none":
            return 0.03
        return float(record["S"]) if record["S"] is not None else float("nan")
    return float(record.get("t_edge", np.nan))


# ---------------------------------------------------------------- main
def analyze_scene(scene, root, cosmos, out, seeds, selftest_only=False):
    manifest = [json.loads(l) for l in (root / "manifest.jsonl").read_text().splitlines() if l.strip()]
    refs, selftest = {}, []
    for record in manifest:
        ref = gt_reference(record, root, scene)
        refs[record["id"]] = ref
        res = adjudicate(scene, record, ref, ref["gt_frames"])
        match = (res["prediction"] == record["outcome"]) or bool(record.get("boundary"))
        selftest.append({"id": record["id"], "cond": record["cond"], "gt_outcome": record["outcome"],
                         "adjudicated": res["prediction"], "match": bool(match), "s_e_gt": res["s_e"],
                         "event_gen_gt": res["event_gen"], "event_frame": record["event_frame"],
                         "extra": {k: v for k, v in res.items()
                                   if k in ("momentum_ratio", "law_slope", "hover", "drop_dia", "fall_g_ratio")}})
    n_match = sum(s["match"] for s in selftest)
    print(f"[{scene}] GT self-test: {n_match}/{len(selftest)} outcomes reproduced", flush=True)
    for s in selftest:
        if not s["match"]:
            print("   MISMATCH:", {k: v for k, v in s.items() if k != "extra"})
    out.mkdir(parents=True, exist_ok=True)
    (out / "gt_selftest.json").write_text(json.dumps(selftest, indent=1, default=float))
    if selftest_only:
        return {"gt_selftest": f"{n_match}/{len(selftest)}"}

    rows, by_cond = [], {}
    for record in manifest:
        ref = refs[record["id"]]
        for seed in seeds:
            clip = cosmos / f"seed_{seed:02d}" / record["id"] / "rollout.mp4"
            if not clip.exists():
                continue
            frames = imageio.mimread(clip, memtest=False)[: record["rollout_frames"]]
            res = adjudicate(scene, record, ref, frames)
            row = {"id": record["id"], "cond": record["cond"], "group": record["group"], "variant": record["variant"],
                   "x": x_of(scene, record), "S": record["S"], "t_edge": record.get("t_edge"),
                   "outcome": record["outcome"], "expected": record["expected"], "boundary": record["boundary"],
                   "event_frame": record["event_frame"], "seed": seed,
                   "prediction": res["prediction"], "correct": int(res["prediction"] == record["outcome"]),
                   "valid": int(bool(res["valid"])), "valid_frac": res["valid_frac"], "s_e": res["s_e"],
                   "event_gen": res["event_gen"]}
            for k in ("law_slope", "momentum_ratio", "hover", "drop_dia", "fall_g_ratio", "passed_edge", "x_final_dia", "v_pre_px"):
                if k in res:
                    row[k] = res[k]
            rows.append(row)
            by_cond.setdefault(record["cond"], {})[seed] = row
    if not rows:
        print(f"[{scene}] no rollouts under {cosmos}")
        return {"gt_selftest": f"{n_match}/{len(selftest)}", "rollouts": 0}

    gt_s = {s["cond"]: s["s_e_gt"] for s in selftest}
    gt_ev = {s["cond"]: s["event_gen_gt"] for s in selftest}
    conds = []
    for record in manifest:
        c = record["cond"]
        rs = list(by_cond.get(c, {}).values())
        if not rs:
            continue
        dec = [r for r in rs if r["prediction"] >= 0]
        se = [r["s_e"] for r in rs if not np.isnan(r["s_e"])]
        ev = [r["event_gen"] for r in rs if r["event_gen"] is not None]
        conds.append({"cond": c, "group": record["group"], "variant": record["variant"], "x": x_of(scene, record),
                      "S": record["S"], "t_edge": record.get("t_edge"), "outcome": record["outcome"],
                      "n": len(rs), "p_event": float(np.mean([r["prediction"] == 1 for r in dec])) if dec else float("nan"),
                      "undecided": float(np.mean([r["prediction"] == -1 for r in rs])),
                      "accuracy": float(np.mean([r["correct"] for r in rs])),
                      "valid_rate": float(np.mean([r["valid"] for r in rs])),
                      "s_e_mean": float(np.mean(se)) if se else float("nan"),
                      "s_e_sd": float(np.std(se, ddof=1)) if len(se) > 1 else float("nan"),
                      "s_e_gt": gt_s.get(c), "event_gt": gt_ev.get(c), "gt_event_frame": record["event_frame"],
                      "event_gen_mean": float(np.mean(ev)) if ev else float("nan"), "event_gen_n": len(ev),
                      "hover_rate": float(np.mean([r.get("hover", False) for r in rs])) if scene == "edge" else None})

    effects = []
    for label, c0, c1, kind in EDITS[scene]:
        e = paired_effect(by_cond, c0, c1)
        dgt = float("nan")
        if c0 in gt_s and c1 in gt_s and gt_s[c0] is not None and gt_s[c1] is not None:
            dgt = float(gt_s[c1] - gt_s[c0])
        e.update({"label": label, "kind": kind, "delta_gt": dgt})
        e["normalised"] = e["delta_mean"] / dgt if not np.isnan(dgt) and abs(dgt) > 1e-9 else float("nan")
        e["direction_agrees"] = (bool(np.sign(e["delta_mean"]) == np.sign(dgt)) if not np.isnan(e["normalised"]) else None)
        effects.append(e)
    ri = [relevance_index(by_cond, ("A", "B"), ("A", "C")), relevance_index(by_cond, ("C", "D"), ("B", "D"))]

    summary = {"scene": scene, "gt_selftest": f"{n_match}/{len(selftest)}", "rollouts": len(rows),
               "seeds": sorted({r["seed"] for r in rows}), "conditions": conds, "effects": effects,
               "relevance_index": ri,
               "overall": {"accuracy_nonboundary": float(np.mean([r["correct"] for r in rows if not r["boundary"]])),
                           "valid_rate": float(np.mean([r["valid"] for r in rows])),
                           "undecided_rate": float(np.mean([r["prediction"] == -1 for r in rows]))}}
    if scene == "edge":
        errs = [r["event_gen"] - gt_ev[r["cond"]] for r in rows
                if r["event_gen"] is not None and gt_ev.get(r["cond"]) is not None and r["outcome"] == 1]
        cens = [r for r in rows if r["outcome"] == 1 and r["event_gen"] is None]
        hold = [r for r in rows if r["outcome"] == 0]
        summary["timing"] = {"n": len(errs), "mean_err_frames": float(np.mean(errs)) if errs else float("nan"),
                             "mae_frames": float(np.mean(np.abs(errs))) if errs else float("nan"),
                             "censored_fall_cells": len(cens),
                             "false_fall_rate_hold": float(np.mean([r["prediction"] == 1 for r in hold])) if hold else float("nan"),
                             "hover_rate_fall_cells": float(np.mean([r.get("hover", False) for r in rows if r["outcome"] == 1]))}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    with (out / "samples.csv").open("w", newline="") as handle:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(handle, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    curve_plot(scene, conds, out / "curves.png")
    sheets(scene, manifest, refs, cosmos, rows, seeds, out)
    print(json.dumps({"scene": scene, "overall": summary["overall"],
                      "effects": [{k: e[k] for k in ("label", "delta_mean", "ci_lo", "ci_hi", "delta_gt", "flip_rate")} for e in effects],
                      "relevance_index": ri, "timing": summary.get("timing")}, indent=1, default=float))
    return summary


def curve_plot(scene, conds, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    main = [c for c in conds if c["group"] in ("core", "nodecoy", "ladder", "isotime", "hold")]
    ext = [c for c in conds if c["group"] == "ext"]
    for ax, key, ylabel in ((axes[0], "p_event", "P(event)"), (axes[1], "s_e_mean", "s_e (model) vs GT")):
        for grp, mk in (("core", "o"), ("nodecoy", "s"), ("ladder", "^"), ("isotime", "^"), ("hold", "v")):
            cs = [c for c in main if c["group"] == grp]
            if cs:
                ax.scatter([c["x"] for c in cs], [c[key] for c in cs], marker=mk, label=grp, s=40)
        if ext:
            ax.scatter([c["x"] for c in ext], [c[key] for c in ext], marker="x", label="ext", s=40, color="k")
        for c in main + ext:
            ax.annotate(c["cond"], (c["x"], c[key]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        if key == "s_e_mean":
            pts = [(c["x"], c["s_e_gt"]) for c in main + ext if c["s_e_gt"] is not None]
            if pts:
                ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker="_", color="0.4", s=120, label="GT")
        ax.set(xlabel={"hill": "main hump height h [m] (wall=0.40, flat=0)", "collide": "target S (log)",
                       "edge": "t_edge [s]"}[scene], ylabel=ylabel, title=f"{scene}: {ylabel}")
        if scene == "collide":
            ax.set_xscale("log")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sheets(scene, manifest, refs, cosmos, rows, seeds, out):
    entries = []
    for record in manifest:
        ref = refs[record["id"]]
        k = min(record["event_frame"], ref["n_frames"] - 1)
        tiles, marks = [np.asarray(ref["gt_frames"][k])], ""
        for seed in seeds:
            clip = cosmos / f"seed_{seed:02d}" / record["id"] / "rollout.mp4"
            if clip.exists():
                frames = imageio.mimread(clip, memtest=False)
                tiles.append(np.asarray(frames[min(k, len(frames) - 1)]))
                row = next((r for r in rows if r["id"] == record["id"] and r["seed"] == seed), None)
                marks += {1: "1", 0: "0", -1: "?"}.get(row["prediction"] if row else -1, "?")
            else:
                tiles.append(np.zeros_like(tiles[0]))
                marks += "."
        rec = dict(record)
        rec["v0"] = float(record["params"].get("v0", float("nan")))
        entries.append({"record": rec, "tiles": tiles, "marks": marks})
    result_sheet(out / "sheets" / f"{scene}.jpg", scene, entries, seeds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/pvr"))
    parser.add_argument("--scenes", nargs="*", default=("hill", "collide", "edge"))
    parser.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    parser.add_argument("--gt-selftest-only", action="store_true")
    parser.add_argument("--output-name", default="analysis")
    args = parser.parse_args()
    for scene in args.scenes:
        root = (args.root / scene).resolve()
        analyze_scene(scene, root, root / "cosmos_v2w", root / args.output_name, list(args.seeds),
                      selftest_only=args.gt_selftest_only)


if __name__ == "__main__":
    main()
