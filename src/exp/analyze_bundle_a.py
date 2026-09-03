"""Adjudicate Cosmos V2W rollouts on the Bundle A threshold benchmark.

Direction selection is read from the 21-frame trajectory, never from a single
last frame.  The adjudicators are validated against the MuJoCo ground-truth
clips first (they must reproduce every manifest outcome) and then applied,
frozen, to the generated videos.  Validity (object loss / gross morphing) is
tracked separately from physics decisions, and each scene carries its law
metric: the (y, v^2) slope for the energy scenes and the momentum-conservation
ratio for the collision scene.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

G = 9.81
FPS = 16.0
FAMILIES = ("hill_roll", "two_ball", "pendulum_rod")     # Bundle A default; main() reads the manifest
# History-extrapolation baseline: uniform motion never turns, never reverses,
# never stops -- each family's constant answer under linear extrapolation.
ENCODER_PREDICTION = {"hill_roll": 1, "two_ball": 0, "pendulum_rod": 1, "kin_roll": 1,
                      "support_edge": 0}


def kind_of(family):
    """Adjudicator kind from the family name (Bundle A names and their *_wb variants)."""
    for prefix in ("hill_roll", "two_ball", "pendulum_rod", "wall_bounce", "kin_roll", "support_edge"):
        if family.startswith(prefix):
            return prefix
    raise ValueError(family)


def encoder_prediction(family):
    return ENCODER_PREDICTION[kind_of(family)]


def adjudicate_kin(traj, gt_traj, dia, n_cond=5):
    """P0 control: continuation fidelity of uniform rolling.

    Fits constant velocity on the generated clip's own conditioning frames,
    extrapolates it over the rollout, and compares with what was generated and
    with the ground truth.  prediction 1 = keeps rolling (speed ratio >= 0.5),
    0 = stalls/reverses (< 0.2), -1 = ambiguous."""
    n = min(len(traj), len(gt_traj))
    t = np.arange(n_cond)
    vx = np.polyfit(t, traj[:n_cond, 0], 1)[0]
    extrap = traj[n_cond - 1, 0] + vx * np.arange(1, n - n_cond + 1)
    kin_err = float(np.mean(np.abs(traj[n_cond:n, 0] - extrap)))
    gt_err = float(np.mean(np.abs(traj[n_cond:n, 0] - gt_traj[n_cond:n, 0])))
    gt_disp = gt_traj[n - 1, 0] - gt_traj[n_cond, 0]
    gen_disp = traj[n - 1, 0] - traj[n_cond, 0]
    ratio = float(gen_disp / gt_disp) if abs(gt_disp) > 1e-6 else float("nan")
    pred = 1 if ratio >= 0.5 else (0 if ratio < 0.2 else -1)
    return pred, {"kin_err_px": kin_err, "gt_err_px": gt_err, "speed_ratio": ratio,
                  "kin_err_dia": kin_err / max(dia, 1e-6), "gt_err_dia": gt_err / max(dia, 1e-6)}


def color_mask(image, color):
    x = np.asarray(image, dtype=np.float32)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    if color == "red":
        return (r > 105) & (r > 1.22 * g) & (r > 1.12 * b)
    if color == "blue":
        # H.264 decoding shifts the sky toward blue, so demand real saturation
        # and ignore the sky band; the blue ball never leaves the floor region.
        mask = (b > 90) & (b > 1.35 * r) & (b - np.minimum(r, g) > 40)
        mask[:140] = False
        return mask
    raise ValueError(color)


def centroid_and_area(mask):
    y, x = np.nonzero(mask)
    if len(x) < 12:
        return None, 0
    return np.array([x.mean(), y.mean()]), int(len(x))


def track(frames, color):
    """Per-frame centroid (px) and mask area; None where the object vanishes."""
    points, areas = [], []
    for frame in frames:
        c, a = centroid_and_area(color_mask(frame, color))
        points.append(c)
        areas.append(a)
    return points, areas


def diameter_px(frame, color):
    """Area-equivalent diameter: robust to stray mask pixels far from the ball."""
    mask = color_mask(frame, color)
    area = int(mask.sum())
    if area < 12:
        return None
    return float(2.0 * np.sqrt(area / np.pi))


def validity(areas, reference_area):
    """Fraction of frames with the object present at a sane size."""
    ok = [a >= 12 and 0.35 <= a / max(reference_area, 1) <= 2.8 for a in areas]
    return float(np.mean(ok)), all(ok)


def fit_circle(points):
    """Least-squares circle through 2D points (Kasa fit)."""
    pts = np.asarray(points, dtype=np.float64)
    A = np.column_stack([2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))])
    b = (pts ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0], sol[1]
    radius = float(np.sqrt(sol[2] + cx ** 2 + cy ** 2))
    return np.array([cx, cy]), radius


def interp_gaps(points):
    """Fill None centroids by linear interpolation so short dropouts survive."""
    idx = [i for i, p in enumerate(points) if p is not None]
    if len(idx) < 2:
        return None
    arr = np.array([points[i] for i in idx])
    full = np.empty((len(points), 2))
    for d in range(2):
        full[:, d] = np.interp(np.arange(len(points)), idx, arr[:, d])
    return full


# ---------------------------------------------------------------- adjudicators

def adjudicate_hill(traj, crest_x, dia):
    x = traj[:, 0]
    if x.max() > crest_x + 0.75 * dia:
        return 1
    tail = x[-4:]
    if tail[-1] < tail[0] - 0.10 * dia:
        return 0
    return -1                         # stalled or ambiguous: no direction selected


def detect_contact(blue_traj, dia_blue):
    """First frame where the resting blue ball starts moving, in this video."""
    base = float(np.median(blue_traj[:4, 0]))
    moved = np.abs(blue_traj[:, 0] - base) > 0.08 * dia_blue
    return int(np.argmax(moved)) if moved.any() else None


def adjudicate_two_ball(traj, blue_traj, dia, dia_blue):
    # Time-dilated generations collide later than the ground truth, so the
    # adjudication window anchors on the collision observed in this video.
    contact = detect_contact(blue_traj, dia_blue) if blue_traj is not None else None
    if contact is None or contact <= 1:
        return -1, None                # no collision drawn (or moving from t=0)
    read0 = min(contact, len(traj) - 2)
    read1 = min(contact + 5, len(traj) - 1)
    span = max(read1 - read0, 1)
    v_post = (traj[read1, 0] - traj[read0, 0]) / span      # px per frame
    if v_post < -0.012 * dia:
        return 1, contact
    if v_post > 0.012 * dia:
        return 0, contact
    return -1, contact


def adjudicate_pendulum(traj, pivot, dia):
    rel = traj - pivot[None, :]
    theta = np.unwrap(np.arctan2(rel[:, 0], rel[:, 1]))    # 0 = straight down
    if np.abs(theta).max() >= np.pi + 0.05:
        return 1
    peak = int(np.argmax(np.abs(theta)))
    if 1.2 <= np.abs(theta).max() < np.pi - 0.05 and peak < len(theta) - 1 \
            and np.abs(theta)[-1] < np.abs(theta).max() - 0.08:
        return 0
    return -1                          # barely swung or ambiguous: no decision


# ---------------------------------------------------------------- law metrics

def adjudicate_edge(traj, y_ctx, dia, n_cond=5):
    """P3-A support edge.  1 = the ball dropped below the plate level seen in the
    context (image y grows downward) by more than 0.6 diameters, 0 = it stayed
    within 0.3 diameters of that level for the whole rollout, -1 = ambiguous."""
    drop = traj[:, 1] - y_ctx
    fell = np.nonzero(drop[n_cond:] > 0.6 * dia)[0]
    if len(fell):
        return 1, int(fell[0] + n_cond)
    if np.nanmax(drop[n_cond:]) < 0.3 * dia:
        return 0, None
    return -1, None


def fall_g_ratio(traj, y_ctx, dia, scale, fps=16):
    """Free-fall law: vertical acceleration over the first frames of the drop
    (before landing, drop < 2 diameters) relative to g.  NaN if too short."""
    drop = traj[:, 1] - y_ctx
    start = np.nonzero(drop > 0.15 * dia)[0]
    if not len(start):
        return float("nan")
    s = max(int(start[0]) - 1, 0)
    seg = drop[s:]
    seg = seg[: max(np.searchsorted(seg > 2.0 * dia, True), 4)]
    if len(seg) < 4 or np.isnan(seg).any():
        return float("nan")
    acc = float(np.mean(np.diff(seg, 2)))          # px / frame^2
    return acc * fps * fps / max(scale, 1e-6) / 9.81


def hill_energy_slope(traj, scale, n_use):
    """Least-squares d(v^2)/dy in metric units; rolling law predicts -(10/7)g."""
    pts = traj[:n_use] / scale
    v = np.gradient(pts, axis=0) * FPS
    speed2 = (v ** 2).sum(axis=1)
    height = (traj[0, 1] - traj[:n_use, 1]) / scale        # screen y is down
    moving = speed2 > 0.02
    if moving.sum() < 5 or height.max() < 0.01:
        return None
    A = np.column_stack([height[moving], np.ones(moving.sum())])
    sol, *_ = np.linalg.lstsq(A, speed2[moving], rcond=None)
    return float(sol[0])


def pendulum_energy_slope(traj, pivot, L_px, scale, n_use):
    rel = traj[:n_use] - pivot[None, :]
    theta = np.unwrap(np.arctan2(rel[:, 0], rel[:, 1]))
    omega = np.gradient(theta) * FPS
    L = L_px / scale
    speed2 = (omega * L) ** 2
    height = L * (1 - np.cos(theta))
    ok = speed2 > 0.05
    if ok.sum() < 5:
        return None
    A = np.column_stack([height[ok], np.ones(ok.sum())])
    sol, *_ = np.linalg.lstsq(A, speed2[ok], rcond=None)
    return float(sol[0])


def momentum_ratio(red_traj, blue_traj, dia_red, dia_blue, contact_frame, n):
    pre0, pre1 = 1, max(contact_frame - 2, 2)
    post0 = min(contact_frame + 1, n - 2)
    post1 = min(contact_frame + 3, n - 1)
    v_red_pre = (red_traj[pre1, 0] - red_traj[pre0, 0]) / max(pre1 - pre0, 1)
    v_red_post = (red_traj[post1, 0] - red_traj[post0, 0]) / max(post1 - post0, 1)
    v_blue_post = (blue_traj[post1, 0] - blue_traj[post0, 0]) / max(post1 - post0, 1)
    m_red, m_blue = dia_red ** 3, dia_blue ** 3
    if abs(v_red_pre) < 1e-6:
        return None
    return float((m_red * v_red_post + m_blue * v_blue_post) / (m_red * v_red_pre))


# ---------------------------------------------------------------- references

def gt_reference(record, root):
    """Per-sample frozen constants measured on the ground-truth clip."""
    frames = imageio.mimread(root / record["gt_clip"], memtest=False)
    red_pts, red_areas = track(frames, "red")
    traj = interp_gaps(red_pts)
    ref = {"n_frames": len(frames), "red_area": red_areas[0]}
    dia = diameter_px(frames[0], "red")
    ref["dia_red"] = dia
    fam = record["family"]
    if fam.startswith("hill_roll"):
        cols = record["pixel_proxy"]["hill_cols"] if record.get("pixel_proxy") else None
        ref["crest_x"] = float(np.mean(cols)) if cols else 416.0
        ref["scale"] = dia / 0.06                            # ball is 6 cm
    elif fam.startswith("two_ball"):
        ref["dia_blue"] = diameter_px(frames[0], "blue")
        ref["blue_area"] = centroid_and_area(color_mask(frames[0], "blue"))[1]
        ref["contact_frame"] = int(record["event_frame"])
        ref["scale"] = dia / 0.08                            # red ball is 8 cm
    elif fam.startswith("pendulum_rod"):
        pivot, L_px = fit_circle(traj)
        ref["pivot"] = pivot.tolist()
        ref["L_px"] = L_px
        ref["scale"] = L_px / record["params"]["length"]
    elif fam.startswith("kin_roll"):
        ref["scale"] = dia / 0.06                            # ball is 6 cm
    elif fam.startswith("support_edge"):
        ref["y_ctx"] = float(np.nanmean(traj[:5, 1]))        # plate-level centroid row
        ref["scale"] = dia / 0.06                            # ball is 6 cm
    ref["gt_traj"] = traj
    ref["gt_frames"] = frames
    return ref


def adjudicate(record, ref, frames):
    fam = record["family"]
    color_checks = [("red", ref["red_area"])]
    if fam.startswith("two_ball"):
        color_checks.append(("blue", ref["blue_area"]))
    valid_frac = 1.0
    all_valid = True
    tracks = {}
    for color, ref_area in color_checks:
        pts, areas = track(frames, color)
        frac, ok = validity(areas, ref_area)
        valid_frac = min(valid_frac, frac)
        all_valid = all_valid and ok
        tracks[color] = interp_gaps(pts)
    traj = tracks["red"]
    if traj is None:
        return {"prediction": -1, "valid": False, "valid_frac": valid_frac}
    n = len(frames)
    out = {"valid": all_valid, "valid_frac": valid_frac}
    if fam.startswith("hill_roll"):
        out["prediction"] = adjudicate_hill(traj, ref["crest_x"], ref["dia_red"])
        out["law_slope"] = hill_energy_slope(traj, ref["scale"], n)
    elif fam.startswith("two_ball"):
        out["prediction"], contact = adjudicate_two_ball(
            traj, tracks.get("blue"), ref["dia_red"], ref["dia_blue"])
        out["gen_contact_frame"] = contact
        if tracks.get("blue") is not None and contact is not None:
            out["momentum_ratio"] = momentum_ratio(
                traj, tracks["blue"], ref["dia_red"], ref["dia_blue"],
                contact, n)
    elif fam.startswith("pendulum_rod"):
        pivot = np.asarray(ref["pivot"])
        out["prediction"] = adjudicate_pendulum(traj, pivot, ref["dia_red"])
        out["law_slope"] = pendulum_energy_slope(traj, pivot, ref["L_px"],
                                                 ref["scale"], n)
    elif fam.startswith("kin_roll"):
        out["prediction"], extra = adjudicate_kin(traj, ref["gt_traj"], ref["dia_red"])
        out.update(extra)
    elif fam.startswith("support_edge"):
        out["prediction"], out["gen_event_frame"] = adjudicate_edge(traj, ref["y_ctx"], ref["dia_red"])
        out["fall_g_ratio"] = fall_g_ratio(traj, ref["y_ctx"], ref["dia_red"], ref["scale"])
    elif fam.startswith("wall_bounce"):
        vx = np.diff(traj[:, 0])
        slow = np.nonzero(vx < 0.2 * np.median(vx[:4]))[0]
        contact = int(slow[0]) if len(slow) else None
        if contact is None or contact <= 1:
            out["prediction"] = -1
        else:
            read1 = min(contact + 5, len(traj) - 1)
            v_post = (traj[read1, 0] - traj[min(contact + 1, read1), 0]) / max(read1 - contact - 1, 1)
            out["prediction"] = 1 if v_post < -0.012 * ref["dia_red"] else (
                0 if v_post > 0.012 * ref["dia_red"] else -1)
        out["gen_contact_frame"] = contact
    return out


# ---------------------------------------------------------------- reporting

def font(size):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def result_sheet(path, family, entries, seeds):
    """Columns: S grid; rows: GT at event, each seed's rollout at event frame."""
    cell_w, cell_h, header, label_w = 190, 110, 40, 110
    rows = ["GT @event"] + [f"seed {s} @event" for s in seeds]
    canvas = Image.new("RGB", (label_w + len(entries) * cell_w,
                               header + len(rows) * cell_h), (23, 26, 30))
    draw = ImageDraw.Draw(canvas)
    fnt = font(11)
    for r, label in enumerate(rows):
        draw.text((6, header + r * cell_h + cell_h // 2 - 6), label,
                  fill="white", font=fnt)
    for c, entry in enumerate(entries):
        x = label_w + c * cell_w
        rec = entry["record"]
        marks = "".join("XO-"[["0", "1", "?"].index(m)] if m in "01?" else m
                        for m in entry["marks"])
        head = (f"S={rec['S']:.2f}" if rec.get("S") is not None
                else f"v0={rec.get('v0', float('nan')):.2f}")
        draw.text((x + 6, 8), f"{head} y={rec['outcome']}", fill="white", font=fnt)
        draw.text((x + 6, 22), marks, fill=(200, 205, 215), font=fnt)
        for r, frame in enumerate(entry["tiles"]):
            tile = Image.fromarray(frame).resize((cell_w, cell_h),
                                                 Image.Resampling.BILINEAR)
            canvas.paste(tile, (x, header + r * cell_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=90)


def curve_plot(path, per_family):
    fams = list(per_family)
    fig, axes = plt.subplots(1, max(len(fams), 1), figsize=(4.7 * max(len(fams), 1), 4.2),
                             constrained_layout=True, squeeze=False)
    axes = axes[0]
    for axis, family in zip(axes, fams):
        data = per_family[family]
        S = np.array([d["S"] for d in data])
        p = np.array([d["p_success"] for d in data], dtype=float)
        gt = np.array([d["outcome"] for d in data], dtype=float)
        axis.plot(S, gt, drawstyle="steps-mid", color="0.55", lw=1.2,
                  label="MuJoCo GT")
        axis.plot(S, p, "o-", color="#C4402C", label="Cosmos P(success)")
        axis.axhline(encoder_prediction(family), color="#3E68A8", ls=":",
                     lw=1.4, label="const-vel encoder")
        axis.axvline(1.0, color="0.3", ls="--", lw=1)
        axis.set(title=family, xlabel="S", ylim=(-0.06, 1.06), xscale="log")
        axis.set_xticks([0.5, 0.7, 1.0, 1.4, 2.0])
        axis.set_xticklabels(["0.5", "0.7", "1.0", "1.4", "2.0"])
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("P(success side)")
    axes[0].legend(fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/bundle_a"))
    parser.add_argument("--cosmos-dir", type=Path, default=None,
                        help="default: <root>/cosmos_v2w")
    parser.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--gt-selftest-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    cosmos = (args.cosmos_dir or root / "cosmos_v2w").resolve()
    out = (args.output_dir or root / "analysis").resolve()
    out.mkdir(parents=True, exist_ok=True)

    manifest = [json.loads(l) for l in (root / "manifest.jsonl").read_text().splitlines() if l.strip()]
    grid_families = list(dict.fromkeys(r["family"] for r in manifest if r["role"] == "grid"))
    control_families = list(dict.fromkeys(r["family"] for r in manifest if r["role"] == "control"))
    refs, selftest = {}, []
    for record in manifest:
        ref = gt_reference(record, root)
        refs[record["id"]] = ref
        res = adjudicate(record, ref, ref["gt_frames"])
        selftest.append({"id": record["id"], "gt_outcome": record["outcome"],
                         "adjudicated": res["prediction"],
                         "match": res["prediction"] == record["outcome"],
                         "law_slope": res.get("law_slope"),
                         "momentum_ratio": res.get("momentum_ratio")})
    n_match = sum(s["match"] for s in selftest)
    print(f"GT self-test: {n_match}/{len(selftest)} outcomes reproduced")
    for s in selftest:
        if not s["match"]:
            print("  MISMATCH:", s)
    (out / "gt_selftest.json").write_text(json.dumps(selftest, indent=2))
    if args.gt_selftest_only:
        return

    rows = []
    for record in manifest:
        ref = refs[record["id"]]
        for seed in args.seeds:
            clip = cosmos / f"seed_{seed:02d}" / record["id"] / "rollout.mp4"
            if not clip.exists():
                continue
            frames = imageio.mimread(clip, memtest=False)[:record["rollout_frames"]]
            res = adjudicate(record, ref, frames)
            rows.append({
                "id": record["id"], "family": record["family"],
                "role": record["role"], "S": record["S"],
                "outcome": record["outcome"], "boundary": record["boundary"],
                "in_map": record["in_map"], "seed": seed,
                "prediction": res["prediction"],
                "correct": int(res["prediction"] == record["outcome"]),
                "valid": int(bool(res["valid"])),
                "valid_frac": res["valid_frac"],
                "law_slope": res.get("law_slope"),
                "momentum_ratio": res.get("momentum_ratio"),
                "event_frame": record["event_frame"],
                "v0": record.get("v0"),
                "kin_err_px": res.get("kin_err_px"), "gt_err_px": res.get("gt_err_px"),
                "speed_ratio": res.get("speed_ratio"),
            })
    if not rows:
        print("no rollouts found under", cosmos)
        return

    per_family_curve = {}
    summary = {}
    for family in grid_families:
        frows = [r for r in rows if r["family"] == family and r["role"] == "grid"]
        curve = []
        for record in manifest:
            if record["family"] != family or record["role"] != "grid":
                continue
            srows = [r for r in frows if r["id"] == record["id"]]
            decided = [r for r in srows if r["prediction"] in (0, 1)]
            p1 = float(np.mean([r["prediction"] for r in decided])) if decided else np.nan
            curve.append({"S": record["S"], "outcome": record["outcome"],
                          "p_success": p1, "n": len(srows),
                          "n_undecided": sum(r["prediction"] == -1 for r in srows),
                          "valid_rate": float(np.mean([r["valid"] for r in srows])) if srows else np.nan})
        per_family_curve[family] = curve
        scored = [r for r in frows if not r["boundary"]]
        gt_slopes = [s["law_slope"] for s in selftest
                     if s["id"].startswith(family) and s["law_slope"] is not None]
        gen_slopes = [r["law_slope"] for r in frows if r["law_slope"] is not None]
        summary[family] = {
            "n_rollouts": len(frows),
            "valid_rate": float(np.mean([r["valid"] for r in frows])),
            "undecided_rate": float(np.mean([r["prediction"] == -1 for r in frows])),
            "accuracy_nonboundary": float(np.mean([r["correct"] for r in scored])) if scored else np.nan,
            "encoder_accuracy_nonboundary": float(np.mean(
                [int(encoder_prediction(family) == r["outcome"]) for r in scored])) if scored else np.nan,
            "gt_law_slope_mean": float(np.mean(gt_slopes)) if gt_slopes else None,
            "gen_law_slope_mean": float(np.mean(gen_slopes)) if gen_slopes else None,
            "momentum_ratio_mean": float(np.nanmean(
                [r["momentum_ratio"] for r in frows if r["momentum_ratio"] is not None]))
                if any(r["momentum_ratio"] is not None for r in frows) else None,
            "curve": curve,
        }

    for family in control_families:
        crows = [r for r in rows if r["family"] == family]
        per_v = []
        for record in manifest:
            if record["family"] != family:
                continue
            srows = [r for r in crows if r["id"] == record["id"]]
            per_v.append({"id": record["id"], "v0": record.get("v0"), "n": len(srows),
                          "p_continue": float(np.mean([r["prediction"] == 1 for r in srows])) if srows else np.nan,
                          "kin_err_px": float(np.nanmean([r["kin_err_px"] for r in srows if r["kin_err_px"] is not None])) if srows else np.nan,
                          "gt_err_px": float(np.nanmean([r["gt_err_px"] for r in srows if r["gt_err_px"] is not None])) if srows else np.nan,
                          "speed_ratio": float(np.nanmean([r["speed_ratio"] for r in srows if r["speed_ratio"] is not None])) if srows else np.nan,
                          "valid_rate": float(np.mean([r["valid"] for r in srows])) if srows else np.nan,
                          "valid_frac": float(np.mean([r["valid_frac"] for r in srows])) if srows else np.nan})
        summary.setdefault("controls", {})[family] = {
            "n_rollouts": len(crows),
            "p_continue": float(np.mean([r["prediction"] == 1 for r in crows])) if crows else np.nan,
            "valid_rate": float(np.mean([r["valid"] for r in crows])) if crows else np.nan,
            "kin_err_px_mean": float(np.nanmean([r["kin_err_px"] for r in crows if r["kin_err_px"] is not None])) if crows else np.nan,
            "gt_err_px_mean": float(np.nanmean([r["gt_err_px"] for r in crows if r["gt_err_px"] is not None])) if crows else np.nan,
            "speed_ratio_mean": float(np.nanmean([r["speed_ratio"] for r in crows if r["speed_ratio"] is not None])) if crows else np.nan,
            "per_sample": per_v,
        }

    pre_rows = [r for r in rows if r["role"] == "precheck"]
    summary["prechecks"] = {r["id"] + f"_seed{r['seed']}":
                            {"prediction": r["prediction"], "gt": r["outcome"],
                             "valid": r["valid"]} for r in pre_rows}

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out / "samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    curve_plot(out / "curves.png", per_family_curve)

    for family in grid_families + control_families:
        entries = []
        for record in manifest:
            if record["family"] != family or record["role"] not in ("grid", "control"):
                continue
            ref = refs[record["id"]]
            k = min(record["event_frame"], ref["n_frames"] - 1)
            tiles = [np.asarray(ref["gt_frames"][k])]
            marks = ""
            for seed in args.seeds:
                clip = cosmos / f"seed_{seed:02d}" / record["id"] / "rollout.mp4"
                if clip.exists():
                    frames = imageio.mimread(clip, memtest=False)
                    tiles.append(np.asarray(frames[min(k, len(frames) - 1)]))
                    row = next((r for r in rows if r["id"] == record["id"]
                                and r["seed"] == seed), None)
                    marks += {1: "1", 0: "0", -1: "?"}.get(
                        row["prediction"] if row else -1, "?")
                else:
                    tiles.append(np.zeros_like(tiles[0]))
                    marks += "."
            entries.append({"record": record, "tiles": tiles, "marks": marks})
        result_sheet(out / "sheets" / f"{family}.jpg", family, entries, args.seeds)

    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("curve", "per_sample")}
                      for k, v in summary.items() if k in grid_families}, indent=2))
    if control_families:
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_sample"}
                          for k, v in summary["controls"].items()}, indent=2))


if __name__ == "__main__":
    main()
