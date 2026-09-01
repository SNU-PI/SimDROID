"""Generate the Bundle A threshold benchmark: native 832x480, exact 16 FPS.

Per scene and per S grid point this writes the five-frame conditioning video,
the key comparison frames, a 21-frame ground-truth clip (the Cosmos horizon),
and a manifest row carrying the physics record (S, outcome, event frame,
t_event, map inclusion) plus a certificate-lite pixel proxy of the decision
variable measured from the rendered frames themselves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# gen.render selects EGL before importing MuJoCo through the scene package.
from gen.render import DIAG_CFG, font, roll, write_video
from gen.bundle_a_spec import (BOUNDARY_TOL, BUNDLE_A, CONDITION_INDICES,
                               FUTURE_OFFSET, ORACLE, PRECHECKS, PRINCIPLE,
                               PROMPTS, ROLLOUT_FRAMES, S_GRID, bundle_specs,
                               in_map)

CURRENT_IDX = CONDITION_INDICES[-1]


def color_mask(image, color):
    x = np.asarray(image, dtype=np.float32)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    if color == "red":
        return (r > 105) & (r > 1.22 * g) & (r > 1.12 * b)
    if color == "blue":
        # Stricter than the analyzer mask: the lit floor and sky are mildly
        # bluish grey, so demand a strong blue excess.
        return (b > 90) & (b > 1.25 * r) & (b > 1.08 * g)
    if color == "white":
        low = np.minimum(np.minimum(r, g), b)
        high = np.maximum(np.maximum(r, g), b)
        return (low > 185) & (high - low < 30)
    raise ValueError(color)


def lower_half(mask):
    out = np.zeros_like(mask)
    out[mask.shape[0] // 2:] = mask[mask.shape[0] // 2:]
    return out


def pixel_proxy(name, frames):
    """Certificate-lite: the decision variable in pixels, from rendered frames."""
    if name.startswith("hill_roll"):
        mask = color_mask(frames[0], "white")   # sky/floor stay below the threshold
        if not mask.any():
            return None
        rows = np.nonzero(mask.any(axis=1))[0]
        cols = np.nonzero(mask.any(axis=0))[0]
        return {"kind": "hill_apex_px", "value": int(rows.max() - rows.min()),
                "hill_cols": [int(cols.min()), int(cols.max())]}
    if name.startswith("two_ball"):
        mask = lower_half(color_mask(frames[0], "blue"))
        if not mask.any():
            return None
        cols = np.nonzero(mask.any(axis=0))[0]
        return {"kind": "blue_diameter_px", "value": int(cols.max() - cols.min() + 1)}
    if name.startswith("pendulum_rod"):
        lows = []
        for frame in frames[:ROLLOUT_FRAMES]:
            mask = color_mask(frame, "red")
            if mask.any():
                lows.append(int(np.nonzero(mask.any(axis=1))[0].max()))
        return {"kind": "bob_lowest_row_px", "value": max(lows)} if lows else None
    return None


def family_preview(path: Path, name, entries):
    cell_w, cell_h, header, label_w = 200, 116, 40, 96
    rows = [("t=0", 0), ("COND END 0.25s", CURRENT_IDX),
            ("+0.31s", CURRENT_IDX + FUTURE_OFFSET),
            ("EVENT", None), ("t=1.25s", ROLLOUT_FRAMES - 1)]
    canvas = Image.new("RGB", (label_w + len(entries) * cell_w,
                               header + len(rows) * cell_h), (23, 26, 30))
    draw = ImageDraw.Draw(canvas)
    fnt = font(11)
    for row, (label, _) in enumerate(rows):
        draw.text((6, header + row * cell_h + cell_h // 2 - 6), label,
                  fill="white", font=fnt)
    for col, (record, frames) in enumerate(entries):
        x = label_w + col * cell_w
        title = f"S={record['S']:.2f} y={record['outcome']}"
        if not record["in_map"]:
            title += " (excl)"
        draw.text((x + 6, 12), title, fill="white", font=fnt)
        for row, (_, idx) in enumerate(rows):
            k = record["event_frame"] if idx is None else idx
            k = min(k, len(frames) - 1)
            tile = Image.fromarray(frames[k]).resize(
                (cell_w, cell_h), Image.Resampling.BILINEAR)
            canvas.paste(tile, (x, header + row * cell_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def build_sample(name, cls, sample_id, S, params, root: Path, role="grid"):
    frames, labels = roll(cls, params, cls.cam, DIAG_CFG)
    if S is not None and abs(cls.S_of(params) - S) > 1e-9:
        raise RuntimeError(f"{sample_id}: S inversion mismatch")
    expected = None if S is None else int(S > 1.0)
    boundary = S is not None and abs(S - 1.0) <= BOUNDARY_TOL
    if expected is not None and not boundary \
            and int(labels["outcome"]) != expected:
        raise RuntimeError(f"{sample_id}: analytic/simulation outcome mismatch")
    if int(labels["event_frame"]) <= CURRENT_IDX:
        raise RuntimeError(f"{sample_id}: event reaches the conditioning window")
    if name == "hill_roll" and labels["cond_end_x"] + cls.BALL_R > cls.BASE_X:
        raise RuntimeError(f"{sample_id}: conditioning touches the slope")

    condition = frames[list(CONDITION_INDICES)]
    future_idx = CURRENT_IDX + FUTURE_OFFSET
    input_rel = Path("inputs") / f"{sample_id}.npz"
    condition_rel = Path("conditions") / f"{sample_id}.mp4"
    gt_rel = Path("gt") / f"{sample_id}.mp4"
    prompt = PROMPTS[name]
    np.savez_compressed(
        root / input_rel,
        name=sample_id, principle=PRINCIPLE, family=name,
        condition_primary=condition,
        current_primary=frames[CURRENT_IDX],
        gt_future_primary=frames[future_idx],
        gt_event_primary=frames[min(labels["event_frame"], len(frames) - 1)],
        gt_final_primary=frames[ROLLOUT_FRAMES - 1],
        S=float("nan") if S is None else S,
        outcome=int(labels["outcome"]),
        event_frame=int(labels["event_frame"]),
        condition_indices=np.asarray(CONDITION_INDICES),
        current_idx=CURRENT_IDX, future_idx=future_idx,
        capture_fps=DIAG_CFG.fps,
    )
    write_video(root / condition_rel, condition, DIAG_CFG)
    write_video(root / gt_rel, frames[:ROLLOUT_FRAMES], DIAG_CFG)
    (root / "conditions" / f"{sample_id}.txt").write_text(prompt)

    record = {
        "id": sample_id, "principle": PRINCIPLE, "family": name,
        # "kind" follows the cosmos_v2w_sweep contract: dynamic = 5-frame video
        # conditioning ("grid" here once silently fell through to image mode).
        "kind": "dynamic", "role": role,
        "S": S, "margin": float(labels["margin"]),
        "params": {k: float(v) for k, v in params.items()},
        "outcome": int(labels["outcome"]),
        "event_frame": int(labels["event_frame"]),
        "t_event_s": float(labels["t_event_s"]),
        "decided": bool(labels["decided"]),
        "in_map": in_map(labels) if S is not None else True,
        "boundary": boundary,
        "condition_indices": list(CONDITION_INDICES),
        "condition_pixel_frames": len(CONDITION_INDICES),
        "current_idx": CURRENT_IDX, "future_idx": future_idx,
        "rollout_frames": ROLLOUT_FRAMES,
        "capture_fps": DIAG_CFG.fps,
        "render_size": [DIAG_CFG.height, DIAG_CFG.width],
        "input_npz": str(input_rel), "condition_path": str(condition_rel),
        "gt_clip": str(gt_rel), "prompt": prompt,
        "pixel_proxy": pixel_proxy(name, frames),
    }
    for key in ("slip_max", "energy_drift", "e_eff", "momentum_ratio",
                "ke_ratio", "bottom_frame", "read_frame"):
        if key in labels:
            record[key] = float(labels[key])
    if name in ORACLE:
        record["oracle_prompt"] = ORACLE[name][record["outcome"]]
        record["wrong_prompt"] = ORACLE[name][1 - record["outcome"]]
    return record, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/bundle_a"))
    parser.add_argument("--families", nargs="*", choices=tuple(BUNDLE_A),
                        default=tuple(BUNDLE_A))
    parser.add_argument("--skip-prechecks", action="store_true")
    args = parser.parse_args()
    root = args.output_dir.resolve()
    for sub in ("inputs", "conditions", "gt", "previews"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    specs = bundle_specs()
    records = []
    for name in args.families:
        cls = BUNDLE_A[name]
        entries = []
        for index, (S, params) in enumerate(zip(S_GRID, specs[name])):
            sample_id = f"{name}_{index:02d}"
            record, frames = build_sample(name, cls, sample_id, float(S),
                                          params, root)
            records.append(record)
            entries.append((record, frames))
            proxy = record["pixel_proxy"]
            print(f"{sample_id}: S={S:.2f} y={record['outcome']} "
                  f"ev={record['event_frame']} map={int(record['in_map'])} "
                  f"proxy={proxy and proxy['value']}")
        family_preview(root / "previews" / f"{name}.png", name, entries)

    if not args.skip_prechecks:
        pre_hill_cls, pre_hill_kw = PRECHECKS["hill_roll_pre"]
        record, _ = build_sample("hill_roll", pre_hill_cls, "hill_roll_pre",
                                 None, pre_hill_cls.params_for_S(pre_hill_kw["S"]),
                                 root, role="precheck")
        record["S"] = pre_hill_kw["S"]
        records.append(record)
        print(f"hill_roll_pre: y={record['outcome']} ev={record['event_frame']}")
        wall_cls, _ = PRECHECKS["wall_bounce_pre"]
        record, _ = build_sample("wall_bounce_pre", wall_cls, "wall_bounce_pre",
                                 None, dict(v0=wall_cls.V0), root,
                                 role="precheck")
        records.append(record)
        print(f"wall_bounce_pre: y={record['outcome']} ev={record['event_frame']}")

    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in records))
    summary = {
        "principle": PRINCIPLE, "samples": len(records),
        "families": list(args.families), "S_grid": list(S_GRID),
        "render_size": [DIAG_CFG.height, DIAG_CFG.width],
        "capture_fps": DIAG_CFG.fps,
        "conditioning_frames": len(CONDITION_INDICES),
        "rollout_horizon_seconds": ROLLOUT_FRAMES / DIAG_CFG.fps,
        "manifest": str(manifest),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
