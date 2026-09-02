"""Render Phase A scenes as VERA DROID-planner inputs (3-view canvas, 29-frame context).

VERA's DROID planner consumes a 576x128 canvas of three camera views (192x128
each) at 15 fps, conditions on 29 frames and generates 24 per call.  Our scenes
decide within ~1 s, so a 29-frame history at 16 fps would already contain the
event.  We therefore *time-stretch*: capture at 100 Hz (an integer multiple of
both scene timesteps), so 29 context frames span 0.29 s of strictly pre-event
motion and each 24-frame chunk covers 0.24 s; three chunks reach 0.72 s after
the context, which covers every grid point's event.  Played at 15 fps the
physics runs 6.7x slower than real time -- a slow roll, plausible in a robot
workspace.  Ground truth is rendered for context + 3 chunks (101 frames).

Outputs (artifacts/phase_a/vera/): <id>.npz with the uint8 canvas [T,128,576,3]
and the side view at 192x128, <id>_side.mp4 (full 832x480 side view, 15 fps
playback), <id>_canvas.mp4, and manifest_vera.jsonl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from gen.render import RenderCfg, roll_views, write_video
from gen.phase_a_spec import CONTROLS, PHASE_A, PRECHECKS, PROMPTS, S_GRID

CTX, CHUNK, N_CHUNKS = 29, 24, 3
CAPTURE_HZ = 100
VIEW_W, VIEW_H = 192, 128
CFG = RenderCfg(height=480, width=832, fps=15, quality=9, capture_dt=1.0 / CAPTURE_HZ)
SUBSET_S = (0.50, 0.79, 0.95, 1.05, 1.26, 2.00)


def to_view(frames):
    return np.stack([np.asarray(Image.fromarray(f).resize((VIEW_W, VIEW_H), Image.LANCZOS)) for f in frames])


def build(name, cls, params, sample_id, root, S=None, role="grid", v0=None):
    n = CTX + N_CHUNKS * CHUNK
    cams = [cls.cam, "wb_iso", "wb_top"]
    if (root / f"{sample_id}.npz").exists():
        # Rendered already: recover the physics record without rendering.
        scene = cls()
        scene.capture_dt = CFG.capture_dt
        scene.n_frames = n
        labels = scene.run(params, render=False)
    else:
        views, labels = roll_views(cls, params, cams, CFG, n_frames=n)
        canvas = np.concatenate([to_view(views[c]) for c in cams], axis=2)   # [T,128,576,3]
        side = views[cls.cam]
        np.savez_compressed(root / f"{sample_id}.npz", canvas=canvas, side_view=to_view(side),
                            ctx_frames=CTX, chunk_frames=CHUNK, capture_hz=CAPTURE_HZ)
        write_video(root / f"{sample_id}_side.mp4", side, CFG)
        write_video(root / f"{sample_id}_canvas.mp4", canvas, CFG)
    # The control family has no event (its event_frame marks the horizon end).
    if role != "control" and int(labels["event_frame"]) <= CTX - 1:
        raise RuntimeError(f"{sample_id}: event inside the 29-frame context (frame {labels['event_frame']})")
    rec = {"id": sample_id, "family": name, "role": role, "S": S, "v0": v0,
           "outcome": int(labels["outcome"]), "event_frame": int(labels["event_frame"]),
           "t_event_s": int(labels["event_frame"]) / CAPTURE_HZ, "decided": bool(labels["decided"]),
           "ctx_frames": CTX, "chunk_frames": CHUNK, "n_chunks": N_CHUNKS, "capture_hz": CAPTURE_HZ,
           "views": cams, "prompt": PROMPTS[name], "params": {k: float(v) for k, v in params.items()}}
    for key in ("energy_drift", "momentum_ratio", "e_eff", "speed_ratio"):
        if key in labels:
            rec[key] = float(labels[key])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/phase_a/vera"))
    ap.add_argument("--full-grid", action="store_true", help="all 11 S points instead of the 6-point subset")
    args = ap.parse_args()
    root = args.output_dir.resolve(); root.mkdir(parents=True, exist_ok=True)
    grid = S_GRID if args.full_grid else SUBSET_S
    recs = []
    for name, cls in PHASE_A.items():
        for S in grid:
            idx = S_GRID.index(S)
            rec = build(name, cls, cls.params_for_S(float(S)), f"{name}_{idx:02d}", root, S=float(S))
            recs.append(rec); print(rec["id"], "y", rec["outcome"], "event", rec["event_frame"], flush=True)
    for name, (cls, vgrid) in CONTROLS.items():
        for v0 in (vgrid if args.full_grid else (0.45, 0.75)):
            idx = list(vgrid).index(v0)
            rec = build(name, cls, dict(v0=float(v0)), f"{name}_{idx:02d}", root, role="control", v0=float(v0))
            recs.append(rec); print(rec["id"], "y", rec["outcome"], "event", rec["event_frame"], flush=True)
    cls, _ = PRECHECKS["wall_bounce_wb_pre"]
    rec = build("wall_bounce_wb_pre", cls, dict(v0=cls.V0), "wall_bounce_wb_pre", root, role="precheck")
    recs.append(rec); print(rec["id"], "y", rec["outcome"], "event", rec["event_frame"], flush=True)
    (root / "manifest_vera.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    print("samples:", len(recs))


if __name__ == "__main__":
    main()
