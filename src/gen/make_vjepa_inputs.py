"""Render V-JEPA 2-AC latent-track inputs from a Bundle A / Phase A manifest.

For every manifest row: physics frames from the square `vj_side` camera (256x256,
16 fps, 24 frames), plus for each context offset the kinematic-continuation
frames (K) and the +1 px jitter frames (J) at the four target frames.  Saves one
npz per sample under <out>/inputs, a GT-vs-K preview strip, and manifest.jsonl.

Run in the MuJoCo env with software rendering:
  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=src python src/gen/make_vjepa_inputs.py \
      --manifest artifacts/phase_a/manifest.jsonl --out artifacts/vjepa_ac/phase_a
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import mujoco

from core.threshold.rolling_hill import RollingHill, RollingHillWB
from core.threshold.two_ball import TwoBall, TwoBallWB, WallBounce, WallBounceWB
from core.threshold.kin_roll import KinRoll
from gen.render import RenderCfg, ensure_shared_gl_context, write_video, write_png
from gen.vjepa_spec import (CAM, SIZE, FPS, N_FRAMES, OFFSETS, ctx_indices, target_indices,
                            kind_of, px_per_m)

CFG = RenderCfg(height=SIZE, width=SIZE, fps=FPS, quality=9, capture_dt=1.0 / FPS)

FAMILY_CLS = {
    "hill_roll_wb": RollingHillWB, "hill_roll_wb_pre": RollingHillWB,
    "two_ball_wb": TwoBallWB, "wall_bounce_wb_pre": WallBounceWB,
    "kin_roll": KinRoll,
    "hill_roll": RollingHill, "hill_roll_pre": RollingHill,
    "two_ball": TwoBall, "wall_bounce_pre": WallBounce,
}


def body_qpos_adr(model, body):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    return int(model.jnt_qposadr[model.body_jntadr[bid]])


def render_placed(scene, placements):
    """Render frames with bodies placed at given (x, z); rendered twice, must agree."""
    out = []
    for pl in placements:
        for body, (x, z) in pl.items():
            a = body_qpos_adr(scene.model, body)
            scene.data.qpos[a] = x
            scene.data.qpos[a + 2] = z
            scene.data.qpos[a + 3:a + 7] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(scene.model, scene.data)
        f1 = scene.render()
        f2 = scene.render()
        if not np.array_equal(f1, f2) or f1.mean() < 1.0:
            raise RuntimeError("non-deterministic or dark placed render")
        out.append(f1)
    return np.stack(out)


def hill_z(scene, x):
    """Ball centre height when resting on the hill profile at x (flat: hfield base)."""
    h = float(scene.p["h"])
    inside = (x >= scene.BASE_X) and (x <= scene.BASE_X + scene.HILL_W)
    prof = np.sin(np.pi * (x - scene.BASE_X) / scene.HILL_W) ** 2 if inside else 0.0
    return 0.001 + h * prof + scene.BALL_R + 0.0005


def placements_for(kind, scene, trace, params, o, t_idx, jitter_px=0.0, ppm=1.0):
    """K placements (jitter_px=0) or J placements (physics re-placed + shift)."""
    dt = 1.0 / FPS
    c = ctx_indices(o)[1]
    delta = jitter_px / ppm
    out = []
    if kind in ("hill", "kin"):
        x_c, v_c = float(trace[c, 0]), float(trace[c, 2])
        for t in t_idx:
            if jitter_px:
                x, z = float(trace[t, 0]) + delta, float(trace[t, 1])
            else:
                x = x_c + v_c * (t - c) * dt
                z = hill_z(scene, x) if kind == "hill" else float(trace[c, 1])
            out.append({"ball": (x, z)})
    elif kind == "two_ball":
        r1, r2 = scene.R1, float(params["r2"])
        zA, zB = r1 + 0.001, r2 + 0.001
        x_c, v_c = float(trace[c, 0]), float(trace[c, 1])
        for t in t_idx:
            if jitter_px:
                out.append({"ballA": (float(trace[t, 0]) + delta, zA),
                            "ballB": (float(trace[t, 2]), zB)})
            else:
                xa = x_c + v_c * (t - c) * dt
                xb = max(0.0, xa + r1 + r2)          # blue rests at 0 until pushed
                out.append({"ballA": (xa, zA), "ballB": (xb, zB)})
    elif kind == "wall":
        r1 = scene.R1
        zA = r1 + 0.001
        x_c, v_c = float(trace[c, 0]), float(trace[c, 1])
        x_stop = 0.02 - 0.02 - r1                  # wall face at x=0 (box pos 0.02, half 0.02)
        for t in t_idx:
            if jitter_px:
                out.append({"ballA": (float(trace[t, 0]) + delta, zA)})
            else:
                out.append({"ballA": (min(x_c + v_c * (t - c) * dt, x_stop), zA)})
    else:
        raise KeyError(kind)
    return out


def build(row, out_root: Path):
    fam = row["family"]
    cls = FAMILY_CLS[fam]
    kind = kind_of(fam)
    params = {k: float(v) for k, v in row["params"].items()}
    ppm = px_per_m(fam)
    ensure_shared_gl_context()
    scene = cls()
    scene.cam = CAM
    scene.render_height = scene.render_width = SIZE
    scene.capture_dt = CFG.capture_dt
    scene.n_frames = N_FRAMES
    t0 = time.time()
    res = scene.run(params, render=True)
    frames, trace = res.pop("frames"), res.pop("trace")
    if frames.shape[0] != N_FRAMES or (frames.reshape(N_FRAMES, -1).mean(axis=1) < 1.0).any():
        raise RuntimeError(f"{row['id']}: bad physics render")
    if int(res["event_frame"]) != int(row["event_frame"]):
        print(f"[warn] {row['id']}: event_frame {res['event_frame']} vs manifest {row['event_frame']}",
              file=sys.stderr)
    # Re-placement check: physics trace positions re-rendered must reproduce the frames.
    mujoco.mj_resetData(scene.model, scene.data)
    scene.init_state(params)
    mujoco.mj_forward(scene.model, scene.data)
    check_idx = target_indices(0)
    re = render_placed(scene, placements_for(kind, scene, trace, params, 0, check_idx, 1e-9, ppm))
    replace_err = float(np.abs(re.astype(np.int16) - frames[list(check_idx)].astype(np.int16)).mean())

    kin, jit, ctx_ok = [], [], []
    for o in OFFSETS:
        t_idx = target_indices(o)
        ctx_ok.append(bool(ctx_indices(o)[1] < int(res["event_frame"])))
        kin.append(render_placed(scene, placements_for(kind, scene, trace, params, o, t_idx, 0.0, ppm)))
        jit.append(render_placed(scene, placements_for(kind, scene, trace, params, o, t_idx, 1.0, ppm)))
    kin, jit = np.stack(kin), np.stack(jit)
    if scene._r is not None:
        scene._r.close()
        scene._r = None

    sid = row["id"]
    np.savez_compressed(
        out_root / "inputs" / f"{sid}.npz",
        name=sid, family=fam, kind=kind, frames_gt=frames, frames_kin=kin, frames_jit=jit,
        trace=trace.astype(np.float32), offsets=np.asarray(OFFSETS),
        ctx_idx=np.asarray([ctx_indices(o) for o in OFFSETS]),
        target_idx=np.asarray([target_indices(o) for o in OFFSETS]),
        ctx_ok=np.asarray(ctx_ok), S=float("nan") if row["S"] is None else float(row["S"]),
        outcome=int(row["outcome"]), event_frame=int(res["event_frame"]),
        px_per_m=ppm, replace_err=replace_err, capture_fps=FPS,
    )
    # Preview: rows = GT | K | J at offset 0 targets, plus the two context frames.
    cidx = ctx_indices(0)
    strip = np.concatenate([
        np.concatenate([frames[cidx[0]], frames[cidx[1]]] + [frames[t] for t in check_idx], axis=1),
        np.concatenate([np.zeros_like(frames[0])] * 2 + list(kin[0]), axis=1),
        np.concatenate([np.zeros_like(frames[0])] * 2 + list(jit[0]), axis=1)], axis=0)
    write_png(out_root / "previews" / f"{sid}.png", strip)
    write_video(out_root / "previews" / f"{sid}_gt.mp4", frames, CFG)
    rec = dict(row)
    rec.update({"vj_input": f"inputs/{sid}.npz", "kind": kind, "event_frame_vj": int(res["event_frame"]),
                "ctx_ok": ctx_ok, "px_per_m": ppm, "replace_err": replace_err,
                "render_s": round(time.time() - t0, 1)})
    print(f"{sid}: event {res['event_frame']} ctx_ok {ctx_ok} replace_err {replace_err:.3f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None, help="sample ids to build")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    for sub in ("inputs", "previews"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    if args.only:
        rows = [r for r in rows if r["id"] in set(args.only)]
    recs = []
    mpath = out / "manifest.jsonl"
    old = {}
    if mpath.exists():
        old = {json.loads(l)["id"]: json.loads(l) for l in mpath.read_text().splitlines() if l.strip()}
    for row in rows:
        if row["family"] not in FAMILY_CLS:
            print(f"skip {row['id']}: family {row['family']} has no kinematic model", file=sys.stderr)
            continue
        if args.skip_existing and (out / "inputs" / f"{row['id']}.npz").exists() and row["id"] in old:
            recs.append(old[row["id"]])
            continue
        recs.append(build(row, out))
        old[recs[-1]["id"]] = recs[-1]
        mpath.write_text("".join(json.dumps(r) + "\n" for r in old.values()))
    print(f"done: {len(recs)} samples -> {mpath}")


if __name__ == "__main__":
    main()
