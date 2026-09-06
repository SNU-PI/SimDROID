"""Generate the PVR PoC benchmark: three scenes x their condition sets, native 832x480,
exact 16 FPS, Bundle A manifest schema (so the Cosmos runner applies unchanged) plus
per-frame segmentation ids for the region masks of the attribution track.

Per condition: physics gate (event after the conditioning window, expected outcome,
decoy never touched, conditioning never touches the structure), then the render (double
render determinism), the five-frame conditioning video, the 21-frame GT clip, the key
frames, the segmentation ids of the 21 GT frames (int16), a preview strip.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gen.render import DIAG_CFG, roll, write_video
from gen.make_bundle_a import family_preview
from gen.pvr_spec import (CONDITIONS, CURRENT_IDX, FAMILY, HORIZON, PRINCIPLE, PROMPTS, SCENES)
from gen.bundle_a_spec import CONDITION_INDICES, FUTURE_OFFSET


def physics_gate(scene, cls, cond):
    """Run the physics (no render) and check the design gates; returns (labels, problems)."""
    p = cond["params"]
    sc = cls()
    lab = sc.run(p, render=False)
    problems = []
    if lab["event_frame"] <= CURRENT_IDX:
        problems.append(f"event at frame {lab['event_frame']} inside the conditioning window")
    exp = cond["expected"]
    if exp is not None and int(lab["outcome"]) != int(exp):
        problems.append(f"outcome {lab['outcome']} != expected {exp}")
    dcf = int(lab.get("decoy_contact_frame", -1))
    if dcf >= 0:
        # Core / ladder cells must never touch the decoy.  Extension stopper cells rebound
        # at ~v0 and may reach it late in the horizon, after the readout window closed.
        read_end = int(lab.get("read_frame", lab["event_frame"])) + 3
        if cond["group"] != "ext" or dcf <= read_end:
            problems.append(f"ball reaches the decoy at frame {dcf} (readout ends {read_end})")
    if scene == "hill":
        if lab["cond_end_x"] + cls.BALL_R > cls.BASE_X:
            problems.append("conditioning touches the slope")
    elif scene == "collide":
        tr = lab["trace"]
        if p.get("variant", "ball") in ("ball", "miss") and tr[CURRENT_IDX, 0] + cls.R1 > -float(p["r2"]):
            problems.append("conditioning touches the target")
    elif scene == "edge":
        if lab["cond_end_x"] + cls.BALL_R > float(p["x_edge"]):
            problems.append("conditioning reaches the edge")
        if exp == 1 and lab["event_frame"] > HORIZON - 1:
            problems.append(f"fall at frame {lab['event_frame']} is outside the horizon")
    return lab, problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/pvr"))
    parser.add_argument("--scenes", nargs="*", choices=tuple(SCENES), default=tuple(SCENES))
    parser.add_argument("--ids", nargs="*", help="subset of condition ids (smoke tests)")
    parser.add_argument("--allow-gate-failures", action="store_true")
    args = parser.parse_args()

    for scene in args.scenes:
        cls = SCENES[scene]
        root = (args.output_dir / scene).resolve()
        for sub in ("inputs", "conditions", "gt", "previews", "seg"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        family = FAMILY[scene]
        prompt = PROMPTS[scene]
        records, entries, gate_log = [], [], {}
        for cond in CONDITIONS[scene]:
            if args.ids and cond["id"] not in args.ids:
                continue
            sample_id = f"{family}_{cond['id']}"
            lab0, problems = physics_gate(scene, cls, cond)
            gate_log[sample_id] = problems
            if problems:
                msg = f"{sample_id}: GATE FAIL: " + "; ".join(problems)
                if not args.allow_gate_failures:
                    raise RuntimeError(msg)
                print(msg, flush=True)
            frames, labels = roll(cls, cond["params"], cls.cam, DIAG_CFG, segment=True)
            seg = labels.pop("seg")
            meta = {k: labels.pop(k) for k in ("geom_names", "geom_bodies", "body_names", "body_parents")}
            condition = frames[list(CONDITION_INDICES)]
            future_idx = CURRENT_IDX + FUTURE_OFFSET
            event_idx = min(int(labels["event_frame"]), len(frames) - 1)
            input_rel = Path("inputs") / f"{sample_id}.npz"
            np.savez_compressed(
                root / input_rel, name=sample_id, principle=PRINCIPLE, family=family,
                condition_primary=condition, current_primary=frames[CURRENT_IDX],
                gt_future_primary=frames[future_idx], gt_event_primary=frames[event_idx],
                gt_final_primary=frames[HORIZON - 1],
                S=float(cls.S_of(cond["params"])), outcome=int(labels["outcome"]),
                event_frame=int(labels["event_frame"]), condition_indices=np.asarray(CONDITION_INDICES),
                current_idx=CURRENT_IDX, future_idx=future_idx, capture_fps=DIAG_CFG.fps)
            seg_rel = Path("seg") / f"{sample_id}.npz"
            np.savez_compressed(root / seg_rel, seg=seg[:HORIZON].astype(np.int16),
                                meta=json.dumps(meta))
            condition_rel = Path("conditions") / f"{sample_id}.mp4"
            gt_rel = Path("gt") / f"{sample_id}.mp4"
            write_video(root / condition_rel, condition, DIAG_CFG)
            write_video(root / gt_rel, frames[:HORIZON], DIAG_CFG)
            (root / "conditions" / f"{sample_id}.txt").write_text(prompt)
            S = float(cls.S_of(cond["params"]))
            record = {
                "id": sample_id, "principle": PRINCIPLE, "family": family, "scene": scene,
                "cond": cond["id"], "group": cond["group"], "note": cond["note"],
                "kind": "dynamic", "role": "grid",
                "S": None if np.isnan(S) else S, "margin": float(labels["margin"]),
                "params": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in cond["params"].items()},
                "variant": cond["params"].get("variant", "plain"),
                "relevant": cond["relevant"], "decoy": cond["decoy"],
                "expected": cond["expected"],
                "outcome": int(labels["outcome"]), "event_frame": int(labels["event_frame"]),
                "t_event_s": float(labels["t_event_s"]), "decided": bool(labels["decided"]),
                "in_map": bool(labels["decided"]) and int(labels["event_frame"]) <= HORIZON - 1,
                "boundary": bool(S is not None and not np.isnan(S) and abs(S - 1.0) <= 0.04),
                "condition_indices": list(CONDITION_INDICES), "condition_pixel_frames": len(CONDITION_INDICES),
                "current_idx": CURRENT_IDX, "future_idx": future_idx, "rollout_frames": HORIZON,
                "capture_fps": DIAG_CFG.fps, "render_size": [DIAG_CFG.height, DIAG_CFG.width],
                "input_npz": str(input_rel), "condition_path": str(condition_rel), "gt_clip": str(gt_rel),
                "seg_npz": str(seg_rel), "prompt": prompt, "gate_problems": problems,
            }
            for key in ("t_edge", "edge_frame", "x_min_horizon", "xA_min_horizon", "decoy_clear", "decoy_contact_frame",
                        "v_post_ratio", "momentum_ratio", "e_eff", "read_frame", "speed_ratio",
                        "energy_drift", "slip_max", "cond_end_x"):
                if key in labels:
                    v = labels[key]
                    record[key] = bool(v) if isinstance(v, (bool, np.bool_)) else float(v)
            records.append(record)
            entries.append((record, frames))
            print(f"{sample_id}: y={record['outcome']} ev={record['event_frame']} S={record['S']} "
                  f"{'GATE:' + ';'.join(problems) if problems else 'ok'}", flush=True)
        manifest = root / "manifest.jsonl"
        manifest.write_text("".join(json.dumps(r) + "\n" for r in records))
        family_preview(root / "previews" / f"{family}.png", family, entries)
        (root / "summary.json").write_text(json.dumps({
            "principle": PRINCIPLE, "scene": scene, "family": family, "samples": len(records),
            "render_size": [DIAG_CFG.height, DIAG_CFG.width], "capture_fps": DIAG_CFG.fps,
            "conditioning_frames": len(CONDITION_INDICES), "rollout_frames": HORIZON,
            "gate_problems": {k: v for k, v in gate_log.items() if v}, "manifest": str(manifest)}, indent=2))
        print(f"[{scene}] {len(records)} samples -> {manifest}", flush=True)


if __name__ == "__main__":
    main()
