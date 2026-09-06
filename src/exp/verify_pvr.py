"""PVR PoC verification (physics without rendering + mask checks on the generated set).

1. Physics identity: the decoy scenes must reproduce the Phase A/B physics.
   collide: TwoBallDecoy(r_d=0) vs TwoBallWB -> identical traces (same layout).
   edge: SupportEdgeDecoy(x_left=-1.10) vs SupportEdgeWB -> identical traces (same plate).
   hill: HillDecoy(h_d=0) vs RollingHillWB -> same outcome per S and the same trace once the
   approach offset (0.35 m vs 0.40 m) is removed; residual = hfield resampling.
2. Decoy clearance and conditioning gates over every condition (the generator gate).
3. Mask check on artifacts/pvr/<scene>/seg: IoU(segmentation ball, red colour mask) per
   conditioning frame >= 0.9 -- catches any orientation mismatch of the segmentation path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core.threshold.hill_decoy import HillDecoy
from core.threshold.rolling_hill import RollingHillWB
from core.threshold.support_edge import SupportEdgeWB
from core.threshold.support_edge_decoy import SupportEdgeDecoy
from core.threshold.two_ball import TwoBallWB
from core.threshold.two_ball_decoy import TwoBallDecoy
from exp.analyze_bundle_a import color_mask
from exp.pvr_regions import region_masks
from gen.make_pvr import physics_gate
from gen.pvr_spec import CONDITIONS, SCENES


def identity_checks():
    out = {}
    rows = []
    for S in (0.5, 1.0, 2.0):
        pa = TwoBallWB.params_for_S(S)
        pb = TwoBallDecoy.params_for_S(S, r_d=0.0)
        ra, rb = TwoBallWB(pa).run(pa), TwoBallDecoy(pb).run(pb)
        diff = float(np.abs(ra["trace"] - rb["trace"]).max())
        rows.append({"S": S, "outcome": [ra["outcome"], rb["outcome"]], "event": [ra["event_frame"], rb["event_frame"]],
                     "trace_max_abs_diff": diff})
        print(f"collide S={S}: outcome {ra['outcome']}/{rb['outcome']} event {ra['event_frame']}/{rb['event_frame']} "
              f"trace diff {diff:.2e}")
    out["collide"] = rows
    rows = []
    for S in (0.63, 1.26, 2.0):
        pa = SupportEdgeWB.params_for_S(S)
        pb = dict(v0=pa["v0"], x_edge=pa["x_edge"], x_left=-1.10, variant="plain")
        ra, rb = SupportEdgeWB(pa).run(pa), SupportEdgeDecoy(pb).run(pb)
        diff = float(np.abs(ra["trace"] - rb["trace"]).max())
        rows.append({"S": S, "outcome": [ra["outcome"], rb["outcome"]], "event": [ra["event_frame"], rb["event_frame"]],
                     "trace_max_abs_diff": diff})
        print(f"edge S={S}: outcome {ra['outcome']}/{rb['outcome']} event {ra['event_frame']}/{rb['event_frame']} "
              f"trace diff {diff:.2e}")
    out["edge"] = rows
    rows = []
    for S in (0.5, 0.79, 1.26, 2.0):
        pa = RollingHillWB.params_for_S(S)
        pb = HillDecoy.params_for_S(S, h_d=0.0)
        ra, rb = RollingHillWB(pa).run(pa), HillDecoy(pb).run(pb)
        xa, xb = ra["trace"][:, 0] - RollingHillWB.BASE_X, rb["trace"][:, 0] - HillDecoy.BASE_X
        ka, kb = int(np.argmax(xa > -0.05)), int(np.argmax(xb > -0.05))
        n = min(len(xa) - ka, len(xb) - kb, 16)
        d = float(np.abs(xa[ka:ka + n] - xb[kb:kb + n]).max())
        rows.append({"S": S, "outcome": [ra["outcome"], rb["outcome"]], "event": [ra["event_frame"], rb["event_frame"]],
                     "aligned_x_max_abs_diff": d, "align_frames": [ka, kb]})
        print(f"hill S={S}: outcome {ra['outcome']}/{rb['outcome']} event {ra['event_frame']}/{rb['event_frame']} "
              f"aligned x diff {d:.4f} m")
    out["hill"] = rows
    return out


def gates():
    out = {}
    for scene, cls in SCENES.items():
        rows = []
        for cond in CONDITIONS[scene]:
            lab, problems = physics_gate(scene, cls, cond)
            r = {"cond": cond["id"], "outcome": lab["outcome"], "expected": cond["expected"],
                 "event_frame": lab["event_frame"], "decoy_contact_frame": lab.get("decoy_contact_frame", -1),
                 "problems": problems}
            for k in ("t_edge", "edge_frame", "v_post_ratio", "x_min_horizon", "xA_min_horizon"):
                if k in lab:
                    r[k] = float(lab[k])
            rows.append(r)
        bad = [r for r in rows if r["problems"]]
        print(f"{scene}: {len(rows)} conditions, gate problems: {len(bad)}"
              + ("" if not bad else f" -> {[(r['cond'], r['problems']) for r in bad]}"))
        out[scene] = rows
    return out


def mask_check(root: Path):
    out = {}
    for scene in SCENES:
        mpath = root / scene / "manifest.jsonl"
        if not mpath.exists():
            continue
        manifest = [json.loads(l) for l in mpath.read_text().splitlines() if l.strip()]
        ious, worst = [], None
        for record in manifest:
            inp = np.load(root / scene / record["input_npz"])
            segz = np.load(root / scene / record["seg_npz"], allow_pickle=False)
            seg, meta = segz["seg"], json.loads(str(segz["meta"]))
            for k in range(5):
                masks = region_masks(seg[k].astype(np.int32), meta, scene, record["params"])
                red = color_mask(inp["condition_primary"][k], "red")
                ball = masks["ball"]
                inter, union = float((red & ball).sum()), float((red | ball).sum())
                iou = inter / union if union > 0 else 0.0
                ious.append(iou)
                if worst is None or iou < worst[0]:
                    worst = (iou, record["id"], k)
        out[scene] = {"n": len(ious), "iou_min": float(min(ious)), "iou_mean": float(np.mean(ious)), "worst": worst}
        print(f"{scene}: ball-mask IoU min {min(ious):.3f} mean {np.mean(ious):.3f} (worst {worst[1]} frame {worst[2]}) "
              f"{'OK' if min(ious) >= 0.9 else 'FAIL'}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/pvr"))
    parser.add_argument("--skip-physics", action="store_true")
    parser.add_argument("--skip-masks", action="store_true")
    args = parser.parse_args()
    out = {}
    if not args.skip_physics:
        out["identity"] = identity_checks()
        out["gates"] = gates()
    if not args.skip_masks:
        out["masks"] = mask_check(args.root)
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "verify.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
