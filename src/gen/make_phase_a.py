"""Generate the Phase A workbench benchmark (PhysicsGen redesign, 2026-09-02).

Output layout and manifest schema are those of Bundle A (make_bundle_a.py), so
the Cosmos runner and the adjudicator apply unchanged.  Families: kin_roll
(P0 control, speed grid, role=control), hill_roll_wb (P1, S grid),
two_ball_wb (P2, S grid), plus the two pre-checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gen.render import DIAG_CFG
from gen.make_bundle_a import build_sample, family_preview
from gen.phase_a_spec import (CONDITION_INDICES, CONTROLS, ORACLE, PHASE_A, PRECHECKS,
                              PRINCIPLE, PROMPTS, ROLLOUT_FRAMES, S_GRID, phase_specs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase_a"))
    parser.add_argument("--families", nargs="*",
                        choices=tuple(PHASE_A) + tuple(CONTROLS),
                        default=tuple(CONTROLS) + tuple(PHASE_A))
    parser.add_argument("--skip-prechecks", action="store_true")
    args = parser.parse_args()
    root = args.output_dir.resolve()
    for sub in ("inputs", "conditions", "gt", "previews"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    kw = dict(prompts=PROMPTS, oracle=ORACLE, principle=PRINCIPLE)

    specs = phase_specs()
    records = []
    for name in args.families:
        entries = []
        if name in CONTROLS:
            cls, grid = CONTROLS[name]
            for index, v0 in enumerate(grid):
                sample_id = f"{name}_{index:02d}"
                record, frames = build_sample(name, cls, sample_id, None, dict(v0=float(v0)),
                                              root, role="control", **kw)
                record["v0"] = float(v0)
                records.append(record)
                entries.append((record, frames))
                print(f"{sample_id}: v0={v0:.2f} y={record['outcome']} ev={record['event_frame']} "
                      f"speed_ratio={record.get('speed_ratio')} drift={record.get('energy_drift'):.4f}")
        else:
            cls = PHASE_A[name]
            for index, (S, params) in enumerate(zip(S_GRID, specs[name])):
                sample_id = f"{name}_{index:02d}"
                record, frames = build_sample(name, cls, sample_id, float(S), params, root, **kw)
                records.append(record)
                entries.append((record, frames))
                proxy = record["pixel_proxy"]
                print(f"{sample_id}: S={S:.2f} y={record['outcome']} ev={record['event_frame']} "
                      f"map={int(record['in_map'])} proxy={proxy and proxy['value']}")
        family_preview(root / "previews" / f"{name}.png", name, entries)

    if not args.skip_prechecks:
        cls, kwargs = PRECHECKS["hill_roll_wb_pre"]
        record, _ = build_sample("hill_roll_wb", cls, "hill_roll_wb_pre", None,
                                 cls.params_for_S(kwargs["S"]), root, role="precheck", **kw)
        record["S"] = kwargs["S"]
        records.append(record)
        print(f"hill_roll_wb_pre: y={record['outcome']} ev={record['event_frame']}")
        cls, _ = PRECHECKS["wall_bounce_wb_pre"]
        record, _ = build_sample("wall_bounce_wb_pre", cls, "wall_bounce_wb_pre", None,
                                 dict(v0=cls.V0), root, role="precheck", **kw)
        records.append(record)
        print(f"wall_bounce_wb_pre: y={record['outcome']} ev={record['event_frame']}")

    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in records))
    summary = {
        "principle": PRINCIPLE, "samples": len(records),
        "families": list(args.families), "S_grid": list(S_GRID),
        "control_grid": {k: list(v[1]) for k, v in CONTROLS.items()},
        "render_size": [DIAG_CFG.height, DIAG_CFG.width], "capture_fps": DIAG_CFG.fps,
        "conditioning_frames": len(CONDITION_INDICES),
        "rollout_horizon_seconds": ROLLOUT_FRAMES / DIAG_CFG.fps,
        "manifest": str(manifest),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
