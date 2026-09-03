"""Generate the Phase B workbench benchmark: pendulum_rod_wb (P1-B) and support_edge_wb (P3-A).

Output layout and manifest schema are those of Bundle A / Phase A
(make_bundle_a.build_sample), so the Cosmos runner and the adjudicator apply
unchanged.  Gates inside build_sample: analytic vs simulated outcome off the
boundary band, event strictly after the conditioning window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gen.render import DIAG_CFG
from gen.make_bundle_a import build_sample, family_preview
from gen.phase_b_spec import (CONDITION_INDICES, ORACLE, PHASE_B, PRINCIPLE, PROMPTS,
                              ROLLOUT_FRAMES, S_GRID, phase_specs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase_b"))
    parser.add_argument("--families", nargs="*", choices=tuple(PHASE_B), default=tuple(PHASE_B))
    args = parser.parse_args()
    root = args.output_dir.resolve()
    for sub in ("inputs", "conditions", "gt", "previews"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    kw = dict(prompts=PROMPTS, oracle=ORACLE, principle=PRINCIPLE)

    specs = phase_specs()
    records = []
    for name in args.families:
        cls = PHASE_B[name]
        entries = []
        for index, (S, params) in enumerate(zip(S_GRID, specs[name])):
            sample_id = f"{name}_{index:02d}"
            record, frames = build_sample(name, cls, sample_id, float(S), params, root, **kw)
            records.append(record)
            entries.append((record, frames))
            print(f"{sample_id}: S={S:.2f} y={record['outcome']} ev={record['event_frame']} "
                  f"map={int(record['in_map'])} drift={record.get('energy_drift')}", flush=True)
        family_preview(root / "previews" / f"{name}.png", name, entries)

    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in records))
    summary = {
        "principle": PRINCIPLE, "samples": len(records),
        "families": list(args.families), "S_grid": list(S_GRID),
        "render_size": [DIAG_CFG.height, DIAG_CFG.width], "capture_fps": DIAG_CFG.fps,
        "condition_indices": list(CONDITION_INDICES), "rollout_frames": ROLLOUT_FRAMES,
        "in_map": {r["id"]: r["in_map"] for r in records},
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {len(records)} samples to {root}")


if __name__ == "__main__":
    main()
