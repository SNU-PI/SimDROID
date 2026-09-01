"""Adjudicate the prompt-contrast arm with the frozen Bundle A rules.

For each boundary-flank sample the same conditioning ran under an oracle
'correct' and an oracle 'wrong' prompt.  If text steers the physics readout,
P(pass) should split between the two arms; if the motion prior dominates,
the arms should match the neutral-prompt curve.
"""
import argparse, csv, json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from exp.analyze_bundle_a import adjudicate, gt_reference

ROOT = Path("artifacts/bundle_a")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    args = ap.parse_args()

    manifest = [json.loads(l) for l in (ROOT/"manifest_oracle.jsonl").read_text().splitlines() if l.strip()]
    refs = {}
    rows = []
    for record in manifest:
        if record["base_id"] not in refs:
            refs[record["base_id"]] = gt_reference(record, ROOT)
        ref = refs[record["base_id"]]
        for seed in args.seeds:
            clip = ROOT/"cosmos_v2w_oracle"/f"seed_{seed:02d}"/record["id"]/"rollout.mp4"
            if not clip.exists():
                continue
            frames = imageio.mimread(clip, memtest=False)[:record["rollout_frames"]]
            res = adjudicate(record, ref, frames)
            rows.append({"id": record["id"], "base_id": record["base_id"],
                         "family": record["family"], "S": record["S"],
                         "outcome": record["outcome"], "oracle": record["oracle"],
                         "seed": seed, "prediction": res["prediction"],
                         "valid_frac": res["valid_frac"]})
    out = ROOT/"analysis"
    with (out/"oracle_samples.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    summary = {}
    for fam in ("hill_roll", "two_ball", "pendulum_rod"):
        for S in sorted({r["S"] for r in rows if r["family"] == fam}):
            cell = {}
            for tag in ("correct", "wrong"):
                sel = [r for r in rows if r["family"] == fam and r["S"] == S and r["oracle"] == tag]
                dec = [r for r in sel if r["prediction"] in (0, 1)]
                cell[tag] = {"n": len(sel), "n_decided": len(dec),
                             "p_pass": float(np.mean([r["prediction"] for r in dec])) if dec else None}
            summary[f"{fam}_S{S}"] = cell
    (out/"oracle_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
