"""Preview strips for DiLA rollouts: GT / kinematic future / RAE reconstruction / DiLA prediction.

Rows per sample: (1) GT frames (context + targets), (2) context + kinematic counterfactual,
(3) RAE reconstruction of the GT latents (decoder ceiling), (4) DiLA prediction
(context reconstruction then the predicted steps).  Prints latent distances for a quick read.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts/dila/phase_a")
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=0.5)
    args = ap.parse_args()
    root = Path(args.root)
    rows = []
    for sid in args.ids:
        d = np.load(root / "latents" / f"{sid}_o{args.offset}{args.tag}.npz")
        inp = np.load(root / "inputs" / f"{sid}.npz")
        oi = list(inp["offsets"]).index(args.offset)
        tidx, cidx = d["target_idx"], d["ctx_idx"]
        kin_off = int(d["kin_offset"]) if "kin_offset" in d.files else 0
        blank = np.zeros_like(inp["frames_gt"][0])
        gt = [inp["frames_gt"][i] for i in cidx] + [inp["frames_gt"][i] for i in tidx]
        kin = [inp["frames_gt"][i] for i in cidx] + list(inp["frames_kin"][oi][kin_off:])
        rec = [blank] * (len(cidx) - 1) + [d["rgb_ctx_recon"][0]] + list(d["rgb_gt_recon"])
        pred = [blank] * (len(cidx) - 1) + [d["rgb_ctx_recon"][0]] + list(d["rgb_pred"])
        for r in (gt, kin, rec, pred):
            rows.append(np.concatenate(r, axis=1))
        zp, zg, zl = d["z_pred"].astype(np.float32), d["z_gt"].astype(np.float32), d["z_ctx"][-1].astype(np.float32)
        zk = d["z_kin"].astype(np.float32)
        print(sid, f"S={float(d['S']):.2f} outcome={int(d['outcome'])} ctx={list(cidx)} tgt={list(tidx)}",
              "la_norm=%.2f" % float(np.linalg.norm(d["la_ctx"][-1])),
              "dP=" + " ".join(f"{np.abs(zp[k] - zg[k]).mean():.3f}" for k in range(len(zp))),
              "dK=" + " ".join(f"{np.abs(zp[k] - zk[k]).mean():.3f}" for k in range(len(zp))),
              "freeze_dP=" + " ".join(f"{np.abs(zl - zg[k]).mean():.3f}" for k in range(len(zp))),
              "sep=" + " ".join(f"{np.abs(zg[k] - zk[k]).mean():.3f}" for k in range(len(zp))))
    img = np.concatenate(rows, axis=0)
    im = Image.fromarray(img)
    if args.scale != 1:
        im = im.resize((int(img.shape[1] * args.scale), int(img.shape[0] * args.scale)))
    im.save(args.out)
    print("saved", args.out, img.shape)


if __name__ == "__main__":
    main()
