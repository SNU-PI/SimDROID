"""V-JEPA 2-AC zero-action latent rollouts on the latent-track inputs.

For each sample and context offset: encode the two context frames, roll the
action-conditioned predictor N_STEPS steps with zero actions and a constant
end-effector state, and encode the candidate futures (physics P, kinematic K,
jitter J) at the same target frames.  Latents (layer-normed, fp16) go to
<root>/latents/<id>_o<offset>.npz; the full 24-frame physics sequence to
<root>/latents/<id>_gtseq.npz.

Run in the torch env (GPU):
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python src/exp/rollout_vjepa_ac.py --root artifacts/vjepa_ac/phase_a
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

VJEPA_DIR = os.environ.get("VJEPA_DIR", "/mnt/nvme/migration/jihun/SimDROID/external/vjepa2")
VJEPA_CKPT = os.environ.get("VJEPA_CKPT", "/mnt/nvme/migration/jihun/SimDROID/data/stage0/vjepa2-ac-vitg.pt")
sys.path.insert(0, VJEPA_DIR)

from gen.vjepa_spec import N_STEPS, POSE, SIZE  # noqa: E402


def load_model(device):
    from src.hub.backbones import _make_vjepa2_ac_model, _clean_backbone_key
    from app.vjepa_droid.transforms import make_transforms
    enc, pred = _make_vjepa2_ac_model(model_name="vit_ac_giant", img_size=SIZE, pretrained=False)
    ck = torch.load(VJEPA_CKPT, map_location="cpu", weights_only=False)
    m_enc = enc.load_state_dict(_clean_backbone_key(ck["encoder"]), strict=False)
    m_pred = pred.load_state_dict(_clean_backbone_key(ck["predictor"]), strict=True)
    info = {"ckpt_keys": sorted(ck.keys()), "encoder_missing": list(m_enc.missing_keys),
            "encoder_unexpected": list(m_enc.unexpected_keys), "predictor": "strict ok",
            "ckpt_epoch": int(ck.get("epoch", -1)) if isinstance(ck.get("epoch", -1), (int, float)) else str(ck.get("epoch"))}
    del ck
    enc = enc.to(device).eval()
    pred = pred.to(device).eval()
    for p in list(enc.parameters()) + list(pred.parameters()):
        p.requires_grad_(False)
    tf = make_transforms(random_horizontal_flip=False, random_resize_aspect_ratio=(1.0, 1.0),
                         random_resize_scale=(1.0, 1.0), reprob=0.0, auto_augment=False,
                         motion_shift=False, crop_size=SIZE)
    return enc, pred, tf, info


@torch.no_grad()
def encode(enc, tf, frames, device):
    """frames: (T,H,W,3) uint8 -> (T, N, D) layer-normed float32 (each frame as a 2-frame tubelet)."""
    clip = tf(np.ascontiguousarray(frames))            # (C, T, H, W)
    c = clip.unsqueeze(0).permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    h = enc(c.to(device))                              # (T, N, D)
    return F.layer_norm(h, (h.size(-1),))


@torch.no_grad()
def rollout(pred, z_ctx, pose, n_steps, device):
    """z_ctx: (T0, N, D) -> (n_steps, N, D) autoregressive predictions with zero actions."""
    T0, N, D = z_ctx.shape
    z = z_ctx.reshape(1, T0 * N, D)
    pose_t = torch.tensor(pose, dtype=torch.float32, device=device).view(1, 1, 7)
    outs = []
    for k in range(n_steps):
        T = z.shape[1] // N
        actions = torch.zeros(1, T, 7, device=device)
        states = pose_t.repeat(1, T, 1)
        nxt = pred(z, actions, states)[:, -N:]
        nxt = F.layer_norm(nxt, (nxt.size(-1),))
        outs.append(nxt[0])
        z = torch.cat([z, nxt], dim=1)
    return torch.stack(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--pose-b", action="store_true", help="also roll out with a second EE pose")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    device = "cuda"
    t0 = time.time()
    enc, pred, tf, info = load_model(device)
    print(f"model loaded in {time.time() - t0:.0f}s: {json.dumps(info)[:600]}", flush=True)
    if args.selftest:
        f = (np.random.RandomState(0).rand(3, SIZE, SIZE, 3) * 255).astype(np.uint8)
        z = encode(enc, tf, f, device)
        zp = rollout(pred, z[:2], POSE, 2, device)
        print("selftest: z", tuple(z.shape), "pred", tuple(zp.shape),
              "vram GB", round(torch.cuda.max_memory_allocated() / 1e9, 2), flush=True)
        return
    root = Path(args.root)
    (root / "latents").mkdir(exist_ok=True)
    rows = [json.loads(l) for l in (root / "manifest.jsonl").read_text().splitlines() if l.strip()]
    if args.only:
        rows = [r for r in rows if r["id"] in set(args.only)]
    pose_b = [0.673, 0.029, 0.332, -3.051, 0.018, -1.912, 0.997]
    for row in rows:
        sid = row["id"]
        d = np.load(root / row["vj_input"])
        frames = d["frames_gt"]
        ts = time.time()
        z_seq = encode(enc, tf, frames, device)                      # (24, N, D)
        np.savez_compressed(root / "latents" / f"{sid}_gtseq.npz", z=z_seq.half().cpu().numpy())
        for oi, o in enumerate(d["offsets"]):
            cidx = d["ctx_idx"][oi]
            tidx = d["target_idx"][oi]
            z_ctx = z_seq[list(cidx)]
            z_pred = rollout(pred, z_ctx, POSE, N_STEPS, device)
            out = {"z_ctx": z_ctx, "z_pred": z_pred, "z_gt": z_seq[list(tidx)],
                   "z_kin": encode(enc, tf, d["frames_kin"][oi], device),
                   "z_jit": encode(enc, tf, d["frames_jit"][oi], device)}
            if args.pose_b:
                out["z_pred_b"] = rollout(pred, z_ctx, pose_b, N_STEPS, device)
            np.savez_compressed(
                root / "latents" / f"{sid}_o{int(o)}.npz",
                **{k: v.half().cpu().numpy() for k, v in out.items()},
                offset=int(o), ctx_idx=cidx, target_idx=tidx, ctx_ok=bool(d["ctx_ok"][oi]),
                family=str(d["family"]), kind=str(d["kind"]), S=float(d["S"]),
                outcome=int(d["outcome"]), event_frame=int(d["event_frame"]),
                px_per_m=float(d["px_per_m"]))
        print(f"{sid}: {len(d['offsets'])} offsets in {time.time() - ts:.1f}s", flush=True)
    (root / "latents" / "model_info.json").write_text(json.dumps(info, indent=1))
    print(f"done in {time.time() - t0:.0f}s, peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
