"""DiLA (Disentangled Latent Action world model) past-only rollouts on the V-JEPA input bundles.

Reuses the 256x256 / 16 fps bundles from make_vjepa_inputs.py (GT frames + kinematic and
jitter counterfactual futures at the 4 fps target grid) and writes latents in the layout
analyze_vjepa_ac.py reads, so the same gate / direction / freeze metrics apply.

DiLA has no past-only predictor: get_latent_actions(x) is an inverse-dynamics model over the
frames it is given.  We feed only the context frames, take their latent action(s), and choose
the future actions by a policy:
  hold  - repeat the last context action (default; == "keep doing what the context did")
  zero  - all-zero latent action
  mean  - mean of the context actions
The structure rollout is then autoregressive (forward_dynamics + content memory) and the
predicted DINOv2 embeddings are decoded with the RAE decoder for a pixel-space readout.

Time base (--stride, frames at 16 fps per model step):
  4 (contract, = V-JEPA track): context {o, o+4}, predict o+8 .. o+20 (4 steps)
  2: context {o, o+2, o+4}, predict o+6 .. o+20 (8 steps)
  1: context {o .. o+4} (5 frames, = Phase A conditioning), predict o+5 .. o+20 (16 steps)
For every stride the saved z_pred / rgb_pred are the predictions at the 4 fps targets
{o+8, o+12, o+16, o+20} so the analyzer contract is unchanged; the full predicted image
sequence is kept as rgb_pred_full / pred_idx.

Latents are DINOv2-with-registers-base features normalised by the RAE statistics,
stored as (256 tokens, 768) per frame (raster order of the 16x16 grid).
"""
from __future__ import annotations

import os

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from gen.vjepa_spec import SIZE

REPO = Path(os.environ.get("DILA_DIR", "external/DiLA"))
CKPT = REPO / "checkpoints" / "model.pt"
DINO = sorted((REPO / "pretrained" / "models--facebook--dinov2-with-registers-base" / "snapshots").glob("*"))
DECODER = REPO / "pretrained" / "decoders" / "dinov2" / "wReg_base" / "ViTXL" / "dinov2_decoder.pt"
STATS = REPO / "pretrained" / "stats" / "dinov2" / "wReg_base" / "imagenet1k" / "stat.pt"
DECODER_CFG = REPO / "pretrained" / "decoder_config"   # ViTXL config with the placeholder patch_size filled in


def load_model(device):
    sys.path.insert(0, str(REPO))
    import hydra
    from hydra import compose, initialize_config_dir
    from models.RAE import RAE

    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base=None):
        cfg = compose(config_name="train_model", overrides=["phase=2"])
    dino = str(DINO[-1])
    rae = RAE(encoder_cls=cfg.RAE.encoder_cls, encoder_config_path=dino,
              encoder_input_size=cfg.RAE.encoder_input_size,
              encoder_params={"dinov2_path": dino, "normalize": bool(cfg.RAE.encoder_params.normalize)},
              decoder_config_path=str(DECODER_CFG),
              decoder_patch_size=cfg.RAE.decoder_patch_size,
              pretrained_decoder_path=str(DECODER), reshape_to_2d=cfg.RAE.reshape_to_2d,
              noise_tau=cfg.RAE.noise_tau, normalization_stat_path=str(STATS)).to(device).eval()
    structure_encoder = hydra.utils.instantiate(cfg.structure_encoder).to(device)
    content_fusion = hydra.utils.instantiate(cfg.content_fusion).to(device)
    model = hydra.utils.instantiate(cfg.world_model, structure_encoder=structure_encoder,
                                    content_fusion=content_fusion, phase=2).to(device).eval()
    data = torch.load(CKPT, map_location="cpu")
    state = data.get("model", data.get("module", data)) if isinstance(data, dict) else data
    msg = model.load_state_dict(state, strict=False)
    info = {"ckpt": str(CKPT), "missing": list(msg.missing_keys), "unexpected": list(msg.unexpected_keys),
            "n_params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
            "structure_dim": int(model.structure_dim), "action_dim": int(model.action_dim)}
    for p in list(rae.parameters()) + list(model.parameters()):
        p.requires_grad_(False)
    return rae, model, info


@torch.no_grad()
def encode(rae, frames, device, bs=16):
    """uint8 (T,H,W,3) -> normalised DINOv2 latents (T, C, 16, 16) float32 on device."""
    out = []
    for i in range(0, len(frames), bs):
        x = torch.from_numpy(np.ascontiguousarray(frames[i:i + bs])).to(device).float().permute(0, 3, 1, 2) / 255.0
        out.append(rae.encode(x).float())
    return torch.cat(out)


@torch.no_grad()
def decode(rae, z, bs=8):
    """(T, C, 16, 16) -> uint8 (T, 256, 256, 3)."""
    out = []
    for i in range(0, len(z), bs):
        img = rae.decode(z[i:i + bs]).clamp(0, 1)
        out.append((img * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy())
    return np.concatenate(out)


def tokens(z):
    """(T, C, H, W) -> (T, H*W, C) raster order."""
    T, C, H, W = z.shape
    return z.reshape(T, C, H * W).transpose(1, 2).contiguous()


@torch.no_grad()
def rollout(model, z_ctx, n_steps, policy, z_future=None):
    """z_ctx (Tc, C, H, W) -> predicted latents (n_steps, C, H, W), context latent actions (Tc-1, A).

    policy 'oracle' is a LEAKY reference, not a benchmark condition: the latent actions
    of the whole sequence (context + true future) are inferred first (the notebook's
    autoregressive demo), so only the structure rollout / decoder are tested."""
    x_ctx = z_ctx.unsqueeze(0)
    if policy == "oracle":
        la_all = model.get_latent_actions(torch.cat([z_ctx, z_future]).unsqueeze(0))   # (1, Tc-1+n_steps, A)
        res = model.autoregressive_forward(x_ctx, la_all)
        emb = res["embedding_gen"][0]
        tc = z_ctx.shape[0]
        return emb[tc - 1:], la_all[0, :tc - 1]
    la_ctx = model.get_latent_actions(x_ctx)                       # (1, Tc-1, A)
    if policy == "hold":
        la_fut = la_ctx[:, -1:].expand(-1, n_steps, -1)
    elif policy == "zero":
        la_fut = torch.zeros(1, n_steps, la_ctx.shape[-1], device=la_ctx.device, dtype=la_ctx.dtype)
    elif policy == "mean":
        la_fut = la_ctx.mean(dim=1, keepdim=True).expand(-1, n_steps, -1)
    else:
        raise ValueError(policy)
    la = torch.cat([la_ctx, la_fut], dim=1)                        # (1, Tc-1+n_steps, A)
    res = model.autoregressive_forward(x_ctx, la)
    emb = res["embedding_gen"][0]                                  # (Tc-1+n_steps, C, H, W)
    tc = z_ctx.shape[0]
    return emb[tc - 1:], la_ctx[0]


def plan(d, oi, stride, extra):
    """Context indices, predicted frame indices, positions of the 4 fps targets, target indices."""
    o = int(d["offsets"][oi])
    ctx4 = [int(x) for x in d["ctx_idx"][oi]]           # (o, o+4)
    tgt4 = [int(x) for x in d["target_idx"][oi]]        # (o+8, o+12, o+16, o+20)
    if stride == 4:
        cidx = ctx4 + tgt4[:extra]
        pred_idx = tgt4[extra:]
    else:
        assert extra == 0, "--ctx-frames only applies to stride 4"
        cidx = list(range(o, ctx4[-1] + 1, stride))
        pred_idx = list(range(ctx4[-1] + stride, tgt4[-1] + 1, stride))
    tidx = tgt4[extra:]
    keep = [pred_idx.index(t) for t in tidx]
    return cidx, pred_idx, keep, tidx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--policy", default="hold", choices=["hold", "zero", "mean", "oracle"])
    ap.add_argument("--stride", type=int, default=4, choices=[1, 2, 4], help="frames per model step (16 fps base)")
    ap.add_argument("--ctx-frames", type=int, default=2, help="stride-4 context frames (2 = contract; 3 = variant)")
    ap.add_argument("--reverse-ctx", action="store_true", help="control: feed the context frames in reversed order")
    ap.add_argument("--no-decode", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda")
    t0 = time.time()
    rae, model, info = load_model(device)
    print(f"loaded in {time.time() - t0:.0f}s: params {info['n_params_M']}M, missing {len(info['missing'])}, "
          f"unexpected {len(info['unexpected'])}", flush=True)
    if info["missing"]:
        print("MISSING:", info["missing"][:20], flush=True)
    if info["unexpected"]:
        print("UNEXPECTED:", info["unexpected"][:20], flush=True)
    if args.selftest:
        f = (np.random.RandomState(0).rand(3, SIZE, SIZE, 3) * 255).astype(np.uint8)
        z = encode(rae, f, device)
        zp, la = rollout(model, z[:2], 2, args.policy)
        img = decode(rae, zp)
        rec = decode(rae, z[:1])
        print("selftest: z", tuple(z.shape), "pred", tuple(zp.shape), "la", tuple(la.shape),
              "decoded", img.shape, "recon |err|", float(np.abs(rec[0].astype(int) - f[0].astype(int)).mean()),
              "vram GB", round(torch.cuda.max_memory_allocated() / 1e9, 2), flush=True)
        return
    root = Path(args.root)
    (root / "latents").mkdir(exist_ok=True)
    rows = [json.loads(l) for l in (root / "manifest.jsonl").read_text().splitlines() if l.strip()]
    if args.only:
        rows = [r for r in rows if r["id"] in set(args.only)]
    extra = args.ctx_frames - 2
    tag = (f"_s{args.stride}" if args.stride != 4 else "") + (f"_c{args.ctx_frames}" if extra else "") \
        + ("_rev" if args.reverse_ctx else "") + ("" if args.policy == "hold" else f"_{args.policy}")
    for row in rows:
        sid = row["id"]
        d = np.load(root / row["vj_input"])
        frames = d["frames_gt"]
        ts = time.time()
        z_seq = encode(rae, frames, device)                          # (24, C, 16, 16)
        for oi, o in enumerate(d["offsets"]):
            cidx, pred_idx, keep, tidx = plan(d, oi, args.stride, extra)
            if extra and not (cidx[-1] < int(d["event_frame"])):
                continue
            z_ctx = z_seq[cidx[::-1]] if args.reverse_ctx else z_seq[cidx]
            z_pred_full, la_ctx = rollout(model, z_ctx, len(pred_idx), args.policy,
                                          z_future=z_seq[list(pred_idx)] if args.policy == "oracle" else None)
            z_pred = z_pred_full[keep]
            z_gt = z_seq[list(tidx)]
            z_kin = encode(rae, d["frames_kin"][oi][extra:], device)
            z_jit = encode(rae, d["frames_jit"][oi][extra:], device)
            out = {"z_ctx": tokens(z_ctx), "z_pred": tokens(z_pred), "z_gt": tokens(z_gt),
                   "z_kin": tokens(z_kin), "z_jit": tokens(z_jit)}
            extra_arrays = {"la_ctx": la_ctx.cpu().numpy(), "pred_idx": np.asarray(pred_idx), "stride": args.stride}
            if not args.no_decode:
                rgb_full = decode(rae, z_pred_full)
                extra_arrays["rgb_pred_full"] = rgb_full
                extra_arrays["rgb_pred"] = rgb_full[keep]
                extra_arrays["rgb_gt_recon"] = decode(rae, z_gt)
                extra_arrays["rgb_ctx_recon"] = decode(rae, z_ctx[-1:])
            np.savez_compressed(
                root / "latents" / f"{sid}_o{int(o)}{tag}.npz",
                **{k: v.half().cpu().numpy() for k, v in out.items()}, **extra_arrays,
                kin_offset=extra,
                offset=int(o), ctx_idx=np.asarray(cidx), target_idx=np.asarray(tidx), ctx_ok=bool(d["ctx_ok"][oi]),
                family=str(d["family"]), kind=str(d["kind"]), S=float(d["S"]),
                outcome=int(d["outcome"]), event_frame=int(d["event_frame"]),
                px_per_m=float(d["px_per_m"]))
        print(f"{sid}: {len(d['offsets'])} offsets in {time.time() - ts:.1f}s", flush=True)
    info["policy"] = args.policy
    info["stride"] = args.stride
    info["vram_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    (root / "latents" / f"model_info{tag}.json").write_text(json.dumps(info, indent=1))
    print(f"done in {time.time() - t0:.0f}s, peak VRAM {info['vram_GB']} GB")


if __name__ == "__main__":
    main()
