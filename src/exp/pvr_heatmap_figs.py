"""Overlay figures of the stored attribution maps (gradient + attention) for the explainer page.

Per scene/condition: seed-averaged maps at one step -> five small overlays on the last conditioning
frame: (1) frame with region outlines, (2) |ds/dx| for the ball-position probe score, (3) |d(irrelevant
scalar)/dx| (sanity), (4) attention from the predicted ball's tokens to the conditioning tokens
(late block), (5) attention averaged over all predicted tokens.
"""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import matplotlib
from PIL import Image
from scipy.ndimage import binary_erosion

from exp.analyze_bundle_a import color_mask
from exp.pvr_regions import downsample_masks, region_masks

TOK_H, TOK_W = 30, 52
COLORS = {"structure": (255, 214, 0), "decoy": (170, 120, 255), "ball": (255, 255, 255)}


def _outline(mask, width=4):
    return mask & ~binary_erosion(mask, iterations=width)


def frame_with_regions(frame, masks):
    img = frame.copy()
    for r, col in COLORS.items():
        if r in masks and masks[r].any():
            img[_outline(masks[r])] = col
    return Image.fromarray(img)


def overlay(frame, heat, cmap="magma", q=99.5, gamma=0.7, base=0.45):
    h = heat.astype(np.float32)
    h = h / (np.percentile(h, q) + 1e-12)
    h = np.clip(h, 0, 1) ** gamma
    gray = frame.astype(np.float32).mean(axis=2, keepdims=True) / 255.0 * base
    color = matplotlib.colormaps[cmap](h)[..., :3]
    out = gray * (1 - h[..., None]) + color * h[..., None]
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


def _upsample(tok, shape):
    return np.array(Image.fromarray(tok.astype(np.float32)).resize((shape[1], shape[0]), Image.BILINEAR))


def ball_query_weights(root, scene, rec, seed):
    clip = root / scene / "cosmos_v2w" / f"seed_{seed:02d}" / rec["id"] / "rollout.mp4"
    frames = imageio.mimread(clip, memtest=False)
    ms = [color_mask(frames[min(k, len(frames) - 1)], "red") for k in range(17, 21)]
    w = np.mean([downsample_masks({"b": m}, TOK_H, TOK_W)["b"] for m in ms], axis=0).reshape(-1)
    return w.astype(np.float32)


def panels(root: Path, scene: str, cond: str, seeds=(1, 2, 3, 4), step=24, block=27):
    manifest = [json.loads(l) for l in (root / scene / "manifest.jsonl").read_text().splitlines() if l.strip()]
    rec = next(r for r in manifest if r["cond"] == cond)
    segz = np.load(root / scene / rec["seg_npz"], allow_pickle=False)
    seg, meta = segz["seg"], json.loads(str(segz["meta"]))
    masks = region_masks(seg[4].astype(np.int32), meta, scene, rec["params"])
    frame = np.load(root / scene / rec["input_npz"])["condition_primary"][4]
    grads, wrongs, att_ball, att_all, used = [], [], [], [], []
    for s in seeds:
        p = root / scene / "attrib" / cond / f"seed_{s:02d}" / "maps.npz"
        if not p.exists():
            continue
        z = np.load(p)
        g = z[f"grad_video_step{step}"][4].astype(np.float32); grads.append(g / (g.max() + 1e-12))
        w = z[f"grad_video_wrong_step{step}"][4].astype(np.float32); wrongs.append(w / (w.max() + 1e-12))
        a = z[f"att_step{step}_block{block}"].astype(np.float32)          # (1560 queries, 3120 cond tokens)
        qw = ball_query_weights(root, scene, rec, s)
        ab = (a * qw[:, None]).sum(0) / (qw.sum() + 1e-12)
        aa = a.mean(0)
        att_ball.append(ab[TOK_H * TOK_W:].reshape(TOK_H, TOK_W))       # latent frame 1 <- pixel frames 1-4
        att_all.append(aa[TOK_H * TOK_W:].reshape(TOK_H, TOK_W))
        used.append(s)
    g, w = np.mean(grads, 0), np.mean(wrongs, 0)
    ab, aa = np.mean(att_ball, 0), np.mean(att_all, 0)
    corr = float(np.corrcoef(g.ravel(), w.ravel())[0, 1])
    shape = frame.shape[:2]
    return {"frame": frame_with_regions(frame, masks), "grad": overlay(frame, g), "wrong": overlay(frame, w),
            "att_ball": overlay(frame, _upsample(ab, shape), cmap="viridis", q=99.0),
            "att_all": overlay(frame, _upsample(aa, shape), cmap="viridis", q=99.0),
            "corr_grad_wrong": corr, "seeds": used, "step": step, "block": block, "id": rec["id"]}
