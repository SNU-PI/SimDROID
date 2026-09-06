"""PVR PoC attribution track for Cosmos-Predict2-2B Video2World (single-step x0 gradient +
attention routing), with region aggregation and sanity checks.

Re-implements the diffusers pipeline's sampling loop outside no_grad so that, at chosen
denoising steps, one denoiser call x0 = D(x_i, cond) can be differentiated with respect to
the five conditioning RGB frames (through the VAE encoder) and the two conditioning latent
frames.  The event score is a latent probe: a logistic "ball" classifier on the 16 latent
channels, turned into a soft-argmax position on latent frame 5 (pixel frames 17-20).
Attention routing: a recording attention processor recomputes softmax(QK^T) for the
queries of latent frame 5 against the 3,120 conditioning-frame keys, per block.

Subcommands:
  fit-probe   fit the latent ball probe on the GT clips of a scene (VAE encode)
  attribute   run the attribution for (scene, conditions, seeds, steps)
  aggregate   per-scene tables and figures from the saved maps

Design: DESIGN_PVR_POC_2026-09-06.md section 7; Cosmos internals: Materials/cosmos_v2w_internals_2026-09-06.md.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import diffusers.pipelines.cosmos.pipeline_cosmos2_video2world as cosmos_pipeline_module
from diffusers import Cosmos2VideoToWorldPipeline
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_cosmos import CosmosAttnProcessor2_0, CosmosTransformer3DModel
from diffusers.pipelines.cosmos.pipeline_cosmos2_video2world import retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor

from exp.cosmos_v2w_sweep import NEGATIVE_PROMPT, DisabledSyntheticOnlySafetyChecker, letterbox
from exp.pvr_regions import downsample_masks, region_fractions, region_masks

cosmos_pipeline_module.CosmosSafetyChecker = DisabledSyntheticOnlySafetyChecker

H, W = 480, 832
NUM_FRAMES, STEPS, GUIDANCE, FPS = 21, 35, 7.0, 16
LAT_T, LAT_H, LAT_W = 6, 60, 104
TOK_H, TOK_W = 30, 52
N_COND_TOK = 2 * TOK_H * TOK_W          # 3,120 conditioning-frame tokens
REGIONS = ("ball", "structure", "decoy", "support", "franka", "clutter", "background", "other")
SCORE_AXIS = {"hill": "col", "collide": "col", "edge": "row"}
DEFAULT_STEPS = (20, 24, 28)


# ---------------------------------------------------------------- model
def load_pipe(model_dir: Path, ckpt: Path, device="cuda"):
    transformer = CosmosTransformer3DModel.from_single_file(
        str(ckpt), config=str(model_dir / "transformer"), torch_dtype=torch.bfloat16, local_files_only=True)
    pipe = Cosmos2VideoToWorldPipeline.from_pretrained(
        str(model_dir), transformer=transformer, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.to(torch.float32)                       # clean gradients through the encoder
    pipe.transformer.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.transformer.enable_gradient_checkpointing()
    return pipe


def latent_stats(pipe, device):
    mean = torch.tensor(pipe.vae.config.latents_mean).view(1, 16, 1, 1, 1).to(device, torch.float32)
    std = torch.tensor(pipe.vae.config.latents_std).view(1, 16, 1, 1, 1).to(device, torch.float32)
    return mean, std, float(pipe.scheduler.config.sigma_data)


def encode_video(pipe, video, generator=None, sample=True, grad=False):
    """video: (1,3,T,H,W) float32 in [-1,1] -> normalised latents (1,16,T',60,104) float32."""
    mean, std, sigma_data = latent_stats(pipe, video.device)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        post = pipe.vae.encode(video).latent_dist
        if sample:
            eps = randn_tensor(post.mean.shape, generator=generator, device=video.device, dtype=post.mean.dtype)
            z = post.mean + post.std * eps
        else:
            z, eps = post.mode(), None
        z = (z - mean) / std * sigma_data
    return z, eps


def frames_to_video(frames_uint8, device):
    """(T,H,W,3) uint8 (already 480x832) -> (1,3,T,H,W) float32 in [-1,1]."""
    x = torch.from_numpy(np.asarray(frames_uint8)).to(device).float() / 127.5 - 1.0
    return x.permute(3, 0, 1, 2).unsqueeze(0).contiguous()


# ---------------------------------------------------------------- latent probe
class BallProbe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(16, 1)

    def logits(self, z):                     # z: (16, H, W) -> (H, W)
        return self.lin(z.permute(1, 2, 0)).squeeze(-1)

    def score(self, z, axis="col", temperature=1.5):
        """Centroid (in latent cells) of a peaked softmax over the ball logits: the mass sits
        on the cells within a few logit units of the maximum (the ball), so the score follows
        the ball and ignores diffuse background probability."""
        lg = self.logits(z)
        p = torch.softmax((lg - lg.max()).flatten() / temperature, dim=0).view_as(lg)
        rows = torch.arange(lg.shape[0], device=lg.device, dtype=lg.dtype)[:, None]
        cols = torch.arange(lg.shape[1], device=lg.device, dtype=lg.dtype)[None, :]
        return (p * (cols if axis == "col" else rows)).sum(), lg.max()


def ball_targets(seg_npz, record, scene):
    """Soft ball target per latent frame k (1..5) at 60x104 from the segmentation masks."""
    segz = np.load(seg_npz, allow_pickle=False)
    seg, meta = segz["seg"], json.loads(str(segz["meta"]))
    targets = []
    for k in range(1, LAT_T):
        frames = range(4 * k - 3, 4 * k + 1)
        acc = np.zeros((LAT_H, LAT_W), np.float32)
        for f in frames:
            m = region_masks(seg[min(f, len(seg) - 1)].astype(np.int32), meta, scene, record["params"])["ball"]
            acc += downsample_masks({"b": m}, LAT_H, LAT_W)["b"]
        targets.append(acc / len(list(frames)))
    return np.stack(targets)                # (5, 60, 104)


def fit_probe(pipe, root, scene, records, device, out_path, iters=600):
    xs, ys = [], []
    with torch.no_grad():
        for record in records:
            gt = imageio.mimread(root / record["gt_clip"], memtest=False)
            video = frames_to_video(np.stack(gt[:NUM_FRAMES]), device)
            z, _ = encode_video(pipe, video, sample=False)              # (1,16,6,60,104)
            tgt = torch.from_numpy(ball_targets(root / record["seg_npz"], record, scene)).to(device)
            for k in range(1, LAT_T):
                xs.append(z[0, :, k].permute(1, 2, 0).reshape(-1, 16))
                ys.append(tgt[k - 1].reshape(-1))
    X, Y = torch.cat(xs), torch.cat(ys)
    probe = BallProbe().to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=0.05)
    pos_weight = torch.tensor([float(((Y < 0.5).sum() / max((Y >= 0.5).sum(), 1)).sqrt())], device=device)
    for it in range(iters):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(probe.lin(X).squeeze(-1), Y, pos_weight=pos_weight)
        loss.backward()
        opt.step()
    # evaluation: soft-argmax error on the training latents (cells)
    errs = []
    with torch.no_grad():
        for x, y in zip(xs, ys):
            z = x.reshape(LAT_H, LAT_W, 16).permute(2, 0, 1)
            t = y.reshape(LAT_H, LAT_W)
            if t.sum() < 0.5:
                continue
            sc, _ = probe.score(z, "col"); sr, _ = probe.score(z, "row")
            rows = torch.arange(LAT_H, device=device, dtype=t.dtype)[:, None]; cols = torch.arange(LAT_W, device=device, dtype=t.dtype)[None, :]
            tc, tr = (t * cols).sum() / t.sum(), (t * rows).sum() / t.sum()
            errs.append(float(torch.sqrt((sc - tc) ** 2 + (sr - tr) ** 2)))
    info = {"scene": scene, "n_latent_frames": len(xs), "loss": float(loss), "pos_weight": float(pos_weight),
            "softargmax_err_cells_median": float(np.median(errs)), "softargmax_err_cells_p90": float(np.percentile(errs, 90)),
            "weight": probe.lin.weight.detach().cpu().tolist(), "bias": float(probe.lin.bias)}
    torch.save(probe.state_dict(), out_path)
    Path(str(out_path) + ".json").write_text(json.dumps(info, indent=1))
    print(f"[probe:{scene}] loss {loss:.4f} soft-argmax err median {info['softargmax_err_cells_median']:.2f} cells "
          f"(p90 {info['softargmax_err_cells_p90']:.2f})", flush=True)
    return probe


# ---------------------------------------------------------------- attention recording
class AttnStore:
    def __init__(self):
        self.active = False
        self.q_idx = None                    # LongTensor of query token indices
        self.maps = {}                       # block -> (Nq, N_COND_TOK) float32 CPU (head mean)

    def reset(self, q_idx):
        self.q_idx, self.maps = q_idx, {}


class RecordingProcessor(CosmosAttnProcessor2_0):
    def __init__(self, store, block_idx):
        super().__init__()
        self.store, self.block_idx = store, block_idx

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        self_attention = encoder_hidden_states is None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)
            key = apply_rotary_emb(key, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)
        if self_attention and self.store.active and self.store.q_idx is not None:
            with torch.no_grad():
                q = query[:, :, self.store.q_idx].float()                      # (1,h,Nq,d)
                k = key[:, :, :].float()
                att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), dim=-1)  # (1,h,Nq,N)
                self.store.maps[self.block_idx] = att[0, :, :, :N_COND_TOK].mean(dim=0).cpu()
                del att
        query_idx, key_idx, value_idx = query.size(3), key.size(3), value.size(3)
        key = key.repeat_interleave(query_idx // key_idx, dim=3)
        value = value.repeat_interleave(query_idx // value_idx, dim=3)
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def install_recorders(pipe, store):
    for i, block in enumerate(pipe.transformer.transformer_blocks):
        block.attn1.set_processor(RecordingProcessor(store, i))


# ---------------------------------------------------------------- sampling replay
class Replay:
    """One sample's sampler state, re-created exactly as the pipeline does (same generator order)."""

    def __init__(self, pipe, device, conditions_uint8, prompt, seed):
        self.pipe, self.device = pipe, device
        with torch.no_grad():
            self.prompt_embeds, self.negative_embeds = pipe.encode_prompt(
                prompt=prompt, negative_prompt=NEGATIVE_PROMPT, do_classifier_free_guidance=True,
                num_videos_per_prompt=1, device=device, max_sequence_length=512)
        self.prompt_embeds = self.prompt_embeds.detach()
        self.negative_embeds = self.negative_embeds.detach()
        sigmas = torch.linspace(0, 1, STEPS, dtype=torch.float64)
        self.timesteps, _ = retrieve_timesteps(pipe.scheduler, device=device, sigmas=sigmas)
        if pipe.scheduler.config.final_sigmas_type == "sigma_min":
            pipe.scheduler.sigmas[-1] = pipe.scheduler.sigmas[-2]
        self.generator = torch.Generator(device=device).manual_seed(seed)
        pil = [letterbox(f, H, W) for f in conditions_uint8]
        video = pipe.video_processor.preprocess_video(pil, H, W).to(device, torch.float32)   # (1,3,5,H,W)
        self.video5 = video
        pad = video[:, :, -1:].repeat(1, 1, NUM_FRAMES - video.shape[2], 1, 1)
        video21 = torch.cat([video, pad], dim=2)
        self.cond_latents, self.eps = encode_video(pipe, video21, generator=self.generator, sample=True)
        shape = (1, 16, LAT_T, LAT_H, LAT_W)
        self.latents0 = randn_tensor(shape, generator=self.generator, device=device, dtype=torch.float32) \
            * float(pipe.scheduler.config.sigma_max)
        ind = torch.zeros(1, 1, LAT_T, 1, 1, device=device)
        ind[:, :, :2] = 1.0
        self.ind = ind
        self.cond_mask = (ind * torch.ones(1, 1, LAT_T, LAT_H, LAT_W, device=device)).to(torch.bfloat16)
        self.padding_mask = torch.zeros(1, 1, H, W, device=device, dtype=torch.bfloat16)
        sc = torch.tensor(1e-4, dtype=torch.float32, device=device)
        self.t_cond = sc / (sc + 1)

    def coeffs(self, i):
        sigma = self.pipe.scheduler.sigmas[i]
        ct = sigma / (sigma + 1)
        return sigma, ct, 1 - ct, 1 - ct, -ct

    def denoise(self, latents, cond_latents, i, embeds):
        sigma, ct, c_in, c_skip, c_out = self.coeffs(i)
        timestep = ct.view(1, 1, 1, 1, 1).expand(1, -1, LAT_T, -1, -1).to(self.device)
        ind = self.ind
        x_in = (ind * cond_latents + (1 - ind) * latents * c_in).to(torch.bfloat16)
        t_in = (ind * self.t_cond + (1 - ind) * timestep).to(torch.bfloat16)
        out = self.pipe.transformer(hidden_states=x_in, timestep=t_in, encoder_hidden_states=embeds, fps=FPS,
                                    condition_mask=self.cond_mask, padding_mask=self.padding_mask,
                                    return_dict=False)[0]
        x0 = (c_skip * latents + c_out * out.float())
        return ind * cond_latents + (1 - ind) * x0

    def x0_cfg(self, latents, cond_latents, i):
        x0_c = self.denoise(latents, cond_latents, i, self.prompt_embeds)
        x0_u = self.denoise(latents, cond_latents, i, self.negative_embeds)
        return x0_c + GUIDANCE * (x0_c - x0_u)

    @torch.no_grad()
    def run(self, hooks=None):
        """Full replay; hooks: {i: fn(i, latents)} called before step i (latents = x_i)."""
        latents = self.latents0.clone()
        states = {}
        for i, t in enumerate(self.timesteps):
            if hooks and i in hooks:
                states[i] = latents.clone()
                hooks[i](i, latents)
            x0 = self.x0_cfg(latents, self.cond_latents, i)
            sigma = self.pipe.scheduler.sigmas[i]
            d = (latents - x0) / sigma
            latents = self.pipe.scheduler.step(d, t, latents, return_dict=False)[0]
        return latents, states


# ---------------------------------------------------------------- attribution core
def cond_latents_from_video5(pipe, video5, eps21):
    """Differentiable conditioning latents (2 latent frames) from the 5 real frames, same eps."""
    mean, std, sigma_data = latent_stats(pipe, video5.device)
    post = pipe.vae.encode(video5).latent_dist
    z = post.mean + post.std * eps21[:, :, :2]
    z = (z - mean) / std * sigma_data
    pad = torch.zeros(1, 16, LAT_T - 2, LAT_H, LAT_W, device=video5.device, dtype=z.dtype)
    return torch.cat([z, pad], dim=2)


def gradient_at_step(rep, probe, axis, i, latents, wrong_target=False):
    """Single-step x0 gradient of the probe score wrt the 5 conditioning frames and the cond latents."""
    pipe = rep.pipe
    with torch.enable_grad():
        video5 = rep.video5.detach().clone().requires_grad_(True)
        cond = cond_latents_from_video5(pipe, video5, rep.eps)
        cond.retain_grad()
        x0 = rep.x0_cfg(latents.detach(), cond, i)
        z5 = x0[0, :, LAT_T - 1]                                # latent frame 5 (pixels 17-20)
        if wrong_target:
            s = z5.mean()
            top = torch.tensor(float("nan"))
        else:
            s, top = probe.score(z5, axis)
        s.backward()
    g_vid = video5.grad.detach().abs().sum(dim=1)[0]            # (5,H,W)
    g_lat = cond.grad.detach().abs().sum(dim=1)[0, :2]           # (2,60,104)
    return g_vid.float().cpu().numpy(), g_lat.float().cpu().numpy(), float(s), float(top)


def region_fraction_table(weight, masks_list):
    """weight (T,H,W) with per-frame masks -> mass fraction per region (all frames, per frame),
    the region's area fraction, and density = mass / area (1 = uniform)."""
    per_frame, total, area = [], {r: 0.0 for r in REGIONS}, {r: 0.0 for r in REGIONS}
    wsum = float(weight.sum()) + 1e-12
    npix = float(weight.shape[0] * weight.shape[1] * weight.shape[2])
    for k in range(weight.shape[0]):
        fr = region_fractions(weight[k], masks_list[k])
        per_frame.append(fr)
        for r in REGIONS:
            total[r] += float(weight[k][masks_list[k][r]].sum()) / wsum
            area[r] += float(masks_list[k][r].sum()) / npix
    density = {r: (total[r] / area[r] if area[r] > 0 else float("nan")) for r in REGIONS}
    frame_share = [float(weight[k].sum()) / wsum for k in range(weight.shape[0])]
    return {"total": total, "area": area, "density": density, "per_frame": per_frame, "frame_share": frame_share}


def token_masks(masks_list):
    """Region masks per conditioning frame -> token-grid area fractions for key tokens:
    latent frame 0 <- pixel frame 0; latent frame 1 <- mean of pixel frames 1-4."""
    m0 = downsample_masks(masks_list[0], TOK_H, TOK_W)
    m1 = {r: np.mean([downsample_masks(masks_list[k], TOK_H, TOK_W)[r] for k in range(1, 5)], axis=0) for r in REGIONS}
    return {r: np.concatenate([m0[r].reshape(-1), m1[r].reshape(-1)]) for r in REGIONS}    # (3120,)


def attention_fractions(store_maps, tok_masks, q_weights=None):
    """Per block: attention mass (from the query set) landing in each region of the cond tokens."""
    out = {}
    for b, att in store_maps.items():                       # att: (Nq, 3120)
        a = att.numpy()
        if q_weights is not None:
            a = (a * q_weights[:, None]).sum(axis=0) / (q_weights.sum() + 1e-12)
        else:
            a = a.mean(axis=0)
        frame_share = [float(a[:TOK_H * TOK_W].sum()), float(a[TOK_H * TOK_W:].sum())]
        regions = {r: float((a * tok_masks[r]).sum()) for r in REGIONS}
        area = {r: float(tok_masks[r].sum() / tok_masks[r].size) for r in REGIONS}
        out[int(b)] = {"regions": regions, "area": area,
                       "density": {r: (regions[r] / area[r] if area[r] > 0 else float("nan")) for r in REGIONS},
                       "frame_share": frame_share}
    return out


def load_inputs(root, record):
    inp = np.load(root / record["input_npz"])
    segz = np.load(root / record["seg_npz"], allow_pickle=False)
    seg, meta = segz["seg"], json.loads(str(segz["meta"]))
    masks = [region_masks(seg[k].astype(np.int32), meta, record["scene"], record["params"]) for k in range(5)]
    return inp["condition_primary"], masks, seg, meta


def generated_ball_tokens(root, record, seed, seg=None, meta=None):
    """Query weights over latent-frame-5 tokens: the red ball in the generated rollout's frames 17-20
    (falls back to the GT ball if the rollout is missing)."""
    from exp.analyze_bundle_a import color_mask
    clip = root / "cosmos_v2w" / f"seed_{seed:02d}" / record["id"] / "rollout.mp4"
    if clip.exists():
        frames = imageio.mimread(clip, memtest=False)
        ms = [color_mask(frames[min(k, len(frames) - 1)], "red") for k in range(17, 21)]
        src = "generated"
    else:
        ms = [region_masks(seg[min(k, len(seg) - 1)].astype(np.int32), meta, record["scene"], record["params"])["ball"]
              for k in range(17, 21)]
        src = "gt"
    w = np.mean([downsample_masks({"b": m}, TOK_H, TOK_W)["b"] for m in ms], axis=0).reshape(-1)
    return w.astype(np.float32), src


def randomize_last_blocks(pipe, k, rng_seed=0):
    """Re-initialise the last k transformer blocks (all their parameters); returns a restore closure."""
    blocks = pipe.transformer.transformer_blocks
    saved = {}
    g = torch.Generator(device="cpu").manual_seed(rng_seed)
    for b in list(range(len(blocks)))[-k:]:
        for name, p in blocks[b].named_parameters():
            saved[(b, name)] = p.detach().clone()
            with torch.no_grad():
                if p.ndim >= 2:
                    p.copy_(torch.randn(p.shape, generator=g).to(p.device, p.dtype) * 0.02)
                else:
                    p.zero_()

    def restore():
        for (b, name), v in saved.items():
            dict(blocks[b].named_parameters())[name].data.copy_(v)
    return restore


def run_attribution(args):
    device = "cuda"
    pipe = load_pipe(args.model_dir, args.original_checkpoint, device)
    store = AttnStore()
    install_recorders(pipe, store)
    steps = tuple(args.steps)
    for scene in args.scenes:
        root = (args.root / scene).resolve()
        manifest = [json.loads(l) for l in (root / "manifest.jsonl").read_text().splitlines() if l.strip()]
        records = [r for r in manifest if (not args.conds or r["cond"] in args.conds)]
        out_root = root / "attrib"
        out_root.mkdir(parents=True, exist_ok=True)
        probe_path = out_root / "probe.pt"
        if not probe_path.exists() or args.refit_probe:
            probe = fit_probe(pipe, root, scene, manifest, device, probe_path)
        else:
            probe = BallProbe().to(device)
            probe.load_state_dict(torch.load(probe_path, map_location=device))
        axis = SCORE_AXIS[scene]
        for record in records:
            conditions, masks, seg, meta = load_inputs(root, record)
            tmasks = token_masks(masks)
            for seed in args.seeds:
                out_dir = out_root / record["cond"] / f"seed_{seed:02d}"
                if (out_dir / "summary.json").exists() and not args.overwrite:
                    print(f"skip {scene}/{record['cond']}/seed {seed} (done)", flush=True)
                    continue
                out_dir.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                rep = Replay(pipe, device, conditions, record["prompt"], seed)
                q_w, q_src = generated_ball_tokens(root, record, seed, seg, meta)
                q_idx_all = torch.arange((LAT_T - 1) * TOK_H * TOK_W, LAT_T * TOK_H * TOK_W, device=device)
                results = {"scene": scene, "cond": record["cond"], "id": record["id"], "seed": seed, "steps": list(steps),
                           "axis": axis, "query_source": q_src, "grad": {}, "attention": {}, "score": {}}
                maps = {}

                def hook(i, latents):
                    # attention routing (no grad) for the cond branch at this step
                    store.reset(q_idx_all)
                    store.active = True
                    with torch.no_grad():
                        rep.denoise(latents, rep.cond_latents, i, rep.prompt_embeds)
                    store.active = False
                    att_all = attention_fractions(store.maps, tmasks, None)
                    att_ball = attention_fractions(store.maps, tmasks, q_w)
                    keep = {b: store.maps[b].numpy().astype(np.float16) for b in (6, 13, 20, 27) if b in store.maps}
                    results["attention"][str(i)] = {"all_queries": att_all, "ball_queries": att_ball}
                    for b, m in keep.items():
                        maps[f"att_step{i}_block{b}"] = m
                    # single-step x0 gradient wrt the conditioning frames / latents
                    g_vid, g_lat, s, top = gradient_at_step(rep, probe, axis, i, latents)
                    g_vid_n = g_vid / (g_vid.sum() + 1e-12)
                    maps[f"grad_video_step{i}"] = g_vid_n.astype(np.float16)
                    maps[f"grad_latent_step{i}"] = (g_lat / (g_lat.sum() + 1e-12)).astype(np.float16)
                    results["grad"][str(i)] = {"video": region_fraction_table(g_vid_n, masks),
                                               "latent_frame_share": [float(g_lat[0].sum() / (g_lat.sum() + 1e-12)),
                                                                      float(g_lat[1].sum() / (g_lat.sum() + 1e-12))],
                                               "grad_norm": float(g_vid.sum())}
                    results["score"][str(i)] = {"s_lat": s, "max_logit": top}
                    if args.wrong_target:
                        g_vid_w, _, s_w, _ = gradient_at_step(rep, probe, axis, i, latents, wrong_target=True)
                        g_vid_w = g_vid_w / (g_vid_w.sum() + 1e-12)
                        results["grad"][str(i)]["wrong_target"] = region_fraction_table(g_vid_w, masks)["total"]
                        results["grad"][str(i)]["wrong_target_corr"] = float(np.corrcoef(g_vid_n.ravel(), g_vid_w.ravel())[0, 1])
                        maps[f"grad_video_wrong_step{i}"] = g_vid_w.astype(np.float16)

                hooks = {i: hook for i in steps}
                final_latents, states = rep.run(hooks)
                # final score from the sampled latents (sigma_min) = the event the model actually drew
                with torch.no_grad():
                    s_final, top_final = probe.score(final_latents[0, :, LAT_T - 1], axis)
                results["score"]["final"] = {"s_lat": float(s_final), "max_logit": float(top_final)}
                # Sobel edge-energy baseline of the last conditioning frame
                fr = conditions[4].astype(np.float32).mean(axis=2)
                gx = np.abs(np.diff(fr, axis=1, prepend=fr[:, :1])); gy = np.abs(np.diff(fr, axis=0, prepend=fr[:1]))
                sob = gx + gy
                results["edge_baseline"] = region_fractions(sob / (sob.sum() + 1e-12), masks[4])
                # model-randomisation sanity check (selected samples)
                if args.randomize and (args.randomize_all or (record["cond"] in ("A", "B") and seed == args.seeds[0])):
                    i = steps[len(steps) // 2]
                    base = maps[f"grad_video_step{i}"].astype(np.float32).ravel()
                    results["randomization"] = {}
                    for k in (4, 8, 16, 28):
                        restore = randomize_last_blocks(pipe, k)
                        try:
                            g_r, _, _, _ = gradient_at_step(rep, probe, axis, i, states[i])
                        finally:
                            restore()
                        g_r = g_r / (g_r.sum() + 1e-12)
                        results["randomization"][str(k)] = {
                            "corr_with_original": float(np.corrcoef(base, g_r.ravel())[0, 1]),
                            "regions": region_fraction_table(g_r, masks)["total"]}
                        maps[f"grad_video_rand{k}_step{i}"] = g_r.astype(np.float16)
                np.savez_compressed(out_dir / "maps.npz", **maps)
                results["seconds"] = time.time() - t0
                (out_dir / "summary.json").write_text(json.dumps(results, indent=1, default=float))
                gsum = results["grad"][str(steps[-1])]["video"]["total"]
                print(f"[{scene}/{record['cond']}/seed {seed}] {results['seconds']:.0f}s  s_lat@final={float(s_final):.2f}  "
                      f"grad@{steps[-1]}: ball {gsum['ball']:.2f} structure {gsum['structure']:.2f} decoy {gsum['decoy']:.2f} "
                      f"support {gsum['support']:.2f} bg {gsum['background']:.2f}", flush=True)


# ---------------------------------------------------------------- aggregation
def aggregate(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for scene in args.scenes:
        root = (args.root / scene).resolve()
        att_root = root / "attrib"
        if not att_root.exists():
            continue
        manifest = {r["cond"]: r for r in (json.loads(l) for l in (root / "manifest.jsonl").read_text().splitlines() if l.strip())}
        rows = []
        for sfile in sorted(att_root.glob("*/seed_*/summary.json")):
            rows.append(json.loads(sfile.read_text()))
        if not rows:
            continue
        steps = rows[0]["steps"]
        per_cond = {}
        for r in rows:
            c = r["cond"]
            d = per_cond.setdefault(c, {"seeds": [], "grad_total": [], "grad_per_frame": [], "att_all": [], "att_ball": [], "edge": [],
                                        "s_final": [], "latent_share": []})
            d["seeds"].append(r["seed"])
            # step-normalised averages of region fractions
            gt = {reg: float(np.mean([r["grad"][str(i)]["video"]["total"][reg] for i in steps])) for reg in REGIONS}
            d["grad_total"].append(gt)
            d.setdefault("grad_density", []).append({reg: float(np.mean([r["grad"][str(i)]["video"]["density"][reg] for i in steps])) for reg in REGIONS})
            d.setdefault("area", []).append(r["grad"][str(steps[0])]["video"]["area"])
            d["grad_per_frame"].append([[float(np.mean([r["grad"][str(i)]["video"]["per_frame"][k][reg] for i in steps]))
                                         for reg in REGIONS] for k in range(5)])
            d["latent_share"].append([float(np.mean([r["grad"][str(i)]["latent_frame_share"][j] for i in steps])) for j in range(2)])
            for key, tag in (("all_queries", "att_all"), ("ball_queries", "att_ball")):
                blocks = {}
                for i in steps:
                    for b, v in r["attention"][str(i)][key].items():
                        blocks.setdefault(int(b), []).append(v["regions"])
                mean_over_blocks = {reg: float(np.mean([np.mean([bv[reg] for bv in vals]) for vals in blocks.values()])) for reg in REGIONS}
                late = {reg: float(np.mean([np.mean([bv[reg] for bv in vals]) for b, vals in blocks.items() if b >= 14])) for reg in REGIONS}
                dens = {}
                for i in steps:
                    for b, v in r["attention"][str(i)][key].items():
                        if int(b) >= 14:
                            for reg in REGIONS:
                                dens.setdefault(reg, []).append(v["density"][reg])
                d[tag].append({"all_blocks": mean_over_blocks, "late_blocks": late,
                               "late_density": {reg: float(np.nanmean(v)) for reg, v in dens.items()}})
            d["edge"].append(r["edge_baseline"])
            d["s_final"].append(r["score"]["final"]["s_lat"])
        table = []
        for c, d in per_cond.items():
            rec = manifest[c]
            g = {reg: (float(np.mean([x[reg] for x in d["grad_total"]])), float(np.std([x[reg] for x in d["grad_total"]]))) for reg in REGIONS}
            a = {reg: float(np.mean([x["all_blocks"][reg] for x in d["att_all"]])) for reg in REGIONS}
            ab = {reg: float(np.mean([x["late_blocks"][reg] for x in d["att_ball"]])) for reg in REGIONS}
            e = {reg: float(np.mean([x[reg] for x in d["edge"]])) for reg in REGIONS}
            pf = np.mean(np.array(d["grad_per_frame"]), axis=0)      # (5, R)
            gd = {reg: float(np.nanmean([x[reg] for x in d["grad_density"]])) for reg in REGIONS}
            abd = {reg: float(np.nanmean([x["late_density"].get(reg, np.nan) for x in d["att_ball"]])) for reg in REGIONS}
            ar = {reg: float(np.mean([x[reg] for x in d["area"]])) for reg in REGIONS}
            table.append({"cond": c, "group": rec["group"], "n_seeds": len(d["seeds"]), "grad": g, "grad_density": gd,
                          "area": ar, "attention_all": a, "attention_ball_late": ab, "attention_ball_late_density": abd,
                          "edge_baseline": e,
                          "grad_per_frame": {reg: [float(pf[k][j]) for k in range(5)] for j, reg in enumerate(REGIONS)},
                          "latent_frame_share": [float(np.mean([x[j] for x in d["latent_share"]])) for j in range(2)],
                          "s_final_mean": float(np.mean(d["s_final"]))})
        # hypotheses: H1 structure > decoy in A/B; H2 structure-zone mass drops without the structure; H3 ordering
        def g_of(c, reg):
            t = next((x for x in table if x["cond"] == c), None)
            return t["grad"][reg][0] if t else float("nan")
        hyp = {"H1_structure_gt_decoy": {c: (g_of(c, "structure"), g_of(c, "decoy")) for c in ("A", "B", "C", "D") if c in per_cond},
               "H2_no_structure_cells": {c: g_of(c, "structure") for c in ("XF", "XW", "A0") if c in per_cond}}
        rand = [r["randomization"] for r in rows if "randomization" in r]
        wrong = [np.mean([r["grad"][str(i)].get("wrong_target_corr", np.nan) for i in steps]) for r in rows if "wrong_target_corr" in r["grad"][str(steps[0])]]
        summary = {"scene": scene, "n_samples": len(rows), "steps": steps, "table": table, "hypotheses": hyp,
                   "randomization": rand, "wrong_target_corr_mean": float(np.nanmean(wrong)) if wrong else None,
                   "seed_cv_structure": {c: float(np.std([x["structure"] for x in d["grad_total"]]) / (np.mean([x["structure"] for x in d["grad_total"]]) + 1e-9))
                                         for c, d in per_cond.items()}}
        (att_root / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        # figure: region fractions per condition (grad vs attention vs edge baseline)
        conds = [t["cond"] for t in table]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
        for ax, key, title in ((axes[0], "grad", "single-step x0 gradient"), (axes[1], "attention_ball_late", "attention (ball queries, late blocks)"), (axes[2], "edge_baseline", "Sobel edge baseline")):
            bottom = np.zeros(len(conds))
            for reg in ("ball", "structure", "decoy", "support", "franka", "clutter", "background"):
                vals = np.array([(t[key][reg][0] if key == "grad" else t[key][reg]) for t in table])
                ax.bar(conds, vals, bottom=bottom, label=reg)
                bottom += vals
            ax.set(title=f"{scene}: {title}", ylabel="mass fraction", ylim=(0, 1))
            ax.tick_params(axis="x", labelrotation=60, labelsize=7)
        axes[0].legend(fontsize=7, ncol=2)
        fig.savefig(att_root / "region_fractions.png", dpi=150)
        plt.close(fig)
        # heatmap overlays for the core conditions (mean over steps and seeds), last conditioning frame
        core = [c for c in ("A", "B", "C", "D", "XW", "XF") if c in per_cond]
        if core:
            fig, axes = plt.subplots(len(core), 2, figsize=(11, 2.6 * len(core)), constrained_layout=True, squeeze=False)
            for r_i, c in enumerate(core):
                rec = manifest[c]
                inp = np.load(root / rec["input_npz"])
                frame = inp["condition_primary"][4]
                acc, n = None, 0
                for sfile in sorted((att_root / c).glob("seed_*/maps.npz")):
                    m = np.load(sfile)
                    for i in steps:
                        g = m[f"grad_video_step{i}"].astype(np.float32)[4]
                        acc = g if acc is None else acc + g
                        n += 1
                acc = acc / max(n, 1)
                blur = acc
                axes[r_i][0].imshow(frame); axes[r_i][0].set_title(f"{c}: last conditioning frame", fontsize=9); axes[r_i][0].axis("off")
                axes[r_i][1].imshow(frame, alpha=0.55)
                axes[r_i][1].imshow(blur / (blur.max() + 1e-12), cmap="magma", alpha=0.6, vmin=0, vmax=0.35)
                axes[r_i][1].set_title(f"{c}: |d s/d x| (frame 4, mean over steps and seeds)", fontsize=9); axes[r_i][1].axis("off")
            fig.savefig(att_root / "heatmaps_core.png", dpi=120)
            plt.close(fig)
        print(f"[{scene}] attribution aggregate: {len(rows)} samples -> {att_root / 'summary.json'}", flush=True)
        for t in table:
            print(f"  {t['cond']:3s} n={t['n_seeds']} grad mass ball {t['grad']['ball'][0]:.2f} struct {t['grad']['structure'][0]:.2f} "
                  f"decoy {t['grad']['decoy'][0]:.2f} support {t['grad']['support'][0]:.2f} bg {t['grad']['background'][0]:.2f} | "
                  f"density ball {t['grad_density']['ball']:.1f} struct {t['grad_density']['structure']:.1f} decoy {t['grad_density']['decoy']:.1f} "
                  f"support {t['grad_density']['support']:.1f} bg {t['grad_density']['background']:.1f} | "
                  f"att(ball q, late) density struct {t['attention_ball_late_density']['structure']:.1f} decoy {t['attention_ball_late_density']['decoy']:.1f} "
                  f"ball {t['attention_ball_late_density']['ball']:.1f}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=Path, default=Path("artifacts/pvr"))
    common.add_argument("--scenes", nargs="*", default=["hill"])
    p = sub.add_parser("attribute", parents=[common])
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--original-checkpoint", type=Path, required=True)
    p.add_argument("--conds", nargs="*", default=None)
    p.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3, 4])
    p.add_argument("--steps", type=int, nargs="*", default=list(DEFAULT_STEPS))
    p.add_argument("--randomize", action="store_true")
    p.add_argument("--randomize-all", action="store_true")
    p.add_argument("--wrong-target", action="store_true")
    p.add_argument("--refit-probe", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    a = sub.add_parser("aggregate", parents=[common])
    args = parser.parse_args()
    if args.cmd == "attribute":
        run_attribution(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
