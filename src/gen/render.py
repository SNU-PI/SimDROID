"""Shared rendering helpers for the sweep generators.

The four generator scripts each grew their own copy of "set up EGL, roll a
scene, write an mp4, tile a preview image". They are collected here so the
render settings live in one place and a change to resolution or codec cannot
silently apply to one output family but not another.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("MUJOCO_GL", "egl")

_SHARED_GL = None


def ensure_shared_gl_context():
    """One EGL context per process, shared by every mujoco.Renderer.

    mujoco.Renderer creates and frees its own EGL context per instance; with one
    renderer per sample that churn produced intermittent all-black renderers
    (2026-09-02, GPU shared with a diffusion job).  MuJoCo draws into its own
    offscreen FBO (sized by <global offwidth/offheight>), so a single tiny
    pbuffer context made current once is enough for every renderer.
    """
    global _SHARED_GL
    if _SHARED_GL is None:
        import mujoco
        from mujoco.rendering.classic import gl_context as _glc
        _SHARED_GL = mujoco.GLContext(64, 64)
        _SHARED_GL.make_current()
        _glc.GLContext = None          # Renderer.__init__ then reuses the current context
    return _SHARED_GL

FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


@dataclass(frozen=True)
class RenderCfg:
    """Render size, video encode rate, and optionally the sim capture rate.

    capture_dt and fps are deliberately separate. The sweep captures the
    simulation at its native 50 Hz and only *plays back* at 16 fps, while the
    diagnostics capture at exactly 16 Hz so each frame is one model timestep.
    Collapsing the two would silently resample the physics.
    """

    height: int = 256
    width: int = 256
    fps: int = 16
    quality: int = 8
    capture_dt: float | None = None      # None keeps the scene's own rate


SWEEP_CFG = RenderCfg(height=256, width=256, fps=16, quality=8)
DIAG_CFG = RenderCfg(height=480, width=832, fps=16, quality=9, capture_dt=1 / 16)


def font(size: int):
    for path in FONT_PATHS:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def roll(scene_cls, params, camera, cfg: RenderCfg = SWEEP_CFG, n_frames=None):
    """Run one episode and return (frames, labels).

    The scene's capture rate and render size are overridden from cfg so every
    generator shares one definition of "a frame".
    """
    ensure_shared_gl_context()
    scene = scene_cls()
    scene.cam = camera
    if cfg.capture_dt is not None:
        scene.capture_dt = cfg.capture_dt
    scene.render_height = cfg.height
    scene.render_width = cfg.width
    if n_frames is not None:
        scene.n_frames = n_frames
    import sys, time
    for attempt in range(6):
        # Rendering is deterministic: two passes must agree exactly.  Under a
        # GPU shared with a diffusion job we saw all-black and partially dark
        # frames (2026-09-02); the double render catches both.
        passes = []
        for _ in range(2):
            result = scene.run(params, render=True)
            passes.append(result.pop("frames"))
            result.pop("trace", None)
            if scene._r is not None:
                scene._r.close()
                scene._r = None
        frames = passes[0]
        dark = frames is not None and bool((frames.reshape(len(frames), -1).mean(axis=1) < 1.0).any())
        agree = frames is None or np.array_equal(passes[0], passes[1])
        if not dark and agree:
            break
        print(f"[render] {scene_cls.__name__}: {'dark' if dark else 'non-deterministic'} render on "
              f"attempt {attempt + 1}; retrying", file=sys.stderr, flush=True)
        time.sleep(3.0 * (attempt + 1))
    else:
        raise RuntimeError(f"{scene_cls.__name__}: corrupt renders 6x -- refusing to emit corrupt data")
    del scene
    return frames, result


def roll_views(scene_cls, params, cameras, cfg: RenderCfg = SWEEP_CFG, n_frames=None):
    """Same episode rendered from several cameras; labels come from the first."""
    views, labels = {}, None
    for cam in cameras:
        frames, res = roll(scene_cls, params, cam, cfg, n_frames)
        views[cam] = frames
        labels = labels or res
    return views, labels


def write_video(path: Path, frames, cfg: RenderCfg = SWEEP_CFG):
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, list(frames), fps=cfg.fps, codec="libx264",
                    quality=cfg.quality, macro_block_size=1)


def write_png(path: Path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(path)


def tile(images, labels=None, cell=256, pad=6, label_h=22):
    """Lay images out in a row with optional captions underneath."""
    n = len(images)
    h = cell + (label_h if labels else 0)
    canvas = Image.new("RGB", (n * cell + (n - 1) * pad, h), (24, 26, 30))
    draw = ImageDraw.Draw(canvas)
    fnt = font(14)
    for i, img in enumerate(images):
        im = Image.fromarray(np.asarray(img, dtype=np.uint8))
        if im.size != (cell, cell):
            im = im.resize((cell, cell), Image.BILINEAR)
        x = i * (cell + pad)
        canvas.paste(im, (x, 0))
        if labels:
            w = draw.textlength(labels[i], font=fnt)
            draw.text((x + (cell - w) / 2, cell + 4), labels[i],
                      font=fnt, fill=(235, 238, 242))
    return canvas


def write_preview(path: Path, images, labels=None, cell=256):
    path.parent.mkdir(parents=True, exist_ok=True)
    tile(images, labels, cell).save(path)
