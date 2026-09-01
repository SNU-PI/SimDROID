"""Below/above-threshold GIFs and pre-check GIFs for Bundle A.

Reads the stored 21-frame ground-truth clips (no re-simulation) and writes
paired below/above GIFs per scene plus single GIFs for the two pre-checks,
sized for embedding in review pages.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from gen.render import font

PAIRS = {"hill_roll": (1, 9), "two_ball": (1, 9), "pendulum_rod": (1, 9)}
PRE = ("hill_roll_pre", "wall_bounce_pre")


def records(path):
    return {row["id"]: row
            for row in (json.loads(line)
                        for line in path.read_text().splitlines() if line.strip())}


def load_clip(root, record):
    return imageio.mimread(root / record["gt_clip"], memtest=False)


def compose_pair(left, right, left_rec, right_rec, index):
    w, h, header = 416, 240, 40
    canvas = Image.new("RGB", (2 * w, h + header), (23, 26, 30))
    draw = ImageDraw.Draw(canvas)
    fnt = font(14)
    for col, (frame, rec) in enumerate(((left, left_rec), (right, right_rec))):
        image = Image.fromarray(np.asarray(frame)).resize((w, h), Image.Resampling.BILINEAR)
        canvas.paste(image, (col * w, header))
        label = (f"S={rec['S']:.2f}  GT="
                 + ("SUCCESS" if rec["outcome"] else "FAILURE"))
        draw.text((col * w + 12, 11), label, fill="white", font=fnt)
    draw.text((2 * w - 64, 11), f"{index / 16:.2f}s", fill=(190, 198, 210), font=fnt)
    return np.asarray(canvas)


def compose_single(frame, title, index):
    w, h, header = 416, 240, 40
    canvas = Image.new("RGB", (w, h + header), (23, 26, 30))
    draw = ImageDraw.Draw(canvas)
    fnt = font(14)
    image = Image.fromarray(np.asarray(frame)).resize((w, h), Image.Resampling.BILINEAR)
    canvas.paste(image, (0, header))
    draw.text((12, 11), title, fill="white", font=fnt)
    draw.text((w - 64, 11), f"{index / 16:.2f}s", fill=(190, 198, 210), font=fnt)
    return np.asarray(canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/bundle_a"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/bundle_a/gifs"))
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = records(root / "manifest.jsonl")

    for family, (below_idx, above_idx) in PAIRS.items():
        below = manifest[f"{family}_{below_idx:02d}"]
        above = manifest[f"{family}_{above_idx:02d}"]
        clip_b, clip_a = load_clip(root, below), load_clip(root, above)
        n = min(len(clip_b), len(clip_a))
        frames = [compose_pair(clip_b[i], clip_a[i], below, above, i)
                  for i in range(n)]
        imageio.mimsave(out / f"{family}_pair.gif", frames, duration=125,
                        loop=0, palettesize=128)
        print(out / f"{family}_pair.gif")

    for pre_id in PRE:
        rec = manifest[pre_id]
        clip = load_clip(root, rec)
        title = {"hill_roll_pre": "A1-0  S=0.35 (deceleration check)",
                 "wall_bounce_pre": "A2-0  elastic wall (sign flip check)"}[pre_id]
        frames = [compose_single(clip[i], title, i) for i in range(len(clip))]
        imageio.mimsave(out / f"{pre_id}.gif", frames, duration=125,
                        loop=0, palettesize=128)
        print(out / f"{pre_id}.gif")


if __name__ == "__main__":
    main()
