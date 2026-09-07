import os
"""Render lo/hi matched-pair strips for the five pusher families (team repo, R3-79 dense pairs).

Reads DENSE_DIR/pairs/<family>/<param>/0000_{lo,hi}.npz (PNG-in-npz, 640x480, 50 Hz) and writes one
JPEG strip per family: top row = lo quantile, bottom row = hi quantile, five frames spanning the clip.
Usage: python src/gen/render_pusher_strips.py <out_dir>
"""
import sys, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

TEAM_SRC = Path(os.environ.get("SIMDROID_CODE_SRC", "../code/src"))
sys.path.insert(0, str(TEAM_SRC))
from core.dense_io import load_episode  # noqa: E402

PAIRS = Path(os.environ.get("SIMDROID_PAIRS", "../data/episodes_dense/pairs"))
CELLS = [("slide", "mu"), ("roll", "roll_fric"), ("bounce", "damping"), ("collide", "mass2"), ("incline", "mu")]


def main(out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    report = {}
    for fam, param in CELLS:
        eps = {s: load_episode(PAIRS / fam / param / f"0000_{s}.npz") for s in ("lo", "hi")}
        n = len(eps["lo"]["clip"])
        idx = [0, n // 5, (2 * n) // 5, (3 * n) // 5, n - 1]
        w, h, gap = 240, 180, 4
        canvas = Image.new("RGB", (5 * w + 4 * gap, 2 * h + gap), (35, 38, 44))
        for r, s in enumerate(("lo", "hi")):
            for c, k in enumerate(idx):
                im = Image.fromarray(eps[s]["clip"][k]).resize((w, h), Image.LANCZOS)
                canvas.paste(im, (c * (w + gap), r * (h + gap)))
        d = ImageDraw.Draw(canvas)
        for c, k in enumerate(idx):
            d.rectangle([c * (w + gap) + 4, 4, c * (w + gap) + 62, 20], fill=(20, 22, 26))
            d.text((c * (w + gap) + 8, 6), f"t={k * 0.02:.2f}s", fill=(235, 235, 235))
        canvas.save(out / f"{fam}_{param}_pair.jpg", quality=82, optimize=True)
        pv = {s: dict(zip(eps[s]["param_names"], [float(x) for x in eps[s]["params"]])) for s in ("lo", "hi")}
        st = {s: eps[s]["state"] for s in ("lo", "hi")}
        report[f"{fam}/{param}"] = dict(n_frames=int(n), frames=idx, params=pv,
                                        travel_lo=float(np.linalg.norm(st["lo"][-1, :2] - st["lo"][0, :2])),
                                        travel_hi=float(np.linalg.norm(st["hi"][-1, :2] - st["hi"][0, :2])),
                                        zmax_lo=float(st["lo"][:, 2].max()), zmax_hi=float(st["hi"][:, 2].max()),
                                        meta=eps["lo"]["meta"])
        print(fam, param, n, pv["lo"], pv["hi"])
    (out / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "artifacts/scene_library/pusher5")
