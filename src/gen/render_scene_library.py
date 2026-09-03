"""Render a 3-frame strip (t=0 / mid / end) for the early threshold scene modules -> artifacts/scene_library/.
Run: MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=src python src/gen/render_scene_library.py"""
import sys, json, traceback
import numpy as np
from pathlib import Path
from PIL import Image
from core.threshold import FAMILIES, ENERGY_FAMILIES
from gen.render import RenderCfg, roll
out = Path("artifacts/scene_library"); out.mkdir(exist_ok=True)
cfg = RenderCfg(height=256, width=256, fps=16, quality=8)
report = {}
mods = dict(FAMILIES); mods.update({f"energy_{k}": v for k, v in ENERGY_FAMILIES.items()})
for name, cls in mods.items():
    try:
        rng = np.random.RandomState(3)
        params = cls.params_for_margin(0.25) if hasattr(cls, 'params_for_margin') else cls.sample(rng)
        frames, labels = roll(cls, params, cls.cam, cfg)
        n = len(frames); idx = [0, n // 2, n - 1]
        strip = np.concatenate([frames[i] for i in idx], axis=1)
        Image.fromarray(strip).save(out / f"{name}.png")
        report[name] = {"n_frames": n, "params": {k: float(v) for k, v in params.items()},
                        "outcome": int(labels.get("outcome", -1)), "margin": float(labels.get("margin", float("nan"))),
                        "event_frame": int(labels.get("event_frame", -1)), "cam": cls.cam, "doc": (cls.__doc__ or "").strip().split("\n")[0]}
        print(name, report[name], flush=True)
    except Exception as e:
        report[name] = {"error": repr(e)}
        print(name, "FAILED", repr(e), flush=True); traceback.print_exc()
(out / "report.json").write_text(json.dumps(report, indent=1))
