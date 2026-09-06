"""Region masks for the PVR scenes from MuJoCo segmentation ids.

Regions: ball, structure (the on-path structure whose variable is the relevant edit),
decoy (the look-alike structure behind the ball), support (floor / plate body), franka,
clutter (tray, fixture, bin, ticks, tape, wall, refcube), background (sky, back wall).
Hump footprints inside the single height field are separated by projecting their world
x-extent to pixel columns with the scene camera (near-orthographic side view).
"""

from __future__ import annotations

import numpy as np

CAMERAS = {
    "hill": dict(pos=(0.0, -2.05, 0.30), fovy=26.0),
    "collide": dict(pos=(-0.15, -1.5, 0.24), fovy=26.0),
    "edge": dict(pos=(-0.30, -2.05, 0.30), fovy=26.0),
}
FRANKA_ROOT = "franka_static"
CLUTTER = {"tray", "tray_part", "fixture", "bin", "tape", "refcube", "plate_lip", "joint"}


def x_to_col(x, cam, width=832, height=480, y=0.0, z=0.0):
    """Pixel column of world point (x, y, z) for a side camera with xyaxes '1 0 0 0 0.1 0.995'."""
    cx, cy, cz = cam["pos"]
    d = np.array([x - cx, y - cy, z - cz], dtype=float)
    xa = np.array([1.0, 0.0, 0.0])
    ya = np.array([0.0, 0.1, 0.995])
    ya = ya / np.linalg.norm(ya)
    fwd = -np.cross(xa, ya)
    f = (height / 2.0) / np.tan(np.radians(cam["fovy"] / 2.0))
    return float(width / 2.0 + f * (d @ xa) / (d @ fwd))


def _franka_geoms(meta):
    names, bodies, body_names, parents = (meta["geom_names"], meta["geom_bodies"],
                                          meta["body_names"], meta["body_parents"])
    root = body_names.index(FRANKA_ROOT) if FRANKA_ROOT in body_names else -1
    out = set()
    if root < 0:
        return out
    for gid, bid in enumerate(bodies):
        b = bid
        while b > 0:
            if b == root:
                out.add(gid)
                break
            b = parents[b]
    return out


def region_masks(seg, meta, scene, params):
    """seg: (H, W) int geom ids (-1 background).  Returns dict name -> bool mask."""
    names = meta["geom_names"]
    H, W = seg.shape
    cam = CAMERAS[scene]
    by_name = {n: i for i, n in enumerate(names) if n}
    cols = np.arange(W)[None, :].repeat(H, axis=0)

    def geom(name):
        gid = by_name.get(name, None)
        return (seg == gid) if gid is not None else np.zeros((H, W), bool)

    franka = np.zeros((H, W), bool)
    for gid in _franka_geoms(meta):
        franka |= seg == gid
    ticks = np.zeros((H, W), bool)
    for n, gid in by_name.items():
        if n.startswith("tick") or n.startswith("ptick"):
            ticks |= seg == gid
    clutter = ticks.copy()
    for n in CLUTTER:
        clutter |= geom(n)
    background = (seg < 0) | geom("backwall")

    if scene == "hill":
        ball = geom("ballg")
        hill = geom("hillg")
        W_h = 0.30
        c0, c1 = x_to_col(0.10, cam), x_to_col(0.10 + W_h, cam)
        d0, d1 = x_to_col(-0.78, cam), x_to_col(-0.78 + W_h, cam)
        structure = hill & (cols >= c0) & (cols <= c1)
        if params.get("variant") == "wall":
            structure = structure | geom("wall")
        decoy = hill & (cols >= d0) & (cols <= d1) if float(params.get("h_d", 0.0)) > 0 else np.zeros((H, W), bool)
        support = (hill | geom("floor")) & ~structure & ~decoy
    elif scene == "collide":
        ball = geom("gA")
        structure = geom("gB") | geom("wall")
        decoy = geom("gD")
        support = geom("floor")
    elif scene == "edge":
        ball = geom("ballg")
        plate = geom("plate")
        xe = float(params["x_edge"])
        xl = float(params.get("x_left", -1.30))
        e0 = x_to_col(xe - 0.12, cam)
        l1 = x_to_col(xl + 0.12, cam)
        structure = plate & (cols >= e0)
        decoy = plate & (cols <= l1) if xl > -1.2 else np.zeros((H, W), bool)
        support = (plate | geom("plate2") | geom("floor")) & ~structure & ~decoy
        clutter = clutter | geom("joint")
    else:
        raise ValueError(scene)

    masks = dict(ball=ball, structure=structure, decoy=decoy, support=support,
                 franka=franka, clutter=clutter & ~ball, background=background)
    covered = np.zeros((H, W), bool)
    for k in ("ball", "structure", "decoy", "support", "franka", "clutter", "background"):
        masks[k] = masks[k] & ~covered
        covered |= masks[k]
    masks["other"] = ~covered
    return masks


def region_fractions(weight, masks):
    """Share of a non-negative weight map (H, W) inside each region."""
    total = float(weight.sum()) + 1e-12
    return {k: float(weight[m].sum()) / total for k, m in masks.items()}


def downsample_masks(masks, grid_h, grid_w):
    """Area fraction of each region per latent-grid cell (grid_h x grid_w)."""
    out = {}
    for k, m in masks.items():
        H, W = m.shape
        a = m.astype(np.float32).reshape(grid_h, H // grid_h, grid_w, W // grid_w).mean(axis=(1, 3))
        out[k] = a
    return out
