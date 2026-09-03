"""V-JEPA 2-AC latent track (2026-09-03): input contract shared by generator, rollout, analysis.

Time base: the AC predictor was post-trained on DROID at 4 fps, so one predictor
step is 0.25 s = 4 capture frames at our 16 fps.  Context = two strictly pre-event
frames (o, o+4); the predictor then rolls 4 steps to frames o+8 .. o+20 (0.5-1.25 s
after the first context frame), all inside the 21-frame Cosmos horizon.  Offsets
o in {0,1,2} replace seeds (the predictor is deterministic).

Candidate futures rendered by MuJoCo for the same target frames:
  P  physics ground truth (the scene's own integration),
  K  kinematic continuation: the red ball keeps the context velocity along the
     terrain (hill: constant horizontal speed up and over the profile; two-ball:
     red continues and rigidly pushes blue once in contact; wall: red stops at the
     wall without rebound; kin: identical to physics by construction),
  J  physics re-placed with the red ball shifted +1 px in x (encoder noise floor).
"""

CAM = "vj_side"
SIZE = 256
FPS = 16
N_FRAMES = 24
STEP = 4                       # frames per predictor step (0.25 s)
N_STEPS = 4
OFFSETS = (0, 1, 2)
POSE = [0.58, -0.002, 0.248, -3.067, 0.031, -1.913, 0.997]   # DROID home-like EE pose (repo example traj)

# family -> (fovy deg, camera distance m); pixel scale for the jitter render
VJ_CAMS = {"hill": (36.0, 2.05), "kin": (36.0, 2.05), "two_ball": (44.0, 1.5), "edge": (36.0, 2.05)}


def kind_of(family: str) -> str:
    if family.startswith("hill_roll"):
        return "hill"
    if family.startswith("kin_roll"):
        return "kin"
    if family.startswith("support_edge"):
        return "edge"
    if family.startswith("two_ball"):
        return "two_ball"
    if family.startswith("wall_bounce"):
        return "wall"
    raise KeyError(family)


def ctx_indices(o: int):
    return (o, o + STEP)


def target_indices(o: int):
    return tuple(o + STEP * (k + 2) for k in range(N_STEPS))


def px_per_m(family: str) -> float:
    import math
    fovy, dist = VJ_CAMS["two_ball" if kind_of(family) in ("two_ball", "wall") else kind_of(family)]
    extent = 2.0 * dist * math.tan(math.radians(fovy) / 2.0)
    return SIZE / extent
