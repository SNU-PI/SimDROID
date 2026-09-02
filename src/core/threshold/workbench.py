"""Workbench appearance layer for the Phase A scenes (PhysicsGen redesign, 2026-09-02).

Appearance only.  The physics geoms, contact pairs, timesteps and cameras of the
Bundle A scenes are untouched: the same 'floor' plane carries the motion (it is
merely re-materialised as a grey inspection-table top) and every added body is
collision-free (contype/conaffinity 0) and outside the motion corridor.

Pieces: laminate table top, a back wall, a static Franka Panda (menagerie meshes
in the team's baked pose, no joints), a parts tray / fixture / bin as context
clutter, and a tape strip under the tick posts as the visible scale cue.

Colours keep the frozen colour masks of the adjudicator exclusive: red ball,
blue ball, green cable-cover hump; nothing else in frame is red, blue or green.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from core.threshold.base import wrap

HERE = Path(__file__).resolve().parent
MENAGERIE = Path(os.environ.get(
    "MENAGERIE_PANDA",
    "/mnt/nvme/migration/jihun/SimDROID/data/stage0/mujoco_menagerie/franka_emika_panda"))
_BAKED_PATH = "/data/pgc/simdroid/stage0/mujoco_menagerie/franka_emika_panda"
FLOOR_GEOM = '<geom name="floor" type="plane" size="3 3 0.1" material="floor"/>'

WB_ASSET = """
    <texture name="laminate" type="2d" builtin="flat" rgb1="0.66 0.67 0.69" width="512" height="512"
             mark="random" markrgb="0.60 0.61 0.63" random="0.08"/>
    <material name="tabletop" texture="laminate" texrepeat="3 3" specular="0.25" shininess="0.35"/>
    <material name="wallpaint" rgba="0.58 0.57 0.55 1" specular="0.05" shininess="0.05"/>
    <material name="cover" rgba="0.18 0.60 0.30 1" specular="0.20" shininess="0.25"/>
    <material name="steel" rgba="0.62 0.64 0.66 1" specular="0.6" shininess="0.6"/>
    <material name="pandawhite" rgba="0.86 0.87 0.89 1" specular="0.3" shininess="0.3"/>
    <material name="pandadark" rgba="0.20 0.21 0.23 1" specular="0.2" shininess="0.2"/>
    <material name="tape" rgba="0.92 0.90 0.80 1" specular="0.1"/>
"""

# Baked pose offset: the team's static Panda was posed for a table top at
# z=0.40 with its base near (-0.52, -0.34); shifting by (+0.05, +0.95, -0.43)
# stands it on this table behind the motion corridor (y=0), hand at ~y=0.94.
FRANKA_POS = "0.05 0.95 -0.43"


def franka_blocks(pos=FRANKA_POS):
    """(asset xml, body xml) for the static Panda; empty strings if meshes are absent."""
    if not (MENAGERIE / "assets").is_dir():
        return "", ""
    asset = (HERE / "franka_asset.xml").read_text().replace(_BAKED_PATH, str(MENAGERIE))
    body = (HERE / "franka_body.xml").read_text()

    def recolour(line):
        mat = "pandadark" if ('mesh="fr_hand' in line or 'mesh="fr_finger' in line) else "pandawhite"
        return re.sub(r'rgba="[^"]*"', f'material="{mat}"', line)

    body = "\n".join(recolour(l) for l in body.splitlines())
    return asset, f'    <body name="franka_static" pos="{pos}">\n{body}\n    </body>\n'


def wb_body(tape_y=0.17, tape_half=0.60):
    # Extra cameras for multi-view consumers (VERA's 3-view canvas): an elevated
    # three-quarter view and a near-top view; the scene's own side camera stays primary.
    return f"""
    <camera name="wb_iso" fovy="28" pos="-1.25 -1.45 0.80" zaxis="-1.25 -1.45 0.75"/>
    <camera name="wb_top" fovy="36" pos="0 -0.55 1.65" zaxis="0 -0.55 1.60"/>
    <geom name="backwall" type="box" size="2.6 0.05 0.9" pos="0 1.65 0.9" material="wallpaint" contype="0" conaffinity="0"/>
    <geom name="tape" type="box" size="{tape_half} 0.012 0.0008" pos="0 {tape_y} 0.0008" material="tape" contype="0" conaffinity="0"/>
    <geom name="tray" type="box" size="0.14 0.10 0.015" pos="-0.90 0.62 0.015" material="steel" contype="0" conaffinity="0"/>
    <geom name="tray_part" type="cylinder" size="0.025 0.02" pos="-0.93 0.60 0.05" material="pandadark" contype="0" conaffinity="0"/>
    <geom name="fixture" type="box" size="0.06 0.05 0.06" pos="0.80 0.55 0.06" material="pandadark" contype="0" conaffinity="0"/>
    <geom name="bin" type="box" size="0.10 0.08 0.07" pos="0.62 0.85 0.07" material="steel" contype="0" conaffinity="0"/>
"""


def wrap_wb(name, body, contact="", extra_asset="", tape_y=0.17, tape_half=0.60):
    """Bundle-A wrap() plus the workbench dressing; physics untouched."""
    fa, fb = franka_blocks()
    xml = wrap(name, body + wb_body(tape_y, tape_half) + fb, contact,
               extra_asset=extra_asset + WB_ASSET + fa)
    assert FLOOR_GEOM in xml, "floor geom signature changed"
    return xml.replace(FLOOR_GEOM, FLOOR_GEOM.replace('material="floor"', 'material="tabletop"'))
