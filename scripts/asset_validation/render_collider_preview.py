"""Render the authored compound colliders over the visual asset."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def arguments():
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True, type=Path)
    parser.add_argument("--physics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(values)


def point_at(obj, target):
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def bounds(objects):
    minimum = Vector((math.inf, math.inf, math.inf))
    maximum = Vector((-math.inf, -math.inf, -math.inf))
    for obj in objects:
        for corner in obj.bound_box:
            point = obj.matrix_world @ Vector(corner)
            for axis in range(3):
                minimum[axis] = min(minimum[axis], point[axis])
                maximum[axis] = max(maximum[axis], point[axis])
    return minimum, maximum


def material(name, color, emission=0.0):
    value = bpy.data.materials.new(name)
    value.diffuse_color = (*color, 1.0)
    value.use_nodes = True
    node = value.node_tree.nodes.get("Principled BSDF")
    if node:
        node.inputs["Base Color"].default_value = (*color, 1.0)
        node.inputs["Roughness"].default_value = 0.55
        if emission:
            emission_input = node.inputs.get("Emission Color") or node.inputs.get("Emission")
            if emission_input:
                emission_input.default_value = (*color, 1.0)
            strength = node.inputs.get("Emission Strength")
            if strength:
                strength.default_value = emission
    return value


def main():
    args = arguments()
    report = json.loads(args.physics.read_text())
    boxes = report.get("authoring", {}).get("collider_boxes", [])
    if not boxes:
        raise RuntimeError("physics report contains no collider boxes")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.asset.resolve()))
    visual = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    minimum, maximum = bounds(visual)
    center = (minimum + maximum) * 0.5
    dimensions = maximum - minimum
    diagonal = max(dimensions.length, 0.05)

    neutral_material = material("MissingVisualMaterial", (0.32, 0.35, 0.38))
    for obj in visual:
        if not obj.data.materials:
            obj.data.materials.append(neutral_material)

    collider_material = material("Collider", (0.02, 1.0, 0.28), emission=1.5)
    for index, box in enumerate(boxes):
        center_y_up = box["center_m"]
        size_y_up = box["size_m"]
        center_z_up = (center_y_up[0], -center_y_up[2], center_y_up[1])
        size_z_up = (size_y_up[0], size_y_up[2], size_y_up[1])
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=center_z_up)
        collider = bpy.context.object
        collider.name = f"Collider_{index:02d}"
        collider.dimensions = size_z_up
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        collider.data.materials.append(collider_material)
        wire = collider.modifiers.new("ColliderWire", "WIREFRAME")
        wire.thickness = max(diagonal * 0.008, 0.001)
        wire.use_replace = True

    floor_material = material("Floor", (0.15, 0.16, 0.18))
    bpy.ops.mesh.primitive_plane_add(
        size=max(max(dimensions) * 4.0, 0.5),
        location=(center.x, center.y, minimum.z - diagonal * 0.01),
    )
    bpy.context.object.data.materials.append(floor_material)

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = max(max(dimensions) * 1.55, 0.08)
    camera.location = center + Vector((1.0, -1.0, 0.8)).normalized() * diagonal * 2.2
    point_at(camera, center)
    bpy.context.scene.camera = camera

    for energy, offset, size in (
        (180.0, (1.0, -1.0, 1.6), diagonal * 2.5),
        (90.0, (-1.0, -0.3, 1.0), diagonal * 2.0),
        (130.0, (0.0, 0.5, 2.0), diagonal * 2.0),
    ):
        location = center + Vector(offset).normalized() * diagonal * 1.8
        bpy.ops.object.light_add(type="AREA", location=location)
        light = bpy.context.object
        light.data.energy = energy
        light.data.shape = "DISK"
        light.data.size = size
        point_at(light, center)

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "WORLD"
    scene.render.resolution_x = 640
    scene.render.resolution_y = 640
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    if scene.world is None:
        scene.world = bpy.data.worlds.new("ValidationWorld")
    scene.world.color = (0.025, 0.03, 0.035)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(args.output.resolve())
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
