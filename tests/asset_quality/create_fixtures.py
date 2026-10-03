#!/usr/bin/env python3
"""Create deterministic good and bad Blender assets for integration tests."""

import argparse
import math
import sys
from pathlib import Path

import bpy


def args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def good_fixture(path):
    reset()
    bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, 0.06), scale=(0.025, 0.025, 0.06))
    obj = bpy.context.object
    obj.name = "textured_test_cup"
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bevel = obj.modifiers.new("manufactured_edges", "BEVEL")
    bevel.width = 0.004
    bevel.segments = 3
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=bevel.name)
    bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR", center="MEDIAN")
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project()
    bpy.ops.object.mode_set(mode="OBJECT")

    image = bpy.data.images.new("qa_albedo", width=32, height=32)
    pixels = []
    for y in range(32):
        for x in range(32):
            light = 0.75 if (x // 4 + y // 4) % 2 else 0.25
            pixels.extend((0.1 + 0.7 * light, 0.2 + 0.4 * light, 0.7, 1.0))
    image.pixels = pixels
    image.pack()
    material = bpy.data.materials.new("painted_plastic")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = image
    principled = nodes.get("Principled BSDF")
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Roughness"].default_value = 0.45
    material.node_tree.links.new(texture.outputs["Color"], principled.inputs["Base Color"])
    obj.data.materials.append(material)
    bpy.data.materials.new("unused_orphan_material")

    bpy.context.view_layer.objects.active = obj
    bpy.ops.rigidbody.object_add()
    obj.rigid_body.mass = 0.22
    obj.rigid_body.friction = 0.65
    obj.rigid_body.restitution = 0.03
    obj.rigid_body.collision_shape = "BOX"
    bpy.ops.wm.save_as_mainfile(filepath=str(path))


def bad_fixture(path):
    reset()
    mesh = bpy.data.meshes.new("broken_mesh")
    mesh.from_pydata(
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (3, 3, 3)],
        [],
        [(0, 1, 2), (0, 2, 3), (0, 0, 1)],
    )
    obj = bpy.data.objects.new("broken_unscaled_asset", mesh)
    bpy.context.collection.objects.link(obj)
    obj.scale = (10.0, 0.01, 4.0)
    material = bpy.data.materials.new("flat_untextured")
    material.use_nodes = False
    obj.data.materials.append(material)
    bpy.ops.wm.save_as_mainfile(filepath=str(path))


def main():
    output = args().out
    output.mkdir(parents=True, exist_ok=True)
    good_fixture(output / "good.blend")
    bad_fixture(output / "bad.blend")


if __name__ == "__main__":
    main()
