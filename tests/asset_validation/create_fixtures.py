"""Create deterministic good and broken GLB fixtures for the validator test."""

import sys
from pathlib import Path

import bpy

output = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
output.mkdir(parents=True, exist_ok=True)


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def material():
    value = bpy.data.materials.new("TestMaterial")
    value.diffuse_color = (0.05, 0.35, 0.8, 1.0)
    value.use_nodes = True
    node = value.node_tree.nodes.get("Principled BSDF")
    node.inputs["Base Color"].default_value = (0.05, 0.35, 0.8, 1.0)
    node.inputs["Roughness"].default_value = 0.55
    return value


reset()
bpy.ops.mesh.primitive_cube_add(size=1.0)
cube = bpy.context.object
cube.dimensions = (0.04, 0.04, 0.10)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
cube.data.materials.append(material())
bpy.context.view_layer.objects.active = cube
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.uv.smart_project()
bpy.ops.object.mode_set(mode="OBJECT")
bpy.ops.rigidbody.object_add()
cube.rigid_body.mass = 0.12
cube.rigid_body.friction = 0.6
cube.rigid_body.restitution = 0.1
cube.rigid_body.collision_shape = "BOX"
bpy.ops.wm.save_as_mainfile(filepath=str(output / "good.blend"))
bpy.ops.export_scene.gltf(filepath=str(output / "good.glb"), export_format="GLB")

reset()
bpy.ops.mesh.primitive_cube_add(size=0.1)
oversized = bpy.context.object
oversized.data.materials.append(material())
bpy.context.view_layer.objects.active = oversized
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.uv.smart_project()
bpy.ops.object.mode_set(mode="OBJECT")
bpy.ops.export_scene.gltf(filepath=str(output / "oversized.glb"), export_format="GLB")

reset()
mesh = bpy.data.meshes.new("BrokenMesh")
mesh.from_pydata([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], [], [(0, 1, 2)])
broken = bpy.data.objects.new("BrokenAsset", mesh)
bpy.context.collection.objects.link(broken)
bpy.context.view_layer.objects.active = broken
broken.select_set(True)
bpy.ops.export_scene.gltf(filepath=str(output / "bad.glb"), export_format="GLB")
