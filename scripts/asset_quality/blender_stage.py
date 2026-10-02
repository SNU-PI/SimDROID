#!/usr/bin/env python3
"""Blender-side source preflight, interchange export, and canonical renders."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def check(check_id, status, message, metrics=None):
    return {
        "id": check_id,
        "status": status,
        "message": message,
        "metrics": metrics or {},
    }


def material_report(material):
    result = {
        "name": material.name,
        "uses_nodes": bool(material.use_nodes),
        "principled": False,
        "base_color_texture": False,
        "metallic_authored": False,
        "roughness_authored": False,
        "normal_texture": False,
        "missing_images": [],
    }
    if not material.use_nodes or not material.node_tree:
        return result
    for node in material.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            result["principled"] = True
            for key, output_key in (
                ("Base Color", "base_color_texture"),
                ("Metallic", "metallic_authored"),
                ("Roughness", "roughness_authored"),
                ("Normal", "normal_texture"),
            ):
                socket = node.inputs.get(key)
                if socket and (socket.is_linked or key in {"Metallic", "Roughness"}):
                    result[output_key] = True
        if node.type == "TEX_IMAGE" and node.image:
            path = bpy.path.abspath(node.image.filepath)
            if node.image.source == "FILE" and path and not os.path.exists(path):
                result["missing_images"].append(path)
            color_space = getattr(node.image.colorspace_settings, "name", "")
            if color_space == "Non-Color":
                result["metallic_authored"] = True
                result["roughness_authored"] = True
            else:
                result["base_color_texture"] = True
    return result


def mesh_report(obj):
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.normal_update()
    non_manifold = sum(1 for edge in bm.edges if not edge.is_manifold)
    loose_vertices = sum(1 for vertex in bm.verts if not vertex.link_edges)
    degenerate_faces = sum(1 for face in bm.faces if face.calc_area() <= 1e-12)
    bm.free()
    uv_layers = len(mesh.uv_layers)
    zero_uv_faces = 0
    if uv_layers:
        layer = mesh.uv_layers.active.data
        for polygon in mesh.polygons:
            coords = [layer[index].uv for index in polygon.loop_indices]
            area = 0.0
            for index, current in enumerate(coords):
                nxt = coords[(index + 1) % len(coords)]
                area += current.x * nxt.y - nxt.x * current.y
            if abs(area) <= 1e-10:
                zero_uv_faces += 1
    return {
        "name": obj.name,
        "vertices": len(mesh.vertices),
        "edges": len(mesh.edges),
        "faces": len(mesh.polygons),
        "non_manifold_edges": non_manifold,
        "loose_vertices": loose_vertices,
        "degenerate_faces": degenerate_faces,
        "uv_layers": uv_layers,
        "zero_area_uv_faces": zero_uv_faces,
        "scale": [float(value) for value in obj.scale],
        "modifiers": [modifier.type for modifier in obj.modifiers],
        "rigid_body": bool(obj.rigid_body),
        "collision_shape": obj.rigid_body.collision_shape if obj.rigid_body else None,
        "mass_kg": float(obj.rigid_body.mass) if obj.rigid_body else None,
        "friction": float(obj.rigid_body.friction) if obj.rigid_body else None,
        "restitution": float(obj.rigid_body.restitution) if obj.rigid_body else None,
        "collision_margin_m": float(obj.rigid_body.collision_margin) if obj.rigid_body else None,
        "material_slots": [slot.material.name for slot in obj.material_slots if slot.material],
    }


def world_bounds(objects):
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    minimum = Vector((min(point.x for point in points), min(point.y for point in points), min(point.z for point in points)))
    maximum = Vector((max(point.x for point in points), max(point.y for point in points), max(point.z for point in points)))
    return minimum, maximum


def pivot_report(objects, minimum, maximum):
    """Measure whether an asset-level origin is near the support-plane center."""
    candidates = [obj for obj in bpy.context.scene.objects if obj.parent is None and obj.type not in {"CAMERA", "LIGHT"}]
    if not candidates:
        candidates = list(objects)
    target = Vector(((minimum.x + maximum.x) * 0.5, (minimum.y + maximum.y) * 0.5, minimum.z))
    dimensions = maximum - minimum
    xy_tolerance = max(0.005, 0.05 * max(dimensions.x, dimensions.y))
    z_tolerance = max(0.005, 0.05 * max(dimensions.z, 1e-6))
    measured = []
    for obj in candidates:
        origin = obj.matrix_world.translation
        xy_error = math.hypot(origin.x - target.x, origin.y - target.y)
        z_error = abs(origin.z - target.z)
        measured.append({
            "object": obj.name,
            "origin_m": [float(value) for value in origin],
            "xy_error_m": float(xy_error),
            "z_error_m": float(z_error),
        })
    best = min(measured, key=lambda item: item["xy_error_m"] + item["z_error_m"])
    return {
        "at_support_center": best["xy_error_m"] <= xy_tolerance and best["z_error_m"] <= z_tolerance,
        "target_m": [float(value) for value in target],
        "xy_tolerance_m": xy_tolerance,
        "z_tolerance_m": z_tolerance,
        "best_candidate": best,
        "candidates": measured,
    }


def point_camera(camera, target):
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()


def render_views(out_dir, meshes):
    render_dir = out_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    minimum, maximum = world_bounds(meshes)
    center = (minimum + maximum) * 0.5
    extent = maximum - minimum
    radius = max(extent.length * 0.8, 0.1)

    camera_data = bpy.data.cameras.new("qa_camera")
    camera = bpy.data.objects.new("qa_camera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    camera_data.lens = 55

    world = bpy.context.scene.world or bpy.data.worlds.new("qa_world")
    bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.035, 0.035, 0.035, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6

    for index, energy in enumerate((1000.0, 700.0, 500.0)):
        light_data = bpy.data.lights.new("qa_light_%d" % index, "AREA")
        light_data.energy = energy
        light_data.shape = "DISK"
        light_data.size = radius
        light = bpy.data.objects.new("qa_light_%d" % index, light_data)
        bpy.context.scene.collection.objects.link(light)
        offsets = ((1.5, -1.5, 2.0), (-1.5, -0.5, 1.0), (0.0, 1.5, 1.5))
        light.location = center + Vector(offsets[index]) * radius
        point_camera(light, center)

    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = 640
    scene.render.resolution_y = 640
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    views = {
        "front": (0.0, -2.4, 0.8),
        "side": (2.4, 0.0, 0.8),
        "iso": (1.8, -1.8, 1.5),
        "top": (0.01, -0.01, 3.0),
    }
    paths = []
    for name, offset in views.items():
        camera.location = center + Vector(offset) * radius
        point_camera(camera, center)
        scene.render.filepath = str(render_dir / (name + ".png"))
        bpy.ops.render.render(write_still=True)
        paths.append(str(Path("renders") / (name + ".png")))
    return paths


def supported_operator_kwargs(operator, values):
    properties = {prop.identifier for prop in operator.get_rna_type().properties}
    return {key: value for key, value in values.items() if key in properties}


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and not obj.hide_render]
    if not meshes:
        raise RuntimeError("blend file has no renderable mesh objects")

    mesh_data = [mesh_report(obj) for obj in meshes]
    materials = [material_report(material) for material in bpy.data.materials]
    minimum, maximum = world_bounds(meshes)
    dimensions = maximum - minimum
    checks = []
    unapplied = [item["name"] for item in mesh_data if any(abs(value - 1.0) > 1e-4 for value in item["scale"])]
    checks.append(check("BLEND_SCALE", "WARN" if unapplied else "PASS", "unapplied object scale" if unapplied else "object scales are applied", {"objects": unapplied}))
    pivot = pivot_report(meshes, minimum, maximum)
    checks.append(check(
        "BLEND_PIVOT",
        "PASS" if pivot["at_support_center"] else "WARN",
        "asset origin is at the support-plane center" if pivot["at_support_center"] else "asset origin is not at the support-plane center",
        pivot,
    ))
    topology_errors = sum(item["non_manifold_edges"] + item["degenerate_faces"] for item in mesh_data)
    checks.append(check("BLEND_TOPOLOGY", "FAIL" if topology_errors else "PASS", "mesh topology defects found" if topology_errors else "no non-manifold or degenerate topology found", {"defect_count": topology_errors}))
    missing_textures = [path for material in materials for path in material["missing_images"]]
    checks.append(check("BLEND_TEXTURE_PATHS", "FAIL" if missing_textures else "PASS", "missing texture files" if missing_textures else "texture paths resolve", {"missing": missing_textures}))
    uv_coverage = sum(1 for item in mesh_data if item["uv_layers"] > 0) / len(mesh_data)
    checks.append(check("BLEND_UV", "PASS" if uv_coverage >= 0.8 else "WARN", "UV coverage measured", {"mesh_coverage": uv_coverage, "zero_area_faces": sum(item["zero_area_uv_faces"] for item in mesh_data)}))
    pbr_coverage = sum(1 for item in materials if item["principled"] and item["metallic_authored"] and item["roughness_authored"]) / max(len(materials), 1)
    texture_coverage = sum(1 for item in materials if item["base_color_texture"]) / max(len(materials), 1)
    checks.append(check("BLEND_PBR", "PASS" if pbr_coverage >= 0.8 else "WARN", "PBR material coverage measured", {"pbr_coverage": pbr_coverage, "texture_coverage": texture_coverage}))

    glb_path = args.out / "asset.glb"
    bpy.ops.export_scene.gltf(**supported_operator_kwargs(bpy.ops.export_scene.gltf, {
        "filepath": str(glb_path),
        "export_format": "GLB",
        "use_selection": False,
        "export_apply": True,
        "export_materials": "EXPORT",
        # Blender is Z-up. Keep that axis through the interchange step so the
        # Isaac converter does not leave the geometry in glTF's Y-up frame.
        "export_yup": False,
    }))
    render_paths = render_views(args.out, meshes)
    source_physics = {
        "rigid_body_objects": [item["name"] for item in mesh_data if item["rigid_body"]],
        "collider_objects": [item["name"] for item in mesh_data if item["collision_shape"]],
        "collision_shapes": {item["name"]: item["collision_shape"] for item in mesh_data if item["collision_shape"]},
        "masses_kg": {item["name"]: item["mass_kg"] for item in mesh_data if item["mass_kg"] is not None},
        "friction": {item["name"]: item["friction"] for item in mesh_data if item["friction"] is not None},
        "restitution": {item["name"]: item["restitution"] for item in mesh_data if item["restitution"] is not None},
        "collision_margin_m": {item["name"]: item["collision_margin_m"] for item in mesh_data if item["collision_margin_m"] is not None},
    }
    report = {
        "stage": "blender_preflight",
        "status": "FAIL" if any(item["status"] == "FAIL" for item in checks) else "PASS_WITH_WARNINGS" if any(item["status"] == "WARN" for item in checks) else "PASS",
        "source": str(args.blend.resolve()),
        "blender_version": bpy.app.version_string,
        "bounds_m": {"min": list(minimum), "max": list(maximum), "dimensions": list(dimensions)},
        "meshes": mesh_data,
        "materials": materials,
        "source_physics": source_physics,
        "checks": checks,
        "outputs": {"glb": glb_path.name, "renders": render_paths},
    }
    (args.out / "blender_preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "meshes": len(meshes), "materials": len(materials)}))


if __name__ == "__main__":
    main()
