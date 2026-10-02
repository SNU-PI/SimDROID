"""Inspect, normalize, and render a 3D asset with Blender."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import bpy
from mathutils import Vector


def script_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-size-m", type=float, default=0.002)
    parser.add_argument("--max-size-m", type=float, default=3.0)
    parser.add_argument("--max-triangles", type=int, default=2_000_000)
    return parser.parse_args(argv)


def add_check(checks, check_id, status, value, message):
    checks.append({"id": check_id, "status": status, "value": value, "message": message})


def load_asset(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".blend":
        bpy.ops.wm.open_mainfile(filepath=str(path))
        return

    bpy.ops.wm.read_factory_settings(use_empty=True)
    if suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    elif suffix in {".usd", ".usda", ".usdc"}:
        bpy.ops.wm.usd_import(filepath=str(path))
    else:
        raise ValueError(f"unsupported asset format: {suffix}")


class DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def inspect_meshes(mesh_objects):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    stats = {
        "mesh_objects": len(mesh_objects),
        "vertices": 0,
        "triangles": 0,
        "degenerate_triangles": 0,
        "boundary_edges": 0,
        "non_manifold_edges": 0,
        "components": 0,
        "objects_with_material": 0,
        "objects_with_uv": 0,
        "non_finite_vertices": 0,
        "volume_m3": 0.0,
    }
    bounds_min = Vector((math.inf, math.inf, math.inf))
    bounds_max = Vector((-math.inf, -math.inf, -math.inf))

    for source_obj in mesh_objects:
        evaluated = source_obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
        matrix = evaluated.matrix_world
        try:
            stats["vertices"] += len(mesh.vertices)
            world_vertices = [matrix @ vertex.co for vertex in mesh.vertices]
            for vertex in world_vertices:
                if not all(math.isfinite(component) for component in vertex):
                    stats["non_finite_vertices"] += 1
                    continue
                bounds_min.x = min(bounds_min.x, vertex.x)
                bounds_min.y = min(bounds_min.y, vertex.y)
                bounds_min.z = min(bounds_min.z, vertex.z)
                bounds_max.x = max(bounds_max.x, vertex.x)
                bounds_max.y = max(bounds_max.y, vertex.y)
                bounds_max.z = max(bounds_max.z, vertex.z)

            mesh.calc_loop_triangles()
            stats["triangles"] += len(mesh.loop_triangles)
            signed_volume = 0.0
            for triangle in mesh.loop_triangles:
                a, b, c = (world_vertices[index] for index in triangle.vertices)
                if (b - a).cross(c - a).length * 0.5 <= 1e-14:
                    stats["degenerate_triangles"] += 1
                signed_volume += a.dot(b.cross(c)) / 6.0
            stats["volume_m3"] += abs(signed_volume)

            # glTF commonly splits one geometric vertex at UV/normal seams. Weld
            # coincident positions before topology checks so seams are not holes.
            welded_ids = {}
            welded_index = []
            for vertex in world_vertices:
                key = tuple(round(component, 8) for component in vertex)
                if key not in welded_ids:
                    welded_ids[key] = len(welded_ids)
                welded_index.append(welded_ids[key])
            incidences = Counter()
            connected = DisjointSet(len(welded_ids))
            for polygon in mesh.polygons:
                vertices = polygon.vertices
                for index, left in enumerate(vertices):
                    right = vertices[(index + 1) % len(vertices)]
                    left, right = welded_index[left], welded_index[right]
                    if left == right:
                        continue
                    edge = (left, right) if left < right else (right, left)
                    incidences[edge] += 1
                    connected.union(left, right)
            stats["boundary_edges"] += sum(count == 1 for count in incidences.values())
            stats["non_manifold_edges"] += sum(count > 2 for count in incidences.values())
            used_vertices = {index for edge in incidences for index in edge}
            stats["components"] += len({connected.find(index) for index in used_vertices})

            if any(slot.material is not None for slot in source_obj.material_slots):
                stats["objects_with_material"] += 1
            if len(mesh.uv_layers) > 0:
                stats["objects_with_uv"] += 1
        finally:
            evaluated.to_mesh_clear()

    stats["volume_m3"] = round(stats["volume_m3"], 9)
    dimensions = bounds_max - bounds_min
    return stats, bounds_min, bounds_max, dimensions


def source_physics(mesh_objects):
    values = []
    for obj in mesh_objects:
        body = obj.rigid_body
        if body is None:
            continue
        values.append(
            {
                "object": obj.name,
                "mass_kg": body.mass,
                "friction": body.friction,
                "restitution": body.restitution,
                "collision_shape": body.collision_shape,
                "kinematic": body.kinematic,
            }
        )
    return values


def texture_stats():
    files = []
    missing = []
    packed = 0
    for image in bpy.data.images:
        if image.source != "FILE":
            continue
        if image.packed_file:
            packed += 1
            continue
        path = Path(bpy.path.abspath(image.filepath))
        files.append(str(path))
        if not path.is_file():
            missing.append(str(path))
    return {"external": files, "packed": packed, "missing": missing}


def export_glb(mesh_objects, output: Path) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]
    bpy.ops.export_scene.gltf(
        filepath=str(output),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
    )


def make_material(name, color):
    material = bpy.data.materials.new(name)
    material.diffuse_color = (*color, 1.0)
    material.use_nodes = True
    node = material.node_tree.nodes.get("Principled BSDF")
    if node:
        node.inputs["Base Color"].default_value = (*color, 1.0)
        node.inputs["Roughness"].default_value = 0.72
    return material


def point_camera(camera, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def render_previews(output: Path, bounds_min: Vector, bounds_max: Vector) -> list[str]:
    center = (bounds_min + bounds_max) * 0.5
    dimensions = bounds_max - bounds_min
    radius = max(dimensions.length * 0.9, 0.05)

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "ValidationCamera"
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = max(max(dimensions) * 1.45, 0.05)
    bpy.context.scene.camera = camera

    for name, energy, size, position in (
        ("Key", 900.0, 4.0, center + Vector((radius, -radius, radius * 1.5))),
        ("Fill", 500.0, 3.0, center + Vector((-radius, -radius * 0.4, radius))),
        ("Top", 700.0, 3.0, center + Vector((0.0, radius * 0.4, radius * 2.0))),
    ):
        bpy.ops.object.light_add(type="AREA", location=position)
        light = bpy.context.object
        light.name = f"Validation{name}"
        light.data.energy = energy
        light.data.shape = "DISK"
        light.data.size = size
        point_camera(light, center)

    bpy.ops.mesh.primitive_plane_add(size=max(max(dimensions) * 5.0, 0.5), location=(center.x, center.y, bounds_min.z - 0.001))
    floor = bpy.context.object
    floor.name = "ValidationFloor"
    floor.data.materials.append(make_material("ValidationFloorMaterial", (0.18, 0.19, 0.20)))

    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = 16
    if scene.world is None:
        scene.world = bpy.data.worlds.new("ValidationWorld")
    scene.world.color = (0.035, 0.04, 0.045)

    preview_dir = output / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    views = {
        "front": Vector((0.0, -1.0, 0.22)),
        "side": Vector((1.0, 0.0, 0.22)),
        "top": Vector((0.0, -0.05, 1.0)),
        "iso": Vector((1.0, -1.0, 0.75)).normalized(),
    }
    rendered = []
    for name, direction in views.items():
        camera.location = center + direction.normalized() * radius * 2.3
        point_camera(camera, center)
        destination = preview_dir / f"{name}.png"
        scene.render.filepath = str(destination)
        bpy.ops.render.render(write_still=True)
        rendered.append(str(destination.relative_to(output)))
    return rendered


def main() -> None:
    args = script_args()
    asset = args.asset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checks = []

    try:
        load_asset(asset)
        add_check(checks, "asset.load", "pass", True, "Asset opened successfully.")
    except Exception as error:
        add_check(checks, "asset.load", "fail", False, str(error))
        (output / "blender.json").write_text(json.dumps({"status": "fail", "checks": checks}, indent=2))
        raise

    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and not obj.hide_render]
    add_check(
        checks,
        "geometry.mesh_present",
        "pass" if mesh_objects else "fail",
        len(mesh_objects),
        f"Found {len(mesh_objects)} renderable mesh object(s).",
    )
    if not mesh_objects:
        report = {"status": "fail", "checks": checks, "asset": {"source": str(asset)}}
        (output / "blender.json").write_text(json.dumps(report, indent=2))
        return

    stats, bounds_min, bounds_max, dimensions = inspect_meshes(mesh_objects)
    textures = texture_stats()
    authored_physics = source_physics(mesh_objects)
    max_size = max(dimensions)
    min_size = min(dimensions)
    degenerate_ratio = stats["degenerate_triangles"] / max(stats["triangles"], 1)

    add_check(checks, "geometry.finite", "pass" if stats["non_finite_vertices"] == 0 else "fail", stats["non_finite_vertices"], "Non-finite vertex count must be zero.")
    add_check(checks, "geometry.nonzero_extent", "pass" if min_size > 1e-6 else "fail", [round(v, 6) for v in dimensions], "Asset must have non-zero extent on all axes.")
    scale_status = "pass" if args.min_size_m <= max_size <= args.max_size_m else "warn"
    add_check(checks, "geometry.scale", scale_status, round(max_size, 6), f"Largest dimension should be within {args.min_size_m:g}-{args.max_size_m:g} m unless overridden.")
    degenerate_status = "fail" if degenerate_ratio > 0.01 else "warn" if stats["degenerate_triangles"] else "pass"
    add_check(checks, "geometry.degenerate", degenerate_status, {"count": stats["degenerate_triangles"], "ratio": round(degenerate_ratio, 6)}, "Degenerate triangles damage rendering and collision generation.")
    add_check(checks, "geometry.non_manifold", "warn" if stats["non_manifold_edges"] else "pass", stats["non_manifold_edges"], "Non-manifold edges should be reviewed.")
    add_check(checks, "geometry.boundary", "warn" if stats["boundary_edges"] else "pass", stats["boundary_edges"], "Open boundaries may be intentional for thin or open objects.")
    add_check(checks, "geometry.polycount", "warn" if stats["triangles"] > args.max_triangles else "pass", stats["triangles"], f"Triangle budget is {args.max_triangles:,}.")
    add_check(checks, "material.binding", "pass" if stats["objects_with_material"] == stats["mesh_objects"] else "warn", f"{stats['objects_with_material']}/{stats['mesh_objects']}", "Every renderable mesh should have a material.")
    add_check(checks, "material.uv", "pass" if stats["objects_with_uv"] == stats["mesh_objects"] else "warn", f"{stats['objects_with_uv']}/{stats['mesh_objects']}", "Every textured mesh should have a UV layer.")
    add_check(checks, "material.textures_resolved", "fail" if textures["missing"] else "pass", textures["missing"], "External texture paths must resolve.")

    normalized = output / "normalized.glb"
    export_glb(mesh_objects, normalized)
    previews = render_previews(output, bounds_min, bounds_max)
    statuses = {check["status"] for check in checks}
    status = "fail" if "fail" in statuses else "review" if "warn" in statuses else "pass"
    report = {
        "schema_version": "simdroid.asset-validation/v1",
        "status": status,
        "asset": {
            "source": str(asset),
            "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
            "bounds_m": {
                "min": [round(v, 6) for v in bounds_min],
                "max": [round(v, 6) for v in bounds_max],
                "dimensions": [round(v, 6) for v in dimensions],
            },
        },
        "stats": stats,
        "textures": textures,
        "source_physics": authored_physics,
        "checks": checks,
        "artifacts": {"normalized_glb": normalized.name, "previews": previews},
    }
    (output / "blender.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"BLENDER_STATUS={status}")


if __name__ == "__main__":
    main()
