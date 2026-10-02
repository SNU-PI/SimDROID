"""Run a small deterministic drop-and-settle test in Isaac Sim."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--asset", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--seconds", type=float, default=3.0)
parser.add_argument("--mass-kg", type=float)
parser.add_argument("--density-kg-m3", type=float, default=700.0)
parser.add_argument("--default-mass-kg", type=float, default=1.0)
parser.add_argument("--static-friction", type=float, default=0.5)
parser.add_argument("--dynamic-friction", type=float, default=0.4)
parser.add_argument("--restitution", type=float, default=0.05)
parser.add_argument("--max-colliders", type=int, default=8)
parser.add_argument("--graspable", action="store_true")
parser.add_argument("--max-grasp-width-m", type=float, default=0.06)
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "disable_viewport_updates": True})

import omni.timeline
import omni.usd
import omni.kit.app
import omni.kit.asset_converter as asset_converter
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade


def add_check(checks, check_id, status, value, message):
    checks.append({"id": check_id, "status": status, "value": value, "message": message})


def aligned_bounds(stage, prim):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    value = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return value.GetMin(), value.GetMax()


def find_root(stage):
    root = stage.GetDefaultPrim()
    if root and root.IsValid() and root.IsA(UsdGeom.Xformable):
        return root
    for prim in stage.GetPseudoRoot().GetChildren():
        if prim.IsA(UsdGeom.Xformable):
            return prim
    return None


def rotate_normalized_gltf_to_z_up(root):
    """Blender exports glTF as Y-up; rotate its root into Isaac's Z-up world."""
    xform = UsdGeom.Xformable(root)
    operation = next(
        (op for op in xform.GetOrderedXformOps() if op.GetName() == "xformOp:rotateX:validationZUp"),
        None,
    )
    if operation is None:
        operation = xform.AddRotateXOp(UsdGeom.XformOp.PrecisionFloat, "validationZUp")
    operation.Set(90.0)


def read_blender_report():
    path = args.output.parent / "blender.json"
    return json.loads(path.read_text()) if path.exists() else {}


def collect_points(stage, root, meshes):
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    world_to_root = cache.GetLocalToWorldTransform(root).GetInverse()
    values = []
    for prim in meshes:
        mesh = UsdGeom.Mesh(prim)
        local_to_world = cache.GetLocalToWorldTransform(prim)
        for point in mesh.GetPointsAttr().Get() or []:
            world = local_to_world.Transform(Gf.Vec3d(*point))
            values.append(tuple(world_to_root.Transform(world)))
    points = np.asarray(values, dtype=np.float64)
    if len(points) > 50_000:
        sampled = np.linspace(0, len(points) - 1, 50_000, dtype=np.int64)
        extrema = np.concatenate((points.argmin(axis=0), points.argmax(axis=0)))
        points = points[np.unique(np.concatenate((sampled, extrema)))]
    return points


def box_volume(points):
    extent = np.maximum(points.max(axis=0) - points.min(axis=0), 1e-6)
    return float(np.prod(extent))


def compound_boxes(points, fill_ratio, max_boxes):
    """Approximate sparse/concave geometry with a small set of AABBs."""
    clusters = [np.arange(len(points))]
    if fill_ratio >= 0.55 or len(points) < 64:
        return [(points.min(axis=0), points.max(axis=0), clusters[0])]

    while len(clusters) < max_boxes:
        best = None
        for cluster_position, indices in enumerate(clusters):
            current = points[indices]
            parent_volume = box_volume(current)
            if len(indices) < 32 or parent_volume <= 1e-12:
                continue
            for axis in range(3):
                coordinates = current[:, axis]
                if np.ptp(coordinates) <= 1e-6:
                    continue
                for quantile in (0.25, 0.5, 0.75):
                    split = float(np.quantile(coordinates, quantile))
                    left_mask = coordinates <= split
                    if left_mask.sum() < 12 or (~left_mask).sum() < 12:
                        continue
                    left = indices[left_mask]
                    right = indices[~left_mask]
                    child_volume = box_volume(points[left]) + box_volume(points[right])
                    improvement = (parent_volume - child_volume) / parent_volume
                    if best is None or improvement > best[0]:
                        best = (improvement, cluster_position, left, right)
        if best is None or best[0] < 0.08:
            break
        _, position, left, right = best
        clusters[position : position + 1] = [left, right]

    minimum_padding = max(float(np.linalg.norm(np.ptp(points, axis=0))) * 0.002, 0.0005)
    boxes = []
    for indices in clusters:
        minimum = points[indices].min(axis=0)
        maximum = points[indices].max(axis=0)
        extent = maximum - minimum
        padding = np.maximum((minimum_padding - extent) * 0.5, 0.0)
        boxes.append((minimum - padding, maximum + padding, indices))
    return boxes


def collider_fidelity(points, boxes):
    depths = []
    for minimum, maximum, indices in boxes:
        selected = points[indices]
        distances = np.minimum(selected - minimum, maximum - selected)
        depths.extend(np.maximum(distances.min(axis=1), 0.0).tolist())
    total_bounds_volume = box_volume(points)
    collider_volume = sum(float(np.prod(maximum - minimum)) for minimum, maximum, _ in boxes)
    return {
        "surface_error_p95_m": float(np.quantile(depths, 0.95)) if depths else math.inf,
        "collider_to_bounds_volume_ratio": collider_volume / max(total_bounds_volume, 1e-12),
        "collider_volume_m3": collider_volume,
    }


def define_physics_material(stage, root, values):
    path = root.GetPath().AppendChild("PhysicsMaterial")
    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(values["static_friction"])
    api.CreateDynamicFrictionAttr(values["dynamic_friction"])
    api.CreateRestitutionAttr(values["restitution"])
    api.CreateDensityAttr(values["density_kg_m3"])
    return material


def author_box_colliders(stage, root, boxes, material):
    scope = UsdGeom.Xform.Define(stage, root.GetPath().AppendChild("CollisionShapes"))
    paths = []
    for index, (minimum, maximum, _) in enumerate(boxes):
        path = scope.GetPath().AppendChild(f"box_{index:02d}")
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        center = (minimum + maximum) * 0.5
        extent = maximum - minimum
        xform = UsdGeom.Xformable(cube)
        xform.AddTranslateOp().Set(Gf.Vec3d(*center))
        xform.AddScaleOp().Set(Gf.Vec3f(*extent))
        cube.CreatePurposeAttr(UsdGeom.Tokens.guide)
        cube.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.9, 0.35)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        binding = UsdShade.MaterialBindingAPI.Apply(cube.GetPrim())
        binding.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
        paths.append(str(path))
    return paths


async def convert_asset(source: Path, destination: Path):
    manager = omni.kit.app.get_app().get_extension_manager()
    manager.set_extension_enabled_immediate("omni.kit.asset_converter", True)
    context = asset_converter.AssetConverterContext()
    context.ignore_materials = False
    context.use_meter_as_world_unit = True
    context.convert_stage_up_z = True
    task = asset_converter.get_instance().create_converter_task(
        str(source.resolve()), str(destination.resolve()), None, context
    )
    ok = await task.wait_until_finished()
    if not ok:
        raise RuntimeError(task.get_error_message())


def main():
    checks = []
    converted = args.output.parent / "normalized.usdc"
    try:
        asyncio.get_event_loop().run_until_complete(convert_asset(args.asset, converted))
        add_check(checks, "physics.conversion", "pass", True, "Normalized GLB converted to USD.")
    except Exception as error:
        add_check(checks, "physics.conversion", "fail", False, str(error))
        args.output.write_text(json.dumps({"status": "fail", "checks": checks}, indent=2) + "\n")
        return
    context = omni.usd.get_context()
    if not context.open_stage(str(converted.resolve())):
        add_check(checks, "physics.stage_open", "fail", False, "Isaac Sim could not open the normalized USD.")
        args.output.write_text(json.dumps({"status": "fail", "checks": checks}, indent=2) + "\n")
        return
    app.update()
    stage = context.get_stage()
    root = find_root(stage)
    meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    add_check(checks, "physics.stage_open", "pass", True, "Normalized USD opened in Isaac Sim.")
    if root is None or not meshes:
        add_check(checks, "physics.testable", "fail", False, "No testable root and mesh combination was found.")
        args.output.write_text(json.dumps({"status": "fail", "checks": checks}, indent=2) + "\n")
        return

    authored_colliders = sum(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in meshes)
    authored_rigid_body = root.HasAPI(UsdPhysics.RigidBodyAPI)
    blender = read_blender_report()
    source_bodies = blender.get("source_physics", [])
    stats = blender.get("stats", {})
    rotate_normalized_gltf_to_z_up(root)
    app.update()
    points = collect_points(stage, root, meshes)
    if len(points) == 0:
        add_check(checks, "physics.testable", "fail", False, "No mesh points were available for collider generation.")
        args.output.write_text(json.dumps({"status": "fail", "checks": checks}, indent=2) + "\n")
        return

    local_dimensions = np.ptp(points, axis=0)
    world_minimum, world_maximum = aligned_bounds(stage, root)
    dimensions = np.asarray(world_maximum - world_minimum, dtype=np.float64)
    bounds_volume = float(np.prod(np.maximum(local_dimensions, 1e-9)))
    mesh_volume = float(stats.get("volume_m3", 0.0))
    topology_closed = stats.get("boundary_edges", 1) == 0 and stats.get("non_manifold_edges", 1) == 0
    fill_ratio = min(mesh_volume / max(bounds_volume, 1e-12), 1.0) if topology_closed else 0.0
    boxes = compound_boxes(points, fill_ratio, args.max_colliders)
    fidelity = collider_fidelity(points, boxes)

    source_material = source_bodies[0] if source_bodies else {}
    material_values = {
        "density_kg_m3": args.density_kg_m3,
        "static_friction": float(source_material.get("friction", args.static_friction)),
        "dynamic_friction": float(source_material.get("friction", args.dynamic_friction)),
        "restitution": float(source_material.get("restitution", args.restitution)),
    }
    if args.mass_kg is not None:
        mass_kg, mass_source = args.mass_kg, "cli"
    elif source_bodies:
        mass_kg, mass_source = sum(float(body["mass_kg"]) for body in source_bodies), "blender"
    elif topology_closed and mesh_volume > 0:
        mass_kg, mass_source = mesh_volume * args.density_kg_m3, "estimated_from_mesh_volume"
    else:
        mass_kg = args.default_mass_kg
        mass_source = "placeholder"
    mass_kg = max(float(mass_kg), 0.001)

    if not authored_rigid_body:
        UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(mass_kg)
    material = define_physics_material(stage, root, material_values)
    collider_paths = author_box_colliders(stage, root, boxes, material)

    fidelity_tolerance = max(0.003, float(np.linalg.norm(dimensions)) * 0.03)
    fidelity_status = "pass" if fidelity["surface_error_p95_m"] <= fidelity_tolerance else "warn"
    add_check(checks, "physics.source_collider", "pass" if authored_colliders else "info", authored_colliders, "Collider count found before API authoring.")
    add_check(checks, "physics.source_rigid_body", "pass" if authored_rigid_body else "info", authored_rigid_body, "Whether a rigid body survived source conversion.")
    add_check(checks, "physics.collider_compound", fidelity_status, len(collider_paths), f"Generated compound box colliders; surface-error tolerance is {fidelity_tolerance:.6f} m.")
    add_check(checks, "physics.mass", "pass" if mass_source in {"cli", "blender"} else "warn", {"kg": round(mass_kg, 6), "source": mass_source}, "Measured or source-authored mass is preferred over an estimate.")
    add_check(checks, "physics.material", "pass" if source_bodies else "warn", material_values, "Source material values are preferred; defaults require review.")
    expected_dimensions = np.asarray(blender.get("asset", {}).get("bounds_m", {}).get("dimensions", []), dtype=np.float64)
    axis_ok = len(expected_dimensions) == 3 and np.allclose(dimensions, expected_dimensions, atol=1e-4, rtol=1e-4)
    add_check(checks, "physics.axis_conversion", "pass" if axis_ok else "fail", [round(float(value), 6) for value in dimensions], "Converted USD dimensions must match the Blender Z-up dimensions axis by axis.")
    if args.graspable:
        grasp_width = float(np.min(dimensions))
        add_check(checks, "grasp.width", "pass" if grasp_width <= args.max_grasp_width_m else "fail", round(grasp_width, 6), f"Smallest candidate grasp width must be <= {args.max_grasp_width_m:g} m.")
    else:
        add_check(checks, "grasp.width", "info", "not_requested", "Run with --graspable to enforce the gripper-width constraint.")

    asset_usd = args.output.parent / "asset.usdc"
    stage.Export(str(asset_usd))
    verified_stage = Usd.Stage.Open(str(asset_usd))
    verified_root = find_root(verified_stage)
    verified_mass = UsdPhysics.MassAPI(verified_root).GetMassAttr().Get() if verified_root else None
    schemas_ok = bool(
        verified_root
        and verified_root.HasAPI(UsdPhysics.RigidBodyAPI)
        and verified_root.HasAPI(UsdPhysics.MassAPI)
        and abs(float(verified_mass) - mass_kg) <= 1e-6
        and all(verified_stage.GetPrimAtPath(path).HasAPI(UsdPhysics.CollisionAPI) for path in collider_paths)
        and verified_stage.GetPrimAtPath(material.GetPath()).HasAPI(UsdPhysics.MaterialAPI)
    )
    add_check(
        checks,
        "physics.usd_schema",
        "pass" if schemas_ok else "fail",
        schemas_ok,
        "Final USD must contain rigid-body, mass, material, and collider schemas after reopening.",
    )

    scene = UsdPhysics.Scene.Define(stage, "/ValidationPhysicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    # Define the test ground outside the asset root. The Isaac helper places
    # its ground below the default prim, which would make it part of a dynamic
    # asset when the default prim is also the rigid-body root.
    ground = UsdGeom.Plane.Define(stage, "/ValidationGround")
    ground.CreateAxisAttr(UsdGeom.Tokens.z)
    ground.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    minimum, maximum = aligned_bounds(stage, root)
    diagonal = max((maximum - minimum).GetLength(), 1e-4)
    xform = UsdGeom.Xformable(root)
    translate_op = next((op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
    if translate_op is None:
        translate_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "validationDrop")
        original_translation = Gf.Vec3d(0.0)
    else:
        original_translation = translate_op.Get() or Gf.Vec3d(0.0)
    translate_op.Set(original_translation + Gf.Vec3d(0.0, 0.0, 0.20 - minimum[2]))

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    app.update()

    samples = []
    contact = False
    penetration = False
    finite = True
    frame_count = int(args.seconds * 60)
    for _ in range(frame_count):
        app.update()
        minimum, maximum = aligned_bounds(stage, root)
        values = [*minimum, *maximum]
        finite = finite and all(math.isfinite(value) for value in values)
        contact = contact or minimum[2] <= max(0.01, diagonal * 0.03)
        penetration = penetration or minimum[2] < -max(0.02, diagonal * 0.10)
        samples.append(tuple(values))
        if not finite:
            break
    timeline.stop()

    tail = samples[-60:] if len(samples) >= 60 else samples
    movement = 0.0
    if len(tail) >= 2:
        first, last = tail[0], tail[-1]
        movement = max(abs(a - b) for a, b in zip(first, last))
    settle_tolerance = max(0.002, diagonal * 0.02)
    settled = bool(tail) and movement <= settle_tolerance

    add_check(checks, "physics.finite", "pass" if finite else "fail", finite, "Physics transforms must remain finite.")
    add_check(checks, "physics.ground_contact", "pass" if contact else "fail", contact, "Asset must fall and contact the ground.")
    add_check(checks, "physics.no_penetration", "fail" if penetration else "pass", not penetration, "Asset must not tunnel through the ground.")
    add_check(checks, "physics.settled", "pass" if settled else "warn", round(movement, 6), f"Last-second bounds movement should be <= {settle_tolerance:.6f} m.")
    statuses = {check["status"] for check in checks}
    status = "fail" if "fail" in statuses else "review" if "warn" in statuses else "pass"
    report = {
        "status": status,
        "engine": "Isaac Sim / PhysX",
        "simulation_seconds": args.seconds,
        "checks": checks,
        "metrics": {
            "last_second_bounds_movement_m": round(movement, 6),
            "settle_tolerance_m": round(settle_tolerance, 6),
            "visual_dimensions_m": [round(float(value), 6) for value in dimensions],
            "visual_mesh_volume_m3": round(mesh_volume, 9),
            "visual_fill_ratio": round(fill_ratio, 6),
            "collider_surface_error_p95_m": round(fidelity["surface_error_p95_m"], 6),
            "collider_to_bounds_volume_ratio": round(fidelity["collider_to_bounds_volume_ratio"], 6),
        },
        "authoring": {
            "asset_usd": asset_usd.name,
            "collider_strategy": "compound_boxes",
            "collider_frame": "normalized_gltf_y_up_local",
            "collider_paths": collider_paths,
            "collider_boxes": [
                {
                    "center_m": [round(float(value), 6) for value in (minimum + maximum) * 0.5],
                    "size_m": [round(float(value), 6) for value in maximum - minimum],
                }
                for minimum, maximum, _ in boxes
            ],
            "mass_kg": round(mass_kg, 6),
            "mass_source": mass_source,
            "material": material_values,
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"ISAAC_STATUS={status}")


try:
    main()
finally:
    app.close()
