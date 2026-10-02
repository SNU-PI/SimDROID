#!/usr/bin/env python3
"""Inspect exported USD and author an explicit robotics physics layer."""

import argparse
import json
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

try:
    from pxr import PhysxSchema
except ImportError:
    PhysxSchema = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def deep_merge(base, override):
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def mean(values):
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else None


def resolve_physics(config, manifest, preflight):
    """Resolve reviewed values first, then Blender values, then defaults."""
    physics = deep_merge(config["physics"], manifest.get("physics", {}))
    if "physics" in manifest:
        return physics, "manifest_reauthored"

    source = preflight.get("source_physics", {})
    masses = list(source.get("masses_kg", {}).values())
    frictions = list(source.get("friction", {}).values())
    restitutions = list(source.get("restitution", {}).values())
    if not source.get("rigid_body_objects") or not masses:
        return physics, "default_generated"

    # A Blender asset may contain several rigid-body mesh pieces. The USD prop
    # has one asset-level body, so its mass is the sum and scalar surface
    # properties are represented by their mean.
    physics["mass_kg"] = sum(float(value) for value in masses)
    source_friction = mean(frictions)
    if source_friction is not None:
        physics["static_friction"] = source_friction
        physics["dynamic_friction"] = source_friction
    source_restitution = mean(restitutions)
    if source_restitution is not None:
        physics["restitution"] = source_restitution
    shapes = set(source.get("collision_shapes", {}).values())
    if shapes == {"BOX"}:
        physics["collider_strategy"] = "primitive_boxes"
    return physics, "blender_source_reauthored"


def validate_physics(physics):
    mass = float(physics["mass_kg"])
    static_friction = float(physics["static_friction"])
    dynamic_friction = float(physics["dynamic_friction"])
    restitution = float(physics["restitution"])
    if mass <= 0.0:
        raise ValueError("physics.mass_kg must be positive")
    if static_friction < 0.0 or dynamic_friction < 0.0:
        raise ValueError("physics friction values must be non-negative")
    if dynamic_friction > static_friction:
        raise ValueError("dynamic friction cannot exceed static friction")
    if not 0.0 <= restitution <= 1.0:
        raise ValueError("physics.restitution must be between 0 and 1")


def count_schemas(stage):
    result = {"rigid_bodies": 0, "colliders": 0, "mass_apis": 0, "physics_materials": 0, "grasp_identifiers": 0}
    for prim in stage.Traverse():
        result["rigid_bodies"] += int(prim.HasAPI(UsdPhysics.RigidBodyAPI))
        result["colliders"] += int(prim.HasAPI(UsdPhysics.CollisionAPI))
        result["mass_apis"] += int(prim.HasAPI(UsdPhysics.MassAPI))
        result["physics_materials"] += int(prim.HasAPI(UsdPhysics.MaterialAPI))
        result["grasp_identifiers"] += int(prim.GetName().startswith("grasp_identifier"))
    return result


def ensure_root(stage):
    root = stage.GetDefaultPrim()
    if root and root.IsValid() and root.IsA(UsdGeom.Xformable):
        return root
    roots = [prim for prim in stage.GetPseudoRoot().GetChildren() if prim.IsA(UsdGeom.Xformable)]
    if len(roots) == 1:
        stage.SetDefaultPrim(roots[0])
        return roots[0]
    if not roots:
        root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
        stage.SetDefaultPrim(root)
        return root
    raise RuntimeError("USD must have one xformable root prim")


def stage_bounds(stage, root):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    bounds = cache.ComputeWorldBound(root).ComputeAlignedRange()
    minimum = bounds.GetMin()
    maximum = bounds.GetMax()
    return minimum, maximum


def root_local_bounds(stage, root):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    bounds = cache.ComputeLocalBound(root).ComputeAlignedRange()
    return bounds.GetMin(), bounds.GetMax()


def bind_physics_material(prim, material):
    api = UsdShade.MaterialBindingAPI.Apply(prim)
    try:
        api.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
    except Exception:
        api.Bind(material)


def author_material(stage, root, physics):
    path = root.GetPath().AppendChild("PhysicsMaterial")
    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(float(physics["static_friction"]))
    api.CreateDynamicFrictionAttr(float(physics["dynamic_friction"]))
    api.CreateRestitutionAttr(float(physics["restitution"]))
    if PhysxSchema is not None:
        physx_api = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
        if hasattr(physx_api, "CreateFrictionCombineModeAttr"):
            physx_api.CreateFrictionCombineModeAttr("average")
        if hasattr(physx_api, "CreateRestitutionCombineModeAttr"):
            physx_api.CreateRestitutionCombineModeAttr("average")
    return material


def author_primitive(stage, path, specification):
    shape = specification.get("shape", "box").lower()
    center = Gf.Vec3d(*[float(value) for value in specification.get("center_m", [0.0, 0.0, 0.0])])
    scale = None
    if shape == "box":
        scale = [max(float(value), 1e-5) for value in specification["size_m"]]
        geometry = UsdGeom.Cube.Define(stage, path)
        geometry.CreateSizeAttr(1.0)
    elif shape == "sphere":
        geometry = UsdGeom.Sphere.Define(stage, path)
        geometry.CreateRadiusAttr(max(float(specification["radius_m"]), 1e-5))
    elif shape == "capsule":
        geometry = UsdGeom.Capsule.Define(stage, path)
        geometry.CreateRadiusAttr(max(float(specification["radius_m"]), 1e-5))
        geometry.CreateHeightAttr(max(float(specification["height_m"]), 1e-5))
        axis = specification.get("axis", "z").upper()
        if axis not in {"X", "Y", "Z"}:
            raise ValueError("capsule axis must be x, y, or z")
        geometry.CreateAxisAttr(axis)
    else:
        raise ValueError("primitive collider shape must be box, sphere, or capsule")
    geometry.AddTranslateOp().Set(center)
    if scale is not None:
        geometry.AddScaleOp().Set(Gf.Vec3d(*scale))
    geometry.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    return geometry.GetPrim()


def author_mesh_colliders(stage, root, material, physics):
    strategy = physics["collider_strategy"]
    meshes = [UsdGeom.Mesh(prim) for prim in Usd.PrimRange(root) if prim.IsA(UsdGeom.Mesh)]
    authored = []
    if strategy == "primitive_compound":
        specifications = physics.get("colliders", [])
        if not specifications:
            raise ValueError("primitive_compound requires physics.colliders")
        scope = UsdGeom.Scope.Define(stage, root.GetPath().AppendChild("Colliders"))
        for index, specification in enumerate(specifications):
            path = scope.GetPath().AppendChild("primitive_%03d" % index)
            prim = author_primitive(stage, path, specification)
            UsdPhysics.CollisionAPI.Apply(prim)
            bind_physics_material(prim, material)
            authored.append(str(path))
    elif strategy == "primitive_boxes":
        scope = UsdGeom.Scope.Define(stage, root.GetPath().AppendChild("Colliders"))
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        for index, mesh in enumerate(meshes):
            bounds = cache.ComputeRelativeBound(mesh.GetPrim(), root).ComputeAlignedRange()
            minimum, maximum = bounds.GetMin(), bounds.GetMax()
            extent = maximum - minimum
            center = (minimum + maximum) * 0.5
            cube = UsdGeom.Cube.Define(stage, scope.GetPath().AppendChild("box_%03d" % index))
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(center))
            cube.AddScaleOp().Set(Gf.Vec3d(max(float(extent[0]), 1e-5), max(float(extent[1]), 1e-5), max(float(extent[2]), 1e-5)))
            cube.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            bind_physics_material(cube.GetPrim(), material)
            authored.append(str(cube.GetPath()))
    elif strategy in {"mesh_sdf", "mesh_convex_decomposition"}:
        approximation = "sdf" if strategy == "mesh_sdf" else "convexDecomposition"
        for mesh in meshes:
            prim = mesh.GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)
            if PhysxSchema is not None:
                PhysxSchema.PhysxCollisionAPI.Apply(prim)
            bind_physics_material(prim, material)
            authored.append(str(prim.GetPath()))
    else:
        raise ValueError("unsupported collider_strategy: %s" % strategy)
    return authored


def author_grasp(stage, root, minimum, maximum, grasp, hard_max):
    if grasp.get("enabled", True) is False:
        return {"status": "MISSING_GRASP_METADATA", "authored": False}
    center = Gf.Vec3d((minimum + maximum) * 0.5)
    dimensions = maximum - minimum
    axis_name = grasp.get("axis", "auto").lower()
    if axis_name == "auto":
        axis_index = min(range(3), key=lambda index: float(dimensions[index]))
        axis_name = "xyz"[axis_index]
    elif axis_name not in "xyz":
        raise ValueError("grasp.axis must be auto, x, y, or z")
    else:
        axis_index = "xyz".index(axis_name)
    width = float(grasp.get("width_m", dimensions[axis_index]))
    if "center_m" in grasp:
        center = Gf.Vec3d(*[float(value) for value in grasp["center_m"]])
    if "points_m" in grasp:
        points = [Gf.Vec3f(*[float(value) for value in point]) for point in grasp["points_m"]]
        width = (Gf.Vec3d(points[1]) - Gf.Vec3d(points[0])).GetLength()
    else:
        delta = Gf.Vec3d(0.0)
        delta[axis_index] = width * 0.5
        points = [Gf.Vec3f(center - delta), Gf.Vec3f(center + delta)]

    identifier = UsdGeom.Xform.Define(stage, root.GetPath().AppendChild("grasp_identifier_01"))
    line = UsdGeom.BasisCurves.Define(stage, identifier.GetPath().AppendChild("grasp_identifier_line"))
    line.CreatePointsAttr(points)
    line.CreateCurveVertexCountsAttr([2])
    line.CreateTypeAttr(UsdGeom.Tokens.linear)
    line.CreateWidthsAttr([0.002])
    line.CreatePurposeAttr(UsdGeom.Tokens.guide)
    extent_min = Gf.Vec3f(*[min(float(point[index]) for point in points) for index in range(3)])
    extent_max = Gf.Vec3f(*[max(float(point[index]) for point in points) for index in range(3)])
    line.CreateExtentAttr([extent_min, extent_max])
    reviewed = bool(grasp.get("reviewed", False))
    valid_width = 0.0 < width <= hard_max
    return {
        "status": "PASS" if reviewed and valid_width else "REVIEW_REQUIRED" if valid_width else "FAIL",
        "authored": True,
        "source": "manifest" if reviewed else "generated",
        "reviewed": reviewed,
        "axis": axis_name,
        "width_m": width,
        "points_m": [[float(value) for value in point] for point in points],
    }


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else {}
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    settings = deep_merge(config, manifest)
    physics, authoring_mode = resolve_physics(config, manifest, preflight)
    validate_physics(physics)
    settings["physics"] = physics

    source_stage = Usd.Stage.Open(str(args.input))
    if not source_stage:
        raise RuntimeError("could not open exported USD")
    before = count_schemas(source_stage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_stage.GetRootLayer().Export(str(args.output))

    stage = Usd.Stage.Open(str(args.output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = ensure_root(stage)
    root.SetMetadata("kind", "component")
    minimum, maximum = stage_bounds(stage, root)
    local_minimum, local_maximum = root_local_bounds(stage, root)

    rigid_api = UsdPhysics.RigidBodyAPI.Apply(root)
    rigid_api.CreateRigidBodyEnabledAttr(True)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(float(physics["mass_kg"]))
    material = author_material(stage, root, physics)
    collider_paths = author_mesh_colliders(stage, root, material, physics)
    bind_physics_material(root, material)
    grasp = author_grasp(
        stage,
        root,
        local_minimum,
        local_maximum,
        settings.get("grasp", {}),
        float(settings["gripper"]["hard_max_width_m"]),
    )

    stage.GetRootLayer().customLayerData = {
        "SimReady_Metadata": {
            "validation": {
                "profile": settings["profile"]["id"],
                "profile_version": settings["profile"]["version"],
            }
        }
    }
    stage.GetRootLayer().Save()
    after = count_schemas(stage)
    source_declared_physics = bool(preflight["source_physics"]["rigid_body_objects"])
    source_physics = preflight["source_physics"]
    source_masses = list(source_physics.get("masses_kg", {}).values())
    source_frictions = list(source_physics.get("friction", {}).values())
    source_restitutions = list(source_physics.get("restitution", {}).values())
    final_verified = (
        after["rigid_bodies"] > 0
        and after["colliders"] > 0
        and after["mass_apis"] > 0
        and after["physics_materials"] > 0
    )
    transfer = {
        "source_declared_physics": source_declared_physics,
        "source_values": source_physics,
        "exported_before_authoring": before,
        "final_after_authoring": after,
        "native_export_preserved": (not source_declared_physics) or before["rigid_bodies"] > 0,
        "authoring_mode": authoring_mode,
        "final_values": {
            "mass_kg": float(physics["mass_kg"]),
            "static_friction": float(physics["static_friction"]),
            "dynamic_friction": float(physics["dynamic_friction"]),
            "restitution": float(physics["restitution"]),
        },
        "source_value_match": {
            "mass": bool(source_masses) and abs(sum(float(value) for value in source_masses) - float(physics["mass_kg"])) <= 1e-6,
            "friction": bool(source_frictions) and abs(mean(source_frictions) - float(physics["static_friction"])) <= 1e-6,
            "restitution": bool(source_restitutions) and abs(mean(source_restitutions) - float(physics["restitution"])) <= 1e-6,
        },
        "final_verified": final_verified,
        "generated_properties": authoring_mode == "default_generated",
    }
    report = {
        "stage": "usd_physics",
        "status": "FAIL" if not final_verified or grasp["status"] == "FAIL" else "PASS_WITH_REVIEW" if grasp["status"] == "REVIEW_REQUIRED" or transfer["authoring_mode"] == "default_generated" else "PASS",
        "input": str(args.input),
        "output": str(args.output),
        "root_prim": str(root.GetPath()),
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "up_axis": UsdGeom.GetStageUpAxis(stage),
        "bounds_m": {"min": list(minimum), "max": list(maximum), "dimensions": list(maximum - minimum)},
        "root_local_bounds_m": {
            "min": list(local_minimum),
            "max": list(local_maximum),
            "dimensions": list(local_maximum - local_minimum),
        },
        "physics": {
            "mass_kg": float(physics["mass_kg"]),
            "static_friction": float(physics["static_friction"]),
            "dynamic_friction": float(physics["dynamic_friction"]),
            "restitution": float(physics["restitution"]),
            "collider_strategy": physics["collider_strategy"],
            "colliders": collider_paths,
            "value_source": authoring_mode,
        },
        "grasp": grasp,
        "transfer": transfer,
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "colliders": len(collider_paths), "grasp": grasp["status"]}))


if __name__ == "__main__":
    main()
