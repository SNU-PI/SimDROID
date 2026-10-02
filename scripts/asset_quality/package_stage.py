#!/usr/bin/env python3
"""Make an Isaac-transformed SimReady package self-contained."""

import argparse
import json
import shutil
from pathlib import Path

from pxr import Sdf, Usd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--gltf-mdl",
        type=Path,
        default=Path("/isaac-sim/kit/mdl/core/mdl/gltf/pbr.mdl"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    materials_path = args.package / "payloads" / "materials.usda"
    if not materials_path.exists():
        raise SystemExit("materials layer not found: %s" % materials_path)
    if not args.gltf_mdl.exists():
        raise SystemExit("Isaac glTF MDL not found: %s" % args.gltf_mdl)

    mdl_target = materials_path.parent / "gltf" / "pbr.mdl"
    mdl_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.gltf_mdl, mdl_target)

    stage = Usd.Stage.Open(str(materials_path), load=Usd.Stage.LoadNone)
    if stage is None:
        raise SystemExit("could not open materials layer: %s" % materials_path)

    rewritten = []
    for prim in stage.Traverse():
        for attribute in prim.GetAttributes():
            value = attribute.Get()
            if isinstance(value, Sdf.AssetPath) and value.path == "gltf/pbr.mdl":
                attribute.Set(Sdf.AssetPath("./gltf/pbr.mdl"))
                rewritten.append(str(attribute.GetPath()))
    stage.GetRootLayer().Save()

    if not rewritten:
        raise SystemExit("no gltf/pbr.mdl references were found in %s" % materials_path)

    report = {
        "stage": "simready_package",
        "status": "PASS",
        "materials_layer": str(materials_path.relative_to(args.package)),
        "bundled_mdl": str(mdl_target.relative_to(args.package)),
        "rewritten_attributes": rewritten,
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
