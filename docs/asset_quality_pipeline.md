# DROID Asset Quality Pipeline

This pipeline keeps every source, simulation, visual-quality, and DROID-specific
check in one reproducible run:

```text
.blend
  -> Blender source preflight and canonical renders
  -> GLB export, Isaac Sim USD conversion, and explicit physics/grasp authoring
  -> NVIDIA SimReady prop package transform and validation (Prop-Robotics-Isaac 1.0.0)
  -> NVIDIA SimReady Benchmark (FET001, FET003, FET005)
  -> PBR/UV checks and released 3D-PAQA model
  -> category scale and Robotiq 2F-85 grasp-width checks
  -> report.json, report.html, images, and raw logs
```

## Setup

Build the SimReady image automatically on the first run. Prepare the 3D-PAQA
runtime once, using a CUDA PyTorch Python environment:

```bash
PAQA_BASE_PYTHON=/path/to/cuda-python \
  ./scripts/asset_quality/setup_paqa.sh
```

Export the three values printed by the setup command. Isaac Sim's home and
shader cache are retained in the Docker volume
`simdroid-asset-quality-home`, so warm-up is shared across asset runs.

## Run

```bash
./scripts/validate_asset_quality.sh asset.blend \
  --out runs/asset_quality/cup \
  --category cup \
  --manifest cup.manifest.json \
  --source-image droid_frame.png \
  --gpu 0
```

Use a new or empty `--out` directory for every run. This prevents stale model
or benchmark results from being mistaken for evidence from the current asset.

The manifest makes generated physical and grasp values reviewable and explicit:

```json
{
  "physics": {
    "mass_kg": 0.28,
    "static_friction": 0.7,
    "dynamic_friction": 0.55,
    "restitution": 0.03,
    "collider_strategy": "primitive_boxes"
  },
  "grasp": {
    "axis": "x",
    "width_m": 0.055,
    "reviewed": true
  }
}
```

`collider_strategy` is `primitive_boxes`, `mesh_sdf`, or
`mesh_convex_decomposition`. Use `primitive_compound` with a `colliders` list
when one bounding box is not accurate enough. Each collider can be a `box`,
`sphere`, or `capsule` in asset-local meters:

```json
{
  "physics": {
    "collider_strategy": "primitive_compound",
    "colliders": [
      {"shape": "box", "center_m": [0, 0, 0.02], "size_m": [0.08, 0.05, 0.04]},
      {"shape": "capsule", "center_m": [0.04, 0, 0.05], "radius_m": 0.01, "height_m": 0.05, "axis": "z"}
    ]
  }
}
```

Physics precedence is reviewed manifest, Blender rigid-body values, then the
versioned default configuration. Blender physics is not native glTF data, so
the report explicitly shows that it disappeared during export and was
re-authored into USD by the pipeline. An automatically proposed grasp is always
`REVIEW_REQUIRED`; missing model execution or source imagery makes the overall
result `INCOMPLETE`, never a silent pass. For the Robotiq 2F-85, a reviewed
grasp width up to 6 cm passes, 6-8.5 cm is retained with a warning, and wider
or zero-width grasps fail.

## Outputs

- `report.json`: one aggregate machine-readable verdict.
- `report.html`: source/render evidence and every stage result.
- `blender_preflight.json`: source scale, modifiers, topology, UV/PBR, textures.
- `usd_physics.json`: transfer evidence and final rigid body/material/collider/grasp data.
- `simready_validation.json` and `benchmark/`: official NVIDIA results.
- `photorealism_paqa.json`: released 3D-PAQA six-axis scores.
- `renders/`: front, side, isometric, and top canonical views.
- `logs/`: complete tool output; no failed stage is hidden.

The SimReady section reports the requested FET001/FET003/FET005 verdict
separately from the full `Prop-Robotics-Isaac` profile exit code. A failure in
an unrequested feature such as FET004 therefore remains visible as a warning
instead of being mislabeled as either a target failure or a clean pass.
3D-PAQA passes only when all six released quality scores meet the configured
threshold; its mean and failed criteria are also stored for diagnosis.
