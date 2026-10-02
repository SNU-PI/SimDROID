# Asset Validation

This pipeline turns a Blender, GLB, OBJ, or USD asset into a physics-authored
Isaac Sim USD and a small validation report.

## Run

```bash
./scripts/validate_asset.sh path/to/asset.blend validation-results/asset --gpu 0
```

For an object intended for a Franka Panda parallel-jaw grasp, enforce the
6 cm candidate grasp-width limit:

```bash
./scripts/validate_asset.sh path/to/cup.blend validation-results/cup \
  --gpu 0 --graspable
```

When measured mass is known, provide it instead of accepting an estimate:

```bash
./scripts/validate_asset.sh path/to/cup.glb validation-results/cup \
  --gpu 0 --graspable --mass-kg 0.18
```

Optional `--density-kg-m3`, `--static-friction`, `--dynamic-friction`, and
`--restitution` values can override the conservative defaults. Values already
authored in a `.blend` file take precedence for friction and restitution.

## Outputs

- `report.json`: final `PASS`, `REVIEW`, or `FAIL` and every check.
- `asset.usdc`: visual mesh plus USD rigid body, mass, physics material, and
  compound primitive colliders.
- `collider_preview.png`: visual geometry with generated colliders in green.
- `previews/{front,side,top,iso}.png`: fixed visual inspection views.
- `normalized.glb`: normalized interchange copy.
- `blender.json`, `physics.json`: detailed stage reports.

`PASS` means all requested checks passed. `REVIEW` means the asset runs but a
human must confirm an estimate, missing material, or loose collider fit.
`FAIL` means the asset should not enter simulation as-is.

## What Is Checked

The Blender stage checks loadability, finite geometry, dimensions, degenerate
triangles, open/non-manifold edges, triangle count, materials, UVs, textures,
and visual volume. If a `.blend` file already contains rigid-body settings,
its mass, friction, and restitution are preserved as authoring inputs.

The Isaac Sim stage:

1. Converts the normalized visual asset to USD at meter scale and Z-up.
2. Creates a small compound of box colliders instead of one convex hull.
3. Authors `RigidBodyAPI`, `MassAPI`, and `MaterialAPI` values.
4. Reopens the final USD and verifies that every physics schema persisted.
5. Runs a deterministic drop, contact, penetration, and settle test.
6. Optionally checks whether a candidate grasp cross-section is at most 6 cm.

Missing mass or surface properties are deliberately marked `REVIEW`. For a
closed mesh the API estimates mass from volume and density; otherwise it uses
a 1 kg test placeholder. Neither is presented as measured ground truth.

## Limits

Compound boxes are intentionally simple and fast. A high collider surface
error is marked `REVIEW`; those assets need a manually supplied primitive
layout, convex decomposition, or SDF collider. The grasp-width check is only a
geometric sanity check, not a substitute for a robot approach, close, lift,
and hold evaluation.

## Tests

```bash
./tests/asset_validation/test_pipeline.sh
GPU=0 ./tests/asset_validation/test_physics_pipeline.sh
```

The first test covers a valid and a degenerate mesh without Isaac Sim. The
second verifies Blender physics-value preservation, final USD authoring,
collider rendering, the 6 cm grasp rule, and the Isaac Sim drop test.
