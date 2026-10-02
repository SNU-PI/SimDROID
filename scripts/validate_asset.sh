#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 ASSET OUTPUT_DIR [--gpu N] [--graspable] [--mass-kg N] [--skip-physics]"
}

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

ASSET=$(realpath "$1")
OUTPUT=$(realpath -m "$2")
shift 2
GPU=0
SKIP_PHYSICS=0
MIN_SIZE=0.002
MAX_SIZE=3.0
GRASPABLE=0
MASS_KG=""
MAX_COLLIDERS=8
DENSITY=700.0
STATIC_FRICTION=0.5
DYNAMIC_FRICTION=0.4
RESTITUTION=0.05

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --skip-physics) SKIP_PHYSICS=1; shift ;;
    --min-size-m) MIN_SIZE="$2"; shift 2 ;;
    --max-size-m) MAX_SIZE="$2"; shift 2 ;;
    --graspable) GRASPABLE=1; shift ;;
    --mass-kg) MASS_KG="$2"; shift 2 ;;
    --max-colliders) MAX_COLLIDERS="$2"; shift 2 ;;
    --density-kg-m3) DENSITY="$2"; shift 2 ;;
    --static-friction) STATIC_FRICTION="$2"; shift 2 ;;
    --dynamic-friction) DYNAMIC_FRICTION="$2"; shift 2 ;;
    --restitution) RESTITUTION="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$OUTPUT"
rm -f "$OUTPUT/physics.json" "$OUTPUT/asset.usdc" "$OUTPUT/collider_preview.png"

env -u PYTHONHOME -u PYTHONPATH \
  PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 \
  /usr/bin/blender --background --python "$ROOT/scripts/asset_validation/blender_check.py" -- \
  --asset "$ASSET" \
  --output "$OUTPUT" \
  --min-size-m "$MIN_SIZE" \
  --max-size-m "$MAX_SIZE"

if [[ "$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$OUTPUT/blender.json")" == "fail" ]]; then
  SKIP_PHYSICS=1
  echo "Static validation failed; Isaac Sim authoring was skipped."
fi

if [[ "$SKIP_PHYSICS" -eq 0 ]]; then
  ISAAC_SIM_IMAGE=${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.1}
  PHYSICS_ARGS="--max-colliders $MAX_COLLIDERS --density-kg-m3 $DENSITY --static-friction $STATIC_FRICTION --dynamic-friction $DYNAMIC_FRICTION --restitution $RESTITUTION"
  [[ "$GRASPABLE" -eq 1 ]] && PHYSICS_ARGS="$PHYSICS_ARGS --graspable"
  [[ -n "$MASS_KG" ]] && PHYSICS_ARGS="$PHYSICS_ARGS --mass-kg $MASS_KG"
  if ! docker run --rm \
    --gpus "device=$GPU" \
    --user 0:0 \
    --entrypoint /bin/bash \
    -e ACCEPT_EULA=Y \
    -e PRIVACY_CONSENT=Y \
    -e HOST_UID="$(id -u)" \
    -e HOST_GID="$(id -g)" \
    -v "$ROOT:/work:ro" \
    -v "$OUTPUT:/validation-output:rw" \
    "$ISAAC_SIM_IMAGE" \
    -lc "/isaac-sim/python.sh /work/scripts/asset_validation/isaac_check.py --asset /validation-output/normalized.glb --output /validation-output/physics.json $PHYSICS_ARGS; status=\$?; chown -R \$HOST_UID:\$HOST_GID /validation-output || true; exit \$status" \
    >"$OUTPUT/isaac.log" 2>&1; then
    tail -40 "$OUTPUT/isaac.log" >&2
    exit 1
  fi
  grep 'ISAAC_STATUS=' "$OUTPUT/isaac.log" | tail -1 || true
  env -u PYTHONHOME -u PYTHONPATH \
    PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 \
    /usr/bin/blender --background --python "$ROOT/scripts/asset_validation/render_collider_preview.py" -- \
    --asset "$OUTPUT/normalized.glb" \
    --physics "$OUTPUT/physics.json" \
    --output "$OUTPUT/collider_preview.png" \
    >"$OUTPUT/collider-render.log" 2>&1
fi

python "$ROOT/scripts/asset_validation/merge_report.py" --output "$OUTPUT"
