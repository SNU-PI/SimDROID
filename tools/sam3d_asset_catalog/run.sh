#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="${SAM3D_PYTHON:-python3}"
HOST="${SAM3D_CATALOG_HOST:-0.0.0.0}"
PORT="${SAM3D_CATALOG_PORT:-8095}"
export SAM3D_CATALOG_ROOT="${SAM3D_CATALOG_ROOT:-$REPOSITORY_ROOT/data/sam3d_assets}"

cd "$HERE"
exec "$PYTHON" -m uvicorn app:app --host "$HOST" --port "$PORT"
