#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${1:-$(mktemp -d)}"
mkdir -p "$WORK/fixtures"
env -u PYTHONHOME -u PYTHONPATH PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 TMPDIR="$WORK" \
  /usr/bin/blender --background --factory-startup \
  --python "$ROOT/tests/asset_quality/create_fixtures.py" -- --out "$WORK/fixtures"
mkdir -p "$WORK/good_preflight"
env -u PYTHONHOME -u PYTHONPATH PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 TMPDIR="$WORK" \
  /usr/bin/blender --background --factory-startup \
  --python "$ROOT/scripts/asset_quality/blender_stage.py" -- \
  --blend "$WORK/fixtures/good.blend" --out "$WORK/good_preflight"
python3 - "$WORK/good_preflight/blender_preflight.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
assert [material["name"] for material in report["materials"]] == ["painted_plastic"]
PY
python3 -m unittest "$ROOT/tests/asset_quality/test_report.py"
echo "$WORK"
