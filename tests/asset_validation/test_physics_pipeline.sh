#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
GPU=${GPU:-0}

env -u PYTHONHOME -u PYTHONPATH \
  PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 \
  /usr/bin/blender --background --python "$ROOT/tests/asset_validation/create_fixtures.py" -- "$TMP/fixtures" >/dev/null
"$ROOT/scripts/validate_asset.sh" "$TMP/fixtures/good.blend" "$TMP/good" --gpu "$GPU" --graspable >/dev/null

python - "$TMP/good" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
report = json.loads((root / "report.json").read_text())
assert report["status"] == "pass", report
assert report["physics"]["authoring"]["mass_source"] == "blender"
assert report["physics"]["authoring"]["mass_kg"] == 0.12
checks = {check["id"]: check for check in report["checks"]}
assert checks["physics.usd_schema"]["status"] == "pass"
assert checks["physics.axis_conversion"]["status"] == "pass"
assert checks["grasp.width"]["status"] == "pass"
assert (root / report["artifacts"]["physics_usd"]).is_file()
assert (root / report["artifacts"]["collider_preview"]).is_file()
print("asset physics validation fixture: PASS")
PY
