#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

env -u PYTHONHOME -u PYTHONPATH \
  PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 \
  /usr/bin/blender --background --python "$ROOT/tests/asset_validation/create_fixtures.py" -- "$TMP/fixtures" >/dev/null
"$ROOT/scripts/validate_asset.sh" "$TMP/fixtures/good.glb" "$TMP/good" --skip-physics >/dev/null
"$ROOT/scripts/validate_asset.sh" "$TMP/fixtures/bad.glb" "$TMP/bad" --skip-physics >/dev/null

python - "$TMP" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
good = json.loads((root / "good/report.json").read_text())
bad = json.loads((root / "bad/report.json").read_text())
assert good["status"] == "pass", good
assert bad["status"] == "fail", bad
assert len(good["artifacts"]["previews"]) == 4
print("asset validation fixtures: PASS")
PY
