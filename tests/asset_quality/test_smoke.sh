#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${1:-$(mktemp -d)}"
mkdir -p "$WORK/fixtures"
env -u PYTHONHOME -u PYTHONPATH PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 TMPDIR="$WORK" \
  /usr/bin/blender --background --factory-startup \
  --python "$ROOT/tests/asset_quality/create_fixtures.py" -- --out "$WORK/fixtures"
python3 -m unittest "$ROOT/tests/asset_quality/test_report.py"
echo "$WORK"
