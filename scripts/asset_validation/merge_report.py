"""Merge Blender and Isaac Sim validation output into one small report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    blender = json.loads((output / "blender.json").read_text())
    physics_path = output / "physics.json"
    physics = json.loads(physics_path.read_text()) if physics_path.exists() else {
        "status": "skipped",
        "checks": [{"id": "physics.run", "status": "info", "value": "skipped", "message": "Physics test was skipped."}],
    }
    checks = blender.get("checks", []) + physics.get("checks", [])
    statuses = {check["status"] for check in checks}
    status = "fail" if "fail" in statuses else "review" if "warn" in statuses else "pass"
    artifacts = dict(blender.get("artifacts", {}))
    authored_asset = physics.get("authoring", {}).get("asset_usd")
    if authored_asset:
        artifacts["physics_usd"] = authored_asset
    if (output / "collider_preview.png").exists():
        artifacts["collider_preview"] = "collider_preview.png"
    report = {
        "schema_version": "simdroid.asset-validation/v1",
        "status": status,
        "summary": {
            "pass": sum(check["status"] == "pass" for check in checks),
            "review": sum(check["status"] == "warn" for check in checks),
            "fail": sum(check["status"] == "fail" for check in checks),
            "info": sum(check["status"] == "info" for check in checks),
        },
        "asset": blender.get("asset", {}),
        "stats": blender.get("stats", {}),
        "checks": checks,
        "artifacts": artifacts,
        "physics": {key: value for key, value in physics.items() if key != "checks"},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"RESULT={status.upper()} pass={report['summary']['pass']} review={report['summary']['review']} fail={report['summary']['fail']}")
    for check in checks:
        if check["status"] in {"warn", "fail"}:
            print(f"  [{check['status'].upper()}] {check['id']}: {check['message']} ({check['value']})")


if __name__ == "__main__":
    main()
