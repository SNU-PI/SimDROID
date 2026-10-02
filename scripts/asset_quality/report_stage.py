#!/usr/bin/env python3
"""Merge all stage artifacts into one machine-readable and visual report."""

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--category", default="default")
    parser.add_argument("--source-image", type=Path)
    return parser.parse_args()


def read_json(path, fallback_status="INCOMPLETE"):
    if not path.exists():
        return {"status": fallback_status, "message": "%s was not produced" % path.name}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        return {"status": "ERROR", "message": "%s: %s" % (type(error).__name__, error)}


def status_rank(status):
    return {"PASS": 0, "PASS_WITH_WARNINGS": 1, "PASS_WITH_REVIEW": 1, "WARN": 1, "REVIEW_REQUIRED": 2, "INCOMPLETE": 3, "SKIPPED": 3, "FAIL": 4, "ERROR": 5}.get(status, 3)


def overall_status(stages):
    worst = max((stage.get("status", "INCOMPLETE") for stage in stages.values()), key=status_rank)
    if status_rank(worst) >= status_rank("FAIL"):
        return "FAIL"
    if status_rank(worst) >= status_rank("INCOMPLETE"):
        return "INCOMPLETE"
    if status_rank(worst) >= status_rank("REVIEW_REQUIRED"):
        return "REVIEW_REQUIRED"
    if status_rank(worst) >= status_rank("WARN"):
        return "PASS_WITH_WARNINGS"
    return "PASS"


def stage_rows(stages):
    rows = []
    for name, stage in stages.items():
        rows.append("<tr><td>%s</td><td class='%s'>%s</td><td><pre>%s</pre></td></tr>" % (
            html.escape(name),
            html.escape(stage.get("status", "INCOMPLETE").lower()),
            html.escape(stage.get("status", "INCOMPLETE")),
            html.escape(json.dumps(stage, indent=2)[:12000]),
        ))
    return "\n".join(rows)


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    blender = read_json(args.out / "blender_preflight.json")
    usd = read_json(args.out / "usd_physics.json")
    paqa = read_json(args.out / "photorealism_paqa.json")
    simready = read_json(args.out / "simready.json")

    dimensions = usd.get("bounds_m", blender.get("bounds_m", {})).get("dimensions", [0.0, 0.0, 0.0])
    longest = max(dimensions) if dimensions else 0.0
    category_range = config["category_size_m"].get(args.category, config["category_size_m"]["default"])
    category_status = "PASS" if category_range[0] <= longest <= category_range[1] else "FAIL"
    grasp = usd.get("grasp", {})
    width = grasp.get("width_m")
    hard_max = config["gripper"]["hard_max_width_m"]
    recommended = config["gripper"]["recommended_width_m"]
    if width is None:
        grasp_status = "INCOMPLETE"
    elif width <= 0.0 or width > hard_max:
        grasp_status = "FAIL"
    elif not grasp.get("reviewed", False):
        grasp_status = "REVIEW_REQUIRED"
    elif width > recommended:
        grasp_status = "WARN"
    else:
        grasp_status = "PASS"
    droid = {
        "stage": "droid_checks",
        "status": "FAIL" if category_status == "FAIL" or grasp_status == "FAIL" else "INCOMPLETE" if grasp_status == "INCOMPLETE" else "REVIEW_REQUIRED" if grasp_status == "REVIEW_REQUIRED" else "PASS_WITH_WARNINGS" if grasp_status == "WARN" else "PASS",
        "category": args.category,
        "category_size": {"status": category_status, "longest_dimension_m": longest, "allowed_range_m": category_range},
        "robotiq_2f_85": {
            "status": grasp_status,
            "grasp_width_m": width,
            "hard_max_width_m": hard_max,
            "recommended_width_m": recommended,
            "within_recommended_width": width is not None and 0.0 < width <= recommended,
            "metadata_reviewed": bool(grasp.get("reviewed", False)),
        },
    }
    pbr_metrics = next((item.get("metrics", {}) for item in blender.get("checks", []) if item.get("id") == "BLEND_PBR"), {})
    uv_metrics = next((item.get("metrics", {}) for item in blender.get("checks", []) if item.get("id") == "BLEND_UV"), {})
    computational = {
        "stage": "photorealism_computational",
        "status": "PASS" if pbr_metrics.get("pbr_coverage", 0.0) >= config["photorealism"]["required_pbr_coverage"] and pbr_metrics.get("texture_coverage", 0.0) >= config["photorealism"]["required_texture_coverage"] else "FAIL",
        "pbr_and_texture": pbr_metrics,
        "uv": uv_metrics,
        "canonical_renders": blender.get("outputs", {}).get("renders", []),
        "source_image": "source.png" if args.source_image else None,
        "source_render_review": "READY" if args.source_image else "INCOMPLETE",
    }
    if not args.source_image:
        computational["status"] = "INCOMPLETE" if computational["status"] == "PASS" else computational["status"]

    stages = {
        "blender_preflight": blender,
        "usd_physics": usd,
        "simready": simready,
        "photorealism_computational": computational,
        "photorealism_3d_paqa": paqa,
        "droid_checks": droid,
    }
    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": overall_status(stages),
        "stages": stages,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    images = []
    if args.source_image:
        images.append("<figure><img src='source.png'><figcaption>DROID source</figcaption></figure>")
    for relative in blender.get("outputs", {}).get("renders", []):
        images.append("<figure><img src='%s'><figcaption>%s</figcaption></figure>" % (html.escape(relative), html.escape(Path(relative).stem)))
    document = """<!doctype html><html><head><meta charset='utf-8'><title>Asset QA</title>
<style>body{font-family:system-ui,sans-serif;margin:24px;background:#111;color:#eee}h1{margin-bottom:6px}.summary{color:#67e8b2}table{width:100%%;border-collapse:collapse}th,td{border:1px solid #444;padding:10px;vertical-align:top}pre{white-space:pre-wrap;max-height:320px;overflow:auto}.fail,.error{color:#ff7777}.incomplete,.review_required{color:#ffd166}.pass{color:#67e8b2}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:20px 0}figure{margin:0}img{width:100%%;background:#222}figcaption{padding:6px;color:#aaa}</style></head><body>
<h1>SimDROID Asset Quality Report</h1><p class='summary'>Overall: %s</p><div class='gallery'>%s</div>
<table><thead><tr><th>Stage</th><th>Status</th><th>Evidence</th></tr></thead><tbody>%s</tbody></table></body></html>""" % (html.escape(report["status"]), "".join(images), stage_rows(stages))
    (args.out / "report.html").write_text(document, encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(args.out / "report.json")}))


if __name__ == "__main__":
    main()
