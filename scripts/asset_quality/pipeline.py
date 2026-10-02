#!/usr/bin/env python3
"""End-to-end SimDROID asset quality pipeline."""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "asset_quality" / "default.json"
SIMREADY_IMAGE = "simdroid-asset-quality:2026.7.1"
ISAAC_HOME_VOLUME = "simdroid-asset-quality-home"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("blend", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--category", default="default")
    parser.add_argument("--manifest", type=Path, help="Reviewed per-asset physics/grasp JSON")
    parser.add_argument("--source-image", type=Path, help="DROID source frame for side-by-side review")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--blender", default=os.environ.get("BLENDER", "/usr/bin/blender"))
    parser.add_argument("--simready-image", default=SIMREADY_IMAGE)
    parser.add_argument("--skip-image-build", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true", help="Developer smoke test only; final report becomes INCOMPLETE")
    parser.add_argument("--skip-paqa", action="store_true", help="Developer smoke test only; final report becomes INCOMPLETE")
    return parser.parse_args()


def run(command, log_path, env=None, check=False):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, text=True)
    if check and process.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeError("command failed (%d): %s\n%s" % (process.returncode, shlex.join(command), tail))
    return process.returncode


def docker_base(args, out):
    return [
        "docker", "run", "--rm", "--network", "host", "--ipc", "host",
        "--gpus", "device=%s" % args.gpu,
        "-e", "ACCEPT_EULA=Y", "-e", "OMNI_KIT_ALLOW_ROOT=1", "-e", "HOME=/simdroid-home",
        "-v", "%s:/simdroid-home" % ISAAC_HOME_VOLUME,
        "-v", "%s:/work:ro" % REPO_ROOT,
        "-v", "%s:/output:rw" % out,
        "-w", "/work", "--entrypoint", "/bin/bash", args.simready_image,
    ]


def docker_shell(args, out, command):
    wrapped = "%s; code=$?; chown -R %d:%d /output; exit $code" % (
        command,
        os.getuid(),
        os.getgid(),
    )
    return docker_base(args, out) + ["-lc", wrapped]


def ensure_image(args, out):
    exists = subprocess.run(["docker", "image", "inspect", args.simready_image], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if exists or args.skip_image_build:
        return
    run(
        ["docker", "build", "-t", args.simready_image, "-f", str(REPO_ROOT / "docker" / "asset-quality.Dockerfile"), str(REPO_ROOT)],
        out / "logs" / "docker_build.log",
        check=True,
    )


def run_blender(args, out):
    env = dict(os.environ)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env["PATH"] = "/usr/bin:/bin"
    env["PYTHONNOUSERSITE"] = "1"
    env["TMPDIR"] = str(out / ".tmp")
    Path(env["TMPDIR"]).mkdir(exist_ok=True)
    command = [
        args.blender, "--background", "--factory-startup", "--python",
        str(REPO_ROOT / "scripts" / "asset_quality" / "blender_stage.py"), "--",
        "--blend", str(args.blend.resolve()), "--out", str(out),
    ]
    run(command, out / "logs" / "blender.log", env=env, check=True)


def run_usd_stage(args, out):
    convert = [
        "/isaac-sim/python.sh", "/work/scripts/asset_quality/convert_glb.py",
        "--input", "/output/asset.glb", "--output", "/output/asset.source.usd",
    ]
    run(
        docker_shell(args, out, shlex.join(convert)),
        out / "logs" / "usd_convert.log",
        check=True,
    )
    command = [
        "/isaac-sim/python.sh", "/work/scripts/asset_quality/usd_stage.py",
        "--input", "/output/asset.source.usd",
        "--output", "/output/asset.usda",
        "--preflight", "/output/blender_preflight.json",
        "--config", "/output/config.json",
        "--report", "/output/usd_physics.json",
    ]
    if args.manifest:
        shutil.copy2(args.manifest, out / "manifest.json")
        command.extend(["--manifest", "/output/manifest.json"])
    docker_command = docker_shell(args, out, shlex.join(command))
    run(docker_command, out / "logs" / "usd_physics.log", check=True)
    transform = (
        "/isaac-sim/python.sh -m simready.asset_transformer.cli "
        "/output/asset.usda /output/simready_package "
        "--profile simready_physx_to_isaac_prop --interface-name asset.usda "
        "--report /output/simready_transform.json --overwrite --verbose"
    )
    run(
        docker_shell(args, out, transform),
        out / "logs" / "simready_transform.log",
        check=True,
    )
    package = (
        "/isaac-sim/python.sh /work/scripts/asset_quality/package_stage.py "
        "--package /output/simready_package --report /output/simready_package.json"
    )
    run(
        docker_shell(args, out, package),
        out / "logs" / "simready_package.log",
        check=True,
    )


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def status_from_items(statuses, advisory_skips=False):
    normalized = [str(status).upper() for status in statuses]
    if any(status in {"FAIL", "FAILED", "ERROR", "BLOCKED"} for status in normalized):
        return "FAIL"
    if any(status in {"INCOMPLETE", "MISSING", "NOT_RUN"} for status in normalized):
        return "INCOMPLETE"
    if any(status == "SKIPPED" for status in normalized):
        return "PASS_WITH_WARNINGS" if advisory_skips else "INCOMPLETE"
    return "PASS" if normalized else "INCOMPLETE"


def summarize_static_validation(validation, validation_code):
    target_features = ["FET_001_STANDARD", "FET_003_STANDARD", "FET_003_PHYSX", "FET_005_STANDARD"]
    asset_result = next(
        (value for value in (validation or {}).values() if isinstance(value, dict) and "features_summary" in value),
        {},
    )
    features = asset_result.get("features_summary", {})
    target_results = {}
    for feature in target_features:
        item = features.get(feature)
        target_results[feature] = {
            "status": "PASS" if item and item.get("passed") is True else "FAIL" if item else "MISSING",
            "result": item,
        }
    target_status = status_from_items(item["status"] for item in target_results.values())
    failed_outside_target = sorted(
        name for name, item in features.items() if name not in target_features and item.get("passed") is False
    )
    return {
        "profile_status": "PASS" if validation_code == 0 else "FAIL",
        "profile_exit_code": validation_code,
        "target_features_status": target_status,
        "target_features": target_results,
        "failed_outside_target": failed_outside_target,
    }


def summarize_benchmark(out, benchmark_code):
    benchmark_dir = out / "benchmark"
    if benchmark_code == -1:
        return {
            "status": "INCOMPLETE",
            "exit_code": None,
            "requested_features": ["FET001", "FET003", "FET005"],
            "tests": {},
            "reports": [],
            "log": "logs/simready_benchmark.log",
        }

    plan = read_json(benchmark_dir / "plan.json") or {}
    tests_by_id = {item.get("id"): item for item in plan.get("tests", [])}
    expected_ids = {
        test_id
        for asset in plan.get("assets", [])
        for test_id in asset.get("tests", [])
    }
    expected_names = {
        tests_by_id[test_id]["name"]
        for test_id in expected_ids
        if test_id in tests_by_id and tests_by_id[test_id].get("name")
    }
    result_paths = sorted(benchmark_dir.rglob("result.json")) if benchmark_dir.exists() else []
    tests = {}
    for path in result_paths:
        result = read_json(path) or {}
        name = result.get("test_name") or path.parent.parent.name
        tests[name] = {
            "status": str(result.get("status", "INCOMPLETE")).upper(),
            "message": result.get("message", ""),
            "metrics": result.get("metrics", {}),
            "report": str(path.relative_to(out)),
        }
    for name in expected_names:
        tests.setdefault(name, {"status": "NOT_RUN", "message": "planned test produced no result"})

    hard_statuses = [item["status"] for name, item in tests.items() if name != "pivot"]
    status = status_from_items(hard_statuses)
    pivot = tests.get("pivot")
    if status == "PASS" and pivot and pivot["status"] == "SKIPPED":
        status = "PASS_WITH_WARNINGS"
    elif status == "PASS" and pivot and pivot["status"] not in {"PASS", "SKIPPED"}:
        status = status_from_items([pivot["status"]])
    if not expected_names:
        status = "FAIL" if benchmark_code else "INCOMPLETE"
    elif status == "PASS" and benchmark_code != 0:
        status = "PASS_WITH_WARNINGS"
    return {
        "status": status,
        "exit_code": benchmark_code,
        "requested_features": ["FET001", "FET003", "FET005"],
        "tests": dict(sorted(tests.items())),
        "reports": [str(path.relative_to(out)) for path in result_paths],
        "plan": "benchmark/plan.json" if (benchmark_dir / "plan.json").exists() else None,
        "log": "logs/simready_benchmark.log",
    }


def summarize_simready(out, validation_code, benchmark_code):
    validation = None
    validation_path = out / "simready_validation.json"
    if validation_path.exists():
        try:
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
        except Exception:
            validation = None
    static_summary = summarize_static_validation(validation, validation_code)
    benchmark_summary = summarize_benchmark(out, benchmark_code)
    status = status_from_items([static_summary["target_features_status"], benchmark_summary["status"]])
    if status == "PASS" and static_summary["profile_status"] == "FAIL":
        status = "PASS_WITH_WARNINGS"
    elif status == "PASS" and benchmark_summary["status"] == "PASS_WITH_WARNINGS":
        status = "PASS_WITH_WARNINGS"
    package = None
    package_path = out / "simready_package.json"
    if package_path.exists():
        try:
            package = json.loads(package_path.read_text(encoding="utf-8"))
        except Exception:
            package = None
    transform = None
    transform_path = out / "simready_transform.json"
    if transform_path.exists():
        try:
            transform = json.loads(transform_path.read_text(encoding="utf-8"))
        except Exception:
            transform = None
    report = {
        "stage": "simready",
        "status": status,
        "profile": {"id": "Prop-Robotics-Isaac", "version": "1.0.0"},
        "transform": {"result": transform, "log": "logs/simready_transform.log"},
        "package": {"result": package, "log": "logs/simready_package.log"},
        "static_validation": {**static_summary, "result": validation, "log": "logs/simready_validate.log"},
        "benchmark": benchmark_summary,
    }
    (out / "simready.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def run_simready(args, out):
    validate = (
        "/isaac-sim/python.sh /isaac-sim/kit/python/bin/simready-validate "
        "--profile Prop-Robotics-Isaac --version 1.0.0 --output /output/simready_validation.json "
        "--stamp-asset-validation --verbose /output/simready_package/asset.usda"
    )
    validation_code = run(docker_shell(args, out, validate), out / "logs" / "simready_validate.log")
    benchmark_code = -1
    if not args.skip_benchmark:
        benchmark_plan = (
            "/isaac-sim/python.sh /isaac-sim/kit/python/bin/simready-benchmark "
            "--assets /output/simready_package/asset.usda --features FET001 FET003 FET005 "
            "--output-dir /output/benchmark --engines-toml /work/configs/asset_quality/engines.toml "
            "--runtime isaac_sim --plan-only"
        )
        benchmark_edit = (
            "/isaac-sim/python.sh /isaac-sim/kit/python/bin/simready-benchmark "
            "--edit-plan --output-dir /output/benchmark --all-assets "
            "--set-config ground_drop:floor_margin=0.002"
        )
        benchmark_run = (
            "/isaac-sim/python.sh /isaac-sim/kit/python/bin/simready-benchmark "
            "--rerun plan --output-dir /output/benchmark "
            "--engines-toml /work/configs/asset_quality/engines.toml --runtime isaac_sim "
            "--max-concurrent 1 --session-mode batch --format both --no-stamp"
        )
        benchmark = "%s && %s && %s" % (benchmark_plan, benchmark_edit, benchmark_run)
        benchmark_code = run(docker_shell(args, out, benchmark), out / "logs" / "simready_benchmark.log")
    summarize_simready(out, validation_code, benchmark_code)


def run_paqa(args, out, config):
    if args.skip_paqa:
        return
    python = os.environ.get("PAQA_PYTHON")
    qt_root = os.environ.get("PAQA_QT_ROOT")
    deps = os.environ.get("PAQA_PYTHONPATH")
    if not python or not qt_root:
        (out / "logs" / "paqa.log").write_text("PAQA_PYTHON and PAQA_QT_ROOT are required. Run setup_paqa.sh.\n", encoding="utf-8")
        return
    env = dict(os.environ)
    env["QT_ROOT"] = qt_root
    if deps:
        env["PYTHONPATH"] = deps + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["TMPDIR"] = str(out / ".tmp")
    command = [
        python, str(REPO_ROOT / "scripts" / "asset_quality" / "paqa_stage.py"),
        "--glb", str(out / "asset.glb"),
        "--output", str(out / "photorealism_paqa.json"),
        "--min-score", str(config["photorealism"]["paqa_min_score"]),
    ]
    run(command, out / "logs" / "paqa.log", env=env)


def main():
    args = parse_args()
    args.blend = args.blend.resolve()
    args.out = args.out.resolve()
    args.config = args.config.resolve()
    if not args.blend.exists():
        raise SystemExit("blend file not found: %s" % args.blend)
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("output directory must be new or empty: %s" % args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "logs").mkdir(exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    shutil.copy2(args.config, args.out / "config.json")
    if args.source_image:
        shutil.copy2(args.source_image, args.out / "source.png")

    run_blender(args, args.out)
    ensure_image(args, args.out)
    run_usd_stage(args, args.out)
    run_simready(args, args.out)
    run_paqa(args, args.out, config)
    report_command = [
        sys.executable, str(REPO_ROOT / "scripts" / "asset_quality" / "report_stage.py"),
        "--out", str(args.out), "--config", str(args.config), "--category", args.category,
    ]
    if args.source_image:
        report_command.extend(["--source-image", str(args.out / "source.png")])
    run(report_command, args.out / "logs" / "report.log", check=True)
    report = json.loads((args.out / "report.json").read_text(encoding="utf-8"))
    print(json.dumps({"status": report["status"], "report_json": str(args.out / "report.json"), "report_html": str(args.out / "report.html")}, indent=2))


if __name__ == "__main__":
    main()
