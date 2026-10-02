#!/usr/bin/env python3

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.asset_quality.pipeline import summarize_benchmark, summarize_static_validation


ROOT = Path(__file__).resolve().parents[2]


class ReportTests(unittest.TestCase):
    def test_static_target_features_are_separate_from_profile_failure(self):
        validation = {
            "asset.usda": {
                "features_summary": {
                    "FET_001_STANDARD": {"passed": True},
                    "FET_003_STANDARD": {"passed": True},
                    "FET_003_PHYSX": {"passed": True},
                    "FET_005_STANDARD": {"passed": True},
                    "FET_004_STANDARD": {"passed": False},
                }
            }
        }
        result = summarize_static_validation(validation, 1)
        self.assertEqual(result["target_features_status"], "PASS")
        self.assertEqual(result["profile_status"], "FAIL")
        self.assertEqual(result["failed_outside_target"], ["FET_004_STANDARD"])

    def test_benchmark_missing_result_is_incomplete(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            benchmark = out / "benchmark"
            benchmark.mkdir()
            (benchmark / "plan.json").write_text(json.dumps({
                "assets": [{"tests": [1]}],
                "tests": [{"id": 1, "name": "ground_drop"}],
            }))
            result = summarize_benchmark(out, 0)
            self.assertEqual(result["status"], "INCOMPLETE")
            self.assertEqual(result["tests"]["ground_drop"]["status"], "NOT_RUN")

    def test_nonzero_benchmark_exit_is_not_silent(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            benchmark = out / "benchmark"
            result_dir = benchmark / "results" / "ground_drop"
            result_dir.mkdir(parents=True)
            (benchmark / "plan.json").write_text(json.dumps({
                "assets": [{"tests": [1]}],
                "tests": [{"id": 1, "name": "ground_drop"}],
            }))
            (result_dir / "result.json").write_text(json.dumps({
                "test_name": "ground_drop",
                "status": "PASS",
            }))
            result = summarize_benchmark(out, 1)
            self.assertEqual(result["status"], "PASS_WITH_WARNINGS")

    def test_missing_mandatory_stages_are_incomplete(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / "blender_preflight.json").write_text(json.dumps({
                "status": "PASS",
                "bounds_m": {"dimensions": [0.04, 0.05, 0.12]},
                "checks": [{"id": "BLEND_PBR", "metrics": {"pbr_coverage": 1.0, "texture_coverage": 1.0}}],
                "outputs": {"renders": []},
            }))
            (out / "usd_physics.json").write_text(json.dumps({
                "status": "PASS",
                "bounds_m": {"dimensions": [0.04, 0.05, 0.12]},
                "grasp": {"width_m": 0.04, "reviewed": True},
            }))
            subprocess.run([
                sys.executable, str(ROOT / "scripts/asset_quality/report_stage.py"),
                "--out", str(out), "--config", str(ROOT / "configs/asset_quality/default.json"),
                "--category", "cup",
            ], check=True)
            report = json.loads((out / "report.json").read_text())
            self.assertEqual(report["status"], "INCOMPLETE")
            self.assertEqual(report["stages"]["photorealism_3d_paqa"]["status"], "INCOMPLETE")
            self.assertEqual(report["stages"]["simready"]["status"], "INCOMPLETE")

    def test_zero_width_grasp_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / "blender_preflight.json").write_text(json.dumps({
                "status": "PASS",
                "bounds_m": {"dimensions": [0.04, 0.05, 0.12]},
                "checks": [],
                "outputs": {"renders": []},
            }))
            (out / "usd_physics.json").write_text(json.dumps({
                "status": "FAIL",
                "bounds_m": {"dimensions": [0.04, 0.05, 0.12]},
                "grasp": {"width_m": 0.0, "reviewed": False},
            }))
            subprocess.run([
                sys.executable, str(ROOT / "scripts/asset_quality/report_stage.py"),
                "--out", str(out), "--config", str(ROOT / "configs/asset_quality/default.json"),
                "--category", "cup",
            ], check=True)
            report = json.loads((out / "report.json").read_text())
            gripper = report["stages"]["droid_checks"]["robotiq_2f_85"]
            self.assertEqual(gripper["status"], "FAIL")
            self.assertFalse(gripper["within_recommended_width"])


if __name__ == "__main__":
    unittest.main()
