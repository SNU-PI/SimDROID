#!/usr/bin/env python3
"""Run the released 3D-PAQA evaluator on the Blender GLB export."""

import argparse
import csv
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from safetensors.torch import load_file


MODEL_REPO = "JiHyuk-Byun/3D-PAQA-evaluator"
CRITERIA = ["geometry", "texture", "material", "plausibility", "artifact", "preference"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=150_000)
    parser.add_argument("--min-score", type=float, default=2.5)
    return parser.parse_args()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    args = parse_args()
    qt_root = os.environ.get("QT_ROOT")
    if not qt_root or not (Path(qt_root) / "qt").is_dir():
        raise RuntimeError("QT_ROOT must point to a clone of https://github.com/JiHyuk-Byun/QT")
    sys.path.insert(0, qt_root)
    from qt.data import ObjaverseDataModule
    from qt.models import PointTransformerV3

    config_path = hf_hub_download(MODEL_REPO, "config.json")
    weights_path = hf_hub_download(MODEL_REPO, "model.safetensors")
    sampler_path = hf_hub_download(MODEL_REPO, "mesh2pc.py")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    model_args = dict(config["model_args"])
    # FlashAttention is only an optimization. This keeps the released weights
    # usable in CUDA environments where its compiled wheel is unavailable.
    model_args["enable_flash"] = False
    model = PointTransformerV3(**model_args).cuda().eval()
    model.load_state_dict(load_file(weights_path), strict=True)

    sampler = load_module(sampler_path, "paqa_mesh2pc")
    data, sampling = sampler.glb_to_npy_dict(str(args.glb), args.samples)
    for key in ("coord", "normal", "color", "metallic", "roughness"):
        if not np.isfinite(data[key]).all():
            raise RuntimeError("3D-PAQA input has non-finite %s values" % key)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        np.save(temp / "asset.npy", data, allow_pickle=True)
        with (temp / "split.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id"] + CRITERIA)
            writer.writerow(["asset.npy"] + [0.0] * len(CRITERIA))
        data_config = OmegaConf.create({
            "root_dir": str(temp),
            "train_split": str(temp / "split.csv"),
            "test_split": str(temp / "split.csv"),
            "criterion": CRITERIA,
            "batch_size": 1,
            "eval_batch_size": 1,
            "num_workers": 0,
            "dataset_config": {
                "manual_seed": 123456,
                "keys": ["coord", "grid_coord", "mos", "id"],
                "feat_keys": ["coord", "color", "normal", "metallic", "roughness"],
                "grid_size": config["grid_size"],
                "hash_type": config["hash_type"],
                "return_grid_coord": True,
                "augments": None,
            },
        })
        module = ObjaverseDataModule(**OmegaConf.to_container(data_config, resolve=True))
        dataset = module._get_dataset(is_train=False)
        batch = module._collate_fn([dataset[0]])
        if batch is None:
            raise RuntimeError("3D-PAQA preprocessing rejected the asset")
        batch = {key: value.cuda() if torch.is_tensor(value) else value for key, value in batch.items()}
        with torch.no_grad():
            values = model(batch).float().cpu().numpy()[0]

    scores = {criterion: float(values[index]) for index, criterion in enumerate(CRITERIA)}
    failed_criteria = [criterion for criterion, value in scores.items() if value < args.min_score]
    report = {
        "stage": "photorealism_3d_paqa",
        "status": "PASS" if not failed_criteria else "FAIL",
        "model": MODEL_REPO,
        "checkpoint_metric": {"mean_validation_srocc": config.get("monitor_value")},
        "threshold": args.min_score,
        "scores": scores,
        "mean_score": sum(scores.values()) / len(scores),
        "failed_criteria": failed_criteria,
        "sampling": sampling,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
