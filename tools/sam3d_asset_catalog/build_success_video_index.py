from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from daft.datasets.droid import raw
except ImportError:
    raw = None


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG_ROOT = Path(
    os.environ.get("SAM3D_CATALOG_ROOT", REPOSITORY_ROOT / "data" / "sam3d_assets")
)
RAW_ROOT = "gs://gresearch/robotics/droid_raw/1.0.1"
LAB_PATH_ALIASES = {"tri": "TRI"}
MAX_PLAYBACK_DURATION_SEC = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a success-first source video index for the SAM 3D catalog."
    )
    parser.add_argument("--catalog-root", type=Path, default=DEFAULT_CATALOG_ROOT)
    parser.add_argument("--lab", action="append", help="Only scan this DROID lab (repeatable).")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def raw_scene_parts(raw_scene_key: str) -> tuple[str, str, str, int]:
    location, robot_serial, scene_id = raw_scene_key.rsplit("/", 2)
    lab, building = location.split("/", 1)
    return lab, building, robot_serial, int(scene_id)


def episode_timestamp(episode: dict[str, Any]) -> datetime:
    return datetime.strptime(episode["timestamp"], "%Y-%m-%d-%Hh-%Mm-%Ss")


def media_url(episode_dir: str, serial: str | None) -> str | None:
    if not serial:
        return None
    path = episode_dir.removeprefix("gs://gresearch/")
    return f"https://storage.googleapis.com/gresearch/{path}/recordings/MP4/{serial}.mp4"


def verified_success_url(episode: dict[str, Any]) -> str | None:
    url = episode.get("ext1_url") or episode.get("preferred_video_url")
    if episode.get("source_success") is not True or not url:
        return None
    return url if "/success/" in url else None


def episode_record(columns: dict[str, list[Any]], index: int) -> dict[str, Any]:
    episode_dir = columns["episode_dir"][index]
    length = int(columns["trajectory_length"][index])
    return {
        "uuid": columns["uuid"][index],
        "date": str(columns["date"][index]),
        "timestamp": columns["timestamp"][index],
        "task": columns["current_task"][index],
        "source_success": True,
        "trajectory_length": length,
        "trajectory_duration_sec": round(length / 15, 3),
        "episode_dir": episode_dir,
        "ext1_url": media_url(episode_dir, columns["ext1_cam_serial"][index]),
        "ext2_url": media_url(episode_dir, columns["ext2_cam_serial"][index]),
        "wrist_url": media_url(episode_dir, columns["wrist_cam_serial"][index]),
        "preferred_video_camera": "ext1",
    }


def scan_lab(lab: str) -> list[dict[str, Any]]:
    if raw is None:
        raise RuntimeError(
            "The DROID metadata environment providing daft.datasets.droid is required"
        )
    print(f"scanning {lab} success metadata", flush=True)
    frame = raw(f"{RAW_ROOT}/{lab}/success").select(
        "uuid",
        "lab",
        "building",
        "robot_serial",
        "scene_id",
        "date",
        "timestamp",
        "trajectory_length",
        "current_task",
        "episode_dir",
        "ext1_cam_serial",
        "ext2_cam_serial",
        "wrist_cam_serial",
    )
    columns = frame.collect().to_pydict()
    episodes = [episode_record(columns, i) for i in range(len(columns["uuid"]))]
    for index, episode in enumerate(episodes):
        episode["source_key"] = "/".join(
            [
                columns["lab"][index],
                columns["building"][index],
                columns["robot_serial"][index],
                str(columns["scene_id"][index]),
            ]
        )
    print(f"found {len(episodes):,} successful episodes for {lab}", flush=True)
    return episodes


def select_episode(
    setup: dict[str, Any],
    source_setup_count: int,
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    representative = setup.get("representative") or {}
    if verified_success_url(representative):
        selected = dict(representative)
        selected["trajectory_duration_sec"] = round(
            float(selected.get("trajectory_length", 0)) / 15, 3
        )
        return selected, "existing_success_representative"

    available = [row for row in candidates if verified_success_url(row)]
    if not available:
        return {
            "source_success": None,
            "setup_id": setup["setup_id"],
            "trajectory_length": None,
            "trajectory_duration_sec": None,
            "preferred_video_url": None,
        }, "no_verified_success_episode"

    if source_setup_count == 1:
        return max(available, key=lambda row: row["trajectory_length"]), "longest_success_in_setup"

    if representative.get("timestamp"):
        target = episode_timestamp(representative)
        return min(
            available,
            key=lambda row: (
                abs((episode_timestamp(row) - target).total_seconds()),
                -row["trajectory_length"],
            ),
        ), "nearest_success_in_split_source_scene"

    return max(available, key=lambda row: row["trajectory_length"]), "longest_success_in_source_scene"


def main() -> None:
    args = parse_args()
    catalog_root = args.catalog_root.resolve()
    bootstrap = read_json(catalog_root / "review_bootstrap.json")
    setups = bootstrap["setups"]
    source_scenes = {row["raw_scene_key"]: row for row in bootstrap["source_scenes"]}
    requested_labs = set(args.lab or [])
    labs = sorted(
        {
            LAB_PATH_ALIASES.get(
                raw_scene_parts(setup["raw_scene_key"])[0],
                raw_scene_parts(setup["raw_scene_key"])[0],
            )
            for setup in setups
            if not requested_labs
            or raw_scene_parts(setup["raw_scene_key"])[0] in requested_labs
        }
    )

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for lab in labs:
        for episode in scan_lab(lab):
            by_source[episode.pop("source_key")].append(episode)

    result: dict[str, Any] = {
        "version": "sam3d_setup_video_index/v2",
        "fps": 15,
        "max_playback_duration_sec": MAX_PLAYBACK_DURATION_SEC,
        "setups": {},
    }
    selected_success = 0
    unavailable_count = 0
    for setup in setups:
        setup_id = setup["setup_id"]
        if requested_labs and raw_scene_parts(setup["raw_scene_key"])[0] not in requested_labs:
            continue
        source = source_scenes[setup["raw_scene_key"]]
        selected, rule = select_episode(
            setup,
            int(source.get("setup_count", 1)),
            by_source.get(setup["raw_scene_key"], []),
        )
        selected["selection_rule"] = rule
        selected["success_candidate_count"] = len(by_source.get(setup["raw_scene_key"], []))
        selected["preferred_video_url"] = verified_success_url(selected)
        result["setups"][setup_id] = selected
        selected_success += bool(selected.get("source_success"))
        unavailable_count += not bool(selected.get("source_success"))

    output_path = catalog_root / "setup_video_index.json"
    temp_path = output_path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    temp_path.replace(output_path)
    print(
        f"wrote {output_path}: {len(result['setups']):,} setups, "
        f"{selected_success:,} success, {unavailable_count:,} unavailable",
        flush=True,
    )


if __name__ == "__main__":
    main()
