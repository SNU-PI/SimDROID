from __future__ import annotations

import json
import os
import subprocess
import threading
import tempfile
import urllib.request
from hashlib import sha1
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG_ROOT = REPOSITORY_ROOT / "data" / "sam3d_assets"
CATALOG_ROOT = Path(
    os.environ.get("SAM3D_CATALOG_ROOT", DEFAULT_CATALOG_ROOT)
).resolve()
BOOTSTRAP_PATH = Path(
    os.environ.get("SAM3D_BOOTSTRAP", CATALOG_ROOT / "review_bootstrap.json")
).resolve()
SETUPS_ROOT = CATALOG_ROOT / "setups"
VIDEO_INDEX_PATH = CATALOG_ROOT / "setup_video_index.json"
VIDEO_CACHE_ROOT = CATALOG_ROOT / "video_cache"
MAX_VIDEO_DURATION_SEC = float(os.environ.get("SAM3D_MAX_VIDEO_DURATION_SEC", "30"))
VIDEO_POLICY_VERSION = "complete-trajectory-max-30s-v1"
STATIC_ROOT = Path(__file__).parent / "static"
video_cache_locks: dict[str, threading.Lock] = {}
video_cache_locks_guard = threading.Lock()


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def asset_parts(dirname: str) -> tuple[str, int]:
    label, separator, instance = dirname.rpartition("__")
    if separator and instance.isdigit():
        return label, int(instance)
    return dirname, 0


def verified_success_url(episode: dict[str, Any]) -> str | None:
    source_url = episode.get("preferred_video_url") or episode.get("ext1_url")
    if episode.get("source_success") is not True or not source_url:
        return None
    return source_url if "/success/" in source_url else None


def playback_duration(episode: dict[str, Any]) -> float | None:
    trajectory_duration = episode.get("trajectory_duration_sec")
    if trajectory_duration is None and episode.get("trajectory_length"):
        trajectory_duration = float(episode["trajectory_length"]) / 15
    if not trajectory_duration:
        return None
    return round(min(float(trajectory_duration), MAX_VIDEO_DURATION_SEC), 3)


class Catalog:
    def __init__(self) -> None:
        if not BOOTSTRAP_PATH.is_file():
            raise RuntimeError(f"Bootstrap file is missing: {BOOTSTRAP_PATH}")

        self.bootstrap = read_json(BOOTSTRAP_PATH, {})
        self.video_index = read_json(VIDEO_INDEX_PATH, {}).get("setups", {})
        self.scenes = {
            row["scene_id"]: row for row in self.bootstrap.get("scene_groups", [])
        }
        self.setups = {
            row["setup_id"]: row for row in self.bootstrap.get("setups", [])
        }
        self.setup_assets: dict[str, list[dict[str, Any]]] = {}
        self.scene_asset_counts: dict[str, int] = {scene_id: 0 for scene_id in self.scenes}
        self.scene_labels: dict[str, set[str]] = {scene_id: set() for scene_id in self.scenes}
        self.completed_setup_count = 0
        self.asset_count = 0
        self.total_glb_bytes = 0
        self.missing_label_count = 0
        self.missing_setup_count = 0
        self._scan_outputs()

    def _scan_outputs(self) -> None:
        for setup_id, setup in self.setups.items():
            assets_root = SETUPS_ROOT / setup_id / "assets"
            assets: list[dict[str, Any]] = []
            if assets_root.is_dir():
                for asset_dir in sorted(assets_root.iterdir()):
                    glb = asset_dir / "object_000.glb"
                    if not asset_dir.is_dir() or not glb.is_file():
                        continue
                    label, instance = asset_parts(asset_dir.name)
                    size = glb.stat().st_size
                    assets.append(
                        {
                            "key": asset_dir.name,
                            "label": label,
                            "instance": instance,
                            "glb_bytes": size,
                        }
                    )
                    self.total_glb_bytes += size

            if assets:
                self.completed_setup_count += 1
            self.setup_assets[setup_id] = assets
            count = len(assets)
            self.asset_count += count
            scene_id = setup.get("scene_id")
            if scene_id in self.scene_asset_counts:
                self.scene_asset_counts[scene_id] += count
                self.scene_labels[scene_id].update(asset["label"] for asset in assets)
            actual_labels = {asset["label"] for asset in assets}
            missing = [
                label
                for label in dict.fromkeys(setup.get("objects", []))
                if label not in actual_labels
            ]
            self.missing_label_count += len(missing)
            self.missing_setup_count += bool(missing)

    def compact_scene(self, scene: dict[str, Any]) -> dict[str, Any]:
        scene_id = scene["scene_id"]
        setup_ids = scene.get("setup_ids", [])
        representative_setup = next(
            (setup_id for setup_id in setup_ids if self.setup_assets.get(setup_id)),
            setup_ids[0] if setup_ids else None,
        )
        return {
            "scene_id": scene_id,
            "title": scene.get("title") or scene.get("title_ko") or scene_id,
            "title_ko": scene.get("title_ko"),
            "setup_count": len(setup_ids),
            "episode_count": scene.get("episode_count", 0),
            "source_scene_count": scene.get("source_scene_count", 0),
            "asset_count": self.scene_asset_counts.get(scene_id, 0),
            "labels": sorted(self.scene_labels.get(scene_id, set())),
            "representative_setup_id": representative_setup,
            "representative_url": (
                f"/media/setups/{representative_setup}/source.jpg"
                if representative_setup
                else None
            ),
            "confidence": scene.get("confidence"),
            "classification_status": scene.get("classification_status"),
        }

    def compact_setup(self, setup: dict[str, Any]) -> dict[str, Any]:
        setup_id = setup["setup_id"]
        assets = self.setup_assets.get(setup_id, [])
        output_root = SETUPS_ROOT / setup_id
        representative = setup.get("representative") or {}
        video_episode = self.video_index.get(setup_id) or representative
        video_duration = video_episode.get("trajectory_duration_sec")
        if video_duration is None and video_episode.get("trajectory_length"):
            video_duration = round(float(video_episode["trajectory_length"]) / 15, 3)
        video_source_url = verified_success_url(video_episode)
        video_playback_duration = playback_duration(video_episode) if video_source_url else None
        video_playback_rate = (
            round(float(video_duration) / video_playback_duration, 2)
            if video_duration and video_playback_duration
            else None
        )
        video_version = (
            sha1(
                f"{VIDEO_POLICY_VERSION}|{video_source_url}|{video_duration}|{video_playback_duration}".encode(
                    "utf-8"
                )
            ).hexdigest()[:12]
            if video_source_url
            else None
        )
        actual_labels = {asset["label"] for asset in assets}
        missing = [
            label
            for label in dict.fromkeys(setup.get("objects", []))
            if label not in actual_labels
        ]
        return {
            "setup_id": setup_id,
            "scene_id": setup.get("scene_id"),
            "title": setup.get("title") or setup_id,
            "objects": setup.get("objects", []),
            "episode_count": setup.get("episode_count", 0),
            "source_scene_count": setup.get("source_scene_count", 0),
            "confidence": setup.get("confidence"),
            "audit_status": setup.get("audit_status"),
            "asset_count": len(assets),
            "labels": sorted({asset["label"] for asset in assets}),
            "missing_labels": missing,
            "has_source": (output_root / "source.jpg").is_file(),
            "has_detections": (output_root / "detections.jpg").is_file(),
            "source_url": f"/media/setups/{setup_id}/source.jpg",
            "detections_url": f"/media/setups/{setup_id}/detections.jpg",
            "video_url": (
                f"/media/setups/{setup_id}/trajectory.mp4?v={video_version}"
                if video_source_url
                else None
            ),
            "raw_video_url": video_source_url,
            "video_camera": video_episode.get("preferred_video_camera"),
            "video_success": True if video_source_url else None,
            "video_duration_sec": video_playback_duration,
            "video_trajectory_duration_sec": video_duration,
            "video_playback_rate": video_playback_rate,
            "video_trajectory_length": video_episode.get("trajectory_length"),
            "video_task": video_episode.get("task") or video_episode.get("current_task"),
            "video_episode_uuid": video_episode.get("uuid"),
            "video_selection_rule": video_episode.get("selection_rule"),
            "video_unavailable_reason": (
                None if video_source_url else "no_verified_success_episode"
            ),
            "video_urls": {
                "ext1": video_episode.get("ext1_url"),
                "ext2": video_episode.get("ext2_url"),
                "wrist": video_episode.get("wrist_url"),
            },
        }

    def video_episode(self, setup_id: str) -> dict[str, Any]:
        setup = self.setups[setup_id]
        return self.video_index.get(setup_id) or setup.get("representative") or {}


catalog = Catalog()
app = FastAPI(title="SAM 3D DROID Asset Catalog v3", version="3.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "version": app.version, "catalog_root": str(CATALOG_ROOT)}


@app.get("/api/summary")
def summary() -> dict[str, Any]:
    source_summary = catalog.bootstrap.get("summary", {})
    return {
        "generated_at": catalog.bootstrap.get("generated_at"),
        "scene_count": len(catalog.scenes),
        "setup_count": len(catalog.setups),
        "completed_setup_count": catalog.completed_setup_count,
        "empty_output_setup_count": len(catalog.setups) - catalog.completed_setup_count,
        "asset_count": catalog.asset_count,
        "missing_label_count": catalog.missing_label_count,
        "missing_setup_count": catalog.missing_setup_count,
        "total_glb_bytes": catalog.total_glb_bytes,
        "episode_count": source_summary.get("episode_count", 0),
        "source_scene_count": source_summary.get("source_scene_count", 0),
    }


@app.post("/api/refresh")
def refresh_catalog() -> dict[str, Any]:
    global catalog
    catalog = Catalog()
    return {
        "ok": True,
        "asset_count": catalog.asset_count,
        "missing_label_count": catalog.missing_label_count,
    }


@app.get("/api/scenes")
def list_scenes(
    q: str = "",
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
) -> dict[str, Any]:
    needle = q.strip().casefold()
    rows = []
    for scene in catalog.scenes.values():
        compact = catalog.compact_scene(scene)
        haystack = " ".join(
            [
                compact["scene_id"],
                compact["title"] or "",
                compact["title_ko"] or "",
                " ".join(scene.get("objects", [])),
                " ".join(compact["labels"]),
            ]
        ).casefold()
        if not needle or needle in haystack:
            rows.append(compact)
    rows.sort(key=lambda row: (-row["asset_count"], row["scene_id"]))
    return {"total": len(rows), "offset": offset, "items": rows[offset : offset + limit]}


@app.get("/api/scenes/{scene_id}")
def get_scene(scene_id: str) -> dict[str, Any]:
    scene = catalog.scenes.get(scene_id)
    if scene is None:
        raise HTTPException(404, f"Unknown scene: {scene_id}")
    setup_rows = [
        catalog.compact_setup(catalog.setups[setup_id])
        for setup_id in scene.get("setup_ids", [])
        if setup_id in catalog.setups
    ]
    setup_rows.sort(key=lambda row: row["setup_id"])
    return {
        **catalog.compact_scene(scene),
        "description": scene.get("description"),
        "reason": scene.get("reason"),
        "fixed_environment": scene.get("fixed_environment"),
        "objects": scene.get("objects", []),
        "setups": setup_rows,
    }


@app.get("/api/setups/{setup_id}")
def get_setup(setup_id: str) -> dict[str, Any]:
    setup = catalog.setups.get(setup_id)
    if setup is None:
        raise HTTPException(404, f"Unknown setup: {setup_id}")

    asset_rows = []
    for indexed in catalog.setup_assets.get(setup_id, []):
        asset_root = SETUPS_ROOT / setup_id / "assets" / indexed["key"]
        result = read_json(asset_root / "result.json", {})
        detection = result.get("detection") or read_json(asset_root / "detection.json", {})
        prefix = f"/media/setups/{setup_id}/assets/{indexed['key']}"
        asset_rows.append(
            {
                **indexed,
                "score": detection.get("score"),
                "mask_score": detection.get("mask_score"),
                "area_ratio": detection.get("area_ratio"),
                "box_xyxy": detection.get("box_xyxy"),
                "transform": result.get("sam3d_transform"),
                "glb_url": f"{prefix}/object_000.glb",
                "ply_url": f"{prefix}/object_000.ply",
                "mask_url": f"{prefix}/mask.png",
                "result_url": f"{prefix}/result.json",
            }
        )

    return {
        **catalog.compact_setup(setup),
        "reason": setup.get("reason"),
        "raw_scene_key": setup.get("raw_scene_key"),
        "representative": setup.get("representative"),
        "assets": asset_rows,
    }


@app.get("/api/integration-map/{scene_id}")
def integration_map(scene_id: str) -> dict[str, Any]:
    scene = catalog.scenes.get(scene_id)
    if scene is None:
        raise HTTPException(404, f"Unknown scene: {scene_id}")
    return {
        "scene_id": scene_id,
        "asset_count": catalog.scene_asset_counts.get(scene_id, 0),
        "catalog_url": f"/?scene={scene_id}",
        "setups": [
            {
                "setup_id": setup_id,
                "asset_count": len(catalog.setup_assets.get(setup_id, [])),
                "catalog_url": f"/?scene={scene_id}&setup={setup_id}",
            }
            for setup_id in scene.get("setup_ids", [])
            if setup_id in catalog.setups
        ],
    }


def cache_lock(setup_id: str) -> threading.Lock:
    with video_cache_locks_guard:
        return video_cache_locks.setdefault(setup_id, threading.Lock())


def trajectory_time_scale(source_url: str, episode: dict[str, Any]) -> float:
    target_duration = playback_duration(episode)
    if not target_duration:
        return 1.0
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                source_url,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        media_duration = float(probe.stdout.strip())
        scale = float(target_duration) / media_duration
        return min(8.0, max(0.001, scale))
    except (OSError, ValueError, subprocess.SubprocessError):
        raise HTTPException(502, "Unable to determine source video duration")


@app.get("/media/setups/{setup_id}/trajectory.mp4")
def normalized_trajectory(setup_id: str) -> FileResponse:
    if setup_id not in catalog.setups:
        raise HTTPException(404, f"Unknown setup: {setup_id}")
    episode = catalog.video_episode(setup_id)
    source_url = verified_success_url(episode)
    if not source_url:
        raise HTTPException(404, "No verified successful trajectory video is available")

    cache_dir = VIDEO_CACHE_ROOT / setup_id
    cache_identity = (
        f"{VIDEO_POLICY_VERSION}|{source_url}|{episode.get('trajectory_duration_sec')}|"
        f"{episode.get('trajectory_length')}|{playback_duration(episode)}"
    )
    cache_key = sha1(cache_identity.encode("utf-8")).hexdigest()[:12]
    cache_path = cache_dir / f"trajectory-{cache_key}.mp4"
    with cache_lock(setup_id):
        if not cache_path.is_file() or cache_path.stat().st_size == 0:
            cache_dir.mkdir(parents=True, exist_ok=True)
            temp_path = cache_dir / "trajectory.mp4.tmp"
            temp_path.unlink(missing_ok=True)
            source_path: str | None = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as source_file:
                    source_path = source_file.name
                urllib.request.urlretrieve(source_url, source_path)
                time_scale = trajectory_time_scale(source_path, episode)
                command = [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-itsscale",
                    f"{time_scale:.9f}",
                    "-i",
                    source_path,
                    "-map",
                    "0:v:0",
                    "-an",
                    "-c",
                    "copy",
                    "-movflags",
                    "+faststart",
                    "-f",
                    "mp4",
                    str(temp_path),
                ]
                subprocess.run(command, check=True, timeout=180)
                if temp_path.stat().st_size == 0:
                    raise RuntimeError("ffmpeg wrote an empty video")
                temp_path.replace(cache_path)
            except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
                temp_path.unlink(missing_ok=True)
                raise HTTPException(502, f"Unable to prepare trajectory video: {exc}") from exc
            finally:
                if source_path:
                    Path(source_path).unlink(missing_ok=True)

    return FileResponse(
        cache_path,
        media_type="video/mp4",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/media/setups/{setup_id}/{relative_path:path}")
def setup_media(setup_id: str, relative_path: str) -> FileResponse:
    if setup_id not in catalog.setups:
        raise HTTPException(404, f"Unknown setup: {setup_id}")
    setup_root = (SETUPS_ROOT / setup_id).resolve()
    target = (setup_root / relative_path).resolve()
    if setup_root not in target.parents or not target.is_file():
        raise HTTPException(404, "Asset file not found")
    return FileResponse(target, filename=None)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_ROOT / "index.html")
