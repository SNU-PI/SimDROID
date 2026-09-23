# SAM 3D asset catalog

Browser and JSON API for reviewing generated DROID SAM 3D object assets. The
server is GPU-free: it reads completed GLB, PLY, mask, detection, and metadata
files from a catalog directory and renders them with Three.js.

## Data layout

Set `SAM3D_CATALOG_ROOT` to a directory containing:

```text
review_bootstrap.json
setup_video_index.json          # optional
setups/
  SETUP-FULL-00001/
    source.jpg
    detections.jpg
    assets/
      object_label__00/
        object_000.glb
        object_000.ply
        mask.png
        detection.json
        result.json
```

Without the environment variable, the server reads
`data/sam3d_assets` from the repository root. Generated data is intentionally
excluded from Git.

## Run

Install the Python dependencies and make `ffmpeg` and `ffprobe` available on
`PATH` if trajectory playback is needed:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r tools/sam3d_asset_catalog/requirements.txt

export SAM3D_CATALOG_ROOT=/path/to/sam3d_assets
tools/sam3d_asset_catalog/run.sh
```

The default address is `http://localhost:8095`. Override it with
`SAM3D_CATALOG_HOST` and `SAM3D_CATALOG_PORT`. Deep links preserve catalog IDs:

```text
/?scene=SCENE-FULL-0281&setup=SETUP-FULL-01379&asset=banana__00
```

Useful endpoints:

- `GET /api/health`
- `GET /api/summary`
- `GET /api/scenes?q=banana`
- `GET /api/scenes/{scene_id}`
- `GET /api/setups/{setup_id}`
- `GET /api/integration-map/{scene_id}`
- `POST /api/refresh`
- `GET /media/setups/{setup_id}/trajectory.mp4`

Scene search results are ordered by generated asset count in descending order,
then by scene ID.

The browser bundles Three.js r167 under `static/vendor/` for offline use. The
vendored files retain their MIT license headers.

## Source trajectory index

`build_success_video_index.py` builds `setup_video_index.json` from public DROID
raw metadata. Run it in the DROID metadata environment that provides
`daft.datasets.droid`:

```bash
python tools/sam3d_asset_catalog/build_success_video_index.py \
  --catalog-root /path/to/sam3d_assets
```

The index keeps existing successful representatives. Otherwise, it chooses the
longest success from a single-setup source scene or the chronologically nearest
success when a source scene contains multiple setups. Failure trajectories are
not used as playback fallbacks.

The media endpoint preserves every source frame and remuxes timestamps so the
full trajectory plays in at most 30 seconds. Normalized videos are cached under
`video_cache/` in the catalog directory.
