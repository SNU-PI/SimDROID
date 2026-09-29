# DROID Scene Review

Static review site for the full DROID Scene -> Setup -> Episode classification.
It has no backend.

## Run locally

From the SimDROID repository root:

```bash
python3 -m http.server 8142 --directory droid_scene_review
```

Open <http://127.0.0.1:8142/>.

Do not open `index.html` through `file://`; the browser must fetch the
paginated JSON files over HTTP.

## Included

- 560 Scene records, 2,455 Setup records, and 2,229 Source Scene records
- Lazy metadata for all 74,896 Episodes
- Top-100 Scene target-asset inventory covering 593 Setups
- 73,732 generated Episode screenshots used by the UI
- Static HTML, CSS, and JavaScript

## Not included

- Original DROID MP4 files or dataset archives
- Pipeline caches, contact sheets, model features, or intermediate review files
- The 293 MB monolithic `review_data.json` build artifact
- Historical UI versions

Episode video buttons retain the original remote URLs stored in the metadata,
so video playback/download requires network access to the DROID data host.

## Data layout

- `review_bootstrap.json`: first 24 Scenes and loading manifest
- `data/scene_pages/`: default Scene pagination
- `data/scene_setups/`: per-Scene Setup details
- `data/setup_index.json`: lazy Setup tab index
- `data/source_index.json`: lazy Source Scene tab index
- `data/source_episodes/`: per-Source-Scene Episode metadata
- `assets/episode_screenshots/`: generated JPEG screenshots

The screenshot directory is tracked through Git LFS because it contains about
1.1 GB across 73,732 small files.
