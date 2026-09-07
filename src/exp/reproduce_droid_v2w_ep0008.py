"""Reproduce the five published DROID/Cosmos examples through episode 0008."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FPS = 15
CONDITION_FRAMES = 49
FUTURE_FRAMES = 16
HEIGHT = 480
WIDTH = 832
STEPS = 20
GUIDANCE_SCALE = 7.0
DEFAULT_HF_REVISION = "68c2b16b5c9dfef9949868c6fff6393c9b5fa941"
VIEWS = {
    "ext1": "exterior_1_left",
    "ext2": "exterior_2_left",
    "wrist": "wrist_left",
}
NEGATIVE_PROMPT = (
    "blurry, low resolution, abrupt camera jitter, discontinuous motion, unintended scene cut, morphing robot, "
    "malformed gripper, disappearing objects, duplicate objects, implausible contact, text, watermark"
)


class LocalBenignDataSafetyChecker:
    def to(self, *args, **kwargs):
        return self

    def check_text_safety(self, prompt):
        return True

    def check_video_safety(self, video):
        return video


def parse_args():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--droid-root",
        type=Path,
        help="Optional full DROID checkout. The HF mini-dataset is used by default.",
    )
    parser.add_argument("--hf-repo", default="Parkprogrammer/droid-v2w-ep0008")
    parser.add_argument("--hf-revision", default=DEFAULT_HF_REVISION)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs/droid_v2w_ep0001_ep0008.json",
    )
    parser.add_argument("--ids", nargs="*", help="Optional config ids to run.")
    parser.add_argument("--dry-run", action="store_true", help="Validate data selection without loading Cosmos.")
    return parser.parse_args()


def motion_windows(cart: np.ndarray, grip: np.ndarray) -> dict[str, int]:
    low = CONDITION_FRAMES - 1
    high = len(cart) - FUTURE_FRAMES - 1
    if high < low:
        raise ValueError(f"Trajectory with {len(cart)} frames is too short.")
    span = high - low
    translation = np.linalg.norm(np.diff(cart[:, :3], axis=0), axis=1)
    rotation = np.linalg.norm(np.diff(np.unwrap(cart[:, 3:], axis=0), axis=0), axis=1)
    gripper = np.abs(np.diff(grip))
    motion = translation + 0.02 * rotation + 0.08 * gripper
    smooth = np.convolve(motion, np.ones(9, dtype=np.float32) / 9, mode="same")
    return {
        "approach": low + round(0.18 * span),
        "interaction": int(np.argmax(smooth[low : high + 1]) + low),
        "late": low + round(0.82 * span),
    }


def letterbox(frame: np.ndarray) -> Image.Image:
    image = Image.fromarray(np.asarray(frame).astype(np.uint8)).convert("RGB")
    scale = min(WIDTH / image.width, HEIGHT / image.height)
    resized = image.resize(
        (round(image.width * scale), round(image.height * scale)),
        Image.Resampling.LANCZOS,
    )
    border = tuple(np.asarray(image.resize((1, 1))).reshape(-1).astype(int))
    canvas = Image.new("RGB", (WIDTH, HEIGHT), border)
    canvas.paste(resized, ((WIDTH - resized.width) // 2, (HEIGHT - resized.height) // 2))
    return canvas


def read_view(droid_root: Path, meta, view: str, indices: np.ndarray) -> np.ndarray:
    full_view = VIEWS[view]
    file_index = int(meta[f"videos/observation.images.{full_view}/file_index"])
    first_frame = round(float(meta[f"videos/observation.images.{full_view}/from_timestamp"]) * FPS)
    video = droid_root / "videos" / f"observation.images.{full_view}" / "chunk-000" / f"file-{file_index:03d}.mp4"
    if not video.is_file():
        raise FileNotFoundError(video)
    reader = imageio.get_reader(video)
    try:
        return np.stack([reader.get_data(first_frame + int(index)) for index in indices])
    finally:
        reader.close()


def load_droid_clip(droid_root: Path, rows, meta, job: dict):
    episode_rows = rows[rows.episode_index == job["episode"]].sort_values("frame_index")
    if episode_rows.empty:
        raise ValueError(f"Episode {job['episode']} is absent from file-000.parquet.")
    task = str(episode_rows.language_instruction.iloc[0]).strip()
    if task != job["task"]:
        raise ValueError(f"Task mismatch for {job['id']}: {task!r}")
    cart = np.stack(episode_rows["observation.state.cartesian_position"].to_numpy()).astype(np.float32)
    grip = np.asarray(
        [np.asarray(value).item() for value in episode_rows["observation.state.gripper_position"]],
        dtype=np.float32,
    )
    end = motion_windows(cart, grip)[job["window"]]
    indices = np.arange(end - CONDITION_FRAMES + 1, end + FUTURE_FRAMES + 1)
    input_frames = read_view(droid_root, meta, job["input_view"], indices)
    if job["input_view"] == job["output_view"]:
        output_frames = input_frames
    else:
        output_frames = read_view(droid_root, meta, job["output_view"], indices)
    return input_frames[:CONDITION_FRAMES], output_frames[CONDITION_FRAMES:], indices


def load_hf_clip(args, job: dict):
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=args.hf_repo,
        filename=job["clip"],
        repo_type="dataset",
        revision=args.hf_revision,
        cache_dir=args.cache_dir,
    )
    with np.load(path, allow_pickle=False) as data:
        condition = np.asarray(data["condition"])
        target = np.asarray(data["target"])
        indices = np.asarray(data["source_frame_indices"])
        metadata = {
            "episode": int(data["episode"]),
            "window": str(data["window"]),
            "input_view": str(data["input_view"]),
            "output_view": str(data["output_view"]),
            "task": str(data["task"]),
            "fps": int(data["fps"]),
        }
    expected = {key: job[key] for key in ("episode", "window", "input_view", "output_view", "task")}
    expected["fps"] = FPS
    if metadata != expected:
        raise ValueError(f"HF clip metadata mismatch for {job['id']}: {metadata!r}")
    if len(condition) != CONDITION_FRAMES or len(target) != FUTURE_FRAMES:
        raise ValueError(f"HF clip has invalid frame counts for {job['id']}.")
    return condition, target, indices


def to_uint8(frames) -> np.ndarray:
    if isinstance(frames, list):
        return np.stack([np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in frames])
    array = np.asarray(frames)
    if array.dtype != np.uint8:
        array = np.clip(array * 255.0 if array.max() <= 1.5 else array, 0, 255).astype(np.uint8)
    return array


def labeled_frame(observed, target, prediction, view: str):
    size = (416, 240)
    header = 42
    canvas = Image.new("RGB", (size[0] * 3, size[1] + header), (21, 23, 27))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    labels = (f"OBSERVED {view} (49f)", "DROID FUTURE GT", "COSMOS V2W")
    for column, (label, frame) in enumerate(zip(labels, (observed, target, prediction))):
        image = Image.fromarray(frame).resize(size, Image.Resampling.BILINEAR)
        x = column * size[0]
        canvas.paste(image, (x, header))
        box = draw.textbbox((0, 0), label, font=font)
        draw.text((x + (size[0] - box[2] + box[0]) / 2, 11), label, fill="white", font=font)
    return np.asarray(canvas)


def save_outputs(output_dir: Path, job: dict, condition: np.ndarray, target: np.ndarray, prediction: np.ndarray, indices):
    sample_dir = output_dir / job["id"]
    sample_dir.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(sample_dir / "prediction.mp4", prediction, fps=FPS, codec="libx264", quality=8)
    comparison = []
    for index in range(FUTURE_FRAMES):
        observed_index = min(index * CONDITION_FRAMES // FUTURE_FRAMES, CONDITION_FRAMES - 1)
        comparison.append(
            labeled_frame(
                np.asarray(letterbox(condition[observed_index])),
                np.asarray(letterbox(target[index])),
                prediction[index],
                job["input_view"],
            )
        )
    imageio.mimsave(sample_dir / "comparison.gif", comparison, duration=180, loop=0)
    metadata = {
        **job,
        "condition_frames": CONDITION_FRAMES,
        "future_frames": FUTURE_FRAMES,
        "fps": FPS,
        "steps": STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "height": HEIGHT,
        "width": WIDTH,
        "negative_prompt": NEGATIVE_PROMPT,
        "source_frame_indices": indices.tolist(),
    }
    (sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_pipeline(args):
    import diffusers.pipelines.cosmos.pipeline_cosmos2_video2world as cosmos_module
    import torch
    from diffusers import Cosmos2VideoToWorldPipeline
    from diffusers.models.transformers.transformer_cosmos import CosmosTransformer3DModel

    cosmos_module.CosmosSafetyChecker = LocalBenignDataSafetyChecker
    transformer = CosmosTransformer3DModel.from_single_file(
        str(args.original_checkpoint),
        config=str(args.model_dir / "transformer"),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    pipeline = Cosmos2VideoToWorldPipeline.from_pretrained(
        str(args.model_dir),
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def main():
    args = parse_args()
    jobs = json.loads(args.config.read_text())
    selected = set(args.ids or [])
    if selected:
        jobs = [job for job in jobs if job["id"] in selected]
    unknown = selected - {job["id"] for job in jobs}
    if unknown:
        raise ValueError(f"Unknown ids: {sorted(unknown)}")
    if args.droid_root is not None:
        import pandas as pd

        data = pd.read_parquet(args.droid_root / "data/chunk-000/file-000.parquet")
        episode_meta = pd.read_parquet(
            args.droid_root / "meta/episodes/chunk-000/file-000.parquet"
        ).set_index("episode_index")
    prepared = []
    for job in jobs:
        if args.droid_root is None:
            clip = load_hf_clip(args, job)
            job = {**job, "hf_dataset": args.hf_repo, "hf_revision": args.hf_revision}
        else:
            clip = load_droid_clip(args.droid_root, data, episode_meta.loc[job["episode"]], job)
        prepared.append((job, *clip))
        print(f"prepared {job['id']} frames={clip[2][0]}..{clip[2][-1]}", flush=True)
    if args.dry_run:
        return

    import torch

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = load_pipeline(args)
    for position, (job, condition_raw, target_raw, indices) in enumerate(prepared, 1):
        condition = [letterbox(frame) for frame in condition_raw]
        generated = pipeline(
            prompt=job["prompt"],
            negative_prompt=NEGATIVE_PROMPT,
            video=condition,
            height=HEIGHT,
            width=WIDTH,
            num_frames=CONDITION_FRAMES + FUTURE_FRAMES,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE_SCALE,
            fps=FPS,
            generator=torch.Generator(device="cuda").manual_seed(job["seed"]),
            output_type="pil",
        ).frames[0]
        prediction = to_uint8(generated)[CONDITION_FRAMES:]
        if len(prediction) != FUTURE_FRAMES:
            raise RuntimeError(f"Expected {FUTURE_FRAMES} future frames, received {len(prediction)}.")
        save_outputs(args.output_dir, job, condition_raw, target_raw, prediction, indices)
        print(f"[{position}/{len(prepared)}] completed {job['id']}", flush=True)


if __name__ == "__main__":
    main()
