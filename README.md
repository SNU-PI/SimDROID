# cosmos-vwm-physics-thresholds

Controlled intuitive-physics evaluation for **Cosmos Predict2 Video2World**:
from pixels only, predict which side of a physical threshold a MuJoCo scene
lands on. Inference only; no weights are trained.

Six environments separate static configuration reading from dynamic state
estimation:

| regime | environment | threshold outcome |
|---|---|---|
| static | Seesaw | tips left / right from visible torque balance |
| static | Lean | holds / slides from rod angle and fixed contact friction |
| static | Tower | stands / falls from cumulative center of mass |
| dynamic | Hill | returns / crosses from visible velocity and hill height |
| dynamic | Collision | separates / bounces after impact |
| dynamic | Domino | propagation stops / reaches the final domino |

Official 480p/16 FPS `Cosmos-Predict2-2B-Video2World`. Static scenes get one
image, dynamic scenes five strictly pre-outcome frames; endpoints are compared
with MuJoCo at a shared +0.3125 s horizon.

## Layout

```text
src/core/
  threshold/                          six MuJoCo environments and shared rollout base

src/gen/
  render.py                            shared EGL/render/video/preview helpers
  sweep_spec.py                        sweep ranges, cameras, prompts, and margins
  make_sweep.py                        seven margins × six environments
  make_diagnostics.py                  native 832×480, exact-16fps Tower/Hill set
  make_gifs.py                         visual physics smoke tests
  make_policy_inputs.py                pixel-conditioning previews
  energy_spec.py                       shared normalized-energy margin design
  make_energy_sweep.py                 matched Hill/Ramp/Pendulum VWM inputs
  make_energy_gifs.py                  energy-family physics smoke tests

src/exp/
  cosmos_v2w_sweep.py                  six-environment Predict2 V2W inference
  collect_v2w_diagnostics.py           resumable multi-seed corrected collection
  analyze_physics_sweeps.py            GT-calibrated outcome decoding
  analyze_corrected_diagnostics.py     threshold, seed, and intervention statistics
  vae_reconstruction_diagnostics.py    frozen-VAE reconstruction ceiling
  verify_diagnostic_collection.py      MP4/PNG/metadata integrity checks
  prioritize_diagnostic_jobs.py        job ordering for the time-limited run
  render_v2w_triplet_gifs.py           LAST / PHYSICS GT / COSMOS GIFs
  render_energy_v2w_triplet_gifs.py    energy-family comparison GIFs
  render_observed_history_gifs.py      exact one- or five-frame model inputs
  render_diagnostic_comparison_gif.py  corrected Tower/Hill GIFs
  render_intuitive_v2w_gifs.py         sweep-wide qualitative GIFs

artifacts/                              generated data and media; never committed
```

`cosmos_policy_pixels.py` and `cosmos_policy_sweep.py` retain the initial
Cosmos-Policy comparison baseline. They are not the main VWM experiment.

## Environment

Python 3.11+, MuJoCo 3.11, PyTorch/CUDA, Diffusers with
`Cosmos2VideoToWorldPipeline`, NumPy, Pillow, imageio/ffmpeg, matplotlib, and
the Hugging Face Hub client and Cosmos Predict2 2B Video2World checkpoint.

Run MuJoCo EGL rendering and the PyTorch runner in **separate processes**;
sharing one corrupts the encoder output silently. Data goes to `artifacts/`.

## Reproduce

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONPATH=src \
python src/gen/make_sweep.py \
  --stats /path/to/libero_dataset_statistics.json \
  --output-dir artifacts/physics_sweep

MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONPATH=src \
python src/gen/make_gifs.py \
  --output-dir artifacts/threshold_gifs
```

Run the model:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
python src/exp/cosmos_v2w_sweep.py \
  --model-dir /path/to/Cosmos-Predict2-2B-Video2World \
  --original-checkpoint /path/to/model-480p-16fps.pt \
  --manifest artifacts/physics_sweep/manifest.jsonl \
  --sweep-root artifacts/physics_sweep \
  --output-dir artifacts/physics_sweep/cosmos_v2w
```

Analyze and render:

```bash
PYTHONPATH=src python src/exp/analyze_physics_sweeps.py \
  --sweep-root artifacts/physics_sweep

MUJOCO_GL=egl PYTHONPATH=src python src/exp/render_v2w_triplet_gifs.py \
  --sweep-root artifacts/physics_sweep
```

The root wrappers `run_sweep.sh` and `run_vwm.sh` provide the same entry
points with fewer arguments.

## Energy-conservation family

The first principle-preserving benchmark uses a conservative bead-on-hill,
frictionless ramp, and rigid pendulum. All three share the dimensionless
margin `available_energy / required_energy - 1`; the model receives only five
pixel frames and the prompt, never the margin.

```bash
MUJOCO_EGL_DEVICE_ID=0 PYTHONPATH=src \
python src/gen/make_energy_sweep.py \
  --output-dir artifacts/energy_conservation_sweep

MUJOCO_EGL_DEVICE_ID=0 PYTHONPATH=src \
python src/gen/make_energy_gifs.py \
  --output-dir artifacts/energy_conservation_gifs

PYTHONPATH=src python src/exp/render_observed_history_gifs.py \
  --artifacts-root artifacts
```

The manifest contains eight matched margins per scene, exact 16 FPS 832x480
conditioning videos, MuJoCo endpoints at +0.3125 seconds, and an
analytic-versus-simulation outcome gate.

## DROID Video2World examples

`src/exp/reproduce_droid_v2w_ep0008.py` reproduces the five gallery examples
through episode 0008. By default it downloads the exact 49+16 frame clips from
[`Parkprogrammer/droid-v2w-ep0008`](https://huggingface.co/datasets/Parkprogrammer/droid-v2w-ep0008)
at pinned revision `68c2b16` and runs the prompts and seeds in
`configs/droid_v2w_ep0001_ep0008.json`.

| example | window | view | seed |
|---|---|---|---:|
| ep0001 | approach | ext1 → ext1 | 1 |
| ep0002 | late | ext2 → ext2 | 1 |
| ep0003 | interaction | ext1 → ext1 | 3 |
| ep0007 | interaction | wrist → wrist | 1 |
| ep0008 | interaction | ext2 → ext2 | 2 |

Each run supplies 49 observed frames and generates 16 future frames at
832×480 and 15 FPS with 20 denoising steps and guidance scale 7.0.

### 1. Clone and install

```bash
git clone --branch feat/probe-physics-vwm https://github.com/SNU-PI/SimDROID.git
cd SimDROID

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install \
  diffusers==0.36.0 transformers==5.16.1 huggingface_hub==1.29.0 \
  numpy==2.2.6 pillow==12.0.0 imageio==2.37.2 imageio-ffmpeg==0.6.0 \
  accelerate safetensors
```

These are the versions used for the recorded run. The default HF mini-dataset
path does not require pandas, PyArrow, or a full DROID checkout.

### 2. Download the model

Accept access to
[`nvidia/Cosmos-Predict2-2B-Video2World`](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World),
then download the repository. The five DROID clips are public and need no HF
login; authentication is required only for this gated model.

```bash
hf auth login
hf download nvidia/Cosmos-Predict2-2B-Video2World \
  --local-dir checkpoints/Cosmos-Predict2-2B-Video2World
```

The resulting directory must contain both the Diffusers component folders and
`model-480p-16fps.pt`.

### 3. Verify the inputs without a GPU

This downloads the five pinned clips and checks their frame counts and
metadata without loading Cosmos:

```bash
python src/exp/reproduce_droid_v2w_ep0008.py \
  --model-dir checkpoints/Cosmos-Predict2-2B-Video2World \
  --original-checkpoint checkpoints/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt \
  --output-dir artifacts/droid_v2w_ep0008 \
  --dry-run
```

### 4. Run inference

```bash
CUDA_VISIBLE_DEVICES=0 python src/exp/reproduce_droid_v2w_ep0008.py \
  --model-dir checkpoints/Cosmos-Predict2-2B-Video2World \
  --original-checkpoint checkpoints/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt \
  --output-dir artifacts/droid_v2w_ep0008
```

The model is loaded once and the five examples run sequentially. To run only
one example, append:

```bash
--ids ep0001_approach_ext1_to_ext1_s1
```

Each example writes the following files under
`artifacts/droid_v2w_ep0008/<example-id>/`:

- `prediction.mp4`: the 16 generated future frames
- `comparison.gif`: observed input / synchronized DROID GT / Cosmos output
- `metadata.json`: prompt, seed, source revision, and inference settings

The mini-dataset contains five selected examples, not eight episodes. To
rebuild the same windows from a full `lerobot/droid_1.0.1` checkout instead,
install `pandas` and `pyarrow`, then pass
`--droid-root /path/to/droid_1.0.1`; this bypasses the HF clip download.

## Result

The corrected run removes the earlier timing and aspect confounds: native
832×480, exact 16 FPS sampling, five pre-event Hill frames, paired
base/oracle prompts, a Hill image-only ablation, and frozen-VAE plus file
integrity checks. Three hours produced 535 rollouts (213 base, 213 oracle,
109 image-only), every video decoding as 832×480, 16 FPS, 21 frames.

| condition | accuracy |
|---|---|
| majority baseline | 57.1% |
| base Tower | 44.2% |
| base Hill | 46.8% |
| base → oracle prompt (paired) | 45.5% → 54.0%, McNemar p=0.004 |
| base → Hill image-only | 90.9% of decisions flip, p=0.34 |

Base Tower nearly always answers **stable**, base Hill **crosses**, image-only
Hill **fails**. Conditioning moves the generated outcome, but the model does
not recover the analytic threshold.

Outcome decoders are calibrated on MuJoCo endpoints only and then frozen.
Tower is read from red/blue/grey block displacement, Hill from red-ball
horizontal displacement; object loss and morphing count as validity failures
rather than physics decisions.
