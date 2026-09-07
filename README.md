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

## feat/commonsense — scene library, frozen readouts, and heatmaps (2026-09-07)

This branch is a **superset of `feat/probe-physics-vwm`** (linear history; a fast-forward
from its tip). Every command in the sections below still works unchanged. It adds three
things: a larger MuJoCo scene library, frozen pixel readouts with paired counterfactual
statistics, and an attribution (heatmap) track for Cosmos-Predict2-2B Video2World.

### What is added

```text
src/core/threshold/
  base.py                     Base.render(segment=True) -> MuJoCo segmentation ids; run(..., segment=True)
  workbench.py                Franka workbench dressing shared by the Phase A/B/PVR scenes (MENAGERIE_PANDA)
  rolling_hill.py two_ball.py rod_pendulum.py      Bundle A threshold scenes on a shared saturation axis S
  kin_roll.py                 P0 kinematic control (flat rolling)
  support_edge.py             Phase B: support-edge departure (contact loss)
  hill_decoy.py two_ball_decoy.py support_edge_decoy.py
                              PVR scenes: same physics + a look-alike decoy behind the ball,
                              fovy-26 camera, threshold extensions (wall / flat / joint / step / tilt)

src/gen/
  bundle_a_spec.py phase_a_spec.py phase_b_spec.py pvr_spec.py   condition designs (S ladders, 2x2 edits, seeds)
  make_bundle_a.py make_phase_a.py make_phase_b.py make_pvr.py   generators: inputs npz + GT mp4 + manifest (+ seg npz)
  make_vjepa_inputs.py make_vera_inputs.py vjepa_spec.py         inputs for the latent tracks (V-JEPA 2-AC, VERA)
  render.py                   roll(..., segment=True) returns per-frame segmentation + geom/body names

src/exp/
  cosmos_v2w_sweep.py         --seeds / --output-root: many seeds per process (model loads once)
  analyze_bundle_a.py analyze_pvr.py verify_*.py               frozen adjudicators, GT self-tests, paired effects, RI
  pvr_regions.py              segmentation -> regions (ball / relevant structure / decoy / support / franka / clutter / background)
  cosmos_attrib.py            heatmap track: single-step x0 gradient + attention routing + sanity checks + aggregate
  pvr_heatmap_figs.py         seed-averaged overlay figures of the stored maps
  build_pvr_report.py build_pvr_explainer.py                    result / explainer pages (self-contained HTML)
  rollout_vjepa_ac.py rollout_dila.py analyze_vjepa_ac.py ...   latent tracks (see the dated sections below)

run_bundle_a_vwm.sh run_phase_a_vwm.sh run_phase_b_vwm.sh run_pvr_vwm.sh   Cosmos collection wrappers
run_vjepa_ac.sh run_dila.sh                                                 latent-track chains
```

### Environments and variables

Two Python environments are used, in separate processes:

| env | used for | needs |
|---|---|---|
| MuJoCo env | scene generation, verification, readouts, pages | mujoco >= 3.1, numpy, scipy, imageio(+ffmpeg), matplotlib, Pillow; OSMesa (`MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`, `apt-get install libosmesa6`) or EGL |
| Cosmos env | `cosmos_v2w_sweep.py`, `cosmos_attrib.py` | torch (CUDA), diffusers >= 0.35 with `Cosmos2VideoToWorldPipeline`, transformers, imageio, matplotlib |

Nothing in the repo hard-codes a machine path; the wrappers and page builders read these variables:

| variable | meaning | default |
|---|---|---|
| `COSMOS_SNAPSHOT` | directory of the `nvidia/Cosmos-Predict2-2B-Video2World` snapshot (contains `model-480p-16fps.pt`) | required by `run_*_vwm.sh` |
| `COSMOS_PY` | python of the Cosmos env | `python` |
| `HF_HOME`, `CUDA_VISIBLE_DEVICES` | as usual | `~/.cache/huggingface`, `0` |
| `MENAGERIE_PANDA` | `mujoco_menagerie/franka_emika_panda` directory (workbench scenes) | `data/stage0/mujoco_menagerie/franka_emika_panda` |
| `VWM_ARTIFACTS` | artifacts root for the page builders | `artifacts` |
| `VWM_EXTERNAL`, `DILA_DIR`, `VJEPA_DIR`, `VJEPA_CKPT` | latent-track assets | `external/...`, `data/stage0/vjepa2-ac-vitg.pt` |
| `SIMDROID_ENV`, `VERA_ENV`, `VERA_PY` | env prefixes for `run_vjepa_ac.sh` / `run_dila.sh` | required / `python` |
| `SIMDROID_CODE_SRC`, `SIMDROID_CODE_OUT`, `SIMDROID_PAIRS` | team-repo assets used only by `render_pusher_strips.py` and the overview page | `../code/src`, `../code/out`, `../data/episodes_dense/pairs` |

### Quick start: PVR (predictive visual reliance) PoC

Three scenes (Hill / Collide / Edge-Fall), each with a relevant edit (changes the true future),
a same-type decoy edit behind the ball (does not), no-decoy cells, a decision-variable ladder and
threshold extensions; 15 / 15 / 18 conditions x 12 seeds, the same seeds in every condition.

```bash
# 1. scenes -> artifacts/pvr/{hill,collide,edge}/ (inputs npz, seg npz, GT/condition mp4, manifest, previews)
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=src python src/gen/make_pvr.py
PYTHONPATH=src python src/exp/verify_pvr.py                 # physics identity vs the base classes, gates, mask IoU
PYTHONPATH=src python src/exp/analyze_pvr.py --gt-selftest-only

# 2. Cosmos rollouts (Cosmos env; one process per scene, all seeds inside)
export COSMOS_SNAPSHOT=/path/to/models--nvidia--Cosmos-Predict2-2B-Video2World/snapshots/<hash>
export COSMOS_PY=/path/to/cosmos_env/bin/python
./run_pvr_vwm.sh hill 1 2 3 4 5 6 7 8 9 10 11 12          # same for collide, edge

# 3. readout: continuous event scores, paired effects with bootstrap CIs, Relevance Index, curves, sheets
PYTHONPATH=src python src/exp/analyze_pvr.py --scenes hill collide edge

# 4. heatmaps (Cosmos env, GPU): gradient + attention for 6 conditions x 4 seeds, then aggregate
PYTHONPATH=src $COSMOS_PY src/exp/cosmos_attrib.py attribute --scenes hill \
  --model-dir $COSMOS_SNAPSHOT --original-checkpoint $COSMOS_SNAPSHOT/model-480p-16fps.pt \
  --conds A B C D XW XF --seeds 1 2 3 4 --steps 20 24 28 --wrong-target --randomize
PYTHONPATH=src $COSMOS_PY src/exp/cosmos_attrib.py aggregate --scenes hill

# 5. pages
PYTHONPATH=src python src/exp/build_pvr_report.py --out artifacts/pvr/report.html
PYTHONPATH=src python src/exp/build_pvr_explainer.py --out artifacts/pvr/explainer.html
```

The heatmap track re-implements the pipeline's sampling loop (`Replay`) so a step can be frozen
and differentiated: conditioning frames -> fp32 VAE encoder -> DiT (CFG) -> x0 prediction ->
latent ball probe (fitted once per scene from GT latents + segmentation) -> `|ds/dx|` over the five
conditioning frames; attention routing re-computes softmax(QK^T) for the predicted-frame queries
in every block. Maps are aggregated by segmentation region with area-normalised densities, and
checked with an irrelevant-scalar control, last-k-block randomisation, seed CV and a Sobel baseline.
Read the sanity checks before the maps: in our runs the raw single-step gradient correlated
0.64-0.85 with the irrelevant-scalar map and did not localise the relevant structure; the
counterfactual (paired-edit) statistics from step 3 are the primary measurement.

### Using this branch from `feat/probe-physics-vwm`

`git checkout feat/commonsense` (or merge it; it fast-forwards). The original six environments,
`make_sweep.py`, `cosmos_v2w_sweep.py` and the analysis scripts are untouched except for
backward-compatible additions (`Base.render(segment=False)`, `roll(..., segment=False)`,
`--seeds/--output-root` on the sweep runner). Generated data lives under `artifacts/` and is
never committed.


## Environment

Python 3.11+, MuJoCo 3.11, PyTorch/CUDA, Diffusers with
`Cosmos2VideoToWorldPipeline`, NumPy, Pillow, imageio/ffmpeg, matplotlib, and
the Cosmos Predict2 2B Video2World checkpoint.

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


## V-JEPA 2-AC latent track (2026-09-03)

```
./run_vjepa_ac.sh artifacts/phase_a/manifest.jsonl artifacts/vjepa_ac/phase_a --pose-b   # render P/K/J -> latents -> analysis
PYTHONPATH=src python src/exp/rollout_vjepa_ac.py --root artifacts/vjepa_ac/phase_a --ctx-frames 3   # 3-frame-context variant
PYTHONPATH=src python src/exp/rollout_vjepa_ac.py --root artifacts/vjepa_ac/phase_a --reverse-ctx    # motion-cue control
PYTHONPATH=src python src/exp/analyze_vjepa_ac.py --root artifacts/vjepa_ac/phase_a --ctx-tag rev
```
Rendering needs the MuJoCo env with OSMesa (`MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`, `apt-get install libosmesa6` after a pod restart);
the model steps need `envs/vera` (torch 2.6) and `data/stage0/vjepa2-ac-vitg.pt`. Pages: `src/exp/build_phase_a_page.py`, `src/exp/build_scene_model_ui.py`.

## DiLA latent track (2026-09-03)

Past-only rollouts of DiLA (Disentangled Latent Action world model) on the V-JEPA 256x256 input
bundles; the root holds `manifest.jsonl` plus an `inputs/` symlink to the V-JEPA bundle.

```
./run_dila.sh artifacts/dila/phase_a          # hold / zero / mean policies, c3 / rev controls, stride 1 / 2 / 4
PYTHONPATH=src envs/vera/bin/python src/exp/rollout_dila.py --root artifacts/dila/phase_a --policy oracle   # leaky reference
PYTHONPATH=src envs/vera/bin/python src/exp/analyze_vjepa_ac.py --root artifacts/dila/phase_a --ctx-tag s1  # latent metrics
PYTHONPATH=src envs/vera/bin/python src/exp/analyze_dila_pixels.py --root artifacts/dila/phase_a --ctx-tag s1  # decoded red-ball readout
PYTHONPATH=src envs/vera/bin/python src/exp/preview_dila.py --ids hill_roll_wb_04 --tag _s1 --out strip.png
```

Assets: `external/DiLA` (repo), `external/DiLA/checkpoints/model.pt`, `external/DiLA/pretrained/` (RAE decoder,
stats, DINOv2-with-registers-base, patched decoder config). Needs `beartype` in `envs/vera`.

## Phase B scenes (2026-09-03)

`pendulum_rod_wb` (P1-B suspended payload, RodPendulumWB) and `support_edge_wb` (P3-A contact loss, SupportEdgeWB).

```
PYTHONPATH=src envs/miniforge3/envs/simdroid/bin/python src/exp/verify_phase_b.py            # physics signatures, S grid, boundary bisection
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=src .../simdroid/bin/python src/gen/make_phase_b.py
PYTHONPATH=src .../simdroid/bin/python src/exp/analyze_bundle_a.py --root artifacts/phase_b --gt-selftest-only
./run_phase_b_vwm.sh 1 2 3                                                                    # Cosmos V2W, one seed per call
PYTHONPATH=src .../simdroid/bin/python src/exp/analyze_bundle_a.py --root artifacts/phase_b --seeds 1 2 3 --output-dir artifacts/phase_b/analysis_pilot
PYTHONPATH=src .../simdroid/bin/python src/exp/build_phase_b_page.py                          # scene review page
```
