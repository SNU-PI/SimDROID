#!/usr/bin/env bash
# V-JEPA 2-AC latent track: render inputs (MuJoCo/OSMesa env) -> latents (torch env, GPU) -> analysis.
# Usage: run_vjepa_ac.sh <manifest.jsonl> <out_root> [extra rollout args]
set -u
cd "$(dirname "$0")"
MANIFEST=$1; ROOT=$2; shift 2
SIM=${SIMDROID_ENV:?set SIMDROID_ENV to the MuJoCo/OSMesa conda env prefix}
TORCH=${VERA_ENV:?set VERA_ENV to the torch env prefix used for V-JEPA 2-AC}
mkdir -p "$ROOT"
STATUS="$ROOT/chain.log"
echo "$(date +%F_%T) CHAIN_START manifest=$MANIFEST" >> "$STATUS"
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH=$SIM/lib PYTHONPATH=src \
  "$SIM/bin/python" src/gen/make_vjepa_inputs.py --manifest "$MANIFEST" --out "$ROOT" --skip-existing >> "$ROOT/gen.log" 2>&1
echo "$(date +%F_%T) GEN_EXIT=$?" >> "$STATUS"
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 "$TORCH/bin/python" src/exp/rollout_vjepa_ac.py --root "$ROOT" "$@" >> "$ROOT/rollout.log" 2>&1
echo "$(date +%F_%T) ROLLOUT_EXIT=$?" >> "$STATUS"
PYTHONPATH=src "$TORCH/bin/python" src/exp/analyze_vjepa_ac.py --root "$ROOT" >> "$ROOT/analysis.log" 2>&1
echo "$(date +%F_%T) ANALYSIS_EXIT=$? CHAIN_DONE" >> "$STATUS"
