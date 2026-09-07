#!/usr/bin/env bash
# Phase B (payload pendulum, support edge) Cosmos V2W collection: one invocation per seed, resumable.
# Usage: run_phase_b_vwm.sh <seed> [<seed> ...]
set -u
cd "$(dirname "$0")"
SNAP=${COSMOS_SNAPSHOT:?set COSMOS_SNAPSHOT to the Cosmos-Predict2-2B-Video2World snapshot dir (contains model-480p-16fps.pt)}
PY=${COSMOS_PY:-python}   # python of the diffusers/torch env
STATUS=artifacts/phase_b/cosmos_v2w/status.txt
mkdir -p artifacts/phase_b/cosmos_v2w
for SEED in "$@"; do
  DIR=$(printf 'artifacts/phase_b/cosmos_v2w/seed_%02d' "$SEED")
  echo "$(date +%F_%T) START seed=$SEED" >> "$STATUS"
  HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONPATH=src \
  "$PY" src/exp/cosmos_v2w_sweep.py \
    --model-dir "$SNAP" \
    --original-checkpoint "$SNAP/model-480p-16fps.pt" \
    --manifest artifacts/phase_b/manifest.jsonl \
    --sweep-root artifacts/phase_b \
    --output-dir "$DIR" \
    --seed "$SEED" \
    >> "$DIR.log" 2>&1
  echo "$(date +%F_%T) EXIT=$? seed=$SEED" >> "$STATUS"
done
echo "$(date +%F_%T) ALL_DONE seeds=$*" >> "$STATUS"
