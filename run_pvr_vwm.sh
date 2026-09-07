#!/usr/bin/env bash
# PVR PoC Cosmos V2W collection: one process per scene, all requested seeds inside
# (the model loads once).  Usage: run_pvr_vwm.sh <scene> <seed> [<seed> ...]
set -u
cd "$(dirname "$0")"
SCENE=$1; shift
SNAP=${COSMOS_SNAPSHOT:?set COSMOS_SNAPSHOT to the Cosmos-Predict2-2B-Video2World snapshot dir (contains model-480p-16fps.pt)}
PY=${COSMOS_PY:-python}   # python of the diffusers/torch env
ROOT=artifacts/pvr/$SCENE
STATUS=$ROOT/cosmos_v2w/status.txt
mkdir -p "$ROOT/cosmos_v2w"
echo "$(date +%F_%T) START scene=$SCENE seeds=$*" >> "$STATUS"
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONPATH=src \
"$PY" src/exp/cosmos_v2w_sweep.py \
  --model-dir "$SNAP" \
  --original-checkpoint "$SNAP/model-480p-16fps.pt" \
  --manifest "$ROOT/manifest.jsonl" \
  --sweep-root "$ROOT" \
  --output-root "$ROOT/cosmos_v2w" \
  --seeds "$@" \
  >> "$ROOT/cosmos_v2w/run_$(date +%m%d_%H%M).log" 2>&1
RC=$?
echo "$(date +%F_%T) EXIT=$RC scene=$SCENE seeds=$*" >> "$STATUS"
exit $RC
