#!/usr/bin/env bash
# PVR PoC Cosmos V2W collection: one process per scene, all requested seeds inside
# (the model loads once).  Usage: run_pvr_vwm.sh <scene> <seed> [<seed> ...]
set -u
cd "$(dirname "$0")"
SCENE=$1; shift
SNAP=/mnt/nvme/cache/jihunmokn/hf/hub/models--nvidia--Cosmos-Predict2-2B-Video2World/snapshots/f50c09f5d8ab133a90cac3f4886a6471e9ba3f18
PY=/mnt/nvme/migration/jihun/envs/cosmos_v2w/bin/python
ROOT=artifacts/pvr/$SCENE
STATUS=$ROOT/cosmos_v2w/status.txt
mkdir -p "$ROOT/cosmos_v2w"
echo "$(date +%F_%T) START scene=$SCENE seeds=$*" >> "$STATUS"
HF_HOME=/mnt/nvme/cache/jihunmokn/hf CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
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
