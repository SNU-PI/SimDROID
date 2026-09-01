#!/usr/bin/env bash
# Bundle A Cosmos V2W collection: one invocation per seed, resumable.
# Usage: run_bundle_a_vwm.sh <seed> [<seed> ...]
set -u
cd "$(dirname "$0")"
SNAP=/mnt/nvme/cache/jihunmokn/hf/hub/models--nvidia--Cosmos-Predict2-2B-Video2World/snapshots/f50c09f5d8ab133a90cac3f4886a6471e9ba3f18
PY=/mnt/nvme/migration/jihun/envs/cosmos_v2w/bin/python
STATUS=artifacts/bundle_a/cosmos_v2w/status.txt
mkdir -p artifacts/bundle_a/cosmos_v2w
for SEED in "$@"; do
  DIR=$(printf 'artifacts/bundle_a/cosmos_v2w/seed_%02d' "$SEED")
  echo "$(date +%F_%T) START seed=$SEED" >> "$STATUS"
  HF_HOME=/mnt/nvme/cache/jihunmokn/hf CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  "$PY" src/exp/cosmos_v2w_sweep.py \
    --model-dir "$SNAP" \
    --original-checkpoint "$SNAP/model-480p-16fps.pt" \
    --manifest artifacts/bundle_a/manifest.jsonl \
    --sweep-root artifacts/bundle_a \
    --output-dir "$DIR" \
    --seed "$SEED" \
    >> "$DIR.log" 2>&1
  echo "$(date +%F_%T) EXIT=$? seed=$SEED" >> "$STATUS"
done
echo "$(date +%F_%T) ALL_DONE seeds=$*" >> "$STATUS"
