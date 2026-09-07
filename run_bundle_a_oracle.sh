#!/usr/bin/env bash
# Bundle A prompt-contrast collection (oracle correct/wrong at boundary flanks).
set -u
cd "$(dirname "$0")"
SNAP=${COSMOS_SNAPSHOT:?set COSMOS_SNAPSHOT to the Cosmos-Predict2-2B-Video2World snapshot dir (contains model-480p-16fps.pt)}
PY=${COSMOS_PY:-python}   # python of the diffusers/torch env
STATUS=artifacts/bundle_a/cosmos_v2w_oracle/status.txt
mkdir -p artifacts/bundle_a/cosmos_v2w_oracle
for SEED in "$@"; do
  DIR=$(printf 'artifacts/bundle_a/cosmos_v2w_oracle/seed_%02d' "$SEED")
  echo "$(date +%F_%T) START seed=$SEED" >> "$STATUS"
  HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONPATH=src \
  "$PY" src/exp/cosmos_v2w_sweep.py \
    --model-dir "$SNAP" \
    --original-checkpoint "$SNAP/model-480p-16fps.pt" \
    --manifest artifacts/bundle_a/manifest_oracle.jsonl \
    --sweep-root artifacts/bundle_a \
    --output-dir "$DIR" \
    --seed "$SEED" \
    >> "$DIR.log" 2>&1
  echo "$(date +%F_%T) EXIT=$? seed=$SEED" >> "$STATUS"
done
echo "$(date +%F_%T) ALL_DONE seeds=$*" >> "$STATUS"
