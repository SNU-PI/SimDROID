#!/bin/bash
# DiLA latent track: rollouts (policies / time bases / controls) + latent and pixel readouts.
# usage: run_dila.sh <root>   (root must hold manifest.jsonl and inputs/ from make_vjepa_inputs.py)
set -u
ROOT=$1
cd /mnt/nvme/migration/jihun/SimDROID/code_vwm
PY=/mnt/nvme/migration/jihun/envs/vera/bin/python
LOG=$ROOT/chain.log
run() {  # run <tag> <rollout args...>
  local tag=$1; shift
  echo "$(date +%F_%T) ROLLOUT $tag" >> "$LOG"
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 $PY src/exp/rollout_dila.py --root "$ROOT" "$@" >> "$ROOT/rollout.log" 2>&1
  echo "$(date +%F_%T) ROLLOUT_EXIT=$? $tag" >> "$LOG"
  local ctx=${tag#_}
  PYTHONPATH=src $PY src/exp/analyze_vjepa_ac.py --root "$ROOT" --ctx-tag "$ctx" > "$ROOT/analysis${tag}.log" 2>&1
  echo "$(date +%F_%T) ANALYZE_EXIT=$? $tag" >> "$LOG"
  PYTHONPATH=src $PY src/exp/analyze_dila_pixels.py --root "$ROOT" --ctx-tag "$ctx" > "$ROOT/pixels${tag}.log" 2>&1
  echo "$(date +%F_%T) PIXELS_EXIT=$? $tag" >> "$LOG"
}
run ""        --policy hold
run "_zero"   --policy zero
run "_mean"   --policy mean
run "_c3"     --policy hold --ctx-frames 3
run "_rev"    --policy hold --reverse-ctx
run "_s1"     --policy hold --stride 1
run "_s2"     --policy hold --stride 2
run "_s1_zero" --policy zero --stride 1
echo "$(date +%F_%T) DILA_CHAIN_DONE" >> "$LOG"
