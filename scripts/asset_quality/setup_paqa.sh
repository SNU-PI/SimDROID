#!/usr/bin/env bash
set -euo pipefail

CACHE_DIR="${PAQA_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/simdroid/asset-quality-paqa}"
BASE_PYTHON="${PAQA_BASE_PYTHON:-python3}"
QT_COMMIT="5d0e3473517051f9d67f7b3326ef3268fe53c030"

mkdir -p "$CACHE_DIR/deps"
if [[ ! -d "$CACHE_DIR/QT/.git" ]]; then
  git clone https://github.com/JiHyuk-Byun/QT.git "$CACHE_DIR/QT"
fi
git -C "$CACHE_DIR/QT" fetch --depth 1 origin "$QT_COMMIT"
git -C "$CACHE_DIR/QT" checkout --detach "$QT_COMMIT"

TORCH_VERSION="$($BASE_PYTHON -c 'import torch; print(torch.__version__.split("+")[0])')"
CUDA_VERSION="$($BASE_PYTHON -c 'import torch; print((torch.version.cuda or "cpu").replace(".", ""))')"
if [[ "$CUDA_VERSION" == "cpu" ]]; then
  echo "3D-PAQA requires a CUDA PyTorch environment" >&2
  exit 1
fi
"$BASE_PYTHON" -m pip install --target "$CACHE_DIR/deps" --upgrade --no-deps \
  torch-scatter -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION}.html"

cat <<EOF
3D-PAQA runtime ready. Use:
  export PAQA_PYTHON=$BASE_PYTHON
  export PAQA_QT_ROOT=$CACHE_DIR/QT
  export PAQA_PYTHONPATH=$CACHE_DIR/deps
EOF
