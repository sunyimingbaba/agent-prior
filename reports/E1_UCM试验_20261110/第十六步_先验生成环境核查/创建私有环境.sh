#!/bin/bash
set -euo pipefail
ROOT=/private/AGSE_E1_20261110/e1_prior_compat_20261111
BASEPY=/opt/conda/envs/Pytorch-2.11.0/bin/python
if [[ -e "$ROOT/venv" ]]; then
  echo 'Existing environment found; refusing overwrite.' >&2
  exit 1
fi
mkdir -p "$ROOT/tmp" "$ROOT/huggingface"
export LD_LIBRARY_PATH="/opt/conda/envs/Pytorch-2.11.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TMPDIR="$ROOT/tmp"
export HF_HOME="$ROOT/huggingface"
export PIP_NO_CACHE_DIR=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"$BASEPY" -m venv --system-site-packages "$ROOT/venv"
"$ROOT/venv/bin/python" -m pip install --no-index --no-deps --find-links "$ROOT/wheelhouse" -r "$ROOT/requirements.txt"
"$ROOT/venv/bin/python" -m pip check
