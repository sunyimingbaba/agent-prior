#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
pid="$1"
while kill -0 "$pid" 2>/dev/null; do sleep 5; done
if [ ! -e logs/gpu.done ] && grep -q "object has no attribute 'open'" logs/gpu.log; then
  bash code/run.sh gpu >> logs/gpu.log 2>&1
fi
