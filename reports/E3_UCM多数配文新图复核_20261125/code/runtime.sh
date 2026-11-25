#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
base=/private/AGSE_E3_1_rebuild_20261112
gpu=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
cpu=/private/AGSE_E1_20261110/legacy/env/bin/python
df -T "$PWD"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"$gpu" -c 'import sys, torch; print(sys.version, torch.__version__, torch.version.cuda)'
"$cpu" -c 'import sys, torch; print(sys.version, torch.__version__)'
sha256sum "$base/r3_after_tool_trainpanel_20261124/code/sample_panel.py" "$base/r3_after_tool_trainpanel_20261124/code/pack.py" "$base/r3_after_tool_trainpanel_20261124/code/base.py" "$base/paired_answer_preference_20261124/code/guard_read.py"
