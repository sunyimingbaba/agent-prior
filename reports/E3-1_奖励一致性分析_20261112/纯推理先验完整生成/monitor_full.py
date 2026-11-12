#!/usr/bin/env python3
"""Read a compact snapshot of the active full-cache job."""
import json
import subprocess
from pathlib import Path

root = Path('/private/AGSE_E3_1_rebuild_20261112/plain_prior_full_20261112')
out = root/'run_01'
progress = json.loads((out/'progress.json').read_text(encoding='utf-8'))
records = []
if (out/'records.jsonl').exists():
    for line in (out/'records.jsonl').read_text(encoding='utf-8').splitlines(keepends=True):
        if line.endswith('\n'):
            records.append(json.loads(line))
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                               '--format=csv,noheader,nounits'], text=True).strip()
summary = {'status': progress['status'], 'saved': len(records), 'total': progress['total'],
           'elapsed_seconds_at_last_update': round(progress['elapsed_seconds_this_attempt'], 1),
           'latest': {k: records[-1][k] for k in ['split', 'row', 'image']} if records else None,
           'anchor_matches_so_far': sum(r['smoke_anchor'] and r['smoke_anchor_match'] for r in records),
           'gpu_used_mib_and_util_percent': gpu}
if (root/'run_01.exit').exists():
    summary['exit_code'] = int((root/'run_01.exit').read_text().strip())
if progress.get('error'):
    summary['error'] = progress['error']
print(json.dumps(summary, ensure_ascii=True))
