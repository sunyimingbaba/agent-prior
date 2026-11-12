#!/usr/bin/env python3
"""Compact, read-only snapshot of this dedicated scorer training run."""
import json
import statistics
import subprocess
from pathlib import Path

root = Path('/private/AGSE_E3_1_rebuild_20261112/scorer_training_20261112/attempt_02')
progress_path = root/'progress.json'
progress = json.loads(progress_path.read_text(encoding='utf-8')) if progress_path.exists() else {'status': 'starting'}
summary = {k: progress[k] for k in ['status', 'completed_epochs', 'epoch', 'batch',
    'optimizer_steps', 'batches_seen', 'latest_loss', 'peak_allocated_gib',
    'elapsed_seconds', 'last_update_utc', 'error'] if k in progress}
summary['gpu'] = subprocess.check_output(['nvidia-smi',
    '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
events_path = root/'run_01/training_events.jsonl'
if events_path.exists():
    events = [json.loads(line) for line in events_path.read_text(encoding='utf-8').splitlines(keepends=True) if line.endswith('\n')]
    starts = [event for event in events if event['event'] == 'train_start']
    completed = [event for event in events if event['event'] == 'validation_end']
    if starts:
        summary['train_batches_per_epoch'] = starts[0]['train_batches']
    if completed:
        summary['latest_validation'] = completed[-1]
        elapsed = [starts[0]['elapsed_seconds'], *[event['elapsed_seconds'] for event in completed]]
        durations = [b-a for a, b in zip(elapsed, elapsed[1:])]
        summary['recent_epoch_seconds_median'] = statistics.median(durations[-3:])
if (root/'run_01.exit').exists():
    summary['exit_code'] = int((root/'run_01.exit').read_text())
print(json.dumps(summary, ensure_ascii=True))
