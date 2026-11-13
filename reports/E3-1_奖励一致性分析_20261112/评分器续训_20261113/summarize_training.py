#!/usr/bin/env python3
"""Check complete epoch/batch coverage and reconcile native TensorBoard recall metrics."""
import csv
import json
import math
import statistics
from pathlib import Path

from tensorboard_scalars import read_scalars

root = Path('/private/AGSE_E3_1_rebuild_20261112/scorer_resume_20261113')
result = json.loads((root/'training_result.json').read_text(encoding='utf-8'))
assert result['ok'] and result['status'] == 'complete' and result['completed_epochs'] == 50
previous = Path('/private/AGSE_E3_1_rebuild_20261112/scorer_training_20261112/attempt_02')
old_losses = [json.loads(line) for line in (previous/'snapshots/epoch_30/batch_losses.jsonl').read_text(encoding='utf-8').splitlines()]
new_losses = [json.loads(line) for line in (root/'run_01/batch_losses.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(old_losses) == 7500 and len(new_losses) == 5000
losses = old_losses + new_losses
old_events = [json.loads(line) for line in (previous/'snapshots/epoch_30/training_events.jsonl').read_text(encoding='utf-8').splitlines()]
new_events = [json.loads(line) for line in (root/'run_01/training_events.jsonl').read_text(encoding='utf-8').splitlines()]
events = old_events + new_events
validation = [row for row in events if row['event'] == 'validation_end']
assert len(losses) == 12500 and len(validation) == 50
assert {(row['epoch'], row['batch']) for row in losses} == {(epoch, batch) for epoch in range(50) for batch in range(250)}
assert [row['epoch'] for row in validation] == list(range(50))
assert [row['optimizer_step'] for row in validation] == [32*(i+1)-1 for i in range(50)]
assert result['optimizer_steps'] == 1600 and result['batches_seen'] == 5000
assert result['gradient_checks_passed'] == 200 and result['backbone_unchanged']
assert result['initial_backbone_sha256'] == result['final_backbone_sha256']
assert result['initial_trainable_sha256'] != result['final_trainable_sha256']
assert all(row['loss'] is not None and math.isfinite(row['loss']) for row in losses)
assert all(math.isfinite(row['lr']) and 0 <= row['lr'] <= 1.000001e-4 for row in losses)

tags = ['recalls/mean', 'recalls/im_mean', 'recalls/ir_r1', 'recalls/ir_r5',
        'recalls/ir_r10', 'recalls/tr_r1', 'recalls/tr_r5', 'recalls/tr_r10',
        'recalls/iir_r1', 'recalls/iir_r5', 'recalls/iir_r10',
        'recalls/ttr_r1', 'recalls/ttr_r5', 'recalls/ttr_r10']
old_scalars, old_record_count = read_scalars(previous/'run_01/logs/plain_prior_scorer/run_01', tags)
new_scalars, new_record_count = read_scalars(root/'run_01/logs/plain_prior_scorer/run_01', tags)
record_count = old_record_count + new_record_count
scalars = {tag: [entry for entry in old_scalars[tag] if entry['step'] < 960] + new_scalars[tag] for tag in tags}
assert all(len(new_scalars[tag]) == 20 for tag in tags)
for tag in tags:
    entries = scalars[tag]
    assert len(entries) == 50, (tag, len(entries))
    assert [entry['step'] for entry in entries] == [32*(i+1)-1 for i in range(50)]
    assert all(math.isfinite(entry['value']) for entry in entries)
rows = []
for epoch, event in enumerate(validation):
    batches = [row for row in losses if row['epoch'] == epoch]
    assert len(batches) == 250
    assert abs(scalars['recalls/im_mean'][epoch]['value']-event['metrics']['val/the_metric']) < 1e-4
    row = {'completed_epoch': epoch+1,
           'native_validation_counter': event['optimizer_step'],
           'optimizer_updates_completed_after_epoch': 32*(epoch+1),
           'batch_loss_arithmetic_mean': statistics.mean(batch['loss'] for batch in batches),
           'batch_loss_min': min(batch['loss'] for batch in batches),
           'batch_loss_max': max(batch['loss'] for batch in batches),
           'lr_min_observed_after_batch': min(batch['lr'] for batch in batches),
           'lr_max_observed_after_batch': max(batch['lr'] for batch in batches),
           'val_clip_loss_callback': event['metrics']['clip/val/clip_loss'],
           'val_the_metric': event['metrics']['val/the_metric']}
    row.update({tag.replace('/', '_'): scalars[tag][epoch]['value'] for tag in tags})
    rows.append(row)
with (root/'epoch_metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
best_index = max(range(50), key=lambda index: rows[index]['val_the_metric'])
assert abs(result['best_monitor_value']-rows[best_index]['val_the_metric']) < 1e-4
summary = {'verified': True, 'completed_epochs': 50, 'batch_records': 12500,
    'all_batch_losses_finite': True, 'resumed_gradient_audit_passes': 200, 'prior_batch_records': 7500, 'resumed_batch_records': 5000,
    'optimizer_updates': 1600, 'all_epoch_batch_pairs_present_once': True,
    'native_tensorboard_tags_verified': tags,
    'native_tensorboard_crc_verified_records': record_count,
    'native_tensorboard_and_validation_monitor_match': True,
    'best_monitor_completed_epoch': best_index+1,
    'best_monitor_value': rows[best_index]['val_the_metric'],
    'last_monitor_value': rows[-1]['val_the_metric'],
    'best_monitor_original_mean_recall': rows[best_index]['recalls_mean'],
    'last_original_mean_recall': rows[-1]['recalls_mean'],
    'last_epoch': rows[-1], 'best_monitor_epoch': rows[best_index],
    'last_observed_lr': losses[-1]['lr'],
    'batch_loss_average_definition': 'arithmetic mean of 250 recorded total batch losses per epoch',
    'callback_val_loss_definition': 'value observed in trainer.callback_metrics at validation_end',
    'final_scorer_selection': 'last after 50 epochs, fixed before training'}
(root/'training_metrics_verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({key: value for key, value in summary.items() if key not in ['last_epoch', 'best_monitor_epoch', 'native_tensorboard_tags_verified']},ensure_ascii=False))
