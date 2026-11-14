"""Descriptive counts for the deliberately selected 18-sequence diagnostic."""
import csv
import json
import statistics
from pathlib import Path

r = Path(__file__).resolve().parent
plan = json.loads((r/'plan.json').read_text())
scores = json.loads((r/'scoring/scores.json').read_text())
extension = json.loads((r/'extension_records.json').read_text())
comparisons = json.loads((r/'state_comparisons.json').read_text())
index = {(x['source_id'], x['variant']): x for x in scores}
metrics = ['r_retrieval', 'utility_rr', 'utility_margin', 'E1_margin_A', 'E1_margin_B']
pairs = [('saved_vs_forward_penultimate', 'saved_offline', 'full_forward_penultimate', 'all'),
         ('saved_vs_grpo_unpadded', 'saved_offline', 'grpo_unpadded_last', 'all'),
         ('grpo_unpadded_vs_group_padded', 'grpo_unpadded_last', 'grpo_group_padded_last', 'all'),
         ('length_extension_offline', 'saved_offline', 'extended_offline', 'failure'),
         ('length_extension_grpo', 'grpo_unpadded_last', 'extended_grpo_last', 'failure'),
         ('extended_saved_vs_grpo_last', 'extended_offline', 'extended_grpo_last', 'failure')]
groups, per_pair = {}, []
for label, before, after, scope in pairs:
    rows = [x for x in plan['records'] if scope == 'all' or x['role'] == 'selected_failure']
    entries = []
    for source in rows:
        x, y = index[source['source_id'], before], index[source['source_id'], after]
        entry = {'comparison': label, 'source_id': source['source_id'], 'source_role': source['role'],
            'row': source['row'], 'image': source['image'], 'before_variant': before, 'after_variant': after,
            'proxy_rank1_before': x['proxy_rank0']+1, 'proxy_rank1_after': y['proxy_rank0']+1,
            'utility_rank1_before': x['utility_rank0']+1, 'utility_rank1_after': y['utility_rank0']+1,
            **{metric+'_delta': y[metric]-x[metric] for metric in metrics}}
        entries.append(entry)
        per_pair.append(entry)
    selected_cosines = [x for x in comparisons if x['comparison'] == label]
    group = {'records': len(entries), 'proxy_rank_changed': sum(x['proxy_rank1_before'] != x['proxy_rank1_after'] for x in entries),
        'utility_rank_changed': sum(x['utility_rank1_before'] != x['utility_rank1_after'] for x in entries),
        'mean_deltas': {metric: statistics.mean(x[metric+'_delta'] for x in entries) for metric in metrics}}
    if selected_cosines:
        group['state_cosine_range'] = [min(x['cosine'] for x in selected_cosines), max(x['cosine'] for x in selected_cosines)]
        group['state_cosine_mean'] = statistics.mean(x['cosine'] for x in selected_cosines)
        group['bitwise_equal_count'] = sum(x['bitwise_equal'] for x in selected_cosines)
    groups[label] = group
result = {'ok': True, 'scope': '18 archived sequences from six previously selected failure images; no optimizer update',
    'states': len(scores), 'comparisons': groups, 'extensions': extension, 'per_pair': per_pair}
(r/'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
for name, rows in [('states_scores.csv', scores), ('paired_state_effects.csv', per_pair), ('state_cosines.csv', comparisons)]:
    with (r/name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader(); w.writerows(rows)
print(json.dumps({'comparisons': groups, 'extensions': extension}, ensure_ascii=False))
