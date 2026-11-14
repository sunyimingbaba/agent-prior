#!/usr/bin/env python3
"""Post-result descriptive attribution; primary comparisons remain unchanged."""
import argparse
import csv
import json
import math
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
r = a.root
result = json.loads((r/'选择收益完整统计.json').read_text())
plan = json.loads((r/'分析方案.json').read_text())
scores = {x['id']: x for x in json.loads((r/'inputs/scores.json').read_text())}
index = {(x['subset'], x['row'], x['policy'], x['metric']): x for x in result['per_image']}
groups, table = {}, []
for group, rows in plan['groups'].items():
    diagnostics = []
    for row in rows:
        reward = index['all', row, 'retrieval_reward', 'utility_rr']
        total = index['all', row, 'total_reward', 'utility_rr']
        good = index['quality_pass', row, 'retrieval_reward', 'utility_rr']
        greedy = index['all', row, 'greedy', 'utility_rr']['expected_utility']
        random = index['all', row, 'uniform_random', 'utility_rr']['expected_utility']
        quality_random = index['quality_pass', row, 'uniform_random', 'utility_rr']['expected_utility']
        bad = [scores[i] for i in reward['selected_ids'] if not scores[i]['quality_usable']]
        d = {'group': group, 'row': row, 'image': reward['image'],
            'reward_selected_ids': reward['selected_ids'], 'reward_tie_count': reward['selected_count'],
            'reward_invalid_probability': 1-reward['quality_pass_probability'],
            'invalid_selected_ids': [x['id'] for x in bad],
            'invalid_token_limit_count': sum(x['any_token_limit'] for x in bad),
            'all_reward_rr': reward['expected_utility'], 'all_total_rr': total['expected_utility'],
            'greedy_rr': greedy, 'all_random_rr': random,
            'quality_reward_rr': good['expected_utility'], 'quality_random_rr': quality_random,
            'all_reward_minus_greedy': reward['expected_utility']-greedy,
            'all_reward_minus_random': reward['expected_utility']-random,
            'quality_reward_minus_random': good['expected_utility']-quality_random,
            'all_reward_minus_total': reward['expected_utility']-total['expected_utility']}
        assert good['expected_utility'] == total['expected_utility']
        diagnostics.append(d)
        table.append(d)
    partitions = {}
    for name, records in [('invalid_selected', [x for x in diagnostics if x['reward_invalid_probability'] > 0]),
                          ('all_selected_pass', [x for x in diagnostics if x['reward_invalid_probability'] == 0])]:
        partitions[name] = {'images': len(records),
            'sum_reward_minus_greedy': math.fsum(x['all_reward_minus_greedy'] for x in records),
            'sum_reward_minus_random': math.fsum(x['all_reward_minus_random'] for x in records)}
    largest = max(diagnostics, key=lambda x: x['all_reward_minus_greedy'])
    sum_greedy = math.fsum(x['all_reward_minus_greedy'] for x in diagnostics)
    sum_random = math.fsum(x['all_reward_minus_random'] for x in diagnostics)
    groups[group] = {'quality_partitions': partitions,
        'total_reward_and_quality_filtered_reward_rr_identical_all_images': True,
        'largest_greedy_gain': {'row': largest['row'], 'image': largest['image'],
            'delta': largest['all_reward_minus_greedy'],
            'share_of_net_sum_delta': largest['all_reward_minus_greedy']/sum_greedy if sum_greedy else None},
        'leave_one_image_out_reward_minus_greedy_range': [
            min((sum_greedy-x['all_reward_minus_greedy'])/(len(rows)-1) for x in diagnostics),
            max((sum_greedy-x['all_reward_minus_greedy'])/(len(rows)-1) for x in diagnostics)],
        'leave_one_image_out_reward_minus_random_range': [
            min((sum_random-x['all_reward_minus_random'])/(len(rows)-1) for x in diagnostics),
            max((sum_random-x['all_reward_minus_random'])/(len(rows)-1) for x in diagnostics)],
        'per_image': diagnostics}
out = {'analysis_type': 'post-result descriptive attribution; no replacement of the fixed primary analysis',
    'metric': 'utility_rr', 'groups': groups}
(r/'选择结果归因诊断.json').write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n')
with (r/'逐图选择诊断.csv').open('w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator='\n')
    w.writeheader()
    w.writerows({k: json.dumps(v) if isinstance(v, list) else v for k, v in x.items()} for x in table)
print(json.dumps({k: v for k, v in groups['all22'].items() if k != 'per_image'}, ensure_ascii=False))
