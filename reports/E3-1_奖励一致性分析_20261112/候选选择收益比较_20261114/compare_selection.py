#!/usr/bin/env python3
"""Exact candidate-selection expectations and paired image bootstrap."""
import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
a = parser.parse_args()
r = a.root
plan = json.loads((r/'分析方案.json').read_text())
for name, expected in plan['source_inputs'].items():
    assert hashlib.sha256((r/'inputs'/name).read_bytes()).hexdigest() == expected
scores = json.loads((r/'inputs/scores.json').read_text())
assert len(scores) == len({x['id'] for x in scores}) == 242
assert sum(x['role'] == 'candidate' for x in scores) == 176


def mean(values):
    return math.fsum(values)/len(values)


def summary(values, draws):
    x = np.asarray(values, dtype=np.float64)
    lo, hi = np.quantile(x[draws].mean(axis=1), [.025, .975])
    return {'mean': mean(values), 'image_std_ddof1': float(x.std(ddof=1)),
            'interval95': [float(lo), float(hi)], 'images': len(values)}


image_rows = []
for subset in plan['subsets']:
    for row in plan['groups']['all22']:
        pool = [x for x in scores if x['role'] == 'candidate' and x['row'] == row]
        assert len(pool) == 8 and sorted(x['seed'] for x in pool) == list(range(8))
        eligible = pool if subset == 'all' else [x for x in pool if x['quality_usable']]
        assert eligible, f'Empty eligible set: {row} {subset}'
        refs = [x for x in scores if x['role'] == 'reference' and x['row'] == row]
        assert len(refs) == 1 and refs[0]['quality_usable']
        ref = refs[0]
        for metric in plan['metrics']:
            assert all(math.isfinite(x[metric]) for x in eligible+[ref])
            best = max(x[metric] for x in eligible)
            for policy in plan['policies']:
                if policy in plan['reward_columns']:
                    reward = plan['reward_columns'][policy]
                    top = max(x[reward] for x in eligible)
                    selected = [x for x in eligible if x[reward] == top]
                elif policy == 'greedy':
                    selected = [ref]
                elif policy == 'uniform_random':
                    selected = eligible
                elif policy == 'candidate_oracle':
                    selected = [x for x in eligible if x[metric] == best]
                else:
                    raise ValueError(policy)
                values = [x[metric] for x in selected]
                expected = mean(values)
                image_rows.append({'row': row, 'image': ref['image'], 'subset': subset,
                    'policy': policy, 'metric': metric, 'eligible_candidates': len(eligible),
                    'selected_count': len(selected), 'selected_ids': [x['id'] for x in selected],
                    'expected_utility': expected, 'selected_min_utility': min(values),
                    'selected_max_utility': max(values),
                    'selected_utility_population_std': float(np.std(values)),
                    'eligible_utility_sample_std': float(np.std([x[metric] for x in eligible], ddof=1)),
                    'quality_pass_probability': mean([float(x['quality_usable']) for x in selected]),
                    'token_limit_probability': mean([float(x['any_token_limit']) for x in selected]),
                    'candidate_pool_oracle_utility': best, 'candidate_pool_regret': best-expected})

index = {(x['subset'], x['row'], x['policy'], x['metric']): x for x in image_rows}
groups, aggregate_rows, delta_rows = {}, [], []
for group, rows in plan['groups'].items():
    draws = np.random.default_rng(plan['bootstrap']['seed']).integers(
        0, len(rows), size=(plan['bootstrap']['resamples'], len(rows)))
    groups[group] = {'images': len(rows), 'subsets': {}}
    for subset in plan['subsets']:
        sub = {'eligible_candidates': sum(index[subset, row, 'uniform_random', 'utility_rr']['eligible_candidates']
                                          for row in rows), 'policies': {}, 'paired_comparisons': {}}
        for policy in plan['policies']:
            sub['policies'][policy] = {}
            sub['paired_comparisons'][policy] = {}
            for metric, spec in plan['metrics'].items():
                records = [index[subset, row, policy, metric] for row in rows]
                values = [x['expected_utility'] for x in records]
                stats = {**summary(values, draws),
                    'expected_quality_pass_rate': mean([x['quality_pass_probability'] for x in records]),
                    'expected_token_limit_rate': mean([x['token_limit_probability'] for x in records]),
                    'multiple_selected_images': sum(x['selected_count'] > 1 for x in records),
                    'mean_candidate_pool_regret': mean([x['candidate_pool_regret'] for x in records])}
                sub['policies'][policy][metric] = stats
                aggregate_rows.append({'group': group, 'subset': subset, 'policy': policy,
                    'metric': metric, **{k: v for k, v in stats.items() if k != 'interval95'},
                    'ci95_lower': stats['interval95'][0], 'ci95_upper': stats['interval95'][1]})
                comparisons = {}
                for baseline in ['greedy', 'uniform_random']:
                    reference = [index[subset, row, baseline, metric]['expected_utility'] for row in rows]
                    deltas = [x-y for x, y in zip(values, reference)]
                    epsilon = spec['epsilon']
                    baseline_mean = mean(reference)
                    delta = {**summary(deltas, draws), 'baseline_mean': baseline_mean,
                        'policy_mean': mean(values),
                        'relative_improvement': mean(deltas)/baseline_mean
                            if metric == 'utility_rr' and baseline_mean > 0 else None,
                        'win_images': sum(d > epsilon for d in deltas),
                        'tie_images': sum(abs(d) <= epsilon for d in deltas),
                        'loss_images': sum(d < -epsilon for d in deltas)}
                    comparisons[baseline] = delta
                    delta_rows.append({'group': group, 'subset': subset, 'policy': policy,
                        'baseline': baseline, 'metric': metric,
                        **{k: v for k, v in delta.items() if k != 'interval95'},
                        'ci95_lower': delta['interval95'][0], 'ci95_upper': delta['interval95'][1]})
                sub['paired_comparisons'][policy][metric] = comparisons
        # The format bonus is constant within the predefined quality subset.
        if subset == 'quality_pass':
            for row in rows:
                for metric in plan['metrics']:
                    assert index[subset, row, 'retrieval_reward', metric]['selected_ids'] == \
                           index[subset, row, 'total_reward', metric]['selected_ids']
        groups[group]['subsets'][subset] = sub

result = {'ok': True, 'analysis_plan_sha256': hashlib.sha256((r/'分析方案.json').read_bytes()).hexdigest(),
    'source_inputs': plan['source_inputs'], 'runtime': {'python': platform.python_version(), 'numpy': np.__version__},
    'bootstrap': plan['bootstrap'], 'groups': groups, 'per_image': image_rows}
(r/'选择收益完整统计.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
for name, rows in [('逐图策略收益.csv', image_rows), ('策略收益汇总.csv', aggregate_rows),
                   ('相对基线配对差.csv', delta_rows)]:
    with (r/name).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
                         for k, v in x.items()} for x in rows)
primary = groups['all22']['subsets']['all']
print(json.dumps({'ok': True, 'per_image_rows': len(image_rows), 'aggregate_rows': len(aggregate_rows),
    'paired_rows': len(delta_rows), 'all22_rr': {p: primary['policies'][p]['utility_rr']['mean']
                                               for p in plan['policies']},
    'retrieval_reward_deltas': primary['paired_comparisons']['retrieval_reward']['utility_rr']}, ensure_ascii=False))
