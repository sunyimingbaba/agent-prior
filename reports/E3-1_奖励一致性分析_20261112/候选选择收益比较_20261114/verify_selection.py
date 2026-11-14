#!/usr/bin/env python3
"""Separate audit: rebuild selection expectations from immutable raw scores."""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
r = a.root
plan = json.loads((r/'分析方案.json').read_text())
result = json.loads((r/'选择收益完整统计.json').read_text())
assert result['analysis_plan_sha256'] == hashlib.sha256((r/'分析方案.json').read_bytes()).hexdigest()
for name, expected in plan['source_inputs'].items():
    assert hashlib.sha256((r/'inputs'/name).read_bytes()).hexdigest() == expected
assert plan['source_inputs']['scores.json'] == '5c3cc7067a7cc8520ca42601f8801df311f11848897479282ac3d5102c9a0b83'
raw = json.loads((r/'inputs/scores.json').read_text())
assert len(raw) == 242
for x in raw:
    assert x['utility_rr'] == float(Fraction(1, x['utility_rank1']))
    assert x['r_retrieval'] == float(Fraction(1, x['proxy_rank1']))
    if x['role'] != 'historical':
        bonus = .2 if re.search(r'<answer>(.*?)</answer>', x['trajectory_text'], re.S) else 0.
        assert x['r_format'] == bonus and x['r_total'] == bonus+x['r_retrieval']


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=0., abs_tol=2e-13), (actual, expected)


def avg(values):
    return sum(values)/len(values)


rebuilt, image_checks = {}, 0
for entry in result['per_image']:
    row, subset, policy, metric = (entry[k] for k in ['row', 'subset', 'policy', 'metric'])
    pool = [x for x in raw if x['row'] == row and x['role'] == 'candidate'
            and (subset == 'all' or x['quality_usable'])]
    assert len(pool) == entry['eligible_candidates']
    if policy == 'greedy':
        selected = [x for x in raw if x['row'] == row and x['role'] == 'reference']
    elif policy == 'uniform_random':
        selected = pool
    else:
        key = metric if policy == 'candidate_oracle' else plan['reward_columns'][policy]
        ranked = sorted(pool, key=lambda x: x[key], reverse=True)
        selected = [x for x in ranked if x[key] == ranked[0][key]]
    assert {x['id'] for x in selected} == set(entry['selected_ids'])
    assert len(selected) == entry['selected_count']
    if metric == 'utility_rr':
        value = float(sum((Fraction(1, x['utility_rank1']) for x in selected), Fraction(0))/len(selected))
    else:
        value = avg([x[metric] for x in selected])
    close(entry['expected_utility'], value)
    close(entry['selected_min_utility'], min(x[metric] for x in selected))
    close(entry['selected_max_utility'], max(x[metric] for x in selected))
    close(entry['quality_pass_probability'], sum(x['quality_usable'] for x in selected)/len(selected))
    close(entry['token_limit_probability'], sum(x['any_token_limit'] for x in selected)/len(selected))
    oracle = max(x[metric] for x in pool)
    close(entry['candidate_pool_oracle_utility'], oracle)
    close(entry['candidate_pool_regret'], oracle-value)
    close(entry['selected_utility_population_std'], math.sqrt(avg([(x[metric]-value)**2 for x in selected])))
    mu = avg([x[metric] for x in pool])
    close(entry['eligible_utility_sample_std'], math.sqrt(sum((x[metric]-mu)**2 for x in pool)/(len(pool)-1)))
    if policy != 'greedy':
        assert value <= oracle+2e-13
    key = (subset, row, policy, metric)
    assert key not in rebuilt
    rebuilt[key] = value
    image_checks += 1
assert image_checks == 22*2*5*4

ci_checks = comparisons = 0
for label, rows in plan['groups'].items():
    g = result['groups'][label]
    assert g['images'] == len(rows)
    draw = np.random.default_rng(plan['bootstrap']['seed']).integers(
        len(rows), size=(plan['bootstrap']['resamples'], len(rows)))
    for subset in plan['subsets']:
        for policy in plan['policies']:
            for metric, spec in plan['metrics'].items():
                values = [rebuilt[subset, row, policy, metric] for row in rows]
                stats = g['subsets'][subset]['policies'][policy][metric]
                close(stats['mean'], avg(values))
                close(stats['image_std_ddof1'], float(np.std(values, ddof=1)))
                interval = np.percentile(np.asarray(values)[draw].sum(axis=1)/len(rows), [2.5, 97.5])
                assert np.allclose(interval, stats['interval95'], atol=2e-13, rtol=0)
                ci_checks += 1
                for baseline in ['greedy', 'uniform_random']:
                    reference = [rebuilt[subset, row, baseline, metric] for row in rows]
                    deltas = [x-y for x, y in zip(values, reference)]
                    declared = g['subsets'][subset]['paired_comparisons'][policy][metric][baseline]
                    close(declared['mean'], avg(deltas))
                    close(declared['policy_mean'], avg(values))
                    close(declared['baseline_mean'], avg(reference))
                    interval = np.percentile(np.asarray(deltas)[draw].sum(axis=1)/len(rows), [2.5, 97.5])
                    assert np.allclose(interval, declared['interval95'], atol=2e-13, rtol=0)
                    close(declared['image_std_ddof1'], float(np.std(deltas, ddof=1)))
                    if metric == 'utility_rr':
                        close(declared['relative_improvement'], avg(deltas)/avg(reference))
                    else:
                        assert declared['relative_improvement'] is None
                    epsilon = spec['epsilon']
                    assert declared['win_images'] == sum(d > epsilon for d in deltas)
                    assert declared['loss_images'] == sum(d < -epsilon for d in deltas)
                    assert declared['tie_images'] == sum(-epsilon <= d <= epsilon for d in deltas)
                    ci_checks += 1
                    comparisons += 1

csv_counts = {}
for name, expected in [('逐图策略收益.csv', 880), ('策略收益汇总.csv', 120), ('相对基线配对差.csv', 240)]:
    with (r/name).open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == expected
    csv_counts[name] = len(rows)
    for x in rows:
        if name == '逐图策略收益.csv':
            close(float(x['expected_utility']), rebuilt[x['subset'], int(x['row']), x['policy'], x['metric']])
        else:
            g = result['groups'][x['group']]['subsets'][x['subset']]
            ref = g['policies'][x['policy']][x['metric']] if name == '策略收益汇总.csv' else \
                g['paired_comparisons'][x['policy']][x['metric']][x['baseline']]
            close(float(x['mean']), ref['mean'])
            close(float(x['ci95_lower']), ref['interval95'][0])
            close(float(x['ci95_upper']), ref['interval95'][1])

attribution_checks = 0
diagnostic_path = r/'选择结果归因诊断.json'
if diagnostic_path.exists():
    diagnostics = json.loads(diagnostic_path.read_text())
    for group, rows in plan['groups'].items():
        details = diagnostics['groups'][group]
        entries = details['per_image']
        assert {x['row'] for x in entries} == set(rows)
        for entry in entries:
            row = entry['row']
            close(entry['all_reward_minus_greedy'], rebuilt['all', row, 'retrieval_reward', 'utility_rr']-
                  rebuilt['all', row, 'greedy', 'utility_rr'])
            close(entry['all_reward_minus_random'], rebuilt['all', row, 'retrieval_reward', 'utility_rr']-
                  rebuilt['all', row, 'uniform_random', 'utility_rr'])
            close(entry['quality_reward_minus_random'], rebuilt['quality_pass', row, 'retrieval_reward', 'utility_rr']-
                  rebuilt['quality_pass', row, 'uniform_random', 'utility_rr'])
            selected = [x for x in raw if x['id'] in entry['reward_selected_ids']]
            invalid = [x for x in selected if not x['quality_usable']]
            assert {x['id'] for x in invalid} == set(entry['invalid_selected_ids'])
            close(entry['reward_invalid_probability'], len(invalid)/len(selected))
            close(entry['all_total_rr'], rebuilt['quality_pass', row, 'retrieval_reward', 'utility_rr'])
            attribution_checks += 1
        for name, selected in [('invalid_selected', [x for x in entries if x['invalid_selected_ids']]),
                               ('all_selected_pass', [x for x in entries if not x['invalid_selected_ids']])]:
            assert details['quality_partitions'][name]['images'] == len(selected)
            for field in ['greedy', 'random']:
                close(details['quality_partitions'][name]['sum_reward_minus_'+field],
                      sum(x['all_reward_minus_'+field] for x in selected))
        largest = max(entries, key=lambda x: x['all_reward_minus_greedy'])
        assert details['largest_greedy_gain']['row'] == largest['row']
        close(details['largest_greedy_gain']['share_of_net_sum_delta'], largest['all_reward_minus_greedy']/
              sum(x['all_reward_minus_greedy'] for x in entries))
        for field in ['greedy', 'random']:
            leave_one_out = [avg([x['all_reward_minus_'+field] for x in entries if x['row'] != row]) for row in rows]
            assert np.allclose([min(leave_one_out), max(leave_one_out)],
                               details['leave_one_image_out_reward_minus_'+field+'_range'], atol=2e-13, rtol=0)

audit = {'ok': True, 'source_scores_sha256': plan['source_inputs']['scores.json'],
    'per_image_policy_metric_checks': image_checks, 'paired_comparison_checks': comparisons,
    'image_bootstrap_interval_recomputations': ci_checks,
    'bootstrap_resamples': plan['bootstrap']['resamples'], 'csv_rows_checked': csv_counts,
    'rr_expectations_rebuilt_from_exact_rank_fractions': True,
    'all_selected_sets_and_quality_probabilities_verified': True,
    'post_result_attribution_image_checks': attribution_checks, 'numpy': np.__version__}
a.out.write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(audit, ensure_ascii=False))
