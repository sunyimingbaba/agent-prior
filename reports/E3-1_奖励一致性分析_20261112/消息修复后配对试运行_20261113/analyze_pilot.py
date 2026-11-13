#!/usr/bin/env python3
"""Per-image paired statistics, preserving ties, duplicates and undefined rho."""
import collections
import csv
import itertools
import json
import math
import statistics

from pilot_common import ROOT, PLAN, dump, sha

assert json.loads((ROOT/'scoring/verification.json').read_text())['ok']
results = json.loads((ROOT/'scoring/scores.json').read_text())
candidates = [r for r in results if r['role'] == 'candidate']
assert len(candidates) == 24
metrics = {'utility_rr': 1e-12, 'utility_margin': 1e-6, 'E1_margin_A': 1e-6, 'E1_margin_B': 1e-6}
rewards = ['r_retrieval', 'r_total']


def average_ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0]*len(values)
    left = 0
    while left < len(order):
        right = left+1
        while right < len(order) and values[order[right]] == values[order[left]]:
            right += 1
        for index in order[left:right]:
            ranks[index] = (left+right-1)/2+1
        left = right
    return ranks


def rho(xs, ys):
    x, y = average_ranks(xs), average_ranks(ys)
    mx, my = statistics.mean(x), statistics.mean(y)
    sx = sum((v-mx)**2 for v in x)
    sy = sum((v-my)**2 for v in y)
    if sx == 0 or sy == 0:
        return None
    return sum((a-mx)*(b-my) for a, b in zip(x, y))/math.sqrt(sx*sy)


def sign(value, epsilon):
    return 0 if abs(value) <= epsilon else (1 if value > 0 else -1)


paired, per_image = [], []
for item in PLAN['images']:
    row = item['row']
    group = sorted([r for r in candidates if r['row'] == row], key=lambda r: r['seed'])
    assert len(group) == 8 and [r['seed'] for r in group] == PLAN['seeds']
    reference = next(r for r in results if r['row'] == row and r['role'] == 'reference')
    historical = next(r for r in results if r['row'] == row and r['role'] == 'historical')
    stats = {'row': row, 'image': item['image'], 'candidate_count': len(group),
        'unique_state_count': len({r['h_end_tensor_sha256'] for r in group}),
        'unique_trajectory_count': len({r['trajectory_text_sha256'] for r in group}),
        'unique_retrieval_reward_count': len({r['r_retrieval'] for r in group}),
        'candidate_state_duplicate_groups': [[r['id'] for r in group if r['h_end_tensor_sha256'] == digest]
            for digest, count in collections.Counter(r['h_end_tensor_sha256'] for r in group).items() if count > 1],
        'greedy_reference': {k: reference[k] for k in ['id', 'r_retrieval', 'r_format', 'r_total', *metrics]},
        'historical_reference': {k: historical[k] for k in ['id', 'r_retrieval', *metrics]},
        'comparisons': {}}
    for reward in rewards:
        stats['comparisons'][reward] = {}
        for metric, epsilon in metrics.items():
            counts = dict.fromkeys(['concordant', 'discordant', 'reward_tie_only', 'utility_tie_only', 'both_tied'], 0)
            for a, b in itertools.combinations(group, 2):
                dr, du = a[reward]-b[reward], a[metric]-b[metric]
                sr, su = sign(dr, 0.0), sign(du, epsilon)
                if sr == su == 0:
                    category = 'both_tied'
                elif sr == 0:
                    category = 'reward_tie_only'
                elif su == 0:
                    category = 'utility_tie_only'
                else:
                    category = 'concordant' if sr == su else 'discordant'
                counts[category] += 1
                paired.append({'row': row, 'image': item['image'], 'reward_metric': reward,
                    'utility_metric': metric, 'candidate_a': a['id'], 'candidate_b': b['id'],
                    'reward_a': a[reward], 'reward_b': b[reward], 'utility_a': a[metric], 'utility_b': b[metric],
                    'reward_difference': dr, 'utility_difference': du, 'utility_tie_epsilon': epsilon,
                    'category': category, 'same_hidden_state': a['h_end_tensor_sha256'] == b['h_end_tensor_sha256']})
            assert sum(counts.values()) == 28
            ys = [round(r[metric], 6) if epsilon == 1e-6 else r[metric] for r in group]
            informative = counts['concordant']+counts['discordant']
            top = max(r[reward] for r in group)
            top_group = [r for r in group if r[reward] == top]
            best_utility = max(r[metric] for r in group)
            stats['comparisons'][reward][metric] = {'pair_counts': counts, 'informative_pairs': informative,
                'concordance_in_informative_pairs': counts['concordant']/informative if informative else None,
                'spearman': rho([r[reward] for r in group], ys), 'unique_rounded_utility_count': len(set(ys)),
                'utility_range': [min(r[metric] for r in group), best_utility],
                'utility_delta_range': [min(r[metric+'_delta'] for r in group), max(r[metric+'_delta'] for r in group)],
                'max_reward': top, 'max_reward_candidate_ids': [r['id'] for r in top_group],
                'max_reward_utility_range': [min(r[metric] for r in top_group), max(r[metric] for r in top_group)],
                'max_reward_regret_range': [best_utility-max(r[metric] for r in top_group), best_utility-min(r[metric] for r in top_group)]}
    per_image.append(stats)
aggregate = {}
for reward in rewards:
    aggregate[reward] = {}
    for metric in metrics:
        comparisons = [r['comparisons'][reward][metric] for r in per_image]
        counts = {key: sum(r['pair_counts'][key] for r in comparisons) for key in comparisons[0]['pair_counts']}
        informative = counts['concordant']+counts['discordant']
        ratios = [r['concordance_in_informative_pairs'] for r in comparisons if r['informative_pairs']]
        aggregate[reward][metric] = {'pair_counts': counts, 'informative_pairs': informative,
            'pooled_pair_concordance': counts['concordant']/informative if informative else None,
            'mean_image_concordance': statistics.mean(ratios) if ratios else None,
            'images_with_informative_pairs': len(ratios),
            'images_with_defined_spearman': sum(r['spearman'] is not None for r in comparisons)}
summary = {'ok': True, 'image_count': 3, 'candidate_count': 24, 'pairs_per_image_per_metric': 28,
    'candidate_usable_count': sum(r['quality_usable'] for r in candidates),
    'candidate_stop_reasons': dict(collections.Counter(r['quality_stop_reason'] for r in candidates)),
    'plan_sha256': sha(ROOT/'plan.json'), 'scores_sha256': sha(ROOT/'scoring/scores.json'),
    'scope': 'bounded messages-repair pilot of E1-selected val images; one attempt per seed, all candidates retained',
    'primary_metric': 'utility_rr', 'analysis_units': 'within-image; ties and duplicates retained',
    'candidate_format_reward_counts': dict(collections.Counter(r['r_format'] for r in candidates)),
    'candidate_tool_used_counts': dict(collections.Counter(r['tools_used'] for r in candidates)),
    'candidate_flags': {'addCriterion': sum(r['contains_addCriterion'] for r in candidates),
        'token_limit': sum(r['any_token_limit'] for r in candidates),
        'empty_trajectory': sum(not r['trajectory_text'].strip() for r in candidates)},
    'per_image': per_image, 'aggregate': aggregate}
dump(ROOT/'analysis.json', summary)
with (ROOT/'pairwise_comparisons.csv').open('w', encoding='utf-8', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(paired[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(paired)
print('ANALYSIS_COMPLETE', json.dumps({'images': 3, 'candidates': 24,
    'retrieval_reward_vs_primary_utility': aggregate['r_retrieval']['utility_rr']}, ensure_ascii=False), flush=True)
