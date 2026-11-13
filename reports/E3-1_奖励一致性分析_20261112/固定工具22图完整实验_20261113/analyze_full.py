#!/usr/bin/env python3
"""Prespecified image-level analysis, quality subsets and image bootstrap."""
import argparse
import collections
import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--pilot', type=Path, required=True)
args = parser.parse_args()
r = args.run
plan = json.loads((r/'plan.json').read_text())
assert json.loads((r/'run_result.json').read_text())['status'] == 'complete'
scores = json.loads((r/'scoring/scores.json').read_text())
records = [json.loads(line) for line in (r/'generation/records.jsonl').read_text().splitlines()]
candidates = [x for x in scores if x['role'] == 'candidate']
assert len(candidates) == plan['expected_counts']['candidates'] == 176
METRICS = {'utility_rr': 1e-12, 'utility_margin': 1e-6, 'E1_margin_A': 1e-6, 'E1_margin_B': 1e-6}
REWARDS = ['r_retrieval', 'r_total']


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    out = [0.0]*len(values)
    start = 0
    while start < len(order):
        end = start+1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for k in order[start:end]:
            out[k] = (start+end+1)/2
        start = end
    return out


def spearman(xs, ys):
    if len(xs) < 2:
        return None
    a, b = ranks(xs), ranks(ys)
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = sum((x-ma)**2 for x in a), sum((x-mb)**2 for x in b)
    return sum((x-ma)*(y-mb) for x,y in zip(a,b))/math.sqrt(va*vb) if va and vb else None


def image_stats(group, reward, metric, epsilon):
    counts = dict.fromkeys(['concordant','discordant','reward_tie_only','utility_tie_only','both_tied'],0)
    for a,b in itertools.combinations(group,2):
        dr, du = a[reward]-b[reward], a[metric]-b[metric]
        rt, ut = dr == 0, abs(du) <= epsilon
        category = ('both_tied' if rt and ut else 'reward_tie_only' if rt else
                    'utility_tie_only' if ut else 'concordant' if dr*du > 0 else 'discordant')
        counts[category] += 1
    informative = counts['concordant']+counts['discordant']
    ys = [round(x[metric],6) if epsilon == 1e-6 else x[metric] for x in group]
    result = {'candidate_count':len(group), 'pair_counts':counts, 'informative_pairs':informative,
        'concordance':counts['concordant']/informative if informative else None,
        'spearman':spearman([x[reward] for x in group],ys), 'utility_varies':False,
        'optimistic_regret':None,'pessimistic_regret':None}
    if group:
        utility_best = max(x[metric] for x in group)
        utility_worst = min(x[metric] for x in group)
        top_reward = max(x[reward] for x in group)
        top = [x for x in group if x[reward] == top_reward]
        result.update(utility_varies=utility_best-utility_worst > epsilon,
            optimistic_regret=utility_best-max(x[metric] for x in top),
            pessimistic_regret=utility_best-min(x[metric] for x in top))
    return result


per_image = []
for item in plan['images']:
    row = item['row']
    group = [x for x in candidates if x['row'] == row]
    generated = [x for x in records if x['row'] == row]
    assert len(group) == 8 and len(generated) == 10
    assert len({x['message_audit']['rounds'][0]['rendered_text_sha256'] for x in generated}) == 1
    subsets = {'all':group, 'quality_pass':[x for x in group if x['quality_usable']]}
    summary = {'row':row, 'image':item['image'], 'quantity':item['confirmed_quantity'],
        'pilot_image':row in plan['gate_rows'], 'usable':len(subsets['quality_pass']),
        'token_limit':sum(x['any_token_limit'] for x in group),
        'unique_states':len({x['h_end_tensor_sha256'] for x in group}),
        'unique_answers':len({x['trajectory_text'] for x in group}), 'subsets':{}}
    for subset, values in subsets.items():
        summary['subsets'][subset] = {reward:{metric:image_stats(values,reward,metric,eps)
            for metric,eps in METRICS.items()} for reward in REWARDS}
    per_image.append(summary)


def bootstrap(values, draws):
    array = np.array([np.nan if v is None else v for v in values], dtype=np.float64)
    picked = array[draws]
    n = np.isfinite(picked).sum(axis=1)
    means = np.divide(np.nansum(picked,axis=1), n, out=np.full(len(draws),np.nan), where=n>0)
    good = means[np.isfinite(means)]
    finite = array[np.isfinite(array)]
    return {'point_mean':float(finite.mean()) if len(finite) else None,
        'images_defined':int(len(finite)), 'interval95':np.quantile(good,[.025,.975]).tolist() if len(good) else None,
        'defined_resamples':int(len(good)), 'undefined_resamples':int(len(draws)-len(good))}


groups = {}
for label, rows in plan['image_groups'].items():
    images = [x for x in per_image if x['row'] in rows]
    assert len(images) == len(rows)
    rng = np.random.default_rng(plan['full_analysis']['bootstrap']['seed'])
    draws = rng.integers(0,len(images),size=(plan['full_analysis']['bootstrap']['resamples'],len(images)))
    group_result = {'image_count':len(images),'subsets':{}}
    for subset in ['all','quality_pass']:
        group_result['subsets'][subset] = {}
        for reward in REWARDS:
            group_result['subsets'][subset][reward] = {}
            for metric,epsilon in METRICS.items():
                values = [x['subsets'][subset][reward][metric] for x in images]
                counts = {k:sum(x['pair_counts'][k] for x in values) for k in values[0]['pair_counts']}
                informative = counts['concordant']+counts['discordant']
                varying = [x for x in values if x['utility_varies']]
                rhos = [x['spearman'] for x in values if x['spearman'] is not None]
                group_result['subsets'][subset][reward][metric] = {'pair_counts':counts,
                    'informative_pairs':informative,
                    'pooled_pair_concordance':counts['concordant']/informative if informative else None,
                    'image_mean_concordance_bootstrap':bootstrap([x['concordance'] for x in values],draws),
                    'image_mean_spearman_bootstrap':bootstrap([x['spearman'] for x in values],draws),
                    'rho_positive_images':sum(x>1e-12 for x in rhos),
                    'rho_negative_images':sum(x< -1e-12 for x in rhos),
                    'rho_zero_images':sum(abs(x)<=1e-12 for x in rhos),
                    'images_with_utility_variation':len(varying),
                    'top_reward_contains_best_utility_images':sum(x['optimistic_regret'] <= epsilon for x in varying),
                    'all_tied_top_reward_are_best_utility_images':sum(x['pessimistic_regret'] <= epsilon for x in varying)}
    groups[label] = group_result

# The pilot is part of the full22 set; explicitly verify its repeated states.
old_records = [json.loads(x) for x in (args.pilot/'generation/records.jsonl').read_text().splitlines()]
by_id = {x['id']:x for x in records}
for old in old_records:
    new = by_id[old['id']]
    assert new['tensor_sha256'] == old['tensor_sha256']
    assert new['trajectory_text'] == old['trajectory_text']
pilot_scores = json.loads((args.pilot/'scoring/scores.json').read_text())
new_scores = {x['id']:x for x in scores}
anchor_score_checks = []
for old in pilot_scores:
    new = new_scores[old['id']]
    error = max(abs(new[k]-old[k]) for k in ['utility_margin','E1_margin_A','E1_margin_B'])
    assert error < 1e-6 and new['proxy_rank0'] == old['proxy_rank0']
    anchor_score_checks.append({'id':old['id'],'max_abs_margin_delta':error,
        'utility_rank_equal':new['utility_rank0'] == old['utility_rank0']})
result = {'ok':True,'image_count':len(per_image),'candidate_count':len(candidates),
    'plan_sha256':hashlib.sha256((r/'plan.json').read_bytes()).hexdigest(),
    'bootstrap_spec':plan['full_analysis']['bootstrap'],
    'quality':{'usable':sum(x['quality_usable'] for x in candidates),
        'token_limit':sum(x['any_token_limit'] for x in candidates),
        'empty':sum(not x['trajectory_text'].strip() for x in candidates),
        'addCriterion':sum(x['contains_addCriterion'] for x in candidates),
        'stop_reasons':dict(collections.Counter(x['quality_stop_reason'] for x in candidates))},
    'pilot_generated_state_replays':len(old_records),'pilot_score_checks':anchor_score_checks,
    'groups':groups,'per_image':per_image}
(r/'完整逐图统计与重采样.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
table = []
for image in per_image:
    for subset in ['all','quality_pass']:
        for reward in REWARDS:
            for metric in METRICS:
                v = image['subsets'][subset][reward][metric]
                table.append({'row':image['row'],'image':image['image'],'pilot_image':image['pilot_image'],
                    'subset':subset,'reward':reward,'utility':metric,'usable_all8':image['usable'],
                    **{k:x for k,x in v.items() if k!='pair_counts'},**v['pair_counts']})
with (r/'逐图指标汇总.csv').open('w',encoding='utf-8',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(table[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(table)
print(json.dumps({'ok':True,'quality':result['quality'],
    'all22_primary':groups['all22']['subsets']['all']['r_retrieval']['utility_rr'],
    'new19_primary':groups['new19']['subsets']['all']['r_retrieval']['utility_rr']},ensure_ascii=False))
