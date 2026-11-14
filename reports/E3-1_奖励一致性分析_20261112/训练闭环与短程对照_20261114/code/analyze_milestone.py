"""Prespecified paired-image diagnostic for the three final policy points."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
plan=json.loads((r/'plan.json').read_text())
assert json.loads((r/'run_result.json').read_text())['status']=='complete'
scores=json.loads((r/'scoring/scores.json').read_text())
assert len(scores)==330 and len({x['id'] for x in scores})==330
policies=plan['evaluation']['policy_points']
modes=['greedy','sample_mean','retrieval_select','total_select']
metrics=['utility_rr','utility_margin','E1_margin_A','E1_margin_B','quality','tool_usage','r_retrieval']
per_image=[]
for policy in policies:
    for row in plan['validation_rows']:
        entries=[x for x in scores if x['policy']==policy and x['row']==row]
        greedy=[x for x in entries if x['role']=='greedy'];candidates=[x for x in entries if x['role']=='candidate']
        assert len(greedy)==1 and len(candidates)==4
        selected={'greedy':greedy,'sample_mean':candidates}
        for mode,key in [('retrieval_select','r_retrieval'),('total_select','r_total')]:
            top=max(x[key] for x in candidates);selected[mode]=[x for x in candidates if x[key]==top]
        for mode,values in selected.items():
            d={'policy':policy,'row':row,'image':entries[0]['image'],'mode':mode,'selected_ids':[x['id'] for x in values]}
            for metric in metrics:
                field=[float(x['quality_usable']) if metric=='quality' else float(x['tools_used']>0)
                       if metric=='tool_usage' else x[metric] for x in values]
                d[metric]=math.fsum(field)/len(field)
            per_image.append(d)
index={(x['policy'],x['row'],x['mode']):x for x in per_image}


def stats(values,draws):
    x=np.asarray(values,dtype=np.float64);interval=np.percentile(x[draws].mean(axis=1),[2.5,97.5])
    return {'mean':float(x.mean()),'image_std_ddof1':float(x.std(ddof=1)),
            'interval95':[float(v) for v in interval],'images':len(x)}


groups={};summary_rows=[];paired_rows=[]
contrasts=[('retrieval','baseline'),('retrieval','format_only'),('format_only','baseline')]
for label,rows in plan['validation_groups'].items():
    draws=np.random.default_rng(plan['evaluation']['bootstrap']['seed']).integers(0,len(rows),
          size=(plan['evaluation']['bootstrap']['resamples'],len(rows)))
    group={'images':len(rows),'policy_means':{},'paired':{}}
    for policy in policies:
        group['policy_means'][policy]={}
        for mode in modes:
            group['policy_means'][policy][mode]={}
            for metric in metrics:
                s=stats([index[policy,row,mode][metric] for row in rows],draws)
                group['policy_means'][policy][mode][metric]=s
                summary_rows.append({'group':label,'policy':policy,'mode':mode,'metric':metric,'mean':s['mean'],
                    'image_std_ddof1':s['image_std_ddof1'],'images':s['images'],
                    'ci95_lower':s['interval95'][0],'ci95_upper':s['interval95'][1]})
    for policy,baseline in contrasts:
        name=policy+'_minus_'+baseline;group['paired'][name]={}
        for mode in modes:
            group['paired'][name][mode]={}
            for metric in metrics:
                x=[index[policy,row,mode][metric] for row in rows]
                y=[index[baseline,row,mode][metric] for row in rows]
                delta=[u-v for u,v in zip(x,y)];epsilon=1e-6 if 'margin' in metric else 1e-12
                s={**stats(delta,draws),'baseline_mean':float(np.mean(y)),
                   'relative_improvement':float(np.mean(delta)/np.mean(y)) if metric=='utility_rr' and np.mean(y)>0 else None,
                   'wins':sum(v>epsilon for v in delta),'ties':sum(abs(v)<=epsilon for v in delta),
                   'losses':sum(v< -epsilon for v in delta)}
                group['paired'][name][mode][metric]=s
                paired_rows.append({'group':label,'policy':policy,'baseline':baseline,'mode':mode,'metric':metric,
                    **{k:v for k,v in s.items() if k!='interval95'},'ci95_lower':s['interval95'][0],'ci95_upper':s['interval95'][1]})
    groups[label]=group
training={arm:json.loads((r/'training'/arm/'result.json').read_text()) for arm in ['smoke','retrieval','format_only']}
initial=training['smoke']['initial_lora_tensor_hash']
assert all(training[arm]['initial_lora_tensor_hash']==initial for arm in ['retrieval','format_only'])
assert json.loads((r/'evaluation/baseline/settings.json').read_text())['lora_tensor_hash']==initial
for arm in ['retrieval','format_only']:
    assert training[arm]['attempted_steps']==20
    assert json.loads((r/'evaluation'/arm/'settings.json').read_text())['lora_tensor_hash']==training[arm]['final_lora_tensor_hash']
main=groups['all22'];paired=main['paired']
rr_baseline=paired['retrieval_minus_baseline']['greedy']['utility_rr']
rr_control=paired['retrieval_minus_format_only']['greedy']['utility_rr']
quality_delta=paired['retrieval_minus_baseline']['greedy']['quality']['mean']
checks={'rr_gain_interval_above_baseline':rr_baseline['interval95'][0]>0,
        'rr_gain_interval_above_format_control':rr_control['interval95'][0]>0,
        'greedy_quality_within_5_percentage_points':quality_delta>=-.05,
        'retrieval_optimizer_updated':training['retrieval']['optimizer_updates']>0}
decision={'expand_criterion_passed':all(checks.values()),'checks':checks,
          'criterion':plan['evaluation']['expand_criterion']}
result={'ok':True,'plan_sha256':hashlib.sha256((r/'plan.json').read_bytes()).hexdigest(),
        'training':training,'groups':groups,'per_image':per_image,'decision':decision,
        'evaluation_trajectories':len(scores),'paired_bootstrap_unit':'image; one training initialization per arm'}
(r/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
for name,rows in [('per_image_policy.csv',per_image),('policy_summary.csv',summary_rows),('paired_policy_deltas.csv',paired_rows),('evaluation_scores.csv',scores)]:
    with (r/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader()
        w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v for k,v in x.items()} for x in rows)
print(json.dumps({'decision':decision,'main_greedy_rr':{p:main['policy_means'][p]['greedy']['utility_rr'] for p in policies},
                  'main_greedy_quality':{p:main['policy_means'][p]['greedy']['quality']['mean'] for p in policies},
                  'rr_vs_baseline':rr_baseline,'rr_vs_format_control':rr_control},ensure_ascii=False))
