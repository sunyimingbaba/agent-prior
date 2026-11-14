import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
plan=json.loads((r/'plan.json').read_text());assert json.loads((r/'run_result.json').read_text())['status']=='complete'
scores=json.loads((r/'scoring/scores.json').read_text());assert len(scores)==864
policies=plan['evaluation']['policy_points'];modes=['greedy','sample_mean']
metrics=['utility_rr','utility_margin','quality','tool_usage','r_proxy'];per_image=[]
for policy in policies:
    for row in plan['validation_rows']:
        entries=[x for x in scores if x['policy']==policy and x['row']==row]
        selected={'greedy':[x for x in entries if x['role']=='greedy'],
                  'sample_mean':[x for x in entries if x['role']=='candidate']}
        assert len(selected['greedy'])==1 and len(selected['sample_mean'])==2
        for mode,values in selected.items():
            d={'policy':policy,'row':row,'image':entries[0]['image'],'mode':mode,'selected_ids':[x['id'] for x in values]}
            for metric in metrics:
                numbers=[float(x['quality_usable']) if metric=='quality' else float(x['tools_used']>0)
                         if metric=='tool_usage' else x[metric] for x in values]
                d[metric]=math.fsum(numbers)/len(numbers)
            per_image.append(d)
index={(x['policy'],x['row'],x['mode']):x for x in per_image}


def stats(values,draws):
    x=np.asarray(values,dtype=np.float64);ci=np.percentile(x[draws].mean(axis=1),[2.5,97.5])
    return {'mean':float(x.mean()),'image_std_ddof1':float(x.std(ddof=1)),
            'interval95':[float(v) for v in ci],'images':len(x)}


groups={};summary=[];paired=[]
contrasts=[('aligned_margin',x) for x in ['baseline','retrieval','format_only']]+[('retrieval','format_only')]
for group,rows in plan['validation_groups'].items():
    draw=np.random.default_rng(plan['evaluation']['bootstrap']['seed']).integers(0,len(rows),size=(10000,len(rows)))
    g={'images':len(rows),'policy_means':{},'paired':{}}
    for policy in policies:
        g['policy_means'][policy]={}
        for mode in modes:
            g['policy_means'][policy][mode]={}
            for metric in metrics:
                s=stats([index[policy,row,mode][metric] for row in rows],draw)
                g['policy_means'][policy][mode][metric]=s
                summary.append({'group':group,'policy':policy,'mode':mode,'metric':metric,'mean':s['mean'],
                                'image_std_ddof1':s['image_std_ddof1'],'images':s['images'],
                                'ci95_lower':s['interval95'][0],'ci95_upper':s['interval95'][1]})
    for policy,baseline in contrasts:
        name=policy+'_minus_'+baseline;g['paired'][name]={}
        for mode in modes:
            g['paired'][name][mode]={}
            for metric in metrics:
                x=[index[policy,row,mode][metric] for row in rows];y=[index[baseline,row,mode][metric] for row in rows]
                delta=[u-v for u,v in zip(x,y)];epsilon=1e-6 if metric=='utility_margin' else 1e-12
                s={**stats(delta,draw),'baseline_mean':float(np.mean(y)),
                   'relative_improvement':float(np.mean(delta)/np.mean(y)) if metric=='utility_rr' and np.mean(y)>0 else None,
                   'wins':sum(v>epsilon for v in delta),'ties':sum(abs(v)<=epsilon for v in delta),'losses':sum(v< -epsilon for v in delta)}
                g['paired'][name][mode][metric]=s
                paired.append({'group':group,'policy':policy,'baseline':baseline,'mode':mode,'metric':metric,
                               **{k:v for k,v in s.items() if k!='interval95'},'ci95_lower':s['interval95'][0],'ci95_upper':s['interval95'][1]})
    groups[group]=g
training=json.loads((r/'training/aligned_margin/result.json').read_text())
assert training['attempted_steps']==20 and training['initial_lora_tensor_hash']==plan['inherited_initial_hash']
g=groups['new50'];means=g['policy_means'];d=g['paired']
checks={'new50_rr_interval_above_proxy':d['aligned_margin_minus_retrieval']['greedy']['utility_rr']['interval95'][0]>0,
        'new50_rr_interval_above_format':d['aligned_margin_minus_format_only']['greedy']['utility_rr']['interval95'][0]>0,
        'new50_greedy_rr_at_least_initial':means['aligned_margin']['greedy']['utility_rr']['mean']>=means['baseline']['greedy']['utility_rr']['mean'],
        'new50_greedy_quality_within_5pp':d['aligned_margin_minus_baseline']['greedy']['quality']['mean']>=-.05,
        'optimizer_updated':training['optimizer_updates']>0}
result={'ok':True,'plan_sha256':hashlib.sha256((r/'plan.json').read_bytes()).hexdigest(),
        'groups':groups,'per_image':per_image,'aligned_training':training,'inherited_controls':plan['inherited_policies'],
        'decision':{'expand_criterion_passed':all(checks.values()),'checks':checks,'criterion':plan['evaluation']['expand_criterion']},
        'evaluation_scores':864,'new_generations':666,'reused_legacy_trajectories':198}
(r/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
for name,rows in [('per_image_policy.csv',per_image),('policy_summary.csv',summary),('paired_policy_deltas.csv',paired),('evaluation_scores.csv',scores)]:
    with (r/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader()
        w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v for k,v in x.items()} for x in rows)
print(json.dumps({'decision':result['decision'],'new50_greedy':{p:{m:means[p]['greedy'][m]['mean'] for m in metrics} for p in policies},
    'vs_proxy':d['aligned_margin_minus_retrieval']['greedy']['utility_rr'],
    'vs_format':d['aligned_margin_minus_format_only']['greedy']['utility_rr']},ensure_ascii=False))
