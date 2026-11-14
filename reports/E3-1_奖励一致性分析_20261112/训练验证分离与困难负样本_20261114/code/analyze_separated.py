import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
plan=json.loads((r/'plan.json').read_text());assert json.loads((r/'run_result.json').read_text())['status']=='complete'
branch=json.loads((r/'pilot/branch_decision.json').read_text());policies=branch['policies'];rows=plan['validation_rows']
scores=json.loads((r/'scoring/scores.json').read_text());assert len(scores)==150*len(policies)
metrics=['utility_rr','utility_margin','quality','tool_usage'];modes=['greedy','sample_mean'];per=[]
for policy in policies:
    for row in rows:
        for mode in modes:
            xs=[x for x in scores if x['policy']==policy and x['row']==row and x['role']==('greedy' if mode=='greedy' else 'candidate')]
            assert len(xs)==(1 if mode=='greedy' else 2)
            item={'policy':policy,'row':row,'mode':mode,'selected_ids':[x['id'] for x in xs]}
            for m in metrics:
                values=[float(x['quality_usable']) if m=='quality' else float(x['tools_used']>0) if m=='tool_usage' else x[m] for x in xs]
                item[m]=math.fsum(values)/len(values)
            per.append(item)
index={(x['policy'],x['row'],x['mode']):x for x in per}
draws=np.random.default_rng(plan['evaluation']['bootstrap']['seed']).integers(0,50,size=(10000,50))

def stats(v):
    v=np.asarray(v,dtype=np.float64);lo,hi=np.percentile(v[draws].mean(axis=1),[2.5,97.5])
    return {'mean':float(v.mean()),'interval95':[float(lo),float(hi)],'images':50,'image_std_ddof1':float(v.std(ddof=1))}

means={};summary=[];paired={};paired_rows=[]
for p in policies:
    means[p]={}
    for mode in modes:
        means[p][mode]={}
        for m in metrics:
            s=stats([index[p,row,mode][m] for row in rows]);means[p][mode][m]=s
            summary.append({'policy':p,'mode':mode,'metric':m,'mean':s['mean'],'ci95_lower':s['interval95'][0],'ci95_upper':s['interval95'][1]})
contrasts=[('global_margin','baseline'),('global_margin','format_only')]
if branch['scene_arm_enabled']:contrasts += [('scene_margin',b) for b in ['baseline','format_only','global_margin']]
for policy,base in contrasts:
    name=policy+'_minus_'+base;paired[name]={}
    for mode in modes:
        paired[name][mode]={}
        for m in metrics:
            diffs=[index[policy,row,mode][m]-index[base,row,mode][m] for row in rows]
            eps=1e-6 if m=='utility_margin' else 1e-12
            s={**stats(diffs),'wins':sum(x>eps for x in diffs),'ties':sum(abs(x)<=eps for x in diffs),'losses':sum(x< -eps for x in diffs)}
            paired[name][mode][m]=s
            paired_rows.append({'policy':policy,'baseline':base,'mode':mode,'metric':m,'mean':s['mean'],
                                'ci95_lower':s['interval95'][0],'ci95_upper':s['interval95'][1],
                                'wins':s['wins'],'ties':s['ties'],'losses':s['losses']})
training={arm:json.loads((r/'training'/arm/'result.json').read_text()) for arm in branch['arms']}
chosen='scene_margin' if branch['scene_arm_enabled'] else 'global_margin'
comparators=['global_margin','format_only'] if branch['scene_arm_enabled'] else ['baseline','format_only']
checks={f'greedy_rr_ci_above_{base}':paired[chosen+'_minus_'+base]['greedy']['utility_rr']['interval95'][0]>0 for base in comparators}
checks.update(greedy_rr_at_least_initial=means[chosen]['greedy']['utility_rr']['mean']>=means['baseline']['greedy']['utility_rr']['mean'],
              greedy_quality_within_5pp=means[chosen]['greedy']['quality']['mean']-means['baseline']['greedy']['quality']['mean']>=-.05,
              optimizer_updated=training[chosen]['optimizer_updates']>0)
result={'ok':True,'plan_sha256':hashlib.sha256((r/'plan.json').read_bytes()).hexdigest(),'pilot':branch,
        'policies':policies,'training':training,'validation_images':50,'evaluation_scores':len(scores),'policy_means':means,
        'paired':paired,'per_image':per,'decision':{'chosen_by_training_only_gate':chosen,'checks':checks,'expand_criterion_passed':all(checks.values())}}
(r/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
for name,xs in [('policy_summary.csv',summary),('paired_policy_deltas.csv',paired_rows),('per_image_policy.csv',per),('evaluation_scores.csv',scores)]:
    with (r/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(xs[0]),lineterminator='\n');w.writeheader()
        for x in xs:w.writerow({k:json.dumps(v) if isinstance(v,list) else v for k,v in x.items()})
print(json.dumps({'decision':result['decision'],'greedy_rr':{p:means[p]['greedy']['utility_rr']['mean'] for p in policies},
                  'greedy_quality':{p:means[p]['greedy']['quality']['mean'] for p in policies}}))
