import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
plan=json.loads((r/'plan.json').read_text());assert json.loads((r/'run_result.json').read_text())['status']=='complete'
cases=json.loads((r/'inputs/cases.json').read_text());scores=json.loads((r/'scoring/scores.json').read_text());assert len(scores)==500
selectors=list(plan['selectors']);metrics=['margin_A','margin_B','pairacc_A','pairacc_B','quality','tool_usage','retrieval_rr','retrieval_margin']
per=[];diagnostics=[]

def rank(v):
    values=np.asarray(v);return np.array([np.count_nonzero(values<x)+(np.count_nonzero(values==x)-1)/2 for x in values])

for item in cases:
    xs=[x for x in scores if x['image']==item['image']];greedy=next(x for x in xs if x['role']=='greedy')
    candidates=sorted([x for x in xs if x['role']=='candidate'],key=lambda x:x['seed']);assert len(candidates)==4
    usable=[x for x in candidates if x['quality_usable']]
    selected={'greedy':[greedy],'sample_mean':candidates,'format_first':[usable[0] if usable else greedy],
        'quantity_raw':[max(candidates,key=lambda x:x['margin_A'])],
        'quantity_quality':[max(usable,key=lambda x:x['margin_A']) if usable else greedy],
        'reverse_quality':[min(usable,key=lambda x:x['margin_A']) if usable else greedy]}
    for selector,values in selected.items():
        entry={'image':item['image'],'split':item['split'],'object':item['object'],'actual_count':item['actual_count'],
               'selector':selector,'selected_ids':[x['id'] for x in values],'greedy_fallback':selector in ['format_first','quantity_quality','reverse_quality'] and not usable}
        for metric in metrics:
            numbers=[float(x['correct_'+metric[-1]]) if metric.startswith('pairacc_') else float(x['quality_usable']) if metric=='quality'
                     else float(x['tools_used']>0) if metric=='tool_usage' else x[metric] for x in values]
            entry[metric]=None if numbers[0] is None else math.fsum(numbers)/len(numbers)
        per.append(entry)
    av=[x['margin_A'] for x in usable];bv=[x['margin_B'] for x in usable]
    informative=len(usable)>=2 and max(av)-min(av)>1e-6
    rho=None
    if len(usable)>=2 and np.ptp(av)>1e-12 and np.ptp(bv)>1e-12:rho=float(np.corrcoef(rank(av),rank(bv))[0,1])
    concordant=discordant=tied=0
    for i in range(len(usable)):
        for k in range(i):
            da=av[i]-av[k];db=bv[i]-bv[k]
            if abs(da)<=1e-12 or abs(db)<=1e-12:tied+=1
            elif da*db>0:concordant+=1
            else:discordant+=1
    diagnostics.append({'image':item['image'],'split':item['split'],'usable_candidates':len(usable),'informative':informative,
        'A_range_usable':float(np.ptp(av)) if av else 0.,'B_range_usable':float(np.ptp(bv)) if bv else 0.,
        'within_image_spearman':rho,'concordant_pairs':concordant,'discordant_pairs':discordant,'tied_pairs':tied,
        'unique_candidate_hidden_states':len({x['generation_state_sha256'] for x in candidates})})
index={(x['image'],x['selector']):x for x in per};groups={};summary=[];paired_rows=[]
comparisons=[('quantity_quality','format_first'),('quantity_quality','reverse_quality'),('quantity_quality','greedy'),('quantity_raw','format_first')]
for split in ['train','val']:
    images=[x['image'] for x in cases if x['split']==split];n=len(images)
    draws=np.random.default_rng(plan['analysis']['bootstrap']['seed']).integers(0,n,size=(10000,n))
    def stats(v):
        v=np.asarray(v,dtype=np.float64);ci=np.percentile(v[draws].mean(axis=1),[2.5,97.5])
        return {'mean':float(v.mean()),'interval95':[float(x) for x in ci],'images':n}
    g={'images':n,'selector_means':{},'paired':{}}
    for selector in selectors:
        g['selector_means'][selector]={}
        for metric in metrics:
            numbers=[index[image,selector][metric] for image in images]
            if numbers[0] is None:continue
            s=stats(numbers);g['selector_means'][selector][metric]=s
            summary.append({'split':split,'selector':selector,'metric':metric,'mean':s['mean'],'ci95_lower':s['interval95'][0],'ci95_upper':s['interval95'][1]})
    for selector,base in comparisons:
        name=selector+'_minus_'+base;g['paired'][name]={}
        for metric in metrics:
            if index[images[0],selector][metric] is None:continue
            delta=[index[image,selector][metric]-index[image,base][metric] for image in images]
            eps=1e-6 if metric.startswith('margin') or metric=='retrieval_margin' else 1e-12
            s={**stats(delta),'wins':sum(x>eps for x in delta),'ties':sum(abs(x)<=eps for x in delta),'losses':sum(x< -eps for x in delta)}
            g['paired'][name][metric]=s
            paired_rows.append({'split':split,'selector':selector,'baseline':base,'metric':metric,'mean':s['mean'],
                'ci95_lower':s['interval95'][0],'ci95_upper':s['interval95'][1],'wins':s['wins'],'ties':s['ties'],'losses':s['losses']})
    ds=[x for x in diagnostics if x['split']==split];rhos=[x['within_image_spearman'] for x in ds if x['within_image_spearman'] is not None]
    g['candidate_diagnostics']={'informative_images':sum(x['informative'] for x in ds),'images_with_2_usable':sum(x['usable_candidates']>=2 for x in ds),
        'mean_within_image_spearman':float(np.mean(rhos)) if rhos else None,'spearman_defined_images':len(rhos),
        'concordant_pairs':sum(x['concordant_pairs'] for x in ds),'discordant_pairs':sum(x['discordant_pairs'] for x in ds),
        'fallback_images':sum(index[image,'quantity_quality']['greedy_fallback'] for image in images)}
    groups[split]=g
g=groups['train'];threshold=plan['analysis']['next_training_gate'];gain=g['paired']['quantity_quality_minus_format_first']['margin_B']
checks={'training_candidate_protocol_quality':g['selector_means']['sample_mean']['quality']['mean']>=threshold['train_generation_quality_min'],
        'at_least_47_informative_training_images':g['candidate_diagnostics']['informative_images']>=threshold['train_images_with_2_usable_and_A_range_above_1e_6_min'],
        'B_margin_gain_at_least_1e_4':gain['mean']>=threshold['heldout_B_margin_gain_min'],
        'B_margin_gain_interval_above_zero':gain['interval95'][0]>0,
        'B_pairacc_at_least_format_first':g['selector_means']['quantity_quality']['pairacc_B']['mean']>=g['selector_means']['format_first']['pairacc_B']['mean'],
        'B_margin_vs_reverse_interval_above_zero':g['paired']['quantity_quality_minus_reverse_quality']['margin_B']['interval95'][0]>0,
        'selected_protocol_quality_at_least_95pct':g['selector_means']['quantity_quality']['quality']['mean']>=threshold['quality_selected_min']}
result={'ok':True,'plan_sha256':hashlib.sha256((r/'plan.json').read_bytes()).hexdigest(),'optimization_updates':0,
        'policy':plan['policy'],'groups':groups,'per_image':per,'candidate_diagnostics':diagnostics,
        'decision':{'next_training_gate_passed':all(checks.values()),'checks':checks,'gate_uses_train_only':True,
                    'label_assisted_selection_diagnostic':True},'full_generation_records':500,'smoke_records':3}
(r/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
for name,rows in [('selector_summary.csv',summary),('paired_selector_deltas.csv',paired_rows),('per_image_selectors.csv',per),('candidate_diagnostics.csv',diagnostics),('quantity_scores.csv',scores)]:
    with (r/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v) if isinstance(v,list) else v for k,v in row.items()})
print(json.dumps({'decision':result['decision'],'train_B_margin_gain':gain,'val_B_margin_gain':groups['val']['paired']['quantity_quality_minus_format_first']['margin_B']}))
