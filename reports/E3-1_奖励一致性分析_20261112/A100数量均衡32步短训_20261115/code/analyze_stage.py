import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np
import torch
from stage_io import read,dump
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;plan=read(r/'plan.json');j=read(r/'scoring/all_scores.json');cases=read(r/'inputs/evaluation_cases.json');by={x['image']:x for x in cases}
torch.set_num_threads(2);t=torch.load(r/'scoring/tensors.pt',map_location='cpu',weights_only=True);ix={x:i for i,x in enumerate(t['record_ids'])};gallery=read(r/'features/gallery.json');words={w:i+1 for i,w in enumerate(['one','two','three','four','five','six','seven'])}
def count_pred(x):
    pos=by[x['image']]['pairs']['B']['positive'];prefix=pos.rsplit(' ',1)[0]+' ';columns={words[s.rsplit(' ',1)[1].rstrip('.')]:i for i,s in enumerate(gallery['quantity_text_order']) if s.startswith(prefix)}
    q=t['quantity_similarities'][ix[x['id']]];return max(sorted(columns),key=lambda k:float(q[columns[k]]))
metrics=['quality','strict_quality','tool_usage','generated_result_failure','margin_A','margin_B','pairacc_A','pairacc_B','retrieval_rr','argmax_two']
per=[]
for arm in ['before','after']:
    for item in cases:
        cohort='training8' if item['image'] in plan['training_images'] else 'confirmation55' if item['split']=='train' else 'existing_val22'
        xs=[x for x in j if x['arm']==arm and x['image']==item['image']];greedy=[x for x in xs if x['role']=='greedy'];samples=[x for x in xs if x['role']=='candidate'];shared=[x for x in j if x['arm']==arm+'_shared' and x['image']==item['image']]
        assert len(greedy)==len(shared)==1 and len(samples)==4
        for selector,values in [('greedy',greedy),('sample_mean',samples),('shared',shared)]:
            row={'arm':arm,'image':item['image'],'split':item['split'],'cohort':cohort,'actual_count':item['actual_count'],'object':item['object'],'selector':selector,'selected_ids':[x['id'] for x in values]}
            for metric in metrics:
                ns=[None if selector=='shared' and metric in ['quality','strict_quality','tool_usage','generated_result_failure'] else float(x['quality_usable']) if metric=='quality' else float(x['strict_protocol_usable']) if metric=='strict_quality' else float(x['tools_used']>0) if metric=='tool_usage' else float(x['stop_reason']=='model_generated_tool_result') if metric=='generated_result_failure' else float(x['correct_'+metric[-1]]) if metric.startswith('pairacc_') else float(count_pred(x)==2) if metric=='argmax_two' else x[metric] for x in values]
                row[metric]=None if ns[0] is None else math.fsum(ns)/len(ns)
            per.append(row)
lookup={(x['arm'],x['image'],x['selector']):x for x in per}
def seed(label):return (plan['analysis']['bootstrap_seed']+int(hashlib.sha256(label.encode()).hexdigest()[:8],16))%2**32
def stats(v,label):
    v=np.asarray(v,dtype=np.float64);rng=np.random.default_rng(seed(label));draw=rng.integers(0,len(v),size=(10000,len(v)))
    return {'mean':float(v.mean()),'interval95':np.percentile(v[draw].mean(axis=1),[2.5,97.5]).tolist(),'images':len(v)}
groups={};summary=[];macro_rows=[]
for cohort in ['training8','confirmation55','existing_val22']:
    items=[x for x in per if x['cohort']==cohort and x['arm']=='before' and x['selector']=='greedy'];names=[x['image'] for x in items];g={'images':len(names),'raw':{},'paired':{},'macro':{},'count_groups':{}}
    for arm in ['before','after']:
        g['raw'][arm]={}
        for selector in ['greedy','sample_mean','shared']:
            entry={}
            for metric in metrics:
                v=[lookup[arm,n,selector][metric] for n in names]
                if v[0] is not None:entry[metric]=stats(v,cohort+'/'+arm+'/'+selector+'/'+metric)
            g['raw'][arm][selector]=entry
    for selector in ['greedy','sample_mean']:
        g['paired'][selector]={}
        for metric in metrics:
            values=[lookup['after',n,selector][metric] for n in names]
            if values[0] is None:continue
            g['paired'][selector][metric]=stats([lookup['after',n,selector][metric]-lookup['before',n,selector][metric] for n in names],cohort+'/paired/'+selector+'/'+metric)
    for count in sorted({x['actual_count'] for x in items}):
        ns=[x['image'] for x in items if x['actual_count']==count];g['count_groups'][str(count)]={}
        for arm in ['before','after']:
            g['count_groups'][str(count)][arm]={s:{m:float(np.mean([lookup[arm,n,s][m] for n in ns])) for m in ['pairacc_B','margin_B','argmax_two']} for s in ['greedy','sample_mean','shared']}
        g['count_groups'][str(count)]['images']=len(ns)
    common=[x for x in items if x['actual_count'] in [1,2,3,4]]
    contrasts={'after_minus_before__sample_mean':('after','sample_mean','before','sample_mean'),'after_minus_before__greedy':('after','greedy','before','greedy'),
      'after_minus_before__shared':('after','shared','before','shared'),'after_own_minus_shared__greedy':('after','greedy','after','shared'),'before_own_minus_shared__greedy':('before','greedy','before','shared')}
    for contrast,(arm,sel,base,bsel) in contrasts.items():
        g['macro'][contrast]={}
        for metric in ['pairacc_B','margin_B','pairacc_A','argmax_two']:
            rng=np.random.default_rng(seed(cohort+'/macro/'+contrast+'/'+metric));boot=np.zeros(10000);point=0.
            for count in [1,2,3,4]:
                ns=[x['image'] for x in common if x['actual_count']==count];assert ns
                vals=np.asarray([lookup[arm,n,sel][metric]-lookup[base,n,bsel][metric] for n in ns]);point+=float(vals.mean())/4;draw=rng.integers(0,len(vals),size=(10000,len(vals)));boot+=vals[draw].mean(axis=1)/4
            value={'mean':point,'interval95':np.percentile(boot,[2.5,97.5]).tolist(),'images':len(common),'counts':[1,2,3,4]};g['macro'][contrast][metric]=value
            macro_rows.append({'cohort':cohort,'contrast':contrast,'metric':metric,**value})
    for arm in ['before','after']:
        for selector in ['greedy','sample_mean','shared']:
            g.setdefault('macro_absolute',{}).setdefault(arm,{})[selector]={m:float(np.mean([np.mean([lookup[arm,x['image'],selector][m] for x in common if x['actual_count']==c]) for c in [1,2,3,4]])) for m in ['pairacc_B','margin_B','pairacc_A','argmax_two']}
    groups[cohort]=g
tr=read(r/'training/result.json');updates=[json.loads(s) for s in (r/'training/updates.jsonl').read_text().splitlines()];gate=plan['acceptance'];g=groups['confirmation55'];m=g['raw']['after']
checks={'optimizer_updates_exactly_32':tr['optimizer_updates']==32,'updates_per_count_exactly_8':all(sum(x['actual_count']==c for x in updates)==8 for c in [1,2,3,4]),
 'updates_per_image_exactly_4':all(sum(x['image']==n for x in updates)==4 for n in plan['training_images']),
 'holdout_sample_strict_quality_at_least_90pct':m['sample_mean']['strict_quality']['mean']>=gate['holdout_sample_strict_quality_min'],
 'holdout_greedy_strict_quality_at_least_95pct':m['greedy']['strict_quality']['mean']>=gate['holdout_greedy_strict_quality_min'],
 'holdout_sample_actual_tools_at_least_80pct':m['sample_mean']['tool_usage']['mean']>=gate['holdout_sample_tool_use_min'],
 'holdout_generated_result_failure_at_most_2pct':m['sample_mean']['generated_result_failure']['mean']<=gate['generated_result_failure_max'],
 'holdout_count_macro_B_gain_interval_above_zero':g['macro']['after_minus_before__sample_mean']['pairacc_B']['interval95'][0]>0,
 'holdout_own_minus_shared_macro_B_interval_above_zero':g['macro']['after_own_minus_shared__greedy']['pairacc_B']['interval95'][0]>0}
dump(r/'analysis.json',{'ok':True,'plan_sha256':__import__('hashlib').sha256((r/'plan.json').read_bytes()).hexdigest(),'training':tr,'groups':groups,'per_image':per,'adoption':{'passed':all(checks.values()),'checks':checks,'scope':'confirmation55 only; common-count macro based on52 images'},'baseline_replay':read(r/'contract_gate/warm_baseline_replay.json'),'scoring':read(r/'scoring/result.json')})
for name,rows in [('per_image_results.csv',per),('macro_contrasts.csv',macro_rows),('all_scores.csv',j)]:
    fields=list(dict.fromkeys(k for x in rows for k in x))
    with (r/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,list) else v for k,v in x.items()} for x in rows)
print(json.dumps({'adoption':checks,'primary_macro_B':g['macro']['after_minus_before__sample_mean']['pairacc_B'],'own_shared_macro_B':g['macro']['after_own_minus_shared__greedy']['pairacc_B']},ensure_ascii=False),flush=True)
