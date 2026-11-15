import collections
import csv
import itertools
import json
import math
import time
import numpy as np
import torch
from analysis_common import ROOT,read,dump,sha,stat,quantity_metrics,select,concordance
start=time.monotonic();torch.set_num_threads(2);plan=read(ROOT/'plan.json');cases=read(ROOT/'inputs/cases.json');by={x['image']:x for x in cases};gallery=read(ROOT/'features/gallery.json');raw=read(ROOT/'scoring/all_scores.json');t=torch.load(ROOT/'scoring/tensors.pt',map_location='cpu',weights_only=True);assert t['record_ids']==[x['id'] for x in raw]
policies=[x['name'] for x in plan['policies']];metrics=['A_pair_correct','B_pair_correct','A_hard_correct','B_hard_correct','A_hard_margin','B_hard_margin'];grid=[]
for i,x in enumerate(raw):
    m=quantity_metrics(t['quantity_similarities'][i],gallery,by[x['target_image']]);assert m['A_pair_margin']==x['margin_A'] and m['B_pair_margin']==x['margin_B']
    grid.append({**x,**m})
lookup={(x['policy'],x['grid_role'],x['sample'],x['donor_image'],x['target_image']):x for x in grid};assert len(lookup)==2176
new={p:[json.loads(s) for s in (ROOT/'generation'/p/'records.jsonl').read_text().splitlines()] for p in policies};historical=read(ROOT/'historical/records.json');ht=torch.load(ROOT/'historical/tensors.pt',map_location='cpu',weights_only=True);historical_metrics=[]
for i,x in enumerate(historical):
    z=quantity_metrics(ht['quantity_similarities'][i],gallery,by[x['image']]);historical_metrics.append({**x,**z})
per=[];selection=[];coverage=[];donor_means=[]
for policy in policies:
    for case in cases:
        target=case['image'];count=case['actual_count'];own=[lookup[policy,'new_sample',k,target,target] for k in range(16)];old=[x for x in historical_metrics if x['policy']==policy and x['image']==target and x['role']=='candidate'];assert len(old)==4
        # Build one decision per image, retaining all sixteen samples within it.
        e={'policy':policy,'image':target,'object':case['object'],'actual_count':count}
        for role,ks in [('new_sample',list(range(16))),('reused_greedy',[None])]:
            xs=[lookup[policy,role,k,target,target] for k in ks];donors=[[lookup[policy,role,k,n,target] for n in plan['images'] if n!=target] for k in ks];same=[[x for x in ds if x['same_object']] for ds in donors]
            row={**e,'role':role,'sample_count':len(ks)}
            for metric in metrics:
                row['own_'+metric]=float(np.mean([x[metric] for x in xs]));row['donor_'+metric]=float(np.mean([x[metric] for ds in donors for x in ds]));row['own_minus_donor_'+metric]=row['own_'+metric]-row['donor_'+metric]
                row['same_object_donor_'+metric]=None if not same[0] else float(np.mean([x[metric] for ds in same for x in ds]));row['own_minus_same_object_'+metric]=None if not same[0] else row['own_'+metric]-row['same_object_donor_'+metric]
            per.append(row)
            for donor in plan['images']:
                ys=[lookup[policy,role,k,donor,target] for k in ks];donor_means.append({**e,'role':role,'donor_image':donor,'donor_actual_count':by[donor]['actual_count'],'same_image':donor==target,'same_object':by[donor]['object']==case['object'],**{m:float(np.mean([x[m] for x in ys])) for m in metrics}})
        generation={x['sample']:x for x in new[policy] if x['image']==target}
        # Native diagonal scores inherit generation quality and seed from their donor.
        for xs in [own,old]:
            for x in xs:
                if x in old:quality=x['quality_usable'];strict=x['strict_protocol_usable'];tool=x['tools_used']
                else:quality=x['donor_quality'];strict=x['donor_strict_quality'];tool=x['donor_tools_used']
                x.update(quality=float(quality),strict_quality=float(strict),tool_usage=float(tool>0),pair_reward=.2*float(quality)+x['A_pair_margin'],hard_reward=.2*float(quality)+x['A_hard_margin'])
        for size,xs in [('new4',own[:4]),('new16',own),('historical4',old)]:
            qual=[x for x in xs if x['quality']];eligible=qual or xs;rank=concordance(xs)
            coverage.append({**e,'pool':size,'candidates':len(xs),'qualified_candidates':len(qual),'B_hard_decision_varies':len({x['B_hard_correct'] for x in xs})>1,'A_hard_decision_varies':len({x['A_hard_correct'] for x in xs})>1,'qualified_B_oracle_success':float(any(x['B_hard_correct'] for x in qual)),'B_hard_uniform':float(np.mean([x['B_hard_correct'] for x in xs])),'strict_quality':float(np.mean([x['strict_quality'] for x in xs])),'quality':float(np.mean([x['quality'] for x in xs])),'tool_usage':float(np.mean([x['tool_usage'] for x in xs])),'A_B_hard_concordance':rank,'A_hard_margin_range':float(np.ptp([x['A_hard_margin'] for x in xs])),'B_hard_margin_range':float(np.ptp([x['B_hard_margin'] for x in xs]))})
            picks={'uniform':xs,'pair_total':[select(xs,'pair_reward')],'hard_total':[select(xs,'hard_reward')],'hard_quality_first':[select(eligible,'A_hard_margin')],'B_oracle_quality_first':[select(eligible,'B_hard_margin')]}
            for selector,ys in picks.items():selection.append({**e,'pool':size,'selector':selector,'selected_ids':[x['donor_id'] if 'donor_id' in x else x['id'] for x in ys],**{m:float(np.mean([x[m] for x in ys])) for m in metrics+['quality','strict_quality','tool_usage']}})
# All bootstrap statistics record the exact paired image values used.
stats=[]
def emit(label,rows,values,macro=True):
    st=stat(values,[x['actual_count'] for x in rows],label,macro);stats.append({'label':label,'scope':'count_macro' if macro else 'image_mean','image_names':[x['image'] for x in rows],'counts':[x['actual_count'] for x in rows],'values':values,**st});return st
summary={}
for policy in policies:
    p={'native_vs_swapped':{},'coverage':{},'selectors':{},'coverage_expansion':{},'same_object_airplanes':{}}
    for role in ['new_sample','reused_greedy']:
        es=[x for x in per if x['policy']==policy and x['role']==role];p['native_vs_swapped'][role]={}
        for metric in metrics:
            p['native_vs_swapped'][role][metric]={kind:emit(policy+'/'+role+'/'+kind+'/'+metric,es,[x[kind+'_'+metric] for x in es]) for kind in ['own','donor','own_minus_donor']}
        plane=[x for x in es if x['object']=='飞机'];assert len(plane)==6
        p['same_object_airplanes'][role]={metric:emit(policy+'/airplanes/'+role+'/'+metric,plane,[x['own_minus_same_object_'+metric] for x in plane],False) for metric in ['B_hard_correct','B_hard_margin','B_pair_correct']}
    for pool in ['historical4','new4','new16']:
        es=[x for x in coverage if x['policy']==policy and x['pool']==pool]
        p['coverage'][pool]={metric:emit(policy+'/coverage/'+pool+'/'+metric,es,[float(x[metric]) for x in es]) for metric in ['B_hard_uniform','qualified_B_oracle_success','quality','strict_quality','tool_usage']}
        p['coverage'][pool].update(B_decision_varies_images=sum(x['B_hard_decision_varies'] for x in es),A_decision_varies_images=sum(x['A_hard_decision_varies'] for x in es),images=8,mean_A_B_concordance=float(np.mean([x['A_B_hard_concordance'] for x in es if x['A_B_hard_concordance'] is not None])))
        p['selectors'][pool]={}
        for selector in ['uniform','pair_total','hard_total','hard_quality_first','B_oracle_quality_first']:
            rows=[x for x in selection if x['policy']==policy and x['pool']==pool and x['selector']==selector]
            p['selectors'][pool][selector]={metric:emit(policy+'/selector/'+pool+'/'+selector+'/'+metric,rows,[x[metric] for x in rows]) for metric in ['B_hard_correct','B_hard_margin','B_pair_correct','strict_quality']}
    for metric in ['qualified_B_oracle_success','B_hard_uniform','quality','strict_quality']:
        small={x['image']:x for x in coverage if x['policy']==policy and x['pool']=='new4'};large=[x for x in coverage if x['policy']==policy and x['pool']=='new16'];p['coverage_expansion'][metric]=emit(policy+'/expansion/'+metric,large,[x[metric]-small[x['image']][metric] for x in large])
    summary[policy]=p
comparisons={}
for metric in metrics:
    before={x['image']:x for x in per if x['policy']==policies[0] and x['role']=='new_sample'};after=[x for x in per if x['policy']==policies[1] and x['role']=='new_sample'];comparisons[metric]=emit('balanced32_minus_SFT30/'+metric,after,[x['own_'+metric]-before[x['image']]['own_'+metric] for x in after])
counts=collections.Counter((x['policy'],x['grid_role'],x['actual_count'],x['donor_actual_count'],x['B_argmax_count'],x['same_image']) for x in grid)
confusion=[{'policy':p,'role':r,'target_count':c,'donor_count':d,'predicted_B_count':b,'same_image':own,'records':n} for (p,r,c,d,b,own),n in sorted(counts.items())]
result={'ok':True,'plan_sha256':sha(ROOT/'plan.json'),'new_evaluation_trajectories':256,'warm_anchor_generations':2,'grid_records':2176,'optimizer_updates':0,'policies':summary,'policy_comparison_native':comparisons,'per_image_comparisons':per,'statistics':stats,'gpu_pipeline':read(ROOT/'gpu_pipeline_result.json'),'elapsed_seconds':time.monotonic()-start};dump(ROOT/'analysis.json',result);dump(ROOT/'grid_quantity_metrics.json',grid);dump(ROOT/'per_image_selections.json',selection)
for name,rows in [('grid_quantity_metrics.csv',grid),('per_image_comparisons.csv',per),('donor_target_means.csv',donor_means),('coverage_by_image.csv',coverage),('per_image_selections.csv',selection),('paired_statistics.csv',stats),('donor_count_confusion.csv',confusion)]:
    fields=list(dict.fromkeys(k for x in rows for k in x))
    with (ROOT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,list) else v for k,v in x.items()} for x in rows)
print('PERMUTATION_ANALYSIS_COMPLETE',len(stats),flush=True)
