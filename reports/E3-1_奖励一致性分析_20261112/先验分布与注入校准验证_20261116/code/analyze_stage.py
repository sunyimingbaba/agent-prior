import collections
import csv
import json
import math
import time
import numpy as np
import torch
from analysis_common import ROOT,read,dump,sha,quantity_metrics
from statistics_contract import stat
start=time.monotonic();torch.set_num_threads(2);plan=read(ROOT/'plan.json');cases=read(ROOT/'inputs/evaluation_cases.json');by={x['image']:x for x in cases};gallery=read(ROOT/'features/gallery.json');new=read(ROOT/'scoring/all_scores.json');nt=torch.load(ROOT/'scoring/tensors.pt',map_location='cpu',weights_only=True);base=read(ROOT/'baseline/records.json');bt=torch.load(ROOT/'baseline/tensors.pt',map_location='cpu',weights_only=True);states=read(ROOT/'states/records.json');pt=torch.load(ROOT/'path/tensors.pt',map_location='cpu',weights_only=True);selection=read(ROOT/'selection.json');stats=[];allrows=[]
def cohort(n):return 'training8' if n in plan['training_images'] else 'confirmation55' if by[n]['split']=='train' else 'existing_val22'
for i,x in enumerate(new):
    met=quantity_metrics(nt['quantity_similarities'][i],gallery,by[x['target_image']]);assert met['A_pair_margin']==x['margin_A'];allrows.append({**x,**met,'cohort':cohort(x['target_image'])})
for i,x in enumerate(base):
    family='initial_SFT30' if x['arm'].startswith('before') else 'balanced32';role='baseline_shared' if x['role']=='shared' else 'baseline_own';met=quantity_metrics(bt['quantity_similarities'][i],gallery,by[x['image']]);allrows.append({**x,**met,'id':'baseline__'+x['id'],'family':family,'phase':'baseline','role':role,'variant':'auto','target_image':x['image'],'target_split':x['split'],'actual_count':by[x['image']]['actual_count'],'cohort':cohort(x['image'])})
lookup={(x['family'],x['role'],x['target_image']):x for x in allrows if x['phase']!='diagnostic_grid'};metrics=['A_hard_correct','B_hard_correct','A_hard_margin','B_hard_margin','B_pair_correct']
def emit(label,names,values,macro=True):
    value=stat(values,[by[n]['actual_count'] for n in names],label,macro);stats.append({'label':label,'image_names':names,'counts':[by[n]['actual_count'] for n in names],'values':values,'scope':'macro' if macro else 'raw',**value});return value
summary={}
for co in ['training8','confirmation55','existing_val22']:
    names=[c['image'] for c in cases if cohort(c['image'])==co];summary[co]={}
    for family in ['initial_SFT30','balanced32']:
        groups={}
        def get(role,n):return lookup['original_cached' if role=='original_cached' else 'balanced32' if role=='learned_only' else family,role,n]
        for role in ['baseline_own','baseline_shared','selected_own','selected_shared','learned_only','original_cached']:
            groups[role]={scope:{m:emit(co+'/'+family+'/'+role+'/'+scope+'/'+m,names,[get(role,n)[m] for n in names],scope=='macro') for m in metrics} for scope in ['raw','macro']}
            if co=='existing_val22':groups[role]['retrieval_rr']=emit(co+'/'+family+'/'+role+'/raw/retrieval_rr',names,[get(role,n)['retrieval_rr'] for n in names],False)
        contrasts={}
        for name,role,ref in [('selected_minus_native','selected_own','baseline_own'),('selected_own_minus_shared','selected_own','selected_shared'),('selected_minus_learned','selected_own','learned_only'),('original_minus_native','original_cached','baseline_own'),('selected_minus_original','selected_own','original_cached')]:
            contrasts[name]={scope:{m:emit(co+'/'+family+'/'+name+'/'+scope+'/'+m,names,[get(role,n)[m]-get(ref,n)[m] for n in names],scope=='macro') for m in metrics} for scope in ['raw','macro']}
        summary[co][family]={'groups':groups,'contrasts':contrasts}
# Source-distribution and gate saturation are evaluated on all511 cached inputs.
pathrows=[]
for i,x in enumerate(states):
    pathrows.append({'id':x['id'],'family':x['family'],'role':x['role'],'image':x['image'],'cohort':cohort(x['image']),'hidden_norm':float(pt['hidden'][i].double().norm()),'linear_norm':float(pt['linear'][i].double().norm()),'projected_norm':float(pt['projected'][i].double().norm()),'gate_raw':float(pt['gate_raw'][i]),'gate_clipped':float(pt['gate_clipped'][i]),'gate_zero':float(pt['gate_clipped'][i]==0),'gate_one':float(pt['gate_clipped'][i]==1),'prompt_norm':float(pt['prompt_norm'][i])})
distribution={}
for family in ['original_cached','initial_SFT30','balanced32']:
    for role in ['original_cached','greedy','candidate']:
        xs=[x for x in pathrows if x['family']==family and x['role']==role]
        if not xs:continue
        distribution[family+'/'+role]={'states':len(xs),**{k:{'mean':float(np.mean([x[k] for x in xs])),'min':min(x[k] for x in xs),'max':max(x[k] for x in xs)} for k in ['hidden_norm','linear_norm','projected_norm','gate_raw','gate_clipped','gate_zero','gate_one','prompt_norm']}}
g=summary['confirmation55']['balanced32']['contrasts'];checks={'confirmation_B_hard_gain_CI_lower_positive':g['selected_minus_native']['macro']['B_hard_correct']['interval95'][0]>0,'confirmation_selected_own_minus_shared_B_hard_CI_lower_positive':g['selected_own_minus_shared']['macro']['B_hard_correct']['interval95'][0]>0,'confirmation_selected_own_minus_learned_B_hard_CI_lower_positive':g['selected_minus_learned']['macro']['B_hard_correct']['interval95'][0]>0,'confirmation_B_hard_margin_gain_CI_lower_positive':g['selected_minus_native']['macro']['B_hard_margin']['interval95'][0]>0,'immutable_model_and_original_sources_preserved':read(ROOT/'final_source_audit.json')['ok']};assert set(checks)==set(plan['adoption_checks'])
grad=read(ROOT/'gradient/records.json');gradsummary={}
for family in ['initial_SFT30','balanced32']:
    for role in ['native','selected']:
        xs=[x for x in grad if x['family']==family and x['role']==role];gradsummary[family+'/'+role]={'witnesses':len(xs),'zero_gradient_count':sum(x['gradient_norm']==0 for x in xs),'gradient_norm_mean':float(np.mean([x['gradient_norm'] for x in xs])),'finite_difference_mean':float(np.mean([x['fd_central_derivative'] for x in xs]))}
result={'ok':True,'plan_sha256':sha(ROOT/'plan.json'),'selection':selection,'groups':summary,'statistics':stats,'distribution':distribution,'gradient_summary':gradsummary,'adoption':{'passed':all(checks.values()),'checks':checks},'new_utility_scores':1470,'new_Qwen_generations':0,'optimizer_updates':0,'elapsed_seconds':time.monotonic()-start};dump(ROOT/'analysis.json',result);dump(ROOT/'quantity_records.json',allrows)
for name,rows in [('quantity_records.csv',allrows),('paired_statistics.csv',stats),('per_state_path.csv',pathrows),('gradient_witnesses.csv',grad)]:
    fields=list(dict.fromkeys(k for x in rows for k in x))
    with (ROOT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,list) else v for k,v in x.items()} for x in rows)
print('INJECTION_ANALYSIS_COMPLETE',len(stats),flush=True)
