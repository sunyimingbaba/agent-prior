"""All prespecified contrasts, paired uncertainty, raw state and injection-path diagnostics."""
import csv
import hashlib
import json
import math
import re
from pathlib import Path
import numpy as np
import torch
from stage_io import read,dump
from tie_metrics import expected_hits

r=Path(__file__).resolve().parent.parent;d=read(r/'diagnostic_plan.json');ep=read(r/'evaluation_plan.json');base=ep['baseline']
assert read(r/'evaluation_result.json')['ok']
g=read(r/'evaluation/gallery.json');owners=np.array(g['owners']);groups=np.array(g['groups']);assert np.all(np.bincount(owners)==5)
cases=read(r/'inputs/quantity_cases.json');qg=read(r/'inputs/quantity_gallery.json');words={s:i+1 for i,s in enumerate('one two three four five six seven'.split())}
def rng(label):return np.random.default_rng((d['uncertainty']['seed']+int(hashlib.sha256(label.encode()).hexdigest()[:8],16))%2**32)
def hit_arrays(t,protocol):
 if protocol=='strict_tie_averaged':
  score=t['scores'].numpy();pos=owners[None,:]==np.arange(210)[:,None]
  return expected_hits(score,pos),expected_hits(score.T,pos.T)
 ti=t['top_i'].numpy();tt=t['top_t'].numpy();ih=[];th=[]
 positive={a:set(owners[groups==a]) for a in set(groups)}
 own=[set(groups[owners==i]) for i in range(210)]
 for k in [1,5,10]:
  if protocol=='repository_duplicate_token_positive':
   ih.append([any(groups[j] in own[i] for j in ti[i,:k]) for i in range(210)])
   th.append([any(i in positive[groups[j]] for i in tt[:k,j]) for j in range(1050)])
  else:
   ih.append([any(owners[j]==i for j in ti[i,:k]) for i in range(210)])
   th.append([owners[j] in tt[:k,j] for j in range(1050)])
 return np.array(ih,dtype=float).T,np.array(th,dtype=float).T
def metrics(a,b):
 vals=np.r_[a.mean(0),b.mean(0)];return {k:float(v) for k,v in zip(['I2T_R1','I2T_R5','I2T_R10','T2I_R1','T2I_R5','T2I_R10','mR'],np.r_[vals,vals.mean()])}
def ci(delta,label):
 draws=rng(label).integers(len(delta),size=(10000,len(delta)));return np.percentile(delta[draws].mean(1),[2.5,97.5]).tolist()
def contrast_specs():
 names=[c['name'] for c in d['conditions']];out=[('condition/'+n,{n:1.,base:-1.}) for n in names if n!=base]
 out += [('permutation/'+str(i).zfill(2),{f'permutation_{i:02d}':1.,base:-1.}) for i in range(20)]
 out += [('permutation_mean',{**{f'permutation_{i:02d}':1/20 for i in range(20)},base:-1.})]
 for field,level in [('example','without_example'),('policy_image','gray127'),('feedback','constant_training102')]:
  out.append(('factor/'+field,{c['name']:(.25 if c[field]==level else -.25) for c in d['conditions']}))
 return out
retrieval={p:{} for p in d['retrieval']['protocols']+['strict_tie_averaged']};values={};quantity={};qvalues={};paths={};per_image=[]
for arm,labels in ep['readout_conditions'].items():
 paths[arm]={};quantity[arm]={}
 bt=torch.load(r/'evaluation'/arm/base/'retrieval.pt',map_location='cpu',weights_only=True)
 bf=bt['image_features'].numpy().astype(float);bs=bt['scores'].numpy().astype(float);bh=bt['prior_hidden'].float().numpy().astype(float)
 for protocol in retrieval:retrieval[protocol][arm]={}
 for label in labels:
  t=torch.load(r/'evaluation'/arm/label/'retrieval.pt',map_location='cpu',weights_only=True)
  for protocol in retrieval:
   a,b=hit_arrays(t,protocol);retrieval[protocol][arm][label]=metrics(a,b)
   by_image=(a.sum(1)+np.stack([b[owners==i].mean(0) for i in range(210)]).sum(1))/6
   values[arm,protocol,label]=by_image
   for i,name in enumerate(g['names']):per_image.append({'arm':arm,'protocol':protocol,'condition':label,'image':name,'mR_image_contribution':float(by_image[i])})
  f=t['image_features'].numpy().astype(float);h=t['prior_hidden'].float().numpy().astype(float);gate=t['gate_clamped'].numpy();proj=t['projected'].numpy().astype(float)
  paths[arm][label]={'gate_mean':float(gate.mean()),'gate_zero_fraction':float((gate==0).mean()),'gate_one_fraction':float((gate==1).mean()),'gate_min':float(gate.min()),'gate_max':float(gate.max()),'projected_centered_rms':float(np.sqrt(np.mean((proj-proj.mean(0))**2))),'hidden_cosine_to_baseline_mean':float(np.mean(np.sum(h*bh,1)/(np.linalg.norm(h,axis=1)*np.linalg.norm(bh,axis=1)))),'image_feature_cosine_to_baseline_mean':float(np.mean(np.sum(f*bf,1)/(np.linalg.norm(f,axis=1)*np.linalg.norm(bf,axis=1)))),'image_feature_max_abs_change':float(np.max(np.abs(f-bf))),'score_max_abs_change':float(np.max(np.abs(t['scores'].numpy().astype(float)-bs)))}
  q=torch.load(r/'evaluation'/arm/label/'quantity.pt',map_location='cpu',weights_only=True);scores=q['scores'].numpy();qr=[]
  for i,c in enumerate(cases):
   row={k:c[k] for k in ['image','split','actual_count']}
   for wording in ['A','B']:
    positive,negative=qg['quantity_columns'][c['image']][wording];prefix=qg['quantity_text_order'][positive].rsplit(' ',1)[0]
    cols={words[s.rsplit(' ',1)[1][:-1]]:j for j,s in enumerate(qg['quantity_text_order']) if s.rsplit(' ',1)[0]==prefix}
    other=[float(scores[i,j]) for n,j in cols.items() if n!=c['actual_count']];pos=float(scores[i,positive])
    row.update({wording+'_pair_correct':pos>float(scores[i,negative]),wording+'_hard_correct':pos>max(other),wording+'_hard_margin':pos-max(other)})
   qr.append(row)
  dump(r/'evaluation'/arm/label/'quantity_records.json',qr);quantity[arm][label]={}
  for split in ['train','val']:
   xs=[x for x in qr if x['split']==split];common=[x for x in xs if 1<=x['actual_count']<=4];counts={str(n):sum(x['actual_count']==n for x in common) for n in range(1,5)}
   summary={'images':len(xs),'counts1to4_images':len(common),'count_sample_sizes':counts}
   for wording in ['A','B']:
    for metric in ['pair_correct','hard_correct','hard_margin']:
     k=wording+'_'+metric;summary[k+'_image_mean']=float(np.mean([x[k] for x in xs]));summary[k+'_counts1to4_macro']=float(np.mean([np.mean([x[k] for x in common if x['actual_count']==n]) for n in range(1,5)]))
    qvalues[arm,label,split,wording]={n:np.array([float(x[wording+'_hard_correct']) for x in common if x['actual_count']==n]) for n in range(1,5)}
   quantity[arm][label][split]=summary
contrasts=[];qcontrasts=[];permutation_summary={}
for arm in d['primary_readouts']:
 permutation_summary[arm]={}
 for protocol in retrieval:
  for name,weights in contrast_specs():
   delta=sum(weight*values[arm,protocol,label] for label,weight in weights.items());label='retrieval/'+arm+'/'+protocol+'/'+name
   contrasts.append({'arm':arm,'protocol':protocol,'contrast':name,'weights':weights,'delta_mR':float(delta.mean()),'interval95':ci(delta,label),'images':210})
  xs=np.array([retrieval[protocol][arm][f'permutation_{i:02d}']['mR'] for i in range(20)])
  permutation_summary[arm][protocol]={'all20_mR':xs.tolist(),'mean_mR':float(xs.mean()),'std_over_permutations':float(xs.std(ddof=1)),'min_mR':float(xs.min()),'max_mR':float(xs.max()),'mean_minus_baseline':float(xs.mean()-retrieval[protocol][arm][base]['mR'])}
 for split in ['train','val']:
  for wording in ['A','B']:
   for name,weights in contrast_specs():
    ds={n:sum(weight*qvalues[arm,label,split,wording][n] for label,weight in weights.items()) for n in range(1,5)}
    label='quantity/'+arm+'/'+split+'/'+wording+'/'+name;random=rng(label)
    boot=sum(v[random.integers(len(v),size=(10000,len(v)))].mean(1) for v in ds.values())/4
    qcontrasts.append({'arm':arm,'split':split,'wording':wording,'contrast':name,'weights':weights,'delta_hard_macro':float(np.mean([x.mean() for x in ds.values()])),'interval95':np.percentile(boot,[2.5,97.5]).tolist(),'paired_strata_sample_sizes':{str(n):len(v) for n,v in ds.items()}})
generation=read(r/'generation/records.json');cohort=read(r/'inputs/cohort.json');genmap={(x['condition'],x['key']):x for x in generation};states={}
for condition in d['conditions']:
 name=condition['name'];states[name]={}
 for split in ['train','val']:
  cs=[c for c in cohort if c['split']==split];xs=[genmap[name,c['key']] for c in cs]
  h=np.stack([torch.load(r/x['state_file'],map_location='cpu',weights_only=True).float().numpy() for x in xs]).astype(float)
  hc=h-h.mean(0);s=np.linalg.svd(hc,compute_uv=False);energy=s*s;prob=energy/energy.sum() if energy.sum()>1e-20 else np.zeros_like(energy);prob=prob[prob>0]
  u=h/np.linalg.norm(h,axis=1)[:,None];pair=(np.square(u.sum(0)).sum()-len(u))/(len(u)*(len(u)-1))
  texts=[read(r/x['trace_file'])['trajectory_text'] for x in xs]
  states[name][split]={'images':len(xs),'unique_state_sha256':len({x['numeric_state_sha256'] for x in xs}),'centered_rms':float(np.sqrt(np.mean(hc**2))),'effective_rank_centered_covariance':float(np.exp(-np.sum(prob*np.log(prob)))) if len(prob) else 0.,'mean_pairwise_cosine':float(pair),'quality_usable':sum(x['quality_usable'] for x in xs),'tool_requested_images':sum(x['tools_used']>0 for x in xs),'tool_requests_total':sum(x['tools_used'] for x in xs),'lexical_mentions_harbor_or_port':sum((any(w in text for w in ['港口','码头']) or bool(re.search(r'\b(?:harbor|port)\b',text,re.I))) for text in texts),'lexical_query_match_phrase':sum('与查询文本相符' in text for text in texts),'feedback_consumed_events':sum(sum(e.get('after_round',0)<len(read(r/x['trace_file'])['rounds']) for e in read(r/x['trace_file'])['tool_events']) for x in xs),'feedback_exposed_images':sum(any(e.get('after_round',0)<len(read(r/x['trace_file'])['rounds']) for e in read(r/x['trace_file'])['tool_events']) for x in xs),'last_round_feedback_not_consumed':sum(sum(e.get('after_round',0)>=len(read(r/x['trace_file'])['rounds']) for e in read(r/x['trace_file'])['tool_events']) for x in xs),'lexical_scope':'bounded English harbor/port or Chinese港口/码头 term counts, no semantic ground-truth accuracy claim'}
dump(r/'analysis.json',{'ok':True,'retrieval':retrieval,'retrieval_contrasts':contrasts,'permutation_summary':permutation_summary,'quantity':quantity,'quantity_contrasts':qcontrasts,'readout_path':paths,'generation_states':states,'optimizer_updates':{'Agent':0,'readout':0},'scope':'train/val diagnostic; single frozen Actor/readout seed; all declared controls retained; nominal pointwise intervals; no test data scored'})
with (r/'per_image_val_metrics.csv').open('w',encoding='utf-8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(per_image[0]),lineterminator='\n');w.writeheader();w.writerows(per_image)
with (r/'retrieval_summary.csv').open('w',encoding='utf-8',newline='') as f:
 rows=[{'arm':a,'protocol':p,'condition':c,**m} for p,x in retrieval.items() for a,y in x.items() for c,m in y.items()]
 w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
print('ALL_DIAGNOSTIC_CONTRASTS_ANALYZED',len(contrasts),len(qcontrasts),flush=True)
