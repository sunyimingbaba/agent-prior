"""All fixed state contrasts, uncertainty, protocol coverage and raw-candidate diagnostics."""
import csv
import hashlib
from pathlib import Path
import numpy as np
import torch
from stage_io import read,dump
from tie_metrics import expected_hits

r=Path(__file__).resolve().parent.parent;d=read(r/'protocol_plan.json');ep=read(r/'evaluation_plan.json');assert read(r/'evaluation_result.json')['ok']
g=read(r/'evaluation/gallery.json');owners=np.array(g['owners']);groups=np.array(g['groups']);cases=read(r/'inputs/quantity_cases.json');qgallery=read(r/'inputs/quantity_gallery.json');words={w:i+1 for i,w in enumerate('one two three four five six seven'.split())};primary='repository_duplicate_token_positive'
def hits(t,protocol):
 score=t['scores'].numpy();ti=t['top_i'].numpy();tt=t['top_t'].numpy();pos=owners[None,:]==np.arange(210)[:,None]
 if protocol==primary:pos=np.stack([np.isin(groups,groups[owners==i]) for i in range(210)])
 if protocol=='strict_tie_averaged':return expected_hits(score,pos),expected_hits(score.T,pos.T)
 a=np.stack([pos[np.arange(210)[:,None],ti[:,:k]].any(1) for k in [1,5,10]],1).astype(float);b=np.stack([pos[tt[:k],np.arange(1050)[None,:]].any(0) for k in [1,5,10]],1).astype(float);return a,b
def metric(a,b):
 v=np.r_[a.mean(0),b.mean(0)];return {k:float(x) for k,x in zip(['I2T_R1','I2T_R5','I2T_R10','T2I_R1','T2I_R5','T2I_R10','mR'],np.r_[v,v.mean()])}
def random(label):return np.random.default_rng((d['analysis']['seed']+int(hashlib.sha256(label.encode()).hexdigest()[:8],16))%2**32)
def weights():
 out=[]
 for f in d['state_families']:
  out.append((f+'/correct-minus-permutation_mean',{f+'/correct':1.,**{f+f'/permutation_{i:02d}':-.05 for i in range(20)}}))
  out.append((f+'/correct-minus-shared',{f+'/correct':1.,f+'/shared_training102':-1.}))
 for p in ['legacy','selected']:
  for s in ['answer_prefill','answer_pool']:out.append((p+'_'+s+'/correct-minus-last',{p+'_'+s+'/correct':1.,p+'_last/correct':-1.}))
 out.append(('selected_last/correct-minus-legacy_last',{'selected_last/correct':1.,'legacy_last/correct':-1.}));return out
summary={p:{} for p in d['evaluation']['protocols']};quantity={};vals={};qvals={};permstats={};rows=[];perimage={}
for arm,declared in ep['readout_cases'].items():
 quantity[arm]={};perimage[arm]={}
 for p in summary:summary[p][arm]={}
 for c in declared:
  f,label=c['family'],c['condition'];key=f+'/'+label;out=r/'evaluation'/arm/f/label;t=torch.load(out/'tensors.pt',map_location='cpu',weights_only=True)
  for protocol in summary:
   a,b=hits(t,protocol);m=metric(a,b);summary[protocol][arm].setdefault(f,{})[label]=m;contrib=(a.sum(1)+b.reshape(210,5,3).mean(1).sum(1))/6;vals[arm,protocol,key]=contrib
   perimage[arm].setdefault(key,{})[protocol]=torch.tensor(contrib);rows.append({'arm':arm,'protocol':protocol,'family':f,'condition':label,**m})
  qs=t['quantity_scores'].numpy();records=[]
  for i,x in enumerate(cases):
   q={k:x[k] for k in ['image','split','actual_count']}
   for word in ['A','B']:
    pos,neg=qgallery['quantity_columns'][x['image']][word];prefix=qgallery['quantity_text_order'][pos].rsplit(' ',1)[0];wrong=[j for j,s in enumerate(qgallery['quantity_text_order']) if s.rsplit(' ',1)[0]==prefix and words[s.rsplit(' ',1)[1][:-1]]!=x['actual_count']];value=float(qs[i,pos]);competitor=max(float(qs[i,j]) for j in wrong)
    q.update({word+'_pair_correct':value>float(qs[i,neg]),word+'_hard_correct':value>competitor,word+'_hard_margin':value-competitor})
   records.append(q)
  dump(out/'quantity_records.json',records);quantity[arm].setdefault(f,{})[label]={}
  for split in ['train','val']:
   xs=[x for x in records if x['split']==split];common=[x for x in xs if 1<=x['actual_count']<=4];entry={'images':len(xs),'count_strata_n':{str(n):sum(x['actual_count']==n for x in common) for n in range(1,5)}}
   for word in ['A','B']:
    for m in ['pair_correct','hard_correct','hard_margin']:
     k=word+'_'+m;entry[k+'_image_mean']=float(np.mean([x[k] for x in xs]));entry[k+'_macro_1to4']=float(np.mean([np.mean([x[k] for x in common if x['actual_count']==n]) for n in range(1,5)]))
    qvals[arm,key,split,word]={n:np.array([float(x[word+'_hard_correct']) for x in common if x['actual_count']==n]) for n in range(1,5)}
   quantity[arm][f][label][split]=entry
def paired_delta(mapping,w):
 positive=next(k for k,v in w.items() if v==1.);negative=[k for k,v in w.items() if v<0]
 if len(negative)==20:return np.mean(np.stack([mapping[positive]-mapping[k] for k in negative]),axis=0)
 assert len(negative)==1;return mapping[positive]-mapping[negative[0]]
retrieval_ci=[];quantity_ci=[]
for arm in d['evaluation']['primary_readouts']:
 permstats[arm]={}
 for protocol in summary:
  permstats[arm][protocol]={}
  for f in d['state_families']:
   x=np.array([summary[protocol][arm][f][f'permutation_{i:02d}']['mR'] for i in range(20)]);permstats[arm][protocol][f]={'all20_mR':x.tolist(),'mean':float(x.mean()),'std':float(x.std(ddof=1)),'min':float(x.min()),'max':float(x.max())}
  for name,w in weights():
   delta=paired_delta({key:vals[arm,protocol,key] for key in w},w);label='retrieval/'+arm+'/'+protocol+'/'+name;rng=random(label);draw=rng.integers(210,size=(10000,210));ci=np.percentile(delta[draw].mean(1),[2.5,97.5]);retrieval_ci.append({'arm':arm,'protocol':protocol,'contrast':name,'weights':w,'delta_mR':float(delta.mean()),'interval95':ci.tolist()})
 for split in ['train','val']:
  for word in ['A','B']:
   for name,w in weights():
    ds={n:paired_delta({key:qvals[arm,key,split,word][n] for key in w},w) for n in range(1,5)};rng=random('quantity/'+arm+'/'+split+'/'+word+'/'+name);boot=sum(v[rng.integers(len(v),size=(10000,len(v)))].mean(1) for v in ds.values())/4;quantity_ci.append({'arm':arm,'split':split,'wording':word,'contrast':name,'weights':w,'delta_hard_macro':float(np.mean([v.mean() for v in ds.values()])),'interval95':np.percentile(boot,[2.5,97.5]).tolist()})
states={};paths={}
for f in d['state_families']:
 bank=torch.load(r/'state_banks'/(f+'.pt'),map_location='cpu',weights_only=True);meta=read(r/'state_banks'/(f+'.json'));states[f]={}
 for split,h in bank.items():
  h=h.double().numpy();center=h-h.mean(0);s=np.linalg.svd(center,compute_uv=False);energy=s*s;positive=energy[energy>0];p=positive/energy.sum() if energy.sum()>1e-20 else np.array([]);u=h/np.linalg.norm(h,axis=1)[:,None]
  states[f][split]={'images':len(h),'available':sum(x['available'] for x in meta[split]),'fallbacks':sum(not x['available'] for x in meta[split]),'unique_states':len({x['numeric_sha256'] for x in meta[split]}),'centered_rms':float(np.sqrt(np.mean(center**2))),'effective_rank':float(np.exp(-np.sum(p*np.log(p)))) if len(p) else 0.,'mean_pair_cosine':float((np.square(u.sum(0)).sum()-len(u))/(len(u)*(len(u)-1)))}
for arm in ep['readout_cases']:
 paths[arm]={}
 for f in d['state_families']:
  x=torch.load(r/'evaluation'/arm/'bank_paths'/(f+'.pt'),map_location='cpu',weights_only=True);gate=x['gate_clamped'][63:].numpy();p=x['projected'][63:].double().numpy();paths[arm][f]={'val_gate_mean':float(gate.mean()),'val_gate_zero_fraction':float((gate==0).mean()),'val_gate_one_fraction':float((gate==1).mean()),'val_projected_centered_rms':float(np.sqrt(np.mean((p-p.mean(0))**2)))}
selection=read(r/'prompt_selection_lock_v2.json');generation=read(r/'generation_result_v2.json');raw=read(r/'generation_v2/records.json');protocols={}
for group in sorted({x['group'] for x in raw}):
 xs=[x for x in raw if x['group']==group];protocols[group]={'images':len(xs),'strict_completed_with_consumed_tool':sum(x['strict_protocol_usable'] and x['tools_used']>0 and x['feedback_consumed'] for x in xs),'generated_result_tags':sum(x['generated_result_tag'] for x in xs),'body_pool_available':sum(x['coverage']['answer_pool_available'] for x in xs),'prefill_available':sum(x['coverage']['answer_prefill_available'] for x in xs)}
dump(r/'analysis.json',{'ok':True,'prompt_selection':selection,'protocol_repair_passed':generation['protocol_repair_passed'],'protocols':protocols,'retrieval':summary,'retrieval_contrasts':retrieval_ci,'permutation_summary':permstats,'quantity':quantity,'quantity_contrasts':quantity_ci,'state_diagnostics':states,'bank_path_diagnostics':paths,'Agent_optimizer_updates':generation['Agent_optimizer_updates'],'readout_optimizer_updates':0,'scope':'fixed readouts, train-selected protocol and val210 diagnostic, nominal conditional intervals; all failed cases retained with explicit state fallbacks; model-generated descriptions not semantically labelled'})
torch.save(perimage,r/'per_image_metrics.pt')
with (r/'retrieval_summary.csv').open('w',encoding='utf-8',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
print('ALL_STATE_CONTRASTS_ANALYZED',len(retrieval_ci),len(quantity_ci),flush=True)
