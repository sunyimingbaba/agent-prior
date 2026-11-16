"""Independent compact-tensor metrics, exact inputs, model paths and paired intervals."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from checkpoint_metadata import load_checkpoint

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--matched',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.root
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(x,y):assert abs(float(x)-float(y))<1e-10,(x,y)
d=read(r/'protocol_plan.json');ep=read(r/'evaluation_plan.json');j=read(r/'analysis.json');g=read(r/'evaluation/gallery.json');assert g==read(a.parent/'evaluation/gallery.json');owners=np.array(g['owners']);groups=np.array(g['groups']);assert np.array_equal(owners,np.repeat(np.arange(210),5))
perms=read(r/'inputs/permutations.json');assert perms['val']==read(a.parent/'inputs/permutations.json')['val'];random=np.random.default_rng(2026111609)
for actual in perms['train']:
 while True:
  expected=random.permutation(63)
  if np.all(expected!=np.arange(63)):break
 assert actual==expected.tolist()
for s,n in [('train',63),('val',210)]:
 assert len(perms[s])==20 and len({tuple(v) for v in perms[s]})==20
 for v in perms[s]:assert sorted(v)==list(range(n)) and all(i!=x for i,x in enumerate(v))
banks={f:torch.load(r/'state_banks'/(f+'.pt'),map_location='cpu',weights_only=True) for f in d['state_families']};bm=read(r/'state_banks/legacy_last.json');positions={s:{x['image']:i for i,x in enumerate(bm[s])} for s in ['train','val']};qcases=read(r/'inputs/quantity_cases.json');qgallery=read(r/'inputs/quantity_gallery.json');num={w:i+1 for i,w in enumerate('one two three four five six seven'.split())};lock=read(r/'inputs/parent_selection_lock.json')
pos={'strict_image_correspondence':owners[None,:]==np.arange(210)[:,None]};pos['strict_tie_averaged']=pos['strict_image_correspondence'];pos['repository_duplicate_token_positive']=np.stack([np.isin(groups,groups[owners==i]) for i in range(210)])
def source(c,s,i):
 if c=='correct':return s,i
 if c=='shared_training102':return 'train',positions['train']['102.tif']
 return s,perms[s][int(c[-2:])][i]
def tie(matrix,positive):
 result=[]
 for k in [1,5,10]:
  output=[]
  for row,good in zip(matrix,positive):
   cutoff=np.sort(row)[-k];above=row>cutoff
   if good[above].any():output.append(1.);continue
   eq=row==cutoff;t=int(eq.sum());p=int(good[eq].sum());q=k-int(above.sum());output.append(1.-(math.comb(t-p,q)/math.comb(t,q) if t-p>=q else 0.))
  result.append(output)
 return np.array(result).T
pathassets=torch.load(r/'inputs/frozen_path_assets.pt',map_location='cpu',weights_only=True);offset=pathassets['offset'].double().numpy().reshape(-1);shared=pathassets['shared_hidden'].double().numpy();values={};qvalues={};metrics_count=q_count=source_count=cases_count=0;maxscore=maxq=maxproj=maxgate=0.;reuse_cases=0
for arm,declared in ep['readout_cases'].items():
 cp=lock['arms'][arm]['checkpoint'];assert sha(a.matched/cp['path'])==cp['sha256'];state=load_checkpoint(a.matched/cp['path'])['state_dict'];assets=torch.load(r/'evaluation'/arm/'assets.pt',map_location='cpu',weights_only=True);text=assets['text_features'].double().numpy();qt=assets['quantity_text_features'].double().numpy();paths={}
 for family in banks:
  saved=torch.load(r/'evaluation'/arm/'bank_paths'/(family+'.pt'),map_location='cpu',weights_only=True);h=torch.cat([banks[family]['train'],banks[family]['val']]).double().numpy()
  if arm in d['evaluation']['negative_readouts']:h=np.broadcast_to(shared,h.shape)
  z=h@state['agent_proj.0.weight'].double().numpy().T+state['agent_proj.0.bias'].double().numpy()
  if arm!='native':z=z+offset
  z=(z-z.mean(1,keepdims=True))/np.sqrt(z.var(1,keepdims=True)+1e-5);z=z*state['agent_proj.1.weight'].double().numpy()+state['agent_proj.1.bias'].double().numpy();gate=z@state['cluster_mapping.weight'].double().numpy().T+state['cluster_mapping.bias'].double().numpy();gate=np.repeat(gate[:,None,:],16,axis=1)
  if arm=='learned_visual_only':gate[:]=1
  pe=float(np.max(np.abs(z-saved['projected'].numpy())));ge=float(np.max(np.abs(gate-saved['cluster_output_after_arm_hooks'].numpy())));maxproj=max(maxproj,pe);maxgate=max(maxgate,ge);assert pe<2e-4 and ge<2e-4;assert torch.equal(saved['gate_clamped'],saved['cluster_output_after_arm_hooks'].clamp(0,1));paths[family]=saved
  sg=saved['gate_clamped'][63:].numpy();sp=saved['projected'][63:].double().numpy();stat=j['bank_path_diagnostics'][arm][family]
  for k,v in {'val_gate_mean':sg.mean(),'val_gate_zero_fraction':(sg==0).mean(),'val_gate_one_fraction':(sg==1).mean(),'val_projected_centered_rms':np.sqrt(np.mean((sp-sp.mean(0))**2))}.items():close(stat[k],v)
 baseline=None
 for c in declared:
  family,cond=c['family'],c['condition'];key=family+'/'+cond;out=r/'evaluation'/arm/family/cond;meta=read(out/'sources.json');t=torch.load(out/'tensors.pt',map_location='cpu',weights_only=True);vs=[source(cond,'val',i) for i in range(210)];qs=[source(cond,x['split'],positions[x['split']][x['image']]) for x in qcases]
  assert [tuple(x) for x in meta['val']]==vs and [tuple(x) for x in meta['quantity']]==qs and meta['bank_file_sha256']==sha(r/'state_banks'/(family+'.pt')) and meta['assets_file_sha256']==sha(r/'evaluation'/arm/'assets.pt');source_count+=295
  raw=paths[family]['cluster_output_after_arm_hooks']
  for label,sources in [('val_gate_raw',vs),('quantity_gate_raw',qs)]:
   idx=[i if s=='train' else 63+i for s,i in sources];assert torch.max(torch.abs(raw[idx]-t[label]))<2e-4
  f=t['image_features'].double().numpy();qf=t['quantity_image_features'].double().numpy();score=t['scores'].numpy();qscore=t['quantity_scores'].numpy();error=float(np.max(np.abs(assets['logit_scale']*f@text.T-score)));qe=float(np.max(np.abs(qf@qt.T-qscore)));maxscore=max(maxscore,error);maxq=max(maxq,qe);assert error<1e-4 and qe<1e-6
  ti=t['top_i'].numpy();tt=t['top_t'].numpy();assert np.array_equal(np.take_along_axis(score,ti,axis=1),np.sort(score,axis=1)[:,-10:][:,::-1]);assert np.array_equal(np.take_along_axis(score,tt,axis=0),np.sort(score,axis=0)[-10:][::-1]);assert all(len(set(v))==10 for v in ti) and all(len(set(v))==10 for v in tt.T)
  for protocol,positive in pos.items():
   if protocol=='strict_tie_averaged':ih=tie(score,positive);th=tie(score.T,positive.T)
   else:ih=np.stack([positive[np.arange(210)[:,None],ti[:,:k]].any(1) for k in [1,5,10]],1).astype(float);th=np.stack([positive[tt[:k],np.arange(1050)[None,:]].any(0) for k in [1,5,10]],1).astype(float)
   v=np.r_[ih.mean(0),th.mean(0)];s=j['retrieval'][protocol][arm][family][cond]
   for k,x in zip(['I2T_R1','I2T_R5','I2T_R10','T2I_R1','T2I_R5','T2I_R10','mR'],np.r_[v,v.mean()]):close(s[k],x);metrics_count+=1
   values[arm,protocol,key]=(ih.sum(1)+th.reshape(210,5,3).mean(1).sum(1))/6
  if meta['validation_reuse']:
   oldlabel='with_example__real__recorded_real' if cond=='correct' else cond;oldpath=a.parent/'evaluation'/arm/oldlabel/'retrieval.pt';old=torch.load(oldpath,map_location='cpu',weights_only=True);assert sha(oldpath)==meta['validation_reuse']['sha256'];assert torch.equal(t['image_features'],old['image_features']) and torch.equal(t['scores'],old['scores']) and torch.equal(t['top_i'],old['top_i']) and torch.equal(t['top_t'],old['top_t']);reuse_cases+=1
  if family=='legacy_last' and cond=='correct':
   baseline=t;old=torch.load(a.parent/'evaluation'/arm/'with_example__real__recorded_real'/'quantity.pt',map_location='cpu',weights_only=True);assert torch.equal(t['quantity_image_features'],old['image_features']) and torch.equal(assets['quantity_text_features'],old['text_features'])
  if arm in d['evaluation']['negative_readouts']:assert torch.equal(t['image_features'],baseline['image_features']) and torch.equal(t['quantity_image_features'],baseline['quantity_image_features'])
  records=read(out/'quantity_records.json');assert len(records)==85
  for i,(record,x) in enumerate(zip(records,qcases)):
   assert record['image']==x['image'] and record['actual_count']==x['actual_count'] and record['split']==x['split']
   for word in ['A','B']:
    target,other=qgallery['quantity_columns'][x['image']][word];assert qgallery['quantity_text_order'][target]==x['pairs'][word]['positive'] and qgallery['quantity_text_order'][other]==x['pairs'][word]['negative'];prefix=qgallery['quantity_text_order'][target].rsplit(' ',1)[0];wrong=[i for i,s in enumerate(qgallery['quantity_text_order']) if s.rsplit(' ',1)[0]==prefix and num[s.rsplit(' ',1)[1][:-1]]!=x['actual_count']];z=float(qscore[i,target]);maximum=float(np.max(qscore[i,wrong]));assert record[word+'_hard_correct']==(z>maximum) and record[word+'_pair_correct']==(z>float(qscore[i,other]));close(record[word+'_hard_margin'],z-maximum)
  for split in ['train','val']:
   xs=[v for v in records if v['split']==split];common=[v for v in xs if 1<=v['actual_count']<=4];s=j['quantity'][arm][family][cond][split]
   for word in ['A','B']:
    for kind in ['pair_correct','hard_correct','hard_margin']:
     k=word+'_'+kind;close(s[k+'_image_mean'],np.mean([v[k] for v in xs]));close(s[k+'_macro_1to4'],np.mean([np.mean([v[k] for v in common if v['actual_count']==n]) for n in range(1,5)]));q_count+=2
    qvalues[arm,key,split,word]={n:np.array([float(v[word+'_hard_correct']) for v in common if v['actual_count']==n]) for n in range(1,5)}
  cases_count+=1
 imm=read(r/'evaluation'/arm/'model_immutability.json');assert imm['ok'] and imm['all_parameters_frozen'] and not imm['optimizer_created'] and imm['checkpoint_sha256']==cp['sha256']
spec={}
for f in d['state_families']:
 spec[f+'/correct-minus-permutation_mean']={f+'/correct':1.,**{f+f'/permutation_{i:02d}':-.05 for i in range(20)}};spec[f+'/correct-minus-shared']={f+'/correct':1.,f+'/shared_training102':-1.}
for p in ['legacy','selected']:
 for s in ['answer_prefill','answer_pool']:spec[p+'_'+s+'/correct-minus-last']={p+'_'+s+'/correct':1.,p+'_last/correct':-1.}
spec['selected_last/correct-minus-legacy_last']={'selected_last/correct':1.,'legacy_last/correct':-1.}
def random(label):return np.random.default_rng((d['analysis']['seed']+int(hashlib.sha256(label.encode()).hexdigest()[:8],16))%2**32)
def delta(values,w):
 ref=next(k for k,v in w.items() if v>0);others=[k for k,v in w.items() if v<0]
 return np.add.reduce([values[ref]-values[k] for k in others])/len(others)
assert len(j['retrieval_contrasts'])==102 and {(c['arm'],c['protocol'],c['contrast']) for c in j['retrieval_contrasts']}=={(arm,p,n) for arm in d['evaluation']['primary_readouts'] for p in pos for n in spec}
assert len(j['quantity_contrasts'])==136 and {(c['arm'],c['split'],c['wording'],c['contrast']) for c in j['quantity_contrasts']}=={(arm,s,w,n) for arm in d['evaluation']['primary_readouts'] for s in ['train','val'] for w in ['A','B'] for n in spec}
for c in j['retrieval_contrasts']:
 assert c['weights']==spec[c['contrast']];dv=delta({k:values[c['arm'],c['protocol'],k] for k in c['weights']},c['weights']);close(dv.mean(),c['delta_mR']);rng=random('retrieval/'+c['arm']+'/'+c['protocol']+'/'+c['contrast']);draw=rng.integers(210,size=(10000,210));bounds=np.quantile(dv[draw].sum(1)/210,[.025,.975]);assert np.max(np.abs(bounds-c['interval95']))<1e-10
for c in j['quantity_contrasts']:
 assert c['weights']==spec[c['contrast']];ds={n:delta({k:qvalues[c['arm'],k,c['split'],c['wording']][n] for k in c['weights']},c['weights']) for n in range(1,5)};close(np.mean([v.mean() for v in ds.values()]),c['delta_hard_macro']);rng=random('quantity/'+c['arm']+'/'+c['split']+'/'+c['wording']+'/'+c['contrast']);z=np.zeros(10000)
 for n,v in ds.items():z+=v[rng.integers(len(v),size=(10000,len(v)))].sum(1)/len(v)/4
 assert np.max(np.abs(np.quantile(z,[.025,.975])-c['interval95']))<1e-10
assert cases_count==270 and source_count==79650 and reuse_cases==42
out={'ok':True,'readout_conditions_verified':cases_count,'prior_source_rows_verified':source_count,'validation_matrices_reused_exact':reuse_cases,'new_image_encodings':70830,'retrieval_values_rebuilt':metrics_count,'quantity_summary_values_rebuilt':q_count,'quantity_cases_verified':270*85,'retrieval_intervals_rebuilt':102,'quantity_intervals_rebuilt':136,'full_split_derangements_verified':40,'float64_score_max_error':maxscore,'float64_quantity_score_max_error':maxq,'float64_projection_max_error':maxproj,'float64_gate_max_error':maxgate,'all_readout_parameters_frozen':True,'readout_optimizer_updates':0}
a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False))
