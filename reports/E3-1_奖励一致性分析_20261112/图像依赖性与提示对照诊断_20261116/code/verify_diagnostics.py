"""Independent CPU reconstruction from saved tensors, original labels and sealed plans."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from checkpoint_metadata import load_checkpoint

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();r=args.root;parent=args.parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def numeric(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def close(a,b):assert abs(float(a)-float(b))<1e-10,(a,b)
def seed(label):return (d['uncertainty']['seed']+int(hashlib.sha256(label.encode()).hexdigest()[:8],16))%2**32
d=read(r/'diagnostic_plan.json');ep=read(r/'evaluation_plan.json');j=read(r/'analysis.json');base=ep['baseline']
for manifest in ['generation_code_hashes.json','evaluation_code_hashes.json','post_code_hashes.json']:
 for n,h in read(r/manifest).items():assert sha(r/n)==h,n
cohort=read(r/'inputs/cohort.json');cases=read(r/'inputs/quantity_cases.json');const=read(r/'inputs/constant_feedback.json');perms=read(r/'inputs/permutations.json');gen=read(r/'generation/records.json');gr=read(r/'generation_result.json')
assert len(gen)==2184 and len({(x['condition'],x['key']) for x in gen})==2184
cm={c['key']:c for c in cohort};conditions={c['name']:c for c in d['conditions']};actual=0
for x in gen:
 c=cm[x['key']];condition=conditions[x['condition']];h=torch.load(r/x['state_file'],map_location='cpu',weights_only=True);tr=read(r/x['trace_file'])
 assert h.shape==(3584,) and h.dtype==torch.bfloat16 and torch.isfinite(h).all()
 assert numeric(h)==x['numeric_state_sha256']==tr['state_tensor_sha256'];assert sha(r/x['state_file'])==x['state_file_sha256'] and sha(r/x['trace_file'])==x['trace_sha256']
 assert (x['split'],x['row'],x['image'],x['rgb_sha256'])==(c['split'],c['row'],c['image'],c['rgb_sha256'])
 first=tr['rounds'][0];assert first['prompt_text'].endswith('查询文本: '+d['query']+'<|im_end|>\n<|im_start|>assistant\n')
 assert ('There is a harbor with several ships.' in first['prompt_text'])==(condition['example']=='with_example')
 assert len(first['message_roles'])==(6 if condition['example']=='with_example' else 2)
 source=const if condition['feedback']=='constant_training102' else c
 for e in tr['tool_events']:
  assert e['text']==source['feedback']['text'] and e['confidence']==source['feedback']['confidence']
  if not x['baseline_reused']:assert e['source_image']==source['image'] and e['feedback_source']==condition['feedback']
 for step in tr['rounds']:
  ids=step['generated_ids'];assert len(ids)==step['generated_token_count']<=256 and step['last_emitted_token_id']==ids[-1]
  assert step['state_input_token_id']==(ids[-2] if len(ids)>1 else step['prompt_token_ids'][-1])
  assert hashlib.sha256(step['prompt_text'].encode()).hexdigest()==step['prompt_text_sha256']
 assert len(tr['rounds'])<=3 and tr['tools_used']==x['tools_used'] and tr['stop_reason']==x['stop_reason'] and tr['quality_usable']==x['quality_usable']
 if x['baseline_reused']:
  assert x['state_file_sha256']==c['state_file_sha256'] and x['trace_sha256']==c['trace_sha256']
 else:
  rgb=c['rgb_sha256'] if condition['policy_image']=='real' else hashlib.sha256(bytes([127])*(c['size'][0]*c['size'][1]*3)).hexdigest()
  signature=hashlib.sha256(json.dumps({'condition':condition['name'],'rgb':rgb,'size':tuple(c['size']),'feedback':source['feedback'],'query':d['query'],'seed':0,'lora':d['policy_lora_hash']},sort_keys=True).encode()).hexdigest();assert signature==x['signature']
  if x['canonical_reused']:assert condition['policy_image']=='gray127' and condition['feedback']=='constant_training102'
 actual+=int(x['new_actual_generate'])
assert actual==gr['new_actual_generations']<=1911 and sum(x['baseline_reused'] for x in gen)==273 and sum(x['canonical_reused'] for x in gen)==gr['canonical_reused']
for c in cohort[:2]:
 tr=read(r/'generation'/('anchor_'+c['key']+'.json'));old=read(parent/c['trace_file']);assert tr['state_tensor_sha256']==c['numeric_state_sha256']
 assert [s['generated_ids'] for s in tr['rounds']]==[s['generated_ids'] for s in old['rounds']]
for split,n in [('train',1680),('val',210)]:
 rng=np.random.default_rng(d['permutations']['seed']+(0 if split=='train' else 1));assert len(perms[split])==20
 for actual_perm in perms[split]:
  while True:
   expected=rng.permutation(n)
   if np.all(expected!=np.arange(n)):break
  assert expected.tolist()==actual_perm
g=read(r/'evaluation/gallery.json');owners=np.array(g['owners']);groups=np.array(g['groups']);assert len(owners)==1050 and len(g['names'])==210
assert all(sum(owners==i)==5 for i in range(210))
positive={'strict_image_correspondence':owners[None,:]==np.arange(210)[:,None]}
positive['strict_tie_averaged']=positive['strict_image_correspondence']
positive['repository_duplicate_token_positive']=np.stack([np.isin(groups,groups[owners==i]) for i in range(210)])
metrics_count=quantity_count=0;maxscore=maxproj=maxgate=0.;values={};qvalues={};qg=read(r/'inputs/quantity_gallery.json');assets=torch.load(r/'inputs/frozen_path_assets.pt',map_location='cpu',weights_only=True)
parent_refs={(x['split'],x['row']):x for x in read(parent/'cache/records.json') if x['split'] in ['train','val']}
assert read(r/'inputs/parent_development_cache_reference.json')==list(parent_refs.values())
gen_refs={(x['condition'],x['split'],x['row']):x for x in gen}
source_rows_verified=0
def source_check(tensor,sources,label,targets):
 global source_rows_verified
 assert len(sources)==len(targets)==tensor['prior_hidden'].shape[0]
 for i,(source,(split,row)) in enumerate(zip(sources,targets)):
  mapped=perms[split][int(label[-2:])][row] if label.startswith('permutation_') else row
  ref=parent_refs[split,mapped] if label.startswith('permutation_') else gen_refs[label,split,row]
  assert source['split']==split and source['target_row']==row and source['prior_source_row']==mapped
  assert source['state_file_sha256']==ref['state_file_sha256'] and source['numeric_state_sha256']==ref['numeric_state_sha256']
  assert numeric(tensor['prior_hidden'][i])==ref['numeric_state_sha256']
  source_rows_verified+=1
assert sha(r/'inputs/frozen_path_assets.pt')==read(r/'inputs/frozen_path_sources.json')['frozen_path_assets_sha256']
assert gr['lora_hash_unchanged']==d['policy_lora_hash'] and gr['Agent_optimizer_updates']==gr['readout_optimizer_updates']==0
offset=assets['offset'].double().numpy().reshape(-1);shared=assets['shared_hidden'].double().numpy();lock=read(r/'inputs/parent_selection_lock.json');all_qrecords={}
numwords={s:i+1 for i,s in enumerate('one two three four five six seven'.split())}
def projection(h,state,arm):
 z=np.broadcast_to(shared,h.shape) if arm in d['negative_control_readouts'] else h
 z=z@state['agent_proj.0.weight'].double().numpy().T+state['agent_proj.0.bias'].double().numpy()
 if arm!='native':z=z+offset
 z=(z-z.mean(1,keepdims=True))/np.sqrt(z.var(1,keepdims=True)+1e-5)
 z=z*state['agent_proj.1.weight'].double().numpy()+state['agent_proj.1.bias'].double().numpy()
 gate=z@state['cluster_mapping.weight'].double().numpy().T+state['cluster_mapping.bias'].double().numpy()
 if arm=='learned_visual_only':gate[:]=1.
 return z,np.repeat(gate[:,None,:],16,axis=1)
for arm,labels in ep['readout_conditions'].items():
 cp=lock['arms'][arm]['checkpoint'];assert sha(parent/cp['path'])==cp['sha256'];state=load_checkpoint(parent/cp['path'])['state_dict'];bt=bq=None
 assert set(j['retrieval'][d['retrieval']['protocols'][0]][arm])==set(labels)
 for label in labels:
  out=r/'evaluation'/arm/label;t=torch.load(out/'retrieval.pt',map_location='cpu',weights_only=True);q=torch.load(out/'quantity.pt',map_location='cpu',weights_only=True)
  source_check(t,read(out/'state_sources.json'),label,[('val',i) for i in range(210)])
  source_check(q,read(out/'quantity_state_sources.json'),label,[(c['split'],c['row']) for c in cases])
  assert torch.equal(t['gate_clamped'],t['cluster_output_after_arm_hooks'].clamp(0,1))
  f=t['image_features'].double().numpy();txt=t['text_features'].double().numpy();score=t['scores'].numpy();ti=t['top_i'].numpy();tt=t['top_t'].numpy()
  error=float(np.max(np.abs(t['logit_scale']*f@txt.T-score)));maxscore=max(maxscore,error);assert error<1e-4
  assert np.all(np.isfinite(score)) and np.max(np.abs(np.linalg.norm(f,axis=1)-1))<1e-5
  assert np.array_equal(np.take_along_axis(score,ti,axis=1),np.sort(score,axis=1)[:,-10:][:,::-1])
  assert np.array_equal(np.take_along_axis(score,tt,axis=0),np.sort(score,axis=0)[-10:][::-1])
  assert all(len(set(v))==10 for v in ti) and all(len(set(v))==10 for v in tt.T)
  for protocol,pos in positive.items():
   if protocol=='strict_tie_averaged':
    def averaged(matrix,positives):
     hits=[]
     for k in [1,5,10]:
      hs=[]
      for row,ps in zip(matrix,positives):
       threshold=np.sort(row)[-k];above=row>threshold
       if ps[above].any():hs.append(1.);continue
       eq=row==threshold;total=int(eq.sum());good=int(ps[eq].sum());take=k-int(above.sum())
       hs.append(1.-(math.comb(total-good,take)/math.comb(total,take) if total-good>=take else 0.))
      hits.append(hs)
     return np.array(hits).T
    a=averaged(score,pos);b=averaged(score.T,pos.T)
   else:
    a=np.stack([pos[np.arange(210)[:,None],ti[:,:k]].any(1) for k in [1,5,10]],1).astype(float)
    b=np.stack([pos[tt[:k],np.arange(1050)[None,:]].any(0) for k in [1,5,10]],1).astype(float)
   vals=np.r_[a.mean(0),b.mean(0)];v=j['retrieval'][protocol][arm][label]
   for k,x in zip(['I2T_R1','I2T_R5','I2T_R10','T2I_R1','T2I_R5','T2I_R10','mR'],np.r_[vals,vals.mean()]):close(v[k],x);metrics_count+=1
   values[arm,protocol,label]=(a.sum(1)+b.reshape(210,5,3).mean(1).sum(1))/6
   if protocol==d['retrieval']['protocols'][0]:close(read(out/'metrics.json')['mR'],v['mR'])
  for tensor in [t,q]:
   proj,gate=projection(tensor['prior_hidden'].double().numpy(),state,arm)
   pe=float(np.max(np.abs(proj-tensor['projected'].numpy())));ge=float(np.max(np.abs(gate-tensor['cluster_output_after_arm_hooks'].numpy())))
   maxproj=max(maxproj,pe);maxgate=max(maxgate,ge);assert pe<2e-4 and ge<2e-4,(arm,label,pe,ge)
  if label==base:
   bt=t;bq=q;old=torch.load(parent/f"training/{arm}/validation/epoch_{lock['arms'][arm]['epoch']:02d}/tensors.pt",map_location='cpu',weights_only=True)
   assert torch.equal(t['image_features'],old['image_features']) and torch.equal(t['text_features'],old['text_features'])
   oldq=torch.load(parent/f'quantity/{arm}/tensors.pt',map_location='cpu',weights_only=True);assert torch.equal(q['image_features'],oldq['image_features']) and torch.equal(q['text_features'],oldq['text_features'])
  if arm in d['negative_control_readouts']:assert torch.equal(t['image_features'],bt['image_features']) and torch.equal(q['image_features'],bq['image_features'])
  assert np.max(np.abs(q['image_features'].double().numpy()@q['text_features'].double().numpy().T-q['scores'].numpy()))<1e-6
  rows=read(out/'quantity_records.json');all_qrecords[arm,label]=rows;assert len(rows)==85
  for i,(row,c) in enumerate(zip(rows,cases)):
   assert row['image']==c['image'] and row['split']==c['split'] and row['actual_count']==c['actual_count']
   for wording in ['A','B']:
    pos,neg=qg['quantity_columns'][c['image']][wording];assert qg['quantity_text_order'][pos]==c['pairs'][wording]['positive'] and qg['quantity_text_order'][neg]==c['pairs'][wording]['negative']
    prefix=qg['quantity_text_order'][pos].rsplit(' ',1)[0];others=[k for k,s in enumerate(qg['quantity_text_order']) if s.rsplit(' ',1)[0]==prefix and numwords[s.rsplit(' ',1)[1][:-1]]!=c['actual_count']]
    scores=q['scores'][i].numpy();target=float(scores[pos]);wrong=float(np.max(scores[others]));assert row[wording+'_hard_correct']==(target>wrong) and row[wording+'_pair_correct']==(target>float(scores[neg]));close(row[wording+'_hard_margin'],target-wrong)
  for split in ['train','val']:
   xs=[x for x in rows if x['split']==split];common=[x for x in xs if 1<=x['actual_count']<=4];s=j['quantity'][arm][label][split]
   for wording in ['A','B']:
    qvalues[arm,label,split,wording]={n:np.array([float(x[wording+'_hard_correct']) for x in common if x['actual_count']==n]) for n in range(1,5)}
    for metric in ['pair_correct','hard_correct','hard_margin']:
     k=wording+'_'+metric;close(s[k+'_image_mean'],np.mean([x[k] for x in xs]));close(s[k+'_counts1to4_macro'],np.mean([np.mean([x[k] for x in common if x['actual_count']==n]) for n in range(1,5)]));quantity_count+=2
specs={}
for c in d['conditions'][1:]:specs['condition/'+c['name']]={c['name']:1.,base:-1.}
for i in range(20):specs[f'permutation/{i:02d}']={f'permutation_{i:02d}':1.,base:-1.}
specs['permutation_mean']={**{f'permutation_{i:02d}':.05 for i in range(20)},base:-1.}
for field,level in [('example','without_example'),('policy_image','gray127'),('feedback','constant_training102')]:specs['factor/'+field]={c['name']:(.25 if c[field]==level else -.25) for c in d['conditions']}
assert len(j['retrieval_contrasts'])==186 and len(j['quantity_contrasts'])==248
assert {(c['arm'],c['protocol'],c['contrast']) for c in j['retrieval_contrasts']}=={(a,p,n) for a in d['primary_readouts'] for p in positive for n in specs}
assert {(c['arm'],c['split'],c['wording'],c['contrast']) for c in j['quantity_contrasts']}=={(a,s,w,n) for a in d['primary_readouts'] for s in ['train','val'] for w in ['A','B'] for n in specs}
for c in j['retrieval_contrasts']:
 assert c['weights']==specs[c['contrast']];delta=sum(w*values[c['arm'],c['protocol'],label] for label,w in c['weights'].items());close(delta.mean(),c['delta_mR'])
 rng=np.random.default_rng(seed('retrieval/'+c['arm']+'/'+c['protocol']+'/'+c['contrast']));draw=rng.integers(210,size=(10000,210));bounds=np.quantile(np.sum(delta[draw],axis=1)/210,[.025,.975]);assert np.max(np.abs(bounds-c['interval95']))<1e-10
for c in j['quantity_contrasts']:
 assert c['weights']==specs[c['contrast']];ds={n:sum(w*qvalues[c['arm'],label,c['split'],c['wording']][n] for label,w in c['weights'].items()) for n in range(1,5)}
 close(np.mean([v.mean() for v in ds.values()]),c['delta_hard_macro']);rng=np.random.default_rng(seed('quantity/'+c['arm']+'/'+c['split']+'/'+c['wording']+'/'+c['contrast']));z=np.zeros(10000)
 for n,v in ds.items():z+=v[rng.integers(len(v),size=(10000,len(v)))].sum(1)/len(v)/4
 assert np.max(np.abs(np.quantile(z,[.025,.975])-c['interval95']))<1e-10
out={'ok':True,'logical_generation_records_verified':2184,'new_actual_generations_verified':actual,'baseline_states_exact':273,'warm_generation_anchors_exact':2,'full_split_derangements_rebuilt':40,'frozen_readout_conditions_verified':64,'parent_selected_baselines_exact':4,'prior_source_rows_verified':source_rows_verified,'negative_control_conditions_exactly_invariant':8,'retrieval_metric_values_rebuilt':metrics_count,'quantity_summary_values_rebuilt':quantity_count,'quantity_case_scores_verified':64*85,'retrieval_intervals_rebuilt':186,'stratified_quantity_intervals_rebuilt':248,'float64_score_max_abs_error':maxscore,'float64_projection_max_abs_error':maxproj,'float64_gate_max_abs_error':maxgate,'test_examples_scored':0,'Agent_optimizer_updates':0,'readout_optimizer_updates':0}
args.out.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False))
