"""Recompute descriptive state/gate/feature statistics directly from saved artifacts."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from stage_io import read
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.root
j=read(r/'analysis.json');d=read(r/'diagnostic_plan.json');ep=read(r/'evaluation_plan.json');base=ep['baseline'];count=0
def close(x,y):
 global count
 assert abs(float(x)-float(y))<1e-9,(x,y);count+=1
for arm,labels in ep['readout_conditions'].items():
 b=torch.load(r/'evaluation'/arm/base/'retrieval.pt',map_location='cpu',weights_only=True);bf=b['image_features'].double().numpy();bh=b['prior_hidden'].double().numpy();bs=b['scores'].double().numpy()
 imm=read(r/'evaluation'/arm/'model_immutability.json');assert imm['ok'] and imm['all_parameters_require_grad_false'] and not imm['optimizer_created']
 for label in labels:
  t=torch.load(r/'evaluation'/arm/label/'retrieval.pt',map_location='cpu',weights_only=True);v=j['readout_path'][arm][label]
  f=t['image_features'].double().numpy();h=t['prior_hidden'].double().numpy();g=t['gate_clamped'].numpy();pr=t['projected'].double().numpy()
  xs={'gate_mean':g.mean(),'gate_zero_fraction':(g==0).mean(),'gate_one_fraction':(g==1).mean(),'gate_min':g.min(),'gate_max':g.max(),'projected_centered_rms':np.sqrt(np.mean((pr-pr.mean(0))**2)),'hidden_cosine_to_baseline_mean':np.mean(np.sum(h*bh,1)/np.linalg.norm(h,axis=1)/np.linalg.norm(bh,axis=1)),'image_feature_cosine_to_baseline_mean':np.mean(np.sum(f*bf,1)/np.linalg.norm(f,axis=1)/np.linalg.norm(bf,axis=1)),'image_feature_max_abs_change':np.abs(f-bf).max(),'score_max_abs_change':np.abs(t['scores'].double().numpy()-bs).max()}
  assert set(v)==set(xs)
  for k,x in xs.items():close(v[k],x)
gen=read(r/'generation/records.json');gr=read(r/'generation_result.json')
for condition in d['conditions']:
 name=condition['name'];assert Counter(x['stop_reason'] for x in gen if x['condition']==name)==gr['quality_by_condition'][name]
 for split in ['train','val']:
  xs=sorted([x for x in gen if x['condition']==name and x['split']==split],key=lambda x:x['row']);v=j['generation_states'][name][split]
  h=np.array([torch.load(r/x['state_file'],map_location='cpu',weights_only=True).double().numpy() for x in xs]);u=h/np.linalg.norm(h,axis=1)[:,None];center=h-h.mean(0)
  close(v['images'],len(xs));close(v['unique_state_sha256'],len({x['numeric_state_sha256'] for x in xs}));close(v['centered_rms'],np.sqrt((center**2).mean()))
  singular=np.linalg.svd(center,compute_uv=False);energy=singular**2;positive=energy[energy>0]
  if energy.sum()>1e-20:
   probability=positive/energy.sum();rank=np.exp(-np.dot(probability,np.log(probability)))
  else:rank=0.
  close(v['effective_rank_centered_covariance'],rank);close(v['mean_pairwise_cosine'],((u.sum(0)**2).sum()-len(u))/(len(u)*(len(u)-1)))
  traces=[read(r/x['trace_file']) for x in xs];texts=[t['trajectory_text'] for t in traces]
  expected={'quality_usable':sum(x['quality_usable'] for x in xs),'tool_requested_images':sum(bool(t['tool_events']) for t in traces),'tool_requests_total':sum(len(t['tool_events']) for t in traces),'feedback_consumed_events':sum(sum(e.get('after_round',0)<len(t['rounds']) for e in t['tool_events']) for t in traces),'feedback_exposed_images':sum(any(e.get('after_round',0)<len(t['rounds']) for e in t['tool_events']) for t in traces),'last_round_feedback_not_consumed':sum(sum(e.get('after_round',0)>=len(t['rounds']) for e in t['tool_events']) for t in traces),'lexical_mentions_harbor_or_port':sum(any(c in text for c in ['港口','码头']) or bool(re.search(r'\b(?:harbor|port)\b',text,re.I)) for text in texts),'lexical_query_match_phrase':sum('与查询文本相符' in text for text in texts)}
  for k,x in expected.items():close(v[k],x)
for arm,ps in j['permutation_summary'].items():
 for protocol,s in ps.items():
  vals=np.array([j['retrieval'][protocol][arm][f'permutation_{i:02d}']['mR'] for i in range(20)])
  assert np.array_equal(vals,np.array(s['all20_mR']))
  for k,x in {'mean_mR':vals.mean(),'std_over_permutations':vals.std(ddof=1),'min_mR':vals.min(),'max_mR':vals.max(),'mean_minus_baseline':vals.mean()-j['retrieval'][protocol][arm][base]['mR']}.items():close(s[k],x)
out={'ok':True,'descriptive_values_independently_rebuilt':count,'generation_state_strata':16,'readout_path_conditions':64,'all_permutation_spreads_rebuilt':True,'model_freeze_records_verified':4}
a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out))
