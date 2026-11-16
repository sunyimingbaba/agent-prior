"""Recompute state-bank diversity, protocol coverage and finite-permutation dispersion."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stage_io import read
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.root;j=read(r/'analysis.json');d=read(r/'protocol_plan.json');count=0
def close(x,y):
 global count
 assert abs(float(x)-float(y))<1e-9,(x,y);count+=1
for family in d['state_families']:
 bank=torch.load(r/'state_banks'/(family+'.pt'),map_location='cpu',weights_only=True);meta=read(r/'state_banks'/(family+'.json'))
 for split,tensor in bank.items():
  h=tensor.double().numpy();v=j['state_diagnostics'][family][split];center=h-h.mean(0);singular=np.linalg.svd(center,compute_uv=False);energy=singular**2;prob=energy[energy>0]/energy.sum() if energy.sum()>1e-20 else np.array([]);u=h/np.linalg.norm(h,axis=1)[:,None]
  expected={'images':len(h),'available':sum(x['available'] for x in meta[split]),'fallbacks':sum(not x['available'] for x in meta[split]),'unique_states':len({x['numeric_sha256'] for x in meta[split]}),'centered_rms':np.sqrt(np.mean(center**2)),'effective_rank':np.exp(-np.dot(prob,np.log(prob))) if len(prob) else 0.,'mean_pair_cosine':((u.sum(0)**2).sum()-len(u))/(len(u)*(len(u)-1))}
  assert set(v)==set(expected)
  for k,x in expected.items():close(v[k],x)
for arm,ps in j['permutation_summary'].items():
 for protocol,fs in ps.items():
  for family,s in fs.items():
   vals=np.array([j['retrieval'][protocol][arm][family][f'permutation_{i:02d}']['mR'] for i in range(20)]);assert vals.tolist()==s['all20_mR']
   for k,x in {'mean':vals.mean(),'std':vals.std(ddof=1),'min':vals.min(),'max':vals.max()}.items():close(s[k],x)
raw=read(r/'generation_v2/records.json')
for group,s in j['protocols'].items():
 xs=[x for x in raw if x['group']==group];expected={'images':len(xs),'strict_completed_with_consumed_tool':sum(x['strict_protocol_usable'] and x['tools_used']>0 and x['feedback_consumed'] for x in xs),'generated_result_tags':sum(x['generated_result_tag'] for x in xs),'body_pool_available':sum(x['coverage']['answer_pool_available'] for x in xs),'prefill_available':sum(x['coverage']['answer_prefill_available'] for x in xs)}
 for k,x in expected.items():close(s[k],x)
out={'ok':True,'descriptive_scalar_values_rebuilt':count,'state_bank_strata':12,'all_permutation_spreads_rebuilt':True,'all_protocol_group_summaries_rebuilt':True};a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out))
