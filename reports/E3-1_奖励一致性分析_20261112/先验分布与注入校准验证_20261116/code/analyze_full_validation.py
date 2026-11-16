from pathlib import Path
import csv
import hashlib
import json
import sys
import time
import numpy as np
import torch
from stage_io import read,dump
r=Path(__file__).resolve().parent.parent;out=r/'full_validation';torch.set_num_threads(2);t=torch.load(out/'retrieval_tensors.pt',map_location='cpu',weights_only=True);g=read(out/'gallery.json');owners=np.array(g['caption_owners']);groups=np.array(g['token_groups']);N=210;arms=list(t['image_features']);caption_counts=np.bincount(owners,minlength=N);positive_images={k:set(owners[groups==k]) for k in set(groups)};own_groups=[set(groups[owners==i]) for i in range(N)];I={};T={};Tsum={}
for arm in arms:
    top=t['topk'][arm];ih=np.array([[any(groups[j] in own_groups[i] for j in top['i2t'][i,:k].tolist()) for k in [1,5,10]] for i in range(N)],dtype=float);th=np.array([[any(i in positive_images[groups[j]] for i in top['t2i'][:k,j].tolist()) for k in [1,5,10]] for j in range(len(owners))],dtype=float);I[arm]=ih;T[arm]=th;Tsum[arm]=np.stack([th[owners==i].sum(0) for i in range(N)])
    # Reproduce the repository's duplicate-token-aware helpers on the exact saved top-k.
    sys.path.insert(0,str(Path(read(r/'plan.json')['base_root'])/'CUP'))
    from vilt.modules.objectives import tr_top_k,ir_top_k
    correspondence={str(k):owners[groups==k].tolist() for k in set(groups)};rep=torch.tensor(groups);ids=torch.arange(N)
    for idx,k in enumerate([1,5,10]):
        actual_i=float(tr_top_k(ids.unsqueeze(0),correspondence,rep[top['i2t'][:,:k]],k));actual_t=float(ir_top_k(rep.unsqueeze(1),correspondence,top['t2i'][:k],k));assert abs(actual_i-ih[:,idx].mean())<1e-6 and abs(actual_t-th[:,idx].mean())<1e-6
reused=set(g['reused_images']);subsets={'all210':np.arange(N),'unseen188':np.array([i for i,n in enumerate(g['images']) if n not in reused]),'previous22':np.array([i for i,n in enumerate(g['images']) if n in reused])};assert len(subsets['unseen188'])==188
summary={};bootstraps=[];per=[]
for subset,ns in subsets.items():
    summary[subset]={}
    for arm in arms:
        ir=I[arm][ns].mean(0);tr=Tsum[arm][ns].sum(0)/caption_counts[ns].sum();summary[subset][arm]={'images':len(ns),'caption_queries':int(caption_counts[ns].sum()),'I2T_R1':float(ir[0]),'I2T_R5':float(ir[1]),'I2T_R10':float(ir[2]),'T2I_R1':float(tr[0]),'T2I_R5':float(tr[1]),'T2I_R10':float(tr[2]),'mR':float(np.r_[ir,tr].mean())}
    for ref in ['native','learned_only','mean_shift_shared102']:
        label=subset+'/mean_shift_auto-minus-'+ref;rng=np.random.default_rng((2026111603+int(hashlib.sha256(label.encode()).hexdigest()[:8],16))%2**32);draw=ns[rng.integers(len(ns),size=(10000,len(ns)))];di=I['mean_shift_auto']-I[ref];dt=Tsum['mean_shift_auto']-Tsum[ref];boot=(di[draw].mean(1).sum(1)+(dt[draw].sum(1)/caption_counts[draw].sum(1)[:,None]).sum(1))/6;point=summary[subset]['mean_shift_auto']['mR']-summary[subset][ref]['mR'];bootstraps.append({'label':label,'subset':subset,'reference':ref,'mean':point,'interval95':np.percentile(boot,[2.5,97.5]).tolist(),'images':len(ns),'caption_queries':int(caption_counts[ns].sum())})
for arm in arms:
    for i,n in enumerate(g['images']):per.append({'arm':arm,'image':n,'image_index':i,'reused':n in reused,'captions':int(caption_counts[i]),'I2T_hits':I[arm][i].tolist(),'T2I_hit_sums':Tsum[arm][i].tolist()})
checks={x['reference']+'_unseen188_mR_CI_lower_positive':x['interval95'][0]>0 for x in bootstraps if x['subset']=='unseen188'}
dump(out/'metrics.json',{'ok':True,'summary':summary,'paired_mR_intervals':bootstraps,'adoption':{'passed':all(checks.values()),'checks':checks},'repository_duplicate_aware_helpers_equal':True,'caption_token_groups':len(set(groups))});torch.save({'i2t_hits':{k:torch.tensor(v) for k,v in I.items()},'t2i_hits':{k:torch.tensor(v) for k,v in T.items()},'caption_counts':torch.tensor(caption_counts)},out/'metric_tensors.pt')
with (out/'per_image_metrics.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(per[0]),lineterminator='\n');w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,list) else v for k,v in x.items()} for x in per)
print('FULL_VAL_RETRIEVAL_METRICS_COMPLETE',len(g['captions']),flush=True)
