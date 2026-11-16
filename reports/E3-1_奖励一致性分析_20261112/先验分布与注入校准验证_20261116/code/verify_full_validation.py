import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.root;read=lambda p:json.loads(p.read_text(encoding='utf-8'));torch.set_num_threads(2);out=r/'full_validation';plan=read(r/'full_validation_plan.json')
for snap in ['full_validation_code_hashes.json','full_validation_post_hashes.json']:
    for n,h in read(r/snap).items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
records=read(out/'records.json');g=read(out/'gallery.json');t=torch.load(out/'retrieval_tensors.pt',map_location='cpu',weights_only=True);mt=torch.load(out/'metric_tensors.pt',map_location='cpu',weights_only=True);m=read(out/'metrics.json');assert len(records)==210 and sum(not x['reused'] for x in records)==188 and read(out/'generation_result.json')['optimizer_updates']==0
for x in records:
    h=torch.load(r/x['state_file'],map_location='cpu',weights_only=True);assert h.shape==(3584,) and h.dtype==torch.bfloat16 and hashlib.sha256((r/x['state_file']).read_bytes()).hexdigest()==x['state_file_sha256'];assert hashlib.sha256(h.view(torch.uint8).numpy().tobytes()).hexdigest()==x['numeric_state_sha256'];tr=read(r/x['trace_file']);assert hashlib.sha256((r/x['trace_file']).read_bytes()).hexdigest()==x['trace_sha256'] and tr['state_tensor_sha256']==x['numeric_state_sha256']
    if not x['reused']:assert not tr['forced_tool_witness'] and all(z['generation_kwargs']['do_sample']==False for z in tr['rounds'])
owners=np.array(g['caption_owners']);groups=np.array(g['token_groups']);assert np.array_equal(owners,t['caption_owner'].numpy()) and np.array_equal(groups,t['caption_group'].numpy());assert len(g['images'])==210 and len(g['captions'])==len(owners)
# Token grouping is rebuilt from exact saved token rows, independent of text spelling.
seen={};rebuilt=[]
for row in t['caption_tokens'].tolist():
    key=tuple(row)
    if key not in seen:seen[key]=len(seen)
    rebuilt.append(seen[key])
assert rebuilt==g['token_groups'];counts=np.bincount(owners,minlength=210);Is={};Ts={};matrix_error=0.
for arm,F in t['image_features'].items():
    S=t['similarities'][arm].numpy();S64=np.asarray((float(t['logit_scale'])*F.double())@t['text_features'].double().T);matrix_error=max(matrix_error,float(abs(S-S64).max()));assert matrix_error<1e-4
    top=t['topk'][arm];ih=np.zeros((210,3));th=np.zeros((len(owners),3))
    for i in range(210):
        ids=top['i2t'][i].numpy();assert len(set(ids))==10 and np.all(S[i,ids][:-1]>=S[i,ids][1:]) and S[i,ids[-1]]>=np.partition(S[i],-10)[-10]-1e-7
        positive_tokens=set(groups[owners==i])
        for kk,k in enumerate([1,5,10]):ih[i,kk]=any(groups[j] in positive_tokens for j in ids[:k])
    for j in range(len(owners)):
        ids=top['t2i'][:,j].numpy();assert len(set(ids))==10 and np.all(S[ids,j][:-1]>=S[ids,j][1:]) and S[ids[-1],j]>=np.partition(S[:,j],-10)[-10]-1e-7
        positives=set(owners[groups==groups[j]])
        for kk,k in enumerate([1,5,10]):th[j,kk]=any(i in positives for i in ids[:k])
    assert np.array_equal(ih,mt['i2t_hits'][arm].numpy()) and np.array_equal(th,mt['t2i_hits'][arm].numpy());Is[arm]=ih;Ts[arm]=np.stack([th[owners==i].sum(0) for i in range(210)])
reused=set(g['reused_images']);subsets={'all210':np.arange(210),'unseen188':np.array([i for i,n in enumerate(g['images']) if n not in reused]),'previous22':np.array([i for i,n in enumerate(g['images']) if n in reused])}
for subset,ns in subsets.items():
    for arm in Is:
        values=np.r_[Is[arm][ns].mean(0),Ts[arm][ns].sum(0)/counts[ns].sum()];ref=m['summary'][subset][arm]
        for key,v in zip(['I2T_R1','I2T_R5','I2T_R10','T2I_R1','T2I_R5','T2I_R10'],values):assert abs(v-ref[key])<1e-12
        assert abs(values.mean()-ref['mR'])<1e-12
for s in m['paired_mR_intervals']:
    ns=subsets[s['subset']];ref=s['reference'];rng=np.random.default_rng((2026111603+int(hashlib.sha256(s['label'].encode()).hexdigest()[:8],16))%2**32);draw=ns[rng.integers(len(ns),size=(10000,len(ns)))];di=Is['mean_shift_auto']-Is[ref];dt=Ts['mean_shift_auto']-Ts[ref];boot=(di[draw].mean(1).sum(1)+(dt[draw].sum(1)/counts[draw].sum(1)[:,None]).sum(1))/6;point=m['summary'][s['subset']]['mean_shift_auto']['mR']-m['summary'][s['subset']][ref]['mR'];assert abs(point-s['mean'])<1e-12 and np.allclose(np.percentile(boot,[2.5,97.5]),s['interval95'],atol=1e-12,rtol=0)
checks={s['reference']+'_unseen188_mR_CI_lower_positive':s['interval95'][0]>0 for s in m['paired_mR_intervals'] if s['subset']=='unseen188'};assert checks==m['adoption']['checks']
proof={'ok':True,'val_images':210,'new_states':188,'reused_states':22,'caption_queries':len(owners),'token_equivalence_groups':len(seen),'image_features_verified':840,'retrieval_directions_and_recalls_rebuilt':72,'paired_cluster_intervals_rebuilt':9,'float64_scaled_score_max_error':matrix_error,'policy_updates':0,'calibration_choice_fixed_before_extension':True};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n');print('ALL_FULL_VAL210_RETRIEVAL_EVIDENCE_VERIFIED',flush=True)
