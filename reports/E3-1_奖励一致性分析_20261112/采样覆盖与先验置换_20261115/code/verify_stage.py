"""Independent CPU reconstruction of the full donor grid and image-paired statistics."""
import argparse
import collections
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.root
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
torch.set_num_threads(2);plan=read(r/'plan.json');analysis=read(r/'analysis.json');cases=read(r/'inputs/cases.json');by={x['image']:x for x in cases};policies=[x['name'] for x in plan['policies']]
assert analysis['plan_sha256']==sha(r/'plan.json') and analysis['optimizer_updates']==plan['optimizer_updates']==0
for n,h in plan['inputs_sha256'].items():assert sha(r/n)==h,n
for snapshot in ['gpu_code_hashes.json','post_code_hashes.json']:
    for n,h in read(r/snapshot).items():assert sha(r/n)==h,n
scores=read(r/'scoring/all_scores.json');derived=read(r/'grid_quantity_metrics.json');t=torch.load(r/'scoring/tensors.pt',map_location='cpu',weights_only=True);f=torch.load(r/'features/tensors.pt',map_location='cpu',weights_only=True);gallery=read(r/'features/gallery.json');q=t['quantity_similarities'].double().numpy();q64=np.asarray(t['image_features'].double()@f['quantity_text_features'].double().T)
assert len(scores)==len(derived)==2176 and t['record_ids']==[x['id'] for x in scores] and np.max(abs(q-q64))<1e-6
new=[json.loads(s) for policy in policies for s in (r/'generation'/policy/'records.jsonl').read_text().splitlines()];old=read(r/'historical/records.json');assert len(new)==256 and len(old)==80;donors={x['id']:x for x in new+old};events=rounds=0;state_ids={}
for x in new+old:
    h=torch.load(r/x['state_file'],map_location='cpu',weights_only=True);assert h.dtype==torch.bfloat16 and h.shape==(3584,) and torch.isfinite(h).all()
    digest=hashlib.sha256(h.view(torch.uint8).numpy().tobytes()).hexdigest();assert sha(r/x['state_file'])==x['state_file_sha256'] and digest==x['generation_state_sha256'];tr=read(r/x['trace_file']);assert sha(r/x['trace_file'])==x['trace_sha256'] and tr['state_tensor_sha256']==digest and not tr['forced_tool_witness'];state_ids[x['id']]=digest
    assert tr['quality_usable']==x['quality_usable'] and tr['strict_protocol_usable']==x['strict_protocol_usable'] and len(tr['tool_events'])==x['tools_used']
    actual=read(r/'scoring/actual_tools.json')[x['image']]
    for e in tr['tool_events']:
        assert e['origin']=='model_requested_actual_tool' and e['text']==actual['text'] and abs(e['confidence']-actual['confidence'])<1e-6;events+=1
    for z in tr['rounds']:
        ids=z['generated_ids'];assert len(ids)==z['generated_token_count'] and ids[0] not in z['eos_token_ids'] and all(k not in z['eos_token_ids'] for k in ids[:-1]);assert (z['prompt_token_ids']+ids)[-2]==z['state_input_token_id'] and ids[-1]==z['last_emitted_token_id'];assert z['generation_kwargs']['min_new_tokens']==1 and z['generation_kwargs']['top_k']==0 and z['generation_kwargs']['top_p']==1.;rounds+=1
    if x['quality_usable']:assert tr['rounds'][-1]['last_emitted_token_id'] in tr['rounds'][-1]['eos_token_ids']
for x in new:
    k=policies.index(x['policy']);i=plan['images'].index(x['image']);assert x['seed']==2100000+k*10000+i*16+x['sample'] and 0<=x['sample']<16 and x['policy_hash']==plan['policies'][k]['lora_hash']
words={w:i+1 for i,w in enumerate('one two three four five six seven'.split())};texts=gallery['quantity_text_order'];numerical=[]
def metric(row,case):
    result={}
    for w in ['A','B']:
        pos,neg=gallery['quantity_columns'][case['image']][w];prefix=texts[pos].rsplit(' ',1)[0];pool={words[s.rsplit(' ',1)[1][:-1]]:i for i,s in enumerate(texts) if s.rsplit(' ',1)[0]==prefix};assert pool[case['actual_count']]==pos
        other=[float(row[col]) for n,col in sorted(pool.items()) if n!=case['actual_count']];positive=float(row[pos]);result.update({w+'_pair_margin':positive-float(row[neg]),w+'_pair_correct':float(positive>float(row[neg])),w+'_hard_margin':positive-max(other),w+'_hard_correct':float(all(positive>v for v in other)),w+'_mean_margin':math.fsum(positive-v for v in other)/len(other),w+'_argmax_count':sorted(pool,key=lambda n:(-float(row[pool[n]]),n))[0]})
    return result
index={};reconstructed={}
for i,(x,z) in enumerate(zip(scores,derived)):
    donor=donors[x['donor_id']];target=by[x['target_image']];assert x['id']==z['id'] and x['donor_image']==donor['image'] and x['policy']==donor['policy'];assert x['generation_state_sha256']==x['scorer_state_sha256']==state_ids[donor['id']];assert x['state_file']==donor['state_file'] and x['trace_sha256']==donor['trace_sha256'] and x['same_image']==(donor['image']==target['image']) and x['same_object']==(by[donor['image']]['object']==target['object'])
    assert x['donor_quality']==donor['quality_usable'] and x['donor_strict_quality']==donor['strict_protocol_usable'] and x['donor_tools_used']==donor['tools_used']
    m=metric(q[i],target);m64=metric(q64[i],target)
    for n,v in m.items():assert abs(v-z[n])<1e-12,(x['id'],n)
    for w in ['A','B']:
        assert m[w+'_pair_margin']==x['margin_'+w] and bool(m[w+'_pair_correct'])==x['correct_'+w]
        if (m[w+'_hard_correct'],m[w+'_argmax_count'])!=(m64[w+'_hard_correct'],m64[w+'_argmax_count']):numerical.append({'id':x['id'],'wording':w,'primary_margin':m[w+'_hard_margin'],'float64_margin':m64[w+'_hard_margin']})
    row={**x,**m,'quality':float(donor['quality_usable']),'strict_quality':float(donor['strict_protocol_usable']),'tool_usage':float(donor['tools_used']>0)};row['pair_reward']=.2*row['quality']+m['A_pair_margin'];row['hard_reward']=.2*row['quality']+m['A_hard_margin'];reconstructed[x['id']]=row
    key=(x['policy'],x['grid_role'],x['sample'],x['donor_image'],x['target_image']);assert key not in index;index[key]=row
expected={(p,'new_sample',k,d,n) for p in policies for k in range(16) for d in plan['images'] for n in plan['images']}|{(p,'reused_greedy',None,d,n) for p in policies for d in plan['images'] for n in plan['images']};assert set(index)==expected
ht=torch.load(r/'historical/tensors.pt',map_location='cpu',weights_only=True);assert ht['record_ids']==[x['id'] for x in old];history={}
for i,x in enumerate(old):
    m=metric(ht['quantity_similarities'][i].numpy(),by[x['image']]);z={**x,**m,'quality':float(x['quality_usable']),'strict_quality':float(x['strict_protocol_usable']),'tool_usage':float(x['tools_used']>0)};z['pair_reward']=.2*z['quality']+z['A_pair_margin'];z['hard_reward']=.2*z['quality']+z['A_hard_margin'];history[x['id']]=z
    if x['role']=='greedy':
        own=index[x['policy'],'reused_greedy',None,x['image'],x['image']]
        for w in ['A','B']:assert abs(own[w+'_pair_margin']-x['margin_'+w])<1e-6
# Check checkpoint hashes and warm anchor outputs, including both immutable policy contracts.
for p in plan['policies']:
    result=read(r/'generation'/p['name']/'result.json');assert result['ok'] and result['optimizer_updates']==0 and result['records']==128 and result['policy_hash_unchanged']==p['lora_hash']
    anchor=r/'anchors'/p['name'];w=read(anchor/'result.json');assert w['ok'] and w['policy_hash']==p['lora_hash'] and w['optimizer_updates']==0 and w['probe_generations']==1
    h=torch.load(anchor/'prior.pt',map_location='cpu',weights_only=True);assert hashlib.sha256(h.view(torch.uint8).numpy().tobytes()).hexdigest()==state_ids[w['parent_greedy_id']];assert [x['generated_ids'] for x in read(anchor/'trace.json')['rounds']]==[x['generated_ids'] for x in read(r/donors[w['parent_greedy_id']]['trace_file'])['rounds']]
def native(p,n,pool):
    if pool=='historical4':return sorted([x for x in history.values() if x['policy']==p and x['image']==n and x['role']=='candidate'],key=lambda x:x['seed'])
    return [index[p,'new_sample',k,n,n] for k in range(4 if pool=='new4' else 16)]
def selected(xs,selector):
    if selector=='uniform':return xs
    good=[x for x in xs if x['quality']] or xs
    if selector=='hard_quality_first':ys,key=good,'A_hard_margin'
    elif selector=='B_oracle_quality_first':ys,key=good,'B_hard_margin'
    else:ys,key=xs,{'pair_total':'pair_reward','hard_total':'hard_reward'}[selector]
    return [sorted(ys,key=lambda x:(-x[key],x['seed']))[0]]
def coverage(xs,key):
    if key=='qualified_B_oracle_success':return float(any(x['quality'] and x['B_hard_correct'] for x in xs))
    if key=='B_hard_uniform':return math.fsum(x['B_hard_correct'] for x in xs)/len(xs)
    return math.fsum(x[key] for x in xs)/len(xs)
# Reconstruct every value array from the raw Cartesian grid, then bootstrap those arrays.
intervals=0
for s in analysis['statistics']:
    parts=s['label'].split('/');ns=s['image_names'];values=[]
    for n in ns:
        if parts[0]=='balanced32_minus_SFT30':
            key=parts[1];v=np.mean([index[policies[1],'new_sample',k,n,n][key] for k in range(16)])-np.mean([index[policies[0],'new_sample',k,n,n][key] for k in range(16)])
        elif parts[1] in ['new_sample','reused_greedy']:
            p,role,kind,key=parts;ks=range(16) if role=='new_sample' else [None];own=np.mean([index[p,role,k,n,n][key] for k in ks]);donor=np.mean([index[p,role,k,d,n][key] for k in ks for d in plan['images'] if d!=n]);v={'own':own,'donor':donor,'own_minus_donor':own-donor}[kind]
        elif parts[1]=='airplanes':
            p,_,role,key=parts;ks=range(16) if role=='new_sample' else [None];assert by[n]['object']=='飞机';v=np.mean([index[p,role,k,n,n][key] for k in ks])-np.mean([index[p,role,k,d,n][key] for k in ks for d in plan['images'] if d!=n and by[d]['object']=='飞机'])
        elif parts[1]=='coverage':
            p,_,pool,key=parts;v=coverage(native(p,n,pool),key)
        elif parts[1]=='selector':
            p,_,pool,sel,key=parts;xs=selected(native(p,n,pool),sel);v=math.fsum(x[key] for x in xs)/len(xs)
        else:
            assert parts[1]=='expansion';p,_,key=parts;v=coverage(native(p,n,'new16'),key)-coverage(native(p,n,'new4'),key)
        values.append(float(v))
    assert np.allclose(values,s['values'],atol=1e-12,rtol=0),s['label'];cs=np.array([by[n]['actual_count'] for n in ns]);v=np.array(values);rng=np.random.default_rng((2026111557+int(hashlib.sha256(s['label'].encode()).hexdigest()[:8],16))%2**32)
    if s['scope']=='count_macro':
        point=0.;boot=np.zeros(10000)
        for c in range(1,5):
            z=v[cs==c];assert len(z);point+=float(z.mean())/4;boot+=z[rng.integers(len(z),size=(10000,len(z)))].mean(axis=1)/4
    else:point=float(v.mean());boot=v[rng.integers(len(v),size=(10000,len(v)))].mean(axis=1)
    assert abs(point-s['mean'])<1e-12 and np.allclose(np.percentile(boot,[2.5,97.5]),s['interval95'],atol=1e-12,rtol=0),s['label'];intervals+=1
selections=read(r/'per_image_selections.json');assert len(selections)==240
for s in selections:
    xs=selected(native(s['policy'],s['image'],s['pool']),s['selector']);assert s['selected_ids']==[x.get('donor_id',x['id']) for x in xs]
    for k in ['A_pair_correct','B_pair_correct','A_hard_correct','B_hard_correct','A_hard_margin','B_hard_margin','quality','strict_quality','tool_usage']:assert abs(s[k]-math.fsum(x[k] for x in xs)/len(xs))<1e-12
proof={'ok':True,'new_evaluation_trajectories_verified':256,'historical_trajectories_verified':80,'warm_anchor_generations_verified':2,'grid_records_verified':2176,'cartesian_grid_complete':True,'generation_rounds_checked':rounds,'actual_tool_events_checked':events,'float64_quantity_dot_max_error':float(abs(q-q64).max()),'float64_multi_count_decision_changes':numerical,'quantity_classifications_checked':4352,'bootstrap_intervals_rebuilt':intervals,'per_image_selector_rows_rebuilt':240,'immutable_policy_hashes_checked':2,'optimizer_updates':0}
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n');print('ALL_PERMUTATION_EVIDENCE_VERIFIED',intervals,flush=True)
