"""Independent reconstruction of source vectors, projection math, scores and confirmation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.root;torch.set_num_threads(2)
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
plan=read(r/'plan.json');analysis=read(r/'analysis.json');assert analysis['plan_sha256']==sha(r/'plan.json')
for snap in ['gpu_code_hashes.json','gpu_retry_code_hashes.json','gradient_code_hashes.json','post_code_hashes.json']:
    for n,h in read(r/snap).items():assert sha(r/n)==h,n
for n,h in plan['inputs_sha256'].items():assert sha(r/n)==h,n
cases=read(r/'inputs/evaluation_cases.json');by={x['image']:x for x in cases};states=read(r/'states/records.json');pt=torch.load(r/'path/tensors.pt',map_location='cpu',weights_only=True);w=torch.load(r/'path/weights.pt',map_location='cpu',weights_only=True);arch=read(r/'path/architecture.json');assert len(states)==511 and pt['record_ids']==[x['id'] for x in states];ix={x['id']:i for i,x in enumerate(states)}
for i,x in enumerate(states):
    h=torch.load(r/x['state_file'],map_location='cpu',weights_only=True);assert h.shape==(3584,) and h.dtype==torch.bfloat16 and torch.equal(h,pt['hidden'][i]);assert sha(r/x['state_file'])==x['state_file_sha256'] and hashlib.sha256(h.view(torch.uint8).numpy().tobytes()).hexdigest()==x['numeric_state_sha256']
    if 'trace_file' in x:assert sha(r/x['trace_file'])==x['trace_sha256']
H=pt['hidden'].double();L=H@w['agent_proj.0.weight'].double().T+w['agent_proj.0.bias'].double();eps=arch['layernorm_eps'];Z=(L-L.mean(1,keepdim=True))/torch.sqrt(L.var(1,unbiased=False,keepdim=True)+eps)*w['agent_proj.1.weight'].double()+w['agent_proj.1.bias'].double();G=(Z@w['cluster_mapping.weight'].double().T+w['cluster_mapping.bias'].double()).flatten();R=G.clamp(0,1)
learned=w['visual_prompt_embeddings'].double()@w['visual_prompt_proj.weight'].double().T+w['visual_prompt_proj.bias'].double();gp=Z@w['visual_prompt_proj_gather.weight'].double().T;gl=learned[0]@w['visual_prompt_proj_gather.weight'].double().T;P=(1-R)[:,None,None]*gp[:,None,:]+R[:,None,None]*gl[None,:,:]+w['visual_prompt_proj_gather.bias'].double();errs={k:float((x-y.double()).abs().max()) for k,x,y in [('linear',L,pt['linear']),('projected',Z,pt['projected']),('gate_raw',G,pt['gate_raw']),('gate_clipped',R,pt['gate_clipped']),('learned_prompt',learned[0],pt['learned_prompt']),('prompt_first',P[:,0],pt['prompt_first']),('prompt_norm',P.norm(dim=(1,2)),pt['prompt_norm'])]}
assert max(errs.values())<read(r/'numeric_contract.json')['projection_and_prompt_float64_abs_max'],errs
original=pt['linear'][[ix['original__'+n] for n in plan['training_images']]].mean(0)
for fam in ['initial_SFT30','balanced32']:
    expected=(original-pt['linear'][[ix['greedy__'+fam+'__'+n] for n in plan['training_images']]].mean(0))[None];assert torch.equal(expected,pt['offsets'][fam])
assert torch.count_nonzero(pt['offsets']['original_cached'])==0
for sid in ['original__102.tif','greedy__initial_SFT30__102.tif','greedy__balanced32__102.tif','original__1937.tif']:
    x=torch.load(r/'path'/('witness_'+sid+'.pt'),map_location='cpu',weights_only=True);seq=x['vision_preln'].permute(1,0,2);assert torch.equal(torch.cat([seq[:1],x['prompt'].permute(1,0,2),seq[1:]]),x['transformer_input']);assert x['transformer_input'].shape==(273,1,1024) and float((x['prompt'].double()-P[ix[sid]][None]).abs().max())<1e-3
f=torch.load(r/'features/tensors.pt',map_location='cpu',weights_only=True);gallery=read(r/'features/gallery.json');texts=gallery['quantity_text_order'];words={s:i+1 for i,s in enumerate('one two three four five six seven'.split())}
def metrics(q,n):
    c=by[n];out={}
    for word in ['A','B']:
        pos,neg=gallery['quantity_columns'][n][word];prefix=texts[pos].rsplit(' ',1)[0];cols={words[s.rsplit(' ',1)[1][:-1]]:i for i,s in enumerate(texts) if s.rsplit(' ',1)[0]==prefix};assert cols[c['actual_count']]==pos;v=float(q[pos]);other=[float(q[col]) for count,col in cols.items() if count!=c['actual_count']]
        out.update({word+'_hard_correct':float(all(v>z for z in other)),word+'_hard_margin':v-max(other),word+'_pair_correct':float(v>float(q[neg])),word+'_pair_margin':v-float(q[neg]),word+'_argmax_count':max(sorted(cols),key=lambda k:float(q[cols[k]]))})
    return out
new=read(r/'scoring/all_scores.json');nt=torch.load(r/'scoring/tensors.pt',map_location='cpu',weights_only=True);base=read(r/'baseline/records.json');bt=torch.load(r/'baseline/tensors.pt',map_location='cpu',weights_only=True);allmetrics={};numerical=[];ranks=0;maxdot=0.
for source,rows,tensor in [('new',new,nt),('base',base,bt)]:
    assert tensor['record_ids']==[x['id'] for x in rows];q=tensor['quantity_similarities'].numpy();qd=np.asarray(tensor['image_features'].double()@f['quantity_text_features'].double().T);maxdot=max(maxdot,float(abs(q-qd).max()));assert maxdot<1e-6;vix={v:i for i,v in enumerate(tensor['val_record_ids'])};vd=tensor['image_features'][[i for i,x in enumerate(rows) if (x['target_split'] if source=='new' else x['split'])=='val']].double()@f['original_val_text_features'].double().T
    for i,x in enumerate(rows):
        n=x['target_image'] if source=='new' else x['image'];m=metrics(q[i],n);m64=metrics(qd[i],n)
        for word in ['A','B']:
            assert m[word+'_pair_margin']==x['margin_'+word] and bool(m[word+'_pair_correct'])==x['correct_'+word]
            if (m[word+'_hard_correct'],m[word+'_argmax_count'])!=(m64[word+'_hard_correct'],m64[word+'_argmax_count']):numerical.append({'source':source,'id':x['id'],'wording':word,'margin':m[word+'_hard_margin']})
        if by[n]['split']=='val':
            row=vd[vix[x['id']]];pos=gallery['val_positive_columns'][n];rank=int((row[:391]>row[pos].max()).sum());assert rank==x['retrieval_rank0'] and x['retrieval_rr']==1/(rank+1);ranks+=1
        if source=='new':
            state=states[ix[x['donor_id']]];assert x['state_numeric_sha256']==x['scorer_state_sha256']==state['numeric_state_sha256'] and x['state_file']==state['state_file'];assert x['same_image']==(x['donor_image']==n) and x['same_object']==(by[x['donor_image']]['object']==by[n]['object'])
        allmetrics[source,x['id']]={**x,**m,'target_image':n}
assert len(new)==1470 and len(base)==340
# Complete3x5x8x8 grid, and data-independent candidate rule.
grid=[x for x in new if x['phase']=='diagnostic_grid'];assert len(grid)==960;keys={(x['family'],x['variant'],x['donor_image'],x['target_image']) for x in grid};assert keys=={(f,v,d,n) for f in ['original_cached','initial_SFT30','balanced32'] for v in plan['variants'] for d in plan['training_images'] for n in plan['training_images']}
choices=[]
for rank,v in enumerate(plan['variants']):
    xs=[allmetrics['new',x['id']] for x in grid if x['family']=='balanced32' and x['variant']==v and x['same_image']];choices.append({'variant':v,'tie_rank':rank,'A_hard_macro':sum(x['A_hard_correct'] for x in xs)/8,'A_margin_macro':sum(x['A_hard_margin'] for x in xs)/8})
sel=read(r/'selection.json');assert choices==sel['A_only_candidates'];assert max(choices,key=lambda x:(x['A_hard_macro'],x['A_margin_macro'],-x['tie_rank']))['variant']==sel['selected_variant'] and not sel['B_used_for_selection'] and sel['records_at_selection']==960
for x in new:
    if x['phase']=='confirmation' and x['role'].startswith('selected'):assert x['variant']==sel['selected_variant']
# Gate1 removes source dependence exactly; original mean-shift offset0 reproduces native.
fi={x:i for i,x in enumerate(nt['record_ids'])};gate1max=0.;zerooffsetmax=0.
for n in plan['training_images']:
    xs=[x for x in grid if x['variant']=='gate_1' and x['target_image']==n];feat=nt['image_features'][[fi[x['id']] for x in xs]];gate1max=max(gate1max,float((feat-feat[:1]).abs().max()))
    for donor in plan['training_images']:
        pair=[x for x in grid if x['family']=='original_cached' and x['donor_image']==donor and x['target_image']==n and x['variant'] in ['auto','mean_shift_auto']];assert len(pair)==2;zerooffsetmax=max(zerooffsetmax,float((nt['image_features'][fi[pair[0]['id']]]-nt['image_features'][fi[pair[1]['id']]]).abs().max()))
assert gate1max==0 and zerooffsetmax==0
# Rebuild every paired-image value and CI from independent frozen-score metrics.
lookup={}
for source,rows in [('new',new),('base',base)]:
    for x in rows:
        if source=='new' and x['phase']=='diagnostic_grid':continue
        m=allmetrics[source,x['id']]
        if source=='base':fam='initial_SFT30' if x['arm'].startswith('before') else 'balanced32';role='baseline_shared' if x['role']=='shared' else 'baseline_own'
        else:fam=x['family'];role=x['role']
        lookup[fam,role,m['target_image']]=m
contrasts={'selected_minus_native':('selected_own','baseline_own'),'selected_own_minus_shared':('selected_own','selected_shared'),'selected_minus_learned':('selected_own','learned_only'),'original_minus_native':('original_cached','baseline_own'),'selected_minus_original':('selected_own','original_cached')};ci=0
for s in analysis['statistics']:
    co,fam,tag,scope,metric=s['label'].split('/');names=s['image_names']
    def get(role,n):return lookup['original_cached' if role=='original_cached' else 'balanced32' if role=='learned_only' else fam,role,n][metric]
    vals=[get(contrasts[tag][0],n)-get(contrasts[tag][1],n) if tag in contrasts else get(tag,n) for n in names];assert np.allclose(vals,s['values'],atol=1e-12,rtol=0);v=np.asarray(vals);counts=np.asarray([by[n]['actual_count'] for n in names]);rng=np.random.default_rng((2026111601+int(hashlib.sha256(s['label'].encode()).hexdigest()[:8],16))%2**32)
    if scope=='macro':
        point=0.;boot=np.zeros(10000)
        for c in range(1,5):z=v[counts==c];assert len(z);point+=float(z.mean())/4;boot+=z[rng.integers(len(z),size=(10000,len(z)))].mean(axis=1)/4
        assert s['images']==sum(np.isin(counts,[1,2,3,4]))
    else:point=float(v.mean());boot=v[rng.integers(len(v),size=(10000,len(v)))].mean(axis=1)
    assert abs(point-s['mean'])<1e-12 and np.allclose(np.percentile(boot,[2.5,97.5]),s['interval95'],atol=1e-12,rtol=0);ci+=1
# Input-gradient records and continuous F32 finite-difference arithmetic.
grad=read(r/'gradient/records.json');gt=torch.load(r/'gradient/tensors.pt',map_location='cpu',weights_only=True);assert len(grad)==32
for i,x in enumerate(grad):
    assert torch.equal(gt['hidden'][i],pt['hidden'][ix[x['source_state_id']]].float()) and torch.isfinite(gt['gradient'][i]).all();assert abs(float(gt['gradient'][i].norm())-x['gradient_norm'])<1e-7;assert abs(float((gt['gradient'][i]*gt['direction'][i]).sum())-x['directional_derivative'])<1e-7;assert abs((x['margin_plus']-x['margin_minus'])/.2-x['fd_central_derivative'])<1e-12
    matches=[z for z in grid if z['family']==x['family'] and z['variant']==x['variant'] and z['donor_image']==x['image'] and z['same_image']];assert len(matches)==1 and abs(x['A_hard_margin']-matches[0]['A_hard_margin'])<1e-6
assert read(r/'gpu_result.json')['original_model_parameter_hash']==read(r/'gpu_result.json')['final_model_parameter_hash']==read(r/'gradient_result.json')['model_hash_unchanged']
g=analysis['groups']['confirmation55']['balanced32']['contrasts'];checks={'confirmation_B_hard_gain_CI_lower_positive':g['selected_minus_native']['macro']['B_hard_correct']['interval95'][0]>0,'confirmation_selected_own_minus_shared_B_hard_CI_lower_positive':g['selected_own_minus_shared']['macro']['B_hard_correct']['interval95'][0]>0,'confirmation_selected_own_minus_learned_B_hard_CI_lower_positive':g['selected_minus_learned']['macro']['B_hard_correct']['interval95'][0]>0,'confirmation_B_hard_margin_gain_CI_lower_positive':g['selected_minus_native']['macro']['B_hard_margin']['interval95'][0]>0,'immutable_model_and_original_sources_preserved':read(r/'final_source_audit.json')['ok']};assert checks==analysis['adoption']['checks'] and all(checks.values())==analysis['adoption']['passed']
proof={'ok':True,'source_states_verified':511,'projection_math_float64_max_errors':errs,'actual_vision_insertion_witnesses':4,'new_scores_verified':1470,'reused_baseline_scores_verified':340,'float64_quantity_dot_max_error':maxdot,'numerical_quantity_decision_changes':numerical,'val_ranks_verified':ranks,'bootstrap_intervals_rebuilt':ci,'A_only_candidate_selection_rebuilt':True,'gate1_source_independence_max_feature_error':gate1max,'zero_offset_replay_max_feature_error':zerooffsetmax,'input_gradient_witnesses_verified':32,'new_Qwen_generations':0,'optimizer_updates':0,'all_adoption_checks_rebuilt':True};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n');print('ALL_INJECTION_CALIBRATION_EVIDENCE_VERIFIED',ci,flush=True)
